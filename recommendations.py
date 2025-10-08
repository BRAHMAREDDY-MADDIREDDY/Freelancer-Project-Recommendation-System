import pandas as pd
import numpy as np
import tensorflow as tf
import random
import joblib
from typing import List, Dict
from sqlalchemy.orm import Session
from models import Recommendation
import logging
import os

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Layers for NCF
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs):
        query, key = inputs
        query = tf.expand_dims(query, 1)
        key = tf.expand_dims(key, 1)
        attention_output = self.mha(query=query, key=key, value=key)
        return tf.squeeze(attention_output, 1)

class Cast(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.cast(inputs, dtype=tf.float32)

def load_ncf_model_and_encoders(model_path: str, user_encoder_path: str, item_encoder_path: str):
    logger.info(colored("Loading NCF model...", "cyan"))
    try:
        with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer, 'Cast': Cast}):
            model = tf.keras.models.load_model(model_path)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()],
            run_eagerly=True
        )
        dummy_input = {
            'user_input': np.zeros((1,), dtype=np.int32),
            'item_input': np.zeros((1,), dtype=np.int32),
            'tfidf_input': np.zeros((1, 100), dtype=np.float32)
        }
        model.evaluate(dummy_input, np.zeros((1,)), verbose=0)
        logger.info(colored("NCF model loaded successfully", "green"))
    except Exception as e:
        logger.error(colored(f"Error loading NCF model: {e}", "red"))
        raise

    logger.info(colored("Loading encoders...", "cyan"))
    try:
        user_encoder = joblib.load(user_encoder_path)
        item_encoder = joblib.load(item_encoder_path)
        logger.info(colored("Encoders loaded successfully", "green"))
    except Exception as e:
        logger.error(colored(f"Error loading encoders: {e}", "red"))
        raise

    return model, user_encoder, item_encoder

def filter_and_encode_ids(df: pd.DataFrame, user_encoder, item_encoder):
    logger.info(colored("Filtering and encoding IDs...", "cyan"))
    valid_users = set(user_encoder.classes_)
    valid_items = set(item_encoder.classes_)
    df = df[df['freelancer_id'].isin(valid_users) & df['job_id'].isin(valid_items)].copy()
    logger.info(colored(f"Rows after filtering valid IDs: {len(df)}", "green"))

    if df.empty:
        logger.error(colored("No valid data after filtering", "red"))
        raise ValueError("No valid data after filtering")

    df['user_id'] = user_encoder.transform(df['freelancer_id'])
    df['item_id'] = item_encoder.transform(df['job_id'])

    return df

def get_popular_jobs(db: Session, df: pd.DataFrame, model, item_encoder, tfidf_cols, top_k=10, sample_users=20, batch_size=1024, seed=None):
    logger.info(colored("Fetching popular jobs using NCF...", "cyan"))
    # Removed database insertion logic
    # db.query(Recommendation).filter_by(user_id=0, model_type='popular').delete()
    # db.commit() is no longer needed

    if not {'job_id', 'job_title', 'freelancer_id'}.issubset(df.columns):
        logger.error(colored("NCF DataFrame missing required columns", "red"))
        if 'job_title' in df.columns:
            logger.warning(colored("Falling back to random jobs due to missing columns", "yellow"))
            random_jobs = df[['job_id', 'job_title']].dropna().sample(n=min(top_k, len(df)), random_state=seed)
            return [
                {'job_id': row['job_id'], 'job_title': row['job_title'], 'score': 0.0}
                for _, row in random_jobs.iterrows()
            ]
        return []

    valid_user_ids = np.unique(df['user_id'])
    if len(valid_user_ids) == 0:
        logger.warning(colored("No valid user IDs found, falling back to random jobs", "yellow"))
        if 'job_title' in df.columns:
            random_jobs = df[['job_id', 'job_title']].dropna().sample(n=min(top_k, len(df)), random_state=seed)
            return [
                {'job_id': row['job_id'], 'job_title': row['job_title'], 'score': 0.0}
                for _, row in random_jobs.iterrows()
            ]
        return []

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    sampled_users = np.random.choice(valid_user_ids, size=min(sample_users, len(valid_user_ids)), replace=False)
    all_items = np.unique(df['item_id'])
    tfidf_df = df[['item_id'] + tfidf_cols].drop_duplicates('item_id')
    tfidf_input_base = tfidf_df[tfidf_cols].values.astype(np.float32)

    job_scores = np.zeros(len(all_items))
    for user_id in sampled_users:
        user_input = np.array([user_id] * len(all_items), dtype=np.int32)
        item_input = all_items.astype(np.int32)
        tfidf_input = tfidf_input_base.copy()

        try:
            predictions = model.predict(
                {'user_input': user_input, 'item_input': item_input, 'tfidf_input': tfidf_input},
                batch_size=batch_size,
                verbose=0
            ).flatten()
            predictions = np.nan_to_num(predictions, nan=0.0)
            job_scores += predictions
        except Exception as e:
            logger.error(colored(f"Error predicting for user {user_id}: {e}", "red"))
            continue

    job_scores /= len(sampled_users)
    noise = np.random.normal(0, 0.01 * job_scores.std(), size=len(job_scores))
    job_scores += noise
    top_k_indices = np.argsort(-job_scores)[:top_k]
    top_k_items = all_items[top_k_indices]
    top_k_scores = job_scores[top_k_indices]
    top_k_job_ids = item_encoder.inverse_transform(top_k_items)

    recommendations = []
    for job_id, score in zip(top_k_job_ids, top_k_scores):
        job_title = df[df['job_id'] == job_id]['job_title'].iloc[0]
        recommendations.append({'job_id': job_id, 'job_title': job_title, 'score': float(score)})

    logger.info(colored(f"Generated {len(recommendations)} popular jobs", "green"))
    return recommendations

def recommend(user_id: int, db: Session, ncf_dataset_path: str, ncf_model_path: str, user_encoder_path: str, item_encoder_path: str, top_k: int = 10, seed=None) -> List[Dict]:
    logger.info(colored(f"Generating recommendations for user_id: {user_id}", "cyan"))
    try:
        # Load preprocessed dataset
        if ncf_dataset_path.endswith('.pkl'):
            df = pd.read_pickle(ncf_dataset_path)
            tfidf_cols_path = ncf_dataset_path.replace('.pkl', '_tfidf_cols.pkl')
            if os.path.exists(tfidf_cols_path):
                tfidf_cols = joblib.load(tfidf_cols_path)
                logger.info(colored(f"Loaded preprocessed NCF dataset with {len(df)} rows", "green"))
            else:
                raise FileNotFoundError(f"TF-IDF columns file not found at {tfidf_cols_path}")
        else:
            raise ValueError(f"Expected .pkl file, got {ncf_dataset_path}")
    except Exception as e:
        logger.error(colored(f"Failed to load preprocessed dataset: {e}", "red"))
        # Fallback to random jobs if preprocessed file is unavailable
        df = pd.read_csv(ncf_dataset_path, low_memory=False, encoding='utf-8') if os.path.exists(ncf_dataset_path.replace('.pkl', '.csv')) else pd.DataFrame()
        if 'job_title' in df.columns:
            logger.warning(colored("Falling back to random jobs due to dataset error", "yellow"))
            random_jobs = df[['job_id', 'job_title']].dropna().sample(n=min(top_k, len(df)), random_state=seed)
            return [
                {'job_id': row['job_id'], 'job_title': row['job_title'], 'score': 0.0}
                for _, row in random_jobs.iterrows()
            ]
        return []

    # Load NCF model and encoders
    model, user_encoder, item_encoder = load_ncf_model_and_encoders(ncf_model_path, user_encoder_path, item_encoder_path)

    # Check for existing recommendations in the database
    existing_recs = db.query(Recommendation).filter_by(user_id=user_id).all()
    if existing_recs:
        logger.info(colored(f"Found {len(existing_recs)} existing recommendations for user_id: {user_id}", "green"))
        recommendations = [
            {'job_id': rec.job_id, 'job_title': rec.job_title, 'score': float(rec.score)}
            for rec in existing_recs
        ]
        # Limit to top_k if more recommendations exist
        return recommendations[:top_k]

    # If no existing recommendations, generate popular jobs
    df = filter_and_encode_ids(df, user_encoder, item_encoder)
    recommendations = get_popular_jobs(db, df, model, item_encoder, tfidf_cols, top_k, seed=seed)

    # Round scores to three decimal places
    for rec in recommendations:
        rec['score'] = round(rec['score'], 3)

    logger.info(colored(f"Returning {len(recommendations)} popular jobs", "green"))
    return recommendations
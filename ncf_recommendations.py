import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from sqlalchemy.orm import Session
from models import Recommendation
import logging
import random

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Layers
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

def load_ncf_model_and_encoders(model_path, user_encoder_path, item_encoder_path):
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
        logger.info(colored("Model loaded successfully", "green"))
    except Exception as e:
        logger.error(colored(f"Error loading model: {e}", "red"))
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

def load_preprocessed_dataset(dataset_path):
    logger.info(colored("Loading preprocessed dataset...", "cyan"))
    try:
        if dataset_path.endswith('.pkl'):
            df = pd.read_pickle(dataset_path)
            tfidf_cols_path = dataset_path.replace('.pkl', '_tfidf_cols.pkl')
            if os.path.exists(tfidf_cols_path):
                tfidf_cols = joblib.load(tfidf_cols_path)
                logger.info(colored(f"Loaded preprocessed NCF dataset with {len(df)} rows", "green"))
            else:
                raise FileNotFoundError(f"TF-IDF columns file not found at {tfidf_cols_path}")
        else:
            raise ValueError(f"Expected .pkl file, got {dataset_path}")
    except Exception as e:
        logger.error(colored(f"Failed to load preprocessed dataset: {e}", "red"))
        raise
    return df, tfidf_cols

def filter_and_encode_ids(df, user_encoder, item_encoder):
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

def prepare_inference_inputs(df, user_id, completed_job_ids, tfidf_cols, item_encoder, user_encoder):
    # Use a random user_id if the provided one is not in the encoder
    try:
        encoded_user_id = user_encoder.transform([str(user_id)])[0]
    except ValueError:
        logger.warning(colored(f"User ID {user_id} not in user_encoder classes. Using a random valid user_id.", "yellow"))
        valid_user_ids = df['user_id'].unique()
        if len(valid_user_ids) == 0:
            raise ValueError("No valid user IDs available in dataset")
        encoded_user_id = np.random.choice(valid_user_ids)
        logger.info(colored(f"Fallback user_id: {encoded_user_id}", "yellow"))

    if encoded_user_id not in df['user_id'].values:
        raise ValueError(f"Encoded User ID {encoded_user_id} not in dataset")

    # Ignore completed_job_ids and use all available items
    all_items = np.unique(df['item_id'])
    user_input = np.array([encoded_user_id] * len(all_items), dtype=np.int32)
    item_input = all_items.astype(np.int32)

    tfidf_df = df[['item_id'] + tfidf_cols].drop_duplicates('item_id')
    tfidf_input = tfidf_df[tfidf_cols].values.astype(np.float32)

    if len(user_input) != len(item_input) or len(user_input) != len(tfidf_input):
        raise ValueError("Input shape mismatch")

    return user_input, item_input, tfidf_input

def generate_ncf_recommendations(model, df, user_id, completed_job_ids, item_encoder, tfidf_cols, user_encoder, top_k=10, batch_size=1024, db: Session = None):
    logger.info(colored("Generating recommendations...", "cyan"))
    user_input, item_input, tfidf_input = prepare_inference_inputs(df, user_id, completed_job_ids, tfidf_cols, item_encoder, user_encoder)

    try:
        predictions = model.predict(
            {'user_input': user_input, 'item_input': item_input, 'tfidf_input': tfidf_input},
            batch_size=batch_size,
            verbose=0
        ).flatten()
    except Exception as e:
        logger.error(colored(f"Error during prediction: {e}", "red"))
        raise

    if np.any(np.isnan(predictions)):
        valid_indices = ~np.isnan(predictions)
        predictions = predictions[valid_indices]
        item_input = item_input[valid_indices]
        tfidf_input = tfidf_input[valid_indices]
        if len(predictions) == 0:
            logger.error(colored("No valid predictions after filtering NaN", "red"))
            raise ValueError("No valid predictions after filtering NaN")

    # Add random noise to predictions
    noise = np.random.normal(0, 0.01, size=predictions.shape)
    perturbed_predictions = predictions + noise

    top_k_indices = np.argsort(-perturbed_predictions)[:top_k]
    top_k_items = item_input[top_k_indices]
    top_k_scores = predictions[top_k_indices]
    top_k_job_ids = item_encoder.inverse_transform(top_k_items)

    recommendations = []
    for job_id, score in zip(top_k_job_ids, top_k_scores):
        job_title = df[df['job_id'] == job_id]['job_title'].iloc[0]
        recommendations.append({
            'job_id': job_id,
            'job_title': job_title,
            'score': float(score)
        })

    # Save to recommendations table if db session is provided
    if db is not None:
        try:
            for rec in recommendations:
                db_rec = Recommendation(user_id=user_id, model_type='interaction', job_id=rec['job_id'], job_title=rec['job_title'], score=str(rec['score']))
                db.add(db_rec)
            db.commit()
            logger.info(colored(f"Saved {len(recommendations)} interaction-based recommendations for user_id {user_id}", "green"))
        except Exception as e:
            logger.error(colored(f"Failed to save recommendations: {e}", "red"))
            db.rollback()
            raise

    logger.info(colored(f"Generated {len(recommendations)} interaction-based recommendations", "green"))
    return recommendations

def get_interaction_based_recommendations(user_id, completed_job_ids, dataset_path, ncf_model_path, user_encoder_path, item_encoder_path, db: Session, top_k=10, batch_size=1024):
    logger.info(colored(f"Generating interaction-based recommendations for user_id: {user_id}", "cyan"))
    try:
        # Load model and encoders
        model, user_encoder, item_encoder = load_ncf_model_and_encoders(ncf_model_path, user_encoder_path, item_encoder_path)

        # Load preprocessed dataset
        df, tfidf_cols = load_preprocessed_dataset(dataset_path)

        # Filter and encode IDs
        df = filter_and_encode_ids(df, user_encoder, item_encoder)

        # Generate recommendations
        recommendations = generate_ncf_recommendations(model, df, user_id, completed_job_ids, item_encoder, tfidf_cols, user_encoder, top_k, batch_size, db)

        return recommendations

    except Exception as e:
        logger.error(colored(f"Error in recommendation pipeline: {e}", "red"))
        raise
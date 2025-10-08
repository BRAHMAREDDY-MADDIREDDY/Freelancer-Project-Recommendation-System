import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from transformers import DistilBertTokenizerFast, TFDistilBertModel
import pickle
import os
from config import DATASET_PATHS, MODEL_PATHS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache
_tokenizer = None
_model = None
_jobs_df = None
_job_vecs = None

# --- Load BERT Model ---
def load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info("Loading DistilBERT model and tokenizer")
        _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATHS["bert_tokenizer"])
        _model = TFDistilBertModel.from_pretrained(MODEL_PATHS["bert_model"])
    return _tokenizer, _model

# --- Text Embedding ---
def get_embedding(text, tokenizer, model):
    if not text.strip():
        logger.warning("Empty text provided for embedding, returning zero vector")
        return np.zeros((1, 768))  # Return zero vector for empty text
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return embedding.numpy()

# --- Load Jobs CSV ---
def load_jobs():
    global _jobs_df
    if _jobs_df is None:
        logger.info("Loading job datasets")
        train_df = pd.read_csv(DATASET_PATHS["train_dataset"], dtype={"projectId": str})
        test_df = pd.read_csv(DATASET_PATHS["test_dataset"], dtype={"projectId": str})
        _jobs_df = pd.concat([train_df, test_df], ignore_index=True)
        if "description" not in _jobs_df.columns:
            logger.warning("Description column missing, using job_title as fallback")
            _jobs_df["description"] = _jobs_df["job_title"].fillna("")
        # Preprocess tags and description to lowercase
        _jobs_df["tags_clean"] = _jobs_df["tags"].apply(
            lambda x: [t.strip().lower() for t in x] if isinstance(x, list) else 
                      [t.strip().lower() for t in x.split(",")] if isinstance(x, str) else []
        )
        _jobs_df["description"] = _jobs_df["description"].apply(lambda x: x.lower() if isinstance(x, str) else "")
        _jobs_df = _jobs_df[_jobs_df["description"].str.strip() != ""]
        # Ensure projectId is string
        _jobs_df["projectId"] = _jobs_df["projectId"].astype(str)
        logger.info(f"Loaded {len(_jobs_df)} jobs")
    return _jobs_df

# --- Load Precomputed Job Vectors ---
def load_job_vecs():
    global _job_vecs
    if _job_vecs is None:
        cache_path = "./data/job_vecs.pkl"
        if os.path.exists(cache_path):
            logger.info(f"Loading precomputed job vectors from {cache_path}")
            with open(cache_path, "rb") as f:
                _job_vecs = pickle.load(f)
            logger.info(f"Loaded {len(_job_vecs)} job vectors")
        else:
            logger.error(f"Precomputed job embeddings not found at {cache_path}")
            raise FileNotFoundError(f"Precomputed job embeddings not found at {cache_path}")
    return _job_vecs

# --- Fast recommendation using precomputed job_vecs ---
def recommend(profile_text, jobs_df, job_vecs, top_n=5, selected_tags=None):
    logger.info(f"Generating recommendations for profile_text: {profile_text}")
    tokenizer, model = load_model()
    profile_vec = get_embedding(profile_text.lower(), tokenizer, model)
    profile_vec = torch.tensor(profile_vec, dtype=torch.float32)
    job_vecs = torch.tensor(job_vecs, dtype=torch.float32)
    
    scores = torch.nn.functional.cosine_similarity(profile_vec, job_vecs, dim=1).numpy()
    jobs_df = jobs_df.copy()
    jobs_df["score"] = scores
    
    logger.info(f"Computed cosine similarity scores, max score: {scores.max():.3f}, min score: {scores.min():.3f}")
    
    filtered_jobs_df = jobs_df
    if selected_tags:
        selected_tags = [tag.lower().strip() for tag in selected_tags]
        logger.info(f"Filtering jobs with tags: {selected_tags}")
        filtered_jobs_df = jobs_df[
            jobs_df["tags_clean"].apply(
                lambda x: any(tag in x for tag in selected_tags) if isinstance(x, list) else False
            )
        ]
        logger.info(f"Number of jobs after tag filtering: {len(filtered_jobs_df)}")
    
    if filtered_jobs_df.empty and selected_tags:
        logger.warning("No jobs matched the selected tags, returning recommendations without tag filtering")
        filtered_jobs_df = jobs_df
    
    if filtered_jobs_df.empty:
        logger.warning("No jobs available after filtering")
        return pd.DataFrame()
    
    recommendations = filtered_jobs_df.sort_values("score", ascending=False).head(top_n)[["projectId", "job_title", "description", "score", "tags"]]
    logger.info(f"Returning {len(recommendations)} recommendations")
    return recommendations

# --- Initialize global variables (to be called once) ---
def initialize_recommendation_system():
    load_model()
    load_jobs()
    load_job_vecs()

if __name__ == "__main__":
    initialize_recommendation_system()
    profile_text = "python, data analysis, machine learning"
    selected_tags = ["python", "machine learning"]
    recommendations = recommend(profile_text, _jobs_df, _job_vecs, top_n=5, selected_tags=selected_tags)
    print(recommendations[["projectId", "job_title", "description", "score", "tags"]])
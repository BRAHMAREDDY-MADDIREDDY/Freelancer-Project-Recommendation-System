import json
from uuid import uuid4
import docx
from fastapi import Depends, FastAPI, File, HTTPException, Form, UploadFile
from pydantic import BaseModel, EmailStr
from requests import Session
from database.db import SessionLocal, engine
import database.crud as crud
import models
from recommendations import recommend
from content_recommendations import recommend as content_recommend, load_jobs, load_job_vecs, load_model
from config import MODEL_PATHS, DATASET_PATHS
from ncf_recommendations import generate_ncf_recommendations, get_interaction_based_recommendations, load_ncf_model_and_encoders, load_preprocessed_dataset, filter_and_encode_ids
from models import Recommendation, Feedback, Watchlist
from datetime import datetime
import logging
import pandas as pd
from io import StringIO, BytesIO
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import process
import tensorflow as tf
import torch
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Cache for jobs and vectors
_jobs_cache = None
_job_vecs_cache = None

# Initialize recommendation system globally
try:
    load_model()
    _jobs_cache = load_jobs()
    _job_vecs_cache = load_job_vecs()
    logger.info(f"Cached {len(_jobs_cache)} jobs and {len(_job_vecs_cache)} job vectors")
except Exception as e:
    logger.error(f"Failed to initialize recommendation system: {str(e)}")
    raise Exception(f"Initialization failed: {str(e)}")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    first_name: str
    last_name: str

class UserLogin(BaseModel):
    username: str
    password: str

class FeedbackCreate(BaseModel):
    job_id: str
    feedback: str
    timestamp: str

class WatchlistCreate(BaseModel):
    job_id: str
    job_title: str

class ContentInput(BaseModel):
    skills: str
    job_description: str = ""

class TagsInput(BaseModel):
    tags: str

# Load NLP model and skill list globally
nlp = spacy.load("en_core_web_sm")
with open("skills_list.txt", encoding="utf-8") as f:
    skills_gazetteer = [line.strip().lower() for line in f if line.strip()]

def extract_text_from_file(uploaded_file):
    ext = uploaded_file.filename.split('.')[-1].lower()
    content = uploaded_file.file.read()
    uploaded_file.file.seek(0)
    if ext == "pdf":
        try:
            import fitz
            pdf = fitz.open(stream=content, filetype="pdf")
            return "".join(page.get_text() for page in pdf)
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return ""
    elif ext == "docx":
        try:
            docf = docx.Document(BytesIO(content))
            return "\n".join([para.text for para in docf.paragraphs])
        except Exception as e:
            logger.error(f"Error processing DOCX: {str(e)}")
            return ""
    elif ext == "txt":
        return content.decode("utf-8", errors='replace')
    logger.warning(f"Unsupported file type: {ext}")
    return ""

def extract_skills_enhanced(text):
    if not text or all(c in "•\n\t " for c in text.strip()):
        return []
    text = text[:10000] if len(text) > 10000 else text
    start_time = datetime.now()
    doc = nlp(text)
    logger.info(f"NLP processing time: {(datetime.now() - start_time).total_seconds():.2f}s")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skills_gazetteer if len(skill.split()) <= 3]
    matcher.add("SKILL", patterns)
    matches = matcher(doc)
    matcher_skills = set([doc[start:end].text.lower() for _, start, end in matches])
    ner_skills = set(ent.text.strip().lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "SKILL"])
    found_gazetteer = set()
    words = [w for w in text.lower().split() if w.strip()]
    start_time = datetime.now()
    for n in [1, 2]:
        for i in range(len(words) - n + 1):
            ng = " ".join(words[i:i + n])
            match, score = process.extractOne(ng, skills_gazetteer)
            if score and score > 95:
                found_gazetteer.add(match)
    logger.info(f"Fuzzy matching time: {(datetime.now() - start_time).total_seconds():.2f}s")
    processed = " ".join([token.lemma_.lower() for token in doc if not token.is_stop and token.text.isalpha()])
    if processed.strip():
        start_time = datetime.now()
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        X = vectorizer.fit_transform([processed])
        features = vectorizer.get_feature_names_out()
        scores = X.toarray()[0]
        tfidf_terms = set(term for term, score in zip(features, scores) if score > 0.1)
        logger.info(f"TF-IDF time: {(datetime.now() - start_time).total_seconds():.2f}s")
    else:
        tfidf_terms = set()
    all_skills = list(found_gazetteer | ner_skills | matcher_skills | tfidf_terms)
    nice_skills = [s for s in all_skills if len(s) > 2]
    logger.info(f"Total skill extraction time: {(datetime.now() - start_time).total_seconds():.2f}s")
    return sorted(set(nice_skills), key=lambda x: nice_skills.index(x))[:15]

# Health check
@app.get("/")
def root():
    return {"message": "Freelancer Job Recommendation API is running!"}

# Get available tags
@app.get("/tags")
def get_available_tags():
    try:
        jobs_df = _jobs_cache
        available_tags = sorted(set(tag.strip().lower() for sublist in jobs_df['tags'].dropna().str.split(",") for tag in sublist if tag.strip()))
        logger.info(f"Fetched {len(available_tags)} unique tags")
        return {"tags": available_tags}
    except Exception as e:
        logger.error(f"Exception in get_available_tags: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Signup
@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    if crud.get_user_by_username(db, user.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    if crud.get_user_by_email(db, user.email):
        raise HTTPException(status_code=400, detail="Email already exists")
    new_user = crud.create_user(db, user.username, user.email, user.password, user.first_name, user.last_name)
    return {"message": "User created successfully", "user_id": new_user.id, "username": new_user.username, "first_name": new_user.first_name, "last_name": new_user.last_name}

# Login
@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, user.username)
    if not db_user or not crud.verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": f"Welcome {db_user.first_name}", "user_id": db_user.id, "first_name": db_user.first_name, "last_name": db_user.last_name}

# Popular jobs recommendation endpoint
@app.get("/recommend/{user_id}")
def recommend_endpoint(user_id: int, db: Session = Depends(get_db)):
    try:
        recs = recommend(
            user_id=user_id,
            db=db,
            ncf_dataset_path=DATASET_PATHS["ncf_dataset"],
            ncf_model_path=MODEL_PATHS["ncf_model"],
            user_encoder_path=MODEL_PATHS["user_encoder"],
            item_encoder_path=MODEL_PATHS["item_encoder"],
            top_k=10
        )
        adjusted_recs = [
            {
                "job_id": str(rec["job_id"]),
                "job_title": rec["job_title"],
                "score": float(rec["score"])
            } for rec in recs
        ]
        for rec in adjusted_recs:
            logger.debug(f"Checking existing NCF recommendation for user_id={user_id}, job_id={rec['job_id']}")
            existing_rec = db.query(Recommendation).filter(
                Recommendation.user_id == user_id,
                Recommendation.job_id == str(rec["job_id"]),
                Recommendation.model_type == "ncf"
            ).first()
            if not existing_rec:
                new_rec = Recommendation(
                    user_id=user_id,
                    job_id=str(rec["job_id"]),
                    job_title=rec["job_title"],
                    score=float(rec["score"]),
                    model_type="ncf",
                    created_at=datetime.utcnow()
                )
                db.add(new_rec)
        db.commit()
        return {"recommendations": adjusted_recs}
    except Exception as e:
        logger.error(f"Exception in recommend_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Interaction-based (NCF) recommendation endpoint
@app.get("/recommend/interaction/{user_id}")
async def get_interaction_recommendations(user_id: int, completed_job_ids: str, db: Session = Depends(get_db)):
    try:
        completed_jobs = [str(job_id.strip()) for job_id in completed_job_ids.split(",") if job_id.strip()]
        if not completed_jobs:
            raise HTTPException(status_code=400, detail="No completed jobs provided")
        recommendations = get_interaction_based_recommendations(
            user_id=user_id,
            completed_job_ids=completed_jobs,
            dataset_path=DATASET_PATHS["ncf_dataset"],
            ncf_model_path=MODEL_PATHS["ncf_model"],
            user_encoder_path=MODEL_PATHS["user_encoder"],
            item_encoder_path=MODEL_PATHS["item_encoder"],
            db=db
        )
        adjusted_recs = [
            {
                "job_id": str(rec["job_id"]),
                "job_title": rec["job_title"],
                "score": float(rec["score"])
            } for rec in recommendations
        ]
        for rec in adjusted_recs:
            logger.debug(f"Checking existing NCF recommendation for user_id={user_id}, job_id={rec['job_id']}")
            existing_rec = db.query(Recommendation).filter(
                Recommendation.user_id == user_id,
                Recommendation.job_id == str(rec["job_id"]),
                Recommendation.model_type == "ncf"
            ).first()
            if not existing_rec:
                new_rec = Recommendation(
                    user_id=user_id,
                    job_id=str(rec["job_id"]),
                    job_title=rec["job_title"],
                    score=float(rec["score"]),
                    model_type="ncf",
                    created_at=datetime.utcnow()
                )
                db.add(new_rec)
        db.commit()
        return {"recommendations": adjusted_recs}
    except ValueError as e:
        logger.error(f"ValueError in get_interaction_recommendations: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Exception in get_interaction_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Fetch last 10 recommendations
@app.get("/recommend/past/{user_id}")
def get_past_recommendations(user_id: int, db: Session = Depends(get_db)):
    try:
        existing_recs = db.query(Recommendation).filter(Recommendation.user_id == user_id).order_by(Recommendation.created_at.desc()).limit(10).all()
        recommendations = [
            {
                "job_id": str(rec.job_id),
                "job_title": rec.job_title,
                "score": float(rec.score),
                "model_type": rec.model_type,
                "created_at": rec.created_at.isoformat()
            } for rec in existing_recs
        ]
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Exception in get_past_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Feedback endpoint (POST)
@app.post("/feedback/{user_id}")
def save_feedback(user_id: int, feedback: FeedbackCreate, db: Session = Depends(get_db)):
    try:
        existing_feedback = db.query(Feedback).filter(Feedback.user_id == user_id, Feedback.job_id == str(feedback.job_id)).first()
        if existing_feedback:
            raise HTTPException(status_code=400, detail="Feedback already exists for this job")
        if feedback.feedback not in ["like", "dislike"]:
            raise HTTPException(status_code=400, detail="Feedback must be 'like' or 'dislike'")
        new_feedback = Feedback(
            user_id=user_id,
            job_id=str(feedback.job_id),
            feedback=feedback.feedback,
            timestamp=datetime.fromisoformat(feedback.timestamp)
        )
        db.add(new_feedback)
        db.commit()
        db.refresh(new_feedback)
        logger.info(f"Feedback saved for user {user_id}, job {feedback.job_id}: {feedback.feedback}")
        return {"message": "Feedback saved successfully", "feedback_id": new_feedback.id}
    except ValueError as e:
        logger.error(f"ValueError in save_feedback: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Exception in save_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# DELETE endpoint to remove feedback
@app.delete("/feedback/{user_id}/{job_id}")
def delete_feedback_endpoint(user_id: int, job_id: str, db: Session = Depends(get_db)):
    try:
        feedback_to_delete = db.query(Feedback).filter(Feedback.user_id == user_id, Feedback.job_id == str(job_id)).first()
        if not feedback_to_delete:
            raise HTTPException(status_code=404, detail="Feedback not found")
        db.delete(feedback_to_delete)
        db.commit()
        logger.info(f"Feedback removed for user {user_id}, job {job_id}")
        return {"message": "Feedback removed successfully"}
    except Exception as e:
        logger.error(f"Exception in delete_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Get liked and disliked jobs
@app.get("/feedback/{user_id}")
def get_user_feedback(user_id: int, db: Session = Depends(get_db)):
    try:
        feedbacks = db.query(Feedback).filter(Feedback.user_id == user_id).all()
        liked_jobs = [str(feedback.job_id) for feedback in feedbacks if feedback.feedback == "like"]
        disliked_jobs = [str(feedback.job_id) for feedback in feedbacks if feedback.feedback == "dislike"]
        return {
            "liked_jobs": liked_jobs,
            "disliked_jobs": disliked_jobs
        }
    except Exception as e:
        logger.error(f"Exception in get_user_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Watchlist endpoints
@app.post("/watchlist/{user_id}")
def add_to_watchlist(user_id: int, watchlist_item: WatchlistCreate, db: Session = Depends(get_db)):
    try:
        existing_watchlist = db.query(Watchlist).filter(Watchlist.user_id == user_id, Watchlist.job_id == str(watchlist_item.job_id)).first()
        if existing_watchlist:
            raise HTTPException(status_code=400, detail="Job already in watchlist")
        new_watchlist = Watchlist(
            user_id=user_id,
            job_id=str(watchlist_item.job_id),
            job_title=watchlist_item.job_title
        )
        db.add(new_watchlist)
        db.commit()
        db.refresh(new_watchlist)
        logger.info(f"Added to watchlist for user {user_id}, job {watchlist_item.job_id}")
        return {"message": "Added to watchlist successfully", "watchlist_id": new_watchlist.id}
    except Exception as e:
        logger.error(f"Exception in add_to_watchlist: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/watchlist/{user_id}/{job_id}")
def remove_from_watchlist(user_id: int, job_id: str, db: Session = Depends(get_db)):
    try:
        watchlist_item = db.query(Watchlist).filter(Watchlist.user_id == user_id, Watchlist.job_id == str(job_id)).first()
        if not watchlist_item:
            raise HTTPException(status_code=404, detail="Job not found in watchlist")
        db.delete(watchlist_item)
        db.commit()
        logger.info(f"Removed from watchlist for user {user_id}, job {job_id}")
        return {"message": "Removed from watchlist successfully"}
    except Exception as e:
        logger.error(f"Exception in remove_from_watchlist: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/watchlist/{user_id}")
def get_watchlist(user_id: int, db: Session = Depends(get_db)):
    try:
        watchlist_items = db.query(Watchlist).filter(Watchlist.user_id == user_id).all()
        watchlist = [{"job_id": str(item.job_id), "job_title": item.job_title, "added_at": item.added_at.isoformat()} for item in watchlist_items]
        return {"watchlist": watchlist}
    except Exception as e:
        logger.error(f"Exception in get_watchlist: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Content-based recommendation endpoint (manual input)
@app.post("/recommend/content/{user_id}")
def get_content_based_recommendations_manual(user_id: int, input_data: ContentInput, db: Session = Depends(get_db)):
    try:
        jobs_df = _jobs_cache
        job_vecs = _job_vecs_cache
        tokenizer, model = load_model()
        profile_text = input_data.skills
        if input_data.job_description:
            profile_text += " " + input_data.job_description
        if not profile_text.strip():
            raise HTTPException(status_code=400, detail="No profile text provided")
        selected_tags = [tag.strip() for tag in input_data.skills.split(",") if tag.strip()]
        logger.info(f"Input profile_text: {profile_text}")
        logger.info(f"Selected tags: {selected_tags}")
        logger.info(f"Number of jobs before filtering: {len(jobs_df)}")
        recommendations_df = content_recommend(
            profile_text=profile_text,
            jobs_df=jobs_df,
            job_vecs=job_vecs,
            top_n=10,
            selected_tags=selected_tags
        )
        logger.info(f"Number of recommendations after filtering: {len(recommendations_df)}")
        if recommendations_df.empty:
            logger.warning("No recommendations found after tag filtering")
            recommendations_df = content_recommend(
                profile_text=profile_text,
                jobs_df=jobs_df,
                job_vecs=job_vecs,
                top_n=10,
                selected_tags=None
            )
            logger.info(f"Number of recommendations without tag filtering: {len(recommendations_df)}")
            if recommendations_df.empty:
                logger.error("No jobs match the provided skills or description")
                raise HTTPException(
                    status_code=404,
                    detail="No jobs match the provided skills or description. Try broader skills like 'python, data science' or check the job dataset for relevant tags."
                )
        recommendations = [
            {
                "job_id": str(row["projectId"]),
                "job_title": row["job_title"],
                "description": row["description"],
                "score": float(row["score"])
            } for _, row in recommendations_df.iterrows()
        ]
        for rec in recommendations:
            logger.debug(f"Checking existing recommendation for user_id={user_id}, job_id={rec['job_id']}")
            existing_rec = db.query(Recommendation).filter(
                Recommendation.user_id == user_id,
                Recommendation.job_id == str(rec["job_id"]),
                Recommendation.model_type == "content-based"
            ).first()
            if not existing_rec:
                new_rec = Recommendation(
                    user_id=user_id,
                    job_id=str(rec["job_id"]),
                    job_title=rec["job_title"],
                    score=float(rec["score"]),
                    model_type="content-based",
                    created_at=datetime.utcnow()
                )
                db.add(new_rec)
        db.commit()
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Exception in get_content_based_recommendations_manual: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Content-based recommendation endpoint (batch input)
@app.post("/recommend/content/batch/{user_id}")
async def get_content_based_recommendations_batch(user_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        content = await file.read()
        ext = file.filename.split('.')[-1].lower()
        if ext == "csv":
            df = pd.read_csv(StringIO(content.decode("utf-8")))
            required_columns = ["skills", "job_description"]
            if not all(col in df.columns for col in required_columns):
                raise HTTPException(status_code=400, detail=f"CSV must contain {required_columns} columns")
            profile_text = " ".join(df["skills"].astype(str)) + " " + " ".join(df["job_description"].astype(str))
        elif ext == "json":
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, list):
                data = data[0]
            profile_text = data.get("skills", "") + " " + data.get("job_description", "")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV or JSON")
        if not profile_text.strip():
            raise HTTPException(status_code=400, detail="No profile text provided in file")
        jobs_df = _jobs_cache
        job_vecs = _job_vecs_cache
        selected_tags = [tag.strip() for tag in profile_text.split(",") if tag.strip()]
        recommendations_df = content_recommend(
            profile_text=profile_text,
            jobs_df=jobs_df,
            job_vecs=job_vecs,
            top_n=10,
            selected_tags=selected_tags
        )
        if recommendations_df.empty:
            recommendations_df = content_recommend(
                profile_text=profile_text,
                jobs_df=jobs_df,
                job_vecs=job_vecs,
                top_n=10,
                selected_tags=None
            )
            if recommendations_df.empty:
                raise HTTPException(
                    status_code=404,
                    detail="No jobs match the provided skills or description. Try broader skills or check the job dataset."
                )
        recommendations = [
            {
                "job_id": str(row["projectId"]),
                "job_title": row["job_title"],
                "description": row["description"],
                "score": float(row["score"])
            } for _, row in recommendations_df.iterrows()
        ]
        for rec in recommendations:
            existing_rec = db.query(Recommendation).filter(
                Recommendation.user_id == user_id,
                Recommendation.job_id == str(rec["job_id"]),
                Recommendation.model_type == "content-based"
            ).first()
            if not existing_rec:
                new_rec = Recommendation(
                    user_id=user_id,
                    job_id=str(rec["job_id"]),
                    job_title=rec["job_title"],
                    score=float(rec["score"]),
                    model_type="content-based",
                    created_at=datetime.utcnow()
                )
                db.add(new_rec)
        db.commit()
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Exception in get_content_based_recommendations_batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Jobs and vectors endpoint for Job Similarity Explorer
@app.get("/jobs")
def get_jobs_and_vectors(page: int = 1, page_size: int = 1000):
    try:
        jobs_df = _jobs_cache
        job_vecs = _job_vecs_cache
        total_jobs = len(jobs_df)
        
        # Validate pagination parameters
        if page < 1:
            raise HTTPException(status_code=400, detail="Page number must be positive")
        if page_size < 1:
            raise HTTPException(status_code=400, detail="Page size must be positive")
        
        # Calculate pagination indices
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_jobs)
        
        if start_idx >= total_jobs:
            raise HTTPException(status_code=404, detail="Page out of range")
        
        # Slice jobs and vectors
        paginated_jobs_df = jobs_df.iloc[start_idx:end_idx]
        paginated_job_vecs = job_vecs[start_idx:end_idx]
        
        logger.info(f"Fetched {len(paginated_jobs_df)} jobs and {len(paginated_job_vecs)} job vectors for page {page}")
        
        # Select only required columns to reduce data transfer
        jobs = paginated_jobs_df[["projectId", "job_title", "description", "tags"]].to_dict(orient="records")
        for job in jobs:
            job["job_id"] = str(job["projectId"])  # Ensure job_id is string
            job["tags"] = str(job["tags"]) if pd.notna(job["tags"]) else ""  # Handle NaN tags
        
        # Convert job_vecs to a list of lists for JSON serialization
        job_vectors = paginated_job_vecs.tolist()
        
        return {
            "jobs": jobs,
            "job_vectors": job_vectors,
            "page": page,
            "page_size": page_size,
            "total_jobs": total_jobs,
            "total_pages": (total_jobs + page_size - 1) // page_size
        }
    except Exception as e:
        logger.error(f"Exception in get_jobs_and_vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Tag-based filtering endpoint
@app.post("/filter/tags")
def filter_jobs_by_tags(tags_input: TagsInput, db: Session = Depends(get_db)):
    try:
        jobs_df = _jobs_cache
        selected_tags = [tag.strip().lower() for tag in tags_input.tags.split(",") if tag.strip()]
        if not selected_tags:
            raise HTTPException(status_code=400, detail="No tags provided")
        filtered_jobs = jobs_df[jobs_df['tags'].str.lower().str.contains('|'.join(selected_tags), na=False)]
        if filtered_jobs.empty:
            raise HTTPException(status_code=404, detail="No jobs found matching the selected tags")
        recommendations = [
            {
                "job_id": str(row["projectId"]),
                "job_title": row["job_title"],
                "description": row["description"],
                "score": 1.0  # Static score since filtering is binary
            } for _, row in filtered_jobs.iterrows()
        ]
        logger.info(f"Filtered {len(recommendations)} jobs by tags: {selected_tags}")
        return {"jobs": recommendations}
    except Exception as e:
        logger.error(f"Exception in filter_jobs_by_tags: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Profile skill extraction endpoint
@app.post("/profile/extract/{user_id}")
async def extract_profile_skills_endpoint(user_id: int, file: UploadFile = File(None), text: str = Form(None), db: Session = Depends(get_db)):
    try:
        profile_text = ""
        if file:
            profile_text = extract_text_from_file(file)
        elif text:
            profile_text = text
        else:
            raise HTTPException(status_code=400, detail="No file or text provided")
        if not profile_text.strip():
            raise HTTPException(status_code=400, detail="No valid text extracted from input")
        skills = extract_skills_enhanced(profile_text)
        if not skills:
            raise HTTPException(status_code=404, detail="No skills extracted from the provided input")
        return {"skills": skills}
    except Exception as e:
        logger.error(f"Exception in extract_profile_skills_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Float, func, ForeignKey
from database.db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)
    job_id = Column(String(50), nullable=False)
    job_title = Column(String(255), nullable=False)
    score = Column(Float, nullable=False)  # Changed to Float
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    job_id = Column(String(50), nullable=False)  # Added length constraint
    feedback = Column(String(50), nullable=False)  # Added length constraint for consistency
    timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

class Watchlist(Base):
    __tablename__ = "watchlist"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    job_id = Column(String(50), nullable=False)
    job_title = Column(String(255), nullable=False)
    added_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
"""Request/Response models with validation"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List


class UploadAndChunkRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=500)
    
    @validator('session_id')
    def session_id_must_be_uuid_like(cls, v):
        # Basic validation - allow alphanumeric, hyphens, underscores
        if not all(c.isalnum() or c in {'-', '_'} for c in v):
            raise ValueError('session_id must be alphanumeric with hyphens/underscores only')
        return v


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=500)
    question: str = Field(..., min_length=1, max_length=5000)
    is_continue: bool = False
    
    @validator('session_id')
    def session_id_validation(cls, v):
        if not all(c.isalnum() or c in {'-', '_'} for c in v):
            raise ValueError('Invalid session_id format')
        return v
    
    @validator('question')
    def question_not_only_spaces(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class QuizGenerationRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=500)
    scope: str = Field("all", pattern="^(all|parts)$")
    difficulty: str = Field("medium", pattern="^(easy|medium|hard)$")
    question_count: int = Field(10, ge=5, le=50)
    start_index: Optional[int] = Field(None, ge=0)
    end_index: Optional[int] = Field(None, ge=0)
    topic_ids: Optional[List[str]] = []
    
    @validator('session_id')
    def session_id_validation(cls, v):
        if not all(c.isalnum() or c in {'-', '_'} for c in v):
            raise ValueError('Invalid session_id format')
        return v


class QuizSubmissionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=500)
    quiz_id: str = Field(..., min_length=1, max_length=500)
    answers: dict = Field(...)
    
    @validator('session_id')
    def session_id_validation(cls, v):
        if not all(c.isalnum() or c in {'-', '_'} for c in v):
            raise ValueError('Invalid session_id format')
        return v


class TopicsExtractionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=500)
    
    @validator('session_id')
    def session_id_validation(cls, v):
        if not all(c.isalnum() or c in {'-', '_'} for c in v):
            raise ValueError('Invalid session_id format')
        return v

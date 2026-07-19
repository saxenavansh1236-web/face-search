from typing import List, Optional
from pydantic import BaseModel


class IndexFaceResponse(BaseModel):
    status: str
    id: str
    person_id: str
    duplicate_warning: Optional[str] = None
    ai_generated_warning: Optional[str] = None
    ai_confidence: str = "none"


class BulkIndexResult(BaseModel):
    id: str
    status: str
    detail: Optional[str] = None


class BulkIndexResponse(BaseModel):
    results: List[BulkIndexResult]
    indexed_count: int
    failed_count: int


class MatchResult(BaseModel):
    id: str
    person_id: str
    source_url: Optional[str] = None
    distance: float
    confidence: str
    match_found: bool


class SearchFaceResponse(BaseModel):
    query_ok: bool
    threshold_used: float
    best_match_found: bool
    matches: List[MatchResult]
    query_image_ai_warning: Optional[str] = None
    query_image_ai_confidence: str = "none"
    rerank_note: Optional[str] = None
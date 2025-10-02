"""
Pydantic models for API responses
"""

from typing import List
from pydantic import BaseModel


class DocumentUploadRequest(BaseModel):
    doc_name: str


class SimilarDocument(BaseModel):
    id: str
    similarity_score: float


class DuplicateCheckResult(BaseModel):
    has_duplicates: bool
    tfidf_duplicates: List[SimilarDocument]
    embedding_duplicates: List[SimilarDocument]
    max_tfidf_similarity: float
    max_embedding_similarity: float


class DocumentUploadResponse(BaseModel):
    id: str
    content: str
    duplicate_check: DuplicateCheckResult


class DocumentResponse(BaseModel):
    id: str
    content: str


class SimilarDocumentsResponse(BaseModel):
    query_document_id: str
    similar_documents: List[SimilarDocument]
    method: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    counts: dict


class RootResponse(BaseModel):
    message: str
    status: str
    documents_loaded: int
    tfidf_vectors: int
    clusters: int
    embeddings: int
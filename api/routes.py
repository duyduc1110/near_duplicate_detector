"""
API routes for the Near Duplicate Detection service
"""

from fastapi import APIRouter, HTTPException
from api.models import DocumentResponse, SimilarDocumentsResponse, HealthResponse, RootResponse
from api.services import SimilarityService
from api.lifespan import get_documents, get_tfidf_vectors, get_clusters, get_embeddings

router = APIRouter()


@router.get("/", response_model=RootResponse)
async def root():
    """Health check endpoint"""
    documents = get_documents()
    tfidf_vectors = get_tfidf_vectors()
    clusters = get_clusters()
    embeddings = get_embeddings()
    
    return RootResponse(
        message="Near Duplicate Detection API",
        status="running",
        documents_loaded=len(documents),
        tfidf_vectors=len(tfidf_vectors),
        clusters=len(clusters),
        embeddings=len(embeddings)
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    documents = get_documents()
    tfidf_vectors = get_tfidf_vectors()
    clusters = get_clusters()
    embeddings = get_embeddings()
    
    return HealthResponse(
        status="healthy",
        models_loaded={
            "documents": len(documents) > 0,
            "tfidf_vectors": len(tfidf_vectors) > 0,
            "clusters": len(clusters) > 0,
            "embeddings": len(embeddings) > 0
        },
        counts={
            "documents": len(documents),
            "tfidf_vectors": len(tfidf_vectors),
            "clusters": len(clusters),
            "embeddings": len(embeddings)
        }
    )


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """Get document content by ID"""
    documents = get_documents()
    
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    
    return DocumentResponse(
        id=doc_id,
        content=documents[doc_id]
    )


@router.get("/documents/{doc_id}/similar", response_model=SimilarDocumentsResponse)
async def find_similar_documents(
    doc_id: str, 
    method: str = "tfidf",
    threshold: float = 0.7,
    limit: int = 10
):
    """
    Find documents similar to the given document ID
    
    Args:
        doc_id: Document ID to find similarities for
        method: "tfidf" or "embedding" 
        threshold: Similarity threshold (0.0 to 1.0)
        limit: Maximum number of results to return
    """
    documents = get_documents()
    tfidf_vectors = get_tfidf_vectors()
    embeddings = get_embeddings()
    
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    
    if method not in ["tfidf", "embedding"]:
        raise HTTPException(status_code=400, detail="Method must be 'tfidf' or 'embedding'")
    
    # Initialize similarity service
    similarity_service = SimilarityService(documents, tfidf_vectors, embeddings)
    
    similar_docs = []
    if method == "tfidf":
        similar_docs = similarity_service.find_similar_tfidf(doc_id, threshold, limit)
    elif method == "embedding":
        similar_docs = similarity_service.find_similar_embedding(doc_id, threshold, limit)
    
    return SimilarDocumentsResponse(
        query_document_id=doc_id,
        similar_documents=similar_docs,
        method=method
    )
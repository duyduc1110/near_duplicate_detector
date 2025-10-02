"""
API routes for the Near Duplicate Detection service
"""

import os
from fastapi import APIRouter, HTTPException
from api.models import DocumentResponse, DocumentUploadRequest, DocumentUploadResponse, SimilarDocumentsResponse, HealthResponse, RootResponse
from api.services import SimilarityService
from api.lifespan import get_documents, get_tfidf_vectors, get_clusters, get_embeddings, get_tfidf_vectorizer
from scripts.read_data import preprocess_text

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


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(document: DocumentUploadRequest):
    """Process an existing document from data/new_docs folder"""
    doc_name = document.doc_name
    
    # Path to the new_docs directory
    new_docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "new_docs")
    file_path = os.path.join(new_docs_dir, f"{doc_name}.txt")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Document {doc_name}.txt not found in data/new_docs folder")
    
    # Check if document already exists in the system
    documents = get_documents()
    if doc_name in documents:
        raise HTTPException(status_code=409, detail=f"Document {doc_name} already exists in the system")
    
    try:
        # Read content from existing file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Preprocess content to match the format of existing documents
        processed_content = preprocess_text(content)
        
        # Use service layer to add document with duplicate checking
        documents = get_documents()
        tfidf_vectors = get_tfidf_vectors()
        embeddings = get_embeddings()
        clusters = get_clusters()
        tfidf_vectorizer = get_tfidf_vectorizer()
        
        similarity_service = SimilarityService(documents, tfidf_vectors, embeddings, clusters, tfidf_vectorizer)
        duplicate_check = similarity_service.add_document(doc_name, processed_content)
        
        return DocumentUploadResponse(
            id=doc_name,
            content=processed_content,
            duplicate_check=duplicate_check
        )
    
    except ValueError as e:
        # Handle duplicate detection errors with 409 Conflict status
        error_message = str(e)
        if "duplicates" in error_message.lower():
            raise HTTPException(status_code=409, detail=error_message)
        # Re-raise other ValueErrors as 400 Bad Request
        raise HTTPException(status_code=400, detail=error_message)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/documents/{doc_id}/similar", response_model=SimilarDocumentsResponse)
async def find_similar_documents(
    doc_id: str, 
    method: str = "embedding",
    threshold: float = 0.7,
    limit: int = 10
):
    """
    Find documents similar to the given document ID.

    - **doc_id**: Document ID to find similarities for
    - **method**: "tfidf" or "embedding"
    - **threshold**: Similarity threshold (0.0 to 1.0)
    - **limit**: Maximum number of results to return
    """
    documents = get_documents()
    tfidf_vectors = get_tfidf_vectors()
    embeddings = get_embeddings()
    clusters = get_clusters()
    tfidf_vectorizer = get_tfidf_vectorizer()
    
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    
    if method not in ["tfidf", "embedding"]:
        raise HTTPException(status_code=400, detail="Method must be 'tfidf' or 'embedding'")
    
    # Initialize similarity service with clusters and vectorizer
    similarity_service = SimilarityService(documents, tfidf_vectors, embeddings, clusters, tfidf_vectorizer)
    
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
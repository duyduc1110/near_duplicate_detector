"""
Service layer for similarity detection logic
"""

import logging
from typing import List
from scripts.finding_similar_doc import cosine_similarity
from scripts.embedding import cosine_similarity_embeddings
from api.models import SimilarDocument

logger = logging.getLogger(__name__)


class SimilarityService:
    """Service for finding similar documents"""
    
    def __init__(self, documents: dict, tfidf_vectors: dict, embeddings: dict):
        self.documents = documents
        self.tfidf_vectors = tfidf_vectors
        self.embeddings = embeddings
    
    def find_similar_tfidf(self, doc_id: str, threshold: float, limit: int) -> List[SimilarDocument]:
        """Find similar documents using TF-IDF vectors"""
        if doc_id not in self.tfidf_vectors:
            return []
        
        target_vector = self.tfidf_vectors[doc_id]
        similar_docs = []
        
        for other_doc_id, other_vector in self.tfidf_vectors.items():
            if other_doc_id != doc_id:
                similarity = cosine_similarity(target_vector, other_vector)
                if similarity >= threshold:
                    similar_docs.append(SimilarDocument(
                        id=other_doc_id,
                        similarity_score=round(similarity, 4)
                    ))
        
        # Sort by similarity (highest first) and limit results
        similar_docs.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_docs[:limit]
    
    def find_similar_embedding(self, doc_id: str, threshold: float, limit: int) -> List[SimilarDocument]:
        """Find similar documents using embedding vectors"""
        if not self.embeddings or doc_id not in self.embeddings:
            return []
        
        target_embedding = self.embeddings[doc_id]
        similar_docs = []
        
        for other_doc_id, other_embedding in self.embeddings.items():
            if other_doc_id != doc_id:
                similarity = cosine_similarity_embeddings(target_embedding, other_embedding)
                if similarity >= threshold:
                    similar_docs.append(SimilarDocument(
                        id=other_doc_id,
                        similarity_score=round(similarity, 4)
                    ))
        
        # Sort by similarity (highest first) and limit results
        similar_docs.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_docs[:limit]
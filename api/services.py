"""
Service layer for similarity detection logic
"""

import logging
from typing import List, Optional
from scripts.finding_similar_doc import cosine_similarity, find_similar_documents
from scripts.embedding import cosine_similarity_embeddings
from api.models import SimilarDocument

logger = logging.getLogger(__name__)


class SimilarityService:
    """Service for finding similar documents"""
    
    def __init__(self, documents: dict, tfidf_vectors: dict, embeddings: dict, clusters: Optional[dict] = None):
        self.documents = documents
        self.tfidf_vectors = tfidf_vectors
        self.embeddings = embeddings
        self.clusters = clusters
    
    def find_similar_tfidf(self, doc_id: str, threshold: float, limit: int) -> List[SimilarDocument]:
        """Find similar documents using TF-IDF vectors"""
        if doc_id not in self.tfidf_vectors:
            return []
        
        # Use clustered approach if clusters are available
        if self.clusters:
            return self._find_similar_clustered_tfidf(doc_id, threshold, limit)
        else:
            return self._find_similar_all_tfidf(doc_id, threshold, limit)
    
    def _find_similar_all_tfidf(self, doc_id: str, threshold: float, limit: int) -> List[SimilarDocument]:
        """Find similar documents using brute force approach (compare all)"""
        logger.debug(f"Finding similar documents for {doc_id} using brute force TF-IDF")
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
    
    def _find_similar_clustered_tfidf(self, doc_id: str, threshold: float, limit: int) -> List[SimilarDocument]:
        """Find similar documents using clustering optimization"""
        logger.debug(f"Finding similar documents for {doc_id} using clustered TF-IDF")
        if not self.clusters or doc_id not in self.clusters:
            # Fallback to all comparison if doc not in clusters
            return self._find_similar_all_tfidf(doc_id, threshold, limit)
        
        target_vector = self.tfidf_vectors[doc_id]
        target_cluster = self.clusters[doc_id]
        similar_docs = []
        
        # Only compare with documents in the same cluster
        for other_doc_id, cluster_id in self.clusters.items():
            if other_doc_id != doc_id and cluster_id == target_cluster and other_doc_id in self.tfidf_vectors:
                other_vector = self.tfidf_vectors[other_doc_id]
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
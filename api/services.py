"""
Service layer for similarity detection logic
"""

import logging
from typing import List, Optional
from scripts.finding_similar_doc import cosine_similarity, find_similar_documents
from scripts.embedding import cosine_similarity_embeddings, create_embeddings
from scripts.tfidf import transform_new_document
from scripts.clustering import create_clusters
from api.models import SimilarDocument, DuplicateCheckResult
from api.lifespan import update_global_state

logger = logging.getLogger(__name__)


class SimilarityService:
    """Service for finding similar documents"""
    
    def __init__(self, documents: dict, tfidf_vectors: dict, embeddings: dict, clusters: Optional[dict] = None, tfidf_vectorizer=None):
        self.documents = documents
        self.tfidf_vectors = tfidf_vectors
        self.embeddings = embeddings
        self.clusters = clusters
        self.tfidf_vectorizer = tfidf_vectorizer
    
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
    
    def check_document_duplicates(self, doc_id: str, content: str, tfidf_threshold: float = 0.7, embedding_threshold: float = 0.8) -> DuplicateCheckResult:
        """
        Check if a document has duplicates or near duplicates using hybrid approach:
        1. Quick content check (exact string matching) - 1-5ms
        2. Semantic analysis (TF-IDF + embeddings) if no exact match - 100-500ms
        
        Args:
            doc_id: Document ID
            content: Document content
            tfidf_threshold: Threshold for TF-IDF similarity
            embedding_threshold: Threshold for embedding similarity
            
        Returns:
            DuplicateCheckResult with duplicate check results
        """
        logger.info(f"Checking for duplicates for document: {doc_id}")
        
        # Phase 1: Quick content check (1-5ms) - check for exact duplicates
        # Normalize content for comparison (strip leading/trailing whitespace)
        normalized_content = content.strip()
        
        for existing_doc_id, existing_content in self.documents.items():
            normalized_existing = existing_content
            if normalized_content == normalized_existing:
                logger.warning(f"Exact content duplicate found: {existing_doc_id}")
                exact_duplicate = SimilarDocument(id=existing_doc_id, similarity_score=1.0)
                return DuplicateCheckResult(
                    has_duplicates=True,
                    tfidf_duplicates=[exact_duplicate],
                    embedding_duplicates=[exact_duplicate],
                    max_tfidf_similarity=1.0,
                    max_embedding_similarity=1.0
                )
        
        logger.info(f"No exact content match found, proceeding to semantic analysis")
        
        # Phase 2: Semantic analysis (100-500ms) - check for near duplicates
        tfidf_duplicates = []
        embedding_duplicates = []
        max_tfidf_similarity = 0.0
        max_embedding_similarity = 0.0
        
        # First, temporarily create vectors for the new document to check similarities
        temp_documents = {doc_id: content}
        
        # Check TF-IDF similarities with existing documents
        if self.tfidf_vectorizer is not None:
            temp_tfidf_vector = transform_new_document(self.tfidf_vectorizer, doc_id, content)
            
            # Temporarily add the new vector to check similarities
            temp_tfidf_vectors = self.tfidf_vectors.copy()
            temp_tfidf_vectors.update(temp_tfidf_vector)
            
            # Create a temporary service with the new document included
            temp_service = SimilarityService(self.documents, temp_tfidf_vectors, {}, self.clusters)
            tfidf_duplicates = temp_service._find_similar_all_tfidf(doc_id, tfidf_threshold, 50)
            
            if tfidf_duplicates:
                max_tfidf_similarity = max(sim.similarity_score for sim in tfidf_duplicates)
        
        # Check embedding similarities with existing documents
        try:
            temp_embeddings = create_embeddings(temp_documents)
            if temp_embeddings and doc_id in temp_embeddings:
                # Check embedding similarities with existing documents
                temp_all_embeddings = self.embeddings.copy()
                temp_all_embeddings.update(temp_embeddings)
                
                temp_service = SimilarityService(self.documents, {}, temp_all_embeddings, {})
                embedding_duplicates = temp_service.find_similar_embedding(doc_id, embedding_threshold, 50)
                
                if embedding_duplicates:
                    max_embedding_similarity = max(sim.similarity_score for sim in embedding_duplicates)
        except Exception as e:
            logger.warning(f"Could not check embedding duplicates: {e}")
        
        # Log duplicate detection results
        if tfidf_duplicates or embedding_duplicates:
            logger.warning(f"Potential duplicates found for document {doc_id}:")
            
            if tfidf_duplicates:
                logger.warning(f"TF-IDF duplicates (max similarity: {max_tfidf_similarity:.4f}):")
                for dup in tfidf_duplicates[:5]:  # Show top 5
                    logger.warning(f"  - {dup.id}: {dup.similarity_score:.4f}")
            
            if embedding_duplicates:
                logger.warning(f"Embedding duplicates (max similarity: {max_embedding_similarity:.4f}):")
                for dup in embedding_duplicates[:5]:  # Show top 5
                    logger.warning(f"  - {dup.id}: {dup.similarity_score:.4f}")
        else:
            logger.info(f"No duplicates found for document {doc_id}")
        
        return DuplicateCheckResult(
            has_duplicates=bool(tfidf_duplicates or embedding_duplicates),
            tfidf_duplicates=tfidf_duplicates,
            embedding_duplicates=embedding_duplicates,
            max_tfidf_similarity=max_tfidf_similarity,
            max_embedding_similarity=max_embedding_similarity
        )
    
    def add_document(self, doc_id: str, content: str) -> DuplicateCheckResult:
        """
        Add a new document with duplicate checking and update global state.
        
        Note: Since self.documents, self.tfidf_vectors, and self.embeddings are references
        to global dictionaries in lifespan.py, updating them here automatically updates
        the global state. We only need to call update_global_state() for clusters (which
        is a new dict) and saving to disk.
        
        Args:
            doc_id: Document ID
            content: Document content
            
        Returns:
            DuplicateCheckResult with duplicate check results
            
        Raises:
            ValueError: If duplicates are detected
        """
        # Check for duplicates (includes quick content check + semantic analysis)
        duplicate_check = self.check_document_duplicates(doc_id, content)
        
        if duplicate_check.has_duplicates:
            logger.warning(f"Duplicate detected for document {doc_id}, rejecting addition")
            
            # Collect all duplicate document IDs
            duplicate_ids = set()
            for dup in duplicate_check.tfidf_duplicates:
                duplicate_ids.add(dup.id)
            for dup in duplicate_check.embedding_duplicates:
                duplicate_ids.add(dup.id)
            
            duplicate_list = sorted(list(duplicate_ids))
            raise ValueError(f"Document {doc_id} has duplicates: {', '.join(duplicate_list)}")
        
        # Transform the new document using existing fitted vectorizer
        if self.tfidf_vectorizer is not None:
            new_tfidf_vector = transform_new_document(self.tfidf_vectorizer, doc_id, content)
            logger.info(f"Created TF-IDF vector for new document: {doc_id}")
        else:
            logger.error("No fitted vectorizer available, cannot add new document vector")
            raise ValueError("No fitted vectorizer available")
        
        # Update in-memory state (these update global state via references)
        self.documents[doc_id] = content
        self.tfidf_vectors.update(new_tfidf_vector)
        
        # Create embedding for the new document
        try:
            single_doc = {doc_id: content}
            new_embeddings = create_embeddings(single_doc)
            if new_embeddings:
                self.embeddings.update(new_embeddings)
            logger.info(f"Created embeddings for new document: {doc_id}")
        except Exception as e:
            logger.warning(f"Could not create embeddings for new document: {e}")
        
        # Create new clusters with all documents (including the newly added one)
        new_clusters = create_clusters(self.tfidf_vectors, self.documents)
        self.clusters = new_clusters
        
        # Update global clusters and save all state to disk
        update_global_state(new_clusters)
        
        logger.info(f"Successfully added document {doc_id}")
        return duplicate_check
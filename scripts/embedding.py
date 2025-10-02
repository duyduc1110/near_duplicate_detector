"""
Simple embedding vectorization using Hugging Face models.
"""

import pickle
import logging
from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up logger
logger = logging.getLogger("near_duplicate_detector.embedding")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")


def create_embeddings(documents: Dict[str, str], 
                     model_name: str = "all-MiniLM-L6-v2") -> Dict[str, List[float]]:
    """
    Create embeddings for documents using Hugging Face model.
    
    Args:
        documents: Dictionary of document_id -> text
        model_name: Hugging Face model name
        
    Returns:
        Dictionary of document_id -> embedding_vector
    """
    if not EMBEDDING_AVAILABLE:
        logger.error("SentenceTransformers not available!")
        return {}
    
    logger.info(f"Creating embeddings with model: {model_name}")
    logger.info(f"Processing {len(documents)} documents...")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Get texts and document IDs
    doc_ids = list(documents.keys())
    texts = [documents[doc_id] for doc_id in doc_ids]
    
    # Create embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Convert to dictionary
    result = {}
    for doc_id, embedding in zip(doc_ids, embeddings):
        result[doc_id] = embedding.tolist()
    
    logger.info(f"Created {len(result)} embeddings")
    return result


def save_embeddings(embeddings: Dict[str, List[float]], filepath: str) -> None:
    """Save embeddings to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)
    logger.info(f"Saved embeddings to {filepath}")


def load_embeddings(filepath: str) -> Dict[str, List[float]]:
    """Load embeddings from file."""
    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)
    logger.info(f"Loaded embeddings from {filepath}")
    return embeddings


def cosine_similarity_embeddings(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two embedding vectors using sklearn."""
    # Convert to numpy arrays and reshape for sklearn
    v1 = np.array(vec1).reshape(1, -1)
    v2 = np.array(vec2).reshape(1, -1)
    
    # Use sklearn's optimized cosine similarity
    similarity = cosine_similarity(v1, v2)[0, 0]
    
    return float(similarity)


def batch_cosine_similarity(embeddings_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise cosine similarities for all embeddings efficiently using sklearn.
    
    Args:
        embeddings_matrix: numpy array of shape (n_docs, embedding_dim)
        
    Returns:
        Similarity matrix of shape (n_docs, n_docs)
    """
    # Use sklearn's optimized implementation
    return cosine_similarity(embeddings_matrix)


def find_similar_embeddings(embeddings: Dict[str, List[float]], 
                           threshold: float = 0.8) -> List[tuple]:
    """
    Find similar documents using embedding cosine similarity with optimized batch processing.
    
    Args:
        embeddings: Dictionary of document_id -> embedding_vector
        threshold: Minimum similarity threshold
        
    Returns:
        List of (doc1_id, doc2_id, similarity_score) tuples
    """
    logger.info(f"Finding similar documents with embedding threshold {threshold}")
    
    doc_ids = list(embeddings.keys())
    if len(doc_ids) < 2:
        return []
    
    # Convert to matrix for batch processing
    embeddings_matrix = np.array([embeddings[doc_id] for doc_id in doc_ids])
    
    # Calculate all pairwise similarities at once
    similarity_matrix = batch_cosine_similarity(embeddings_matrix)
    
    similar_pairs = []
    n_docs = len(doc_ids)
    
    # Extract pairs above threshold (upper triangle only to avoid duplicates)
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:
                similar_pairs.append((doc_ids[i], doc_ids[j], float(similarity)))
    
    # Sort by similarity (highest first)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    logger.info(f"Found {len(similar_pairs)} similar pairs with embeddings")
    return similar_pairs
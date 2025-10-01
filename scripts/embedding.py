"""
Simple embedding vectorization using Hugging Face models.
"""

import pickle
import logging
from typing import Dict, List
import numpy as np

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
    """Calculate cosine similarity between two embedding vectors."""
    # Convert to numpy arrays
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    # Calculate cosine similarity
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norm_product == 0:
        return 0.0
    
    return dot_product / norm_product


def find_similar_embeddings(embeddings: Dict[str, List[float]], 
                           threshold: float = 0.8) -> List[tuple]:
    """
    Find similar documents using embedding cosine similarity.
    
    Args:
        embeddings: Dictionary of document_id -> embedding_vector
        threshold: Minimum similarity threshold
        
    Returns:
        List of (doc1_id, doc2_id, similarity_score) tuples
    """
    logger.info(f"Finding similar documents with embedding threshold {threshold}")
    
    doc_ids = list(embeddings.keys())
    similar_pairs = []
    
    for i, doc1_id in enumerate(doc_ids):
        for j in range(i + 1, len(doc_ids)):
            doc2_id = doc_ids[j]
            
            similarity = cosine_similarity_embeddings(
                embeddings[doc1_id], 
                embeddings[doc2_id]
            )
            
            if similarity >= threshold:
                similar_pairs.append((doc1_id, doc2_id, similarity))
    
    # Sort by similarity (highest first)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    logger.info(f"Found {len(similar_pairs)} similar pairs with embeddings")
    return similar_pairs


if __name__ == "__main__":
    # Test
    test_docs = {
        "1": "This is a test document about machine learning.",
        "2": "Machine learning is a subset of artificial intelligence.",
        "3": "The weather is nice today."
    }
    
    if EMBEDDING_AVAILABLE:
        embeddings = create_embeddings(test_docs)
        similar_pairs = find_similar_embeddings(embeddings, threshold=0.3)
        
        print("Similar pairs found:")
        for doc1, doc2, sim in similar_pairs:
            print(f"  {doc1} <-> {doc2}: {sim:.3f}")
    else:
        print("SentenceTransformers not available for testing")
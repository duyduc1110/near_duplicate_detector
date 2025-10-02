"""
Simple document similarity finder.
"""

import pickle
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

logger = logging.getLogger("near_duplicate_detector.finding_similar_doc")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors using optimized sklearn."""
    if len(vec1) != len(vec2):
        return 0.0
    
    # Use sklearn's optimized implementation
    v1 = np.array(vec1).reshape(1, -1)
    v2 = np.array(vec2).reshape(1, -1)
    similarity = sklearn_cosine_similarity(v1, v2)[0, 0]
    
    return max(0.0, min(1.0, float(similarity)))


def find_similar_documents(vectors: Dict[str, List[float]], 
                         threshold: float = 0.7,
                         clusters: Optional[Dict[str, int]] = None) -> List[Tuple[str, str, float]]:
    """
    Find similar document pairs using cosine similarity.
    
    Args:
        vectors: Dictionary of document_id -> tfidf_vector
        threshold: Minimum similarity threshold
        clusters: Optional clusters to reduce comparisons
        
    Returns:
        List of (doc1_id, doc2_id, similarity_score) tuples
    """
    logger.info(f"Finding similar documents with threshold {threshold}...")
    
    if clusters:
        return _find_clustered(vectors, threshold, clusters)
    else:
        return _find_all(vectors, threshold)


def _find_all(vectors: Dict[str, List[float]], threshold: float) -> List[Tuple[str, str, float]]:
    """Compare all document pairs."""
    doc_ids = list(vectors.keys())
    similar_pairs = []
    total = len(doc_ids) * (len(doc_ids) - 1) // 2
    
    logger.info(f"Comparing {total} document pairs...")
    
    processed = 0
    for i, doc1_id in enumerate(doc_ids):
        if processed % 1000 == 0:
            logger.debug(f"Progress: {100 * processed / total:.1f}%")
        
        for j in range(i + 1, len(doc_ids)):
            doc2_id = doc_ids[j]
            similarity = cosine_similarity(vectors[doc1_id], vectors[doc2_id])
            
            if similarity >= threshold:
                similar_pairs.append((doc1_id, doc2_id, similarity))
            
            processed += 1
    
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"Found {len(similar_pairs)} similar pairs")
    return similar_pairs


def _find_clustered(vectors: Dict[str, List[float]], 
                   threshold: float,
                   clusters: Dict[str, int]) -> List[Tuple[str, str, float]]:
    """Compare only documents within same cluster."""
    # Group by cluster
    cluster_docs = defaultdict(list)
    for doc_id, cluster_id in clusters.items():
        if doc_id in vectors:
            cluster_docs[cluster_id].append(doc_id)
    
    similar_pairs = []
    total_comparisons = 0
    
    for cluster_id, doc_ids in cluster_docs.items():
        if len(doc_ids) < 2:
            continue
        
        comparisons_in_cluster = len(doc_ids) * (len(doc_ids) - 1) // 2
        total_comparisons += comparisons_in_cluster
        
        for i, doc1_id in enumerate(doc_ids):
            for j in range(i + 1, len(doc_ids)):
                doc2_id = doc_ids[j]
                similarity = cosine_similarity(vectors[doc1_id], vectors[doc2_id])
                
                if similarity >= threshold:
                    similar_pairs.append((doc1_id, doc2_id, similarity))
    
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"Clustered comparison: {total_comparisons} pairs, found {len(similar_pairs)} similar")
    return similar_pairs


def save_similar_pairs(pairs: List[Tuple[str, str, float]], filepath: str) -> None:
    """Save similar pairs to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(pairs, f)
    logger.info(f"Saved {len(pairs)} similar pairs to {filepath}")


def load_similar_pairs(filepath: str) -> List[Tuple[str, str, float]]:
    """Load similar pairs from file."""
    with open(filepath, 'rb') as f:
        pairs = pickle.load(f)
    logger.info(f"Loaded {len(pairs)} similar pairs from {filepath}")
    return pairs

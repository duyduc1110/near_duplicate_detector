"""
Simple document clustering script.
"""

import pickle
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans

logger = logging.getLogger("near_duplicate_detector.clustering")


def create_clusters(vectors: Dict[str, List[float]], documents: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    """
    Create document clusters using K-means.
    
    Args:
        vectors: Document TF-IDF vectors
        documents: Original document texts (optional)
        
    Returns:
        Dictionary mapping document_id -> cluster_id
    """
    logger.info("Creating document clusters...")
    
    doc_ids = list(vectors.keys())
    vector_matrix = np.array([vectors[doc_id] for doc_id in doc_ids])
    
    # Determine number of clusters (roughly sqrt of document count)
    n_clusters = max(10, min(50, int(len(doc_ids) ** 0.5)))
    
    logger.info(f"Clustering {len(doc_ids)} documents into {n_clusters} clusters")
    
    # Create clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vector_matrix)
    
    # Map documents to clusters
    clusters = {}
    for doc_id, cluster_id in zip(doc_ids, cluster_labels):
        clusters[doc_id] = int(cluster_id)
    
    # Print cluster statistics
    cluster_sizes = defaultdict(int)
    for cluster_id in clusters.values():
        cluster_sizes[cluster_id] += 1
    
    logger.info(f"Created {len(cluster_sizes)} clusters")
    logger.debug(f"Average cluster size: {len(doc_ids) / len(cluster_sizes):.1f}")
    
    return clusters


def save_clusters(clusters: Dict[str, int], filepath: str) -> None:
    """Save clusters to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(clusters, f)
    logger.info(f"Saved clusters for {len(clusters)} documents to {filepath}")


def load_clusters(filepath: str) -> Dict[str, int]:
    """Load clusters from file."""
    with open(filepath, 'rb') as f:
        clusters = pickle.load(f)
    logger.info(f"Loaded clusters for {len(clusters)} documents from {filepath}")
    return clusters


def get_cluster_members(clusters: Dict[str, int], cluster_id: int) -> List[str]:
    """Get all document IDs in a specific cluster."""
    return [doc_id for doc_id, cid in clusters.items() if cid == cluster_id]


if __name__ == "__main__":
    # Test
    from read_data import read_documents
    from tfidf import create_tfidf_vectors
    
    data_dir = "/Users/le.duy.duc.nguyen/Documents/Github/happeo/data/all_docs"
    documents = read_documents(data_dir)
    
    if documents:
        vectors = create_tfidf_vectors(documents)
        clusters = create_clusters(vectors, documents)
        save_clusters(clusters, "clusters.pkl")


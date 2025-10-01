"""
Document clustering script.
Groups documents into clusters before similarity comparison to reduce computational complexity.
"""

import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def kmeans_clustering(vectors: Dict[str, List[float]], 
                     n_clusters: int = 50,
                     random_state: int = 42) -> Dict[str, int]:
    """
    K-means clustering using sklearn.
    
    Args:
        vectors: Dictionary of document_id -> tfidf_vector
        n_clusters: Number of clusters to create
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping document_id -> cluster_id
    """
    print(f"Clustering {len(vectors)} documents into {n_clusters} clusters using sklearn...")
    
    doc_ids = list(vectors.keys())
    vector_matrix = np.array([vectors[doc_id] for doc_id in doc_ids])
    
    # Use sklearn's KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = kmeans.fit_predict(vector_matrix)
    
    # Create assignments dictionary
    assignments = {doc_id: int(label) for doc_id, label in zip(doc_ids, cluster_labels)}
    
    # Calculate silhouette score for quality assessment
    if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
        silhouette_avg = silhouette_score(vector_matrix, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.3f} (higher is better, range: -1 to 1)")
    
    # Print cluster statistics
    cluster_sizes = defaultdict(int)
    for cluster_id in assignments.values():
        cluster_sizes[cluster_id] += 1
    
    print(f"\nCluster sizes:")
    for cluster_id in sorted(cluster_sizes.keys()):
        size = cluster_sizes[cluster_id]
        print(f"Cluster {cluster_id}: {size} documents")
    
    return assignments


def simple_clustering_by_length(vectors: Dict[str, List[float]], 
                               documents: Dict[str, str],
                               n_clusters: int = 20) -> Dict[str, int]:
    """
    Simple clustering based on document length - faster alternative.
    
    Args:
        vectors: Document vectors
        documents: Original document texts
        n_clusters: Number of clusters
        
    Returns:
        Dictionary mapping document_id -> cluster_id
    """
    print(f"Clustering documents by length into {n_clusters} groups...")
    
    # Calculate document lengths
    doc_lengths = [(doc_id, len(documents[doc_id])) for doc_id in vectors.keys()]
    doc_lengths.sort(key=lambda x: x[1])  # Sort by length
    
    # Divide into equal-sized groups
    docs_per_cluster = len(doc_lengths) // n_clusters
    assignments = {}
    
    for i, (doc_id, length) in enumerate(doc_lengths):
        cluster_id = min(i // docs_per_cluster, n_clusters - 1)
        assignments[doc_id] = cluster_id
    
    # Print cluster statistics
    cluster_info = defaultdict(list)
    for doc_id, cluster_id in assignments.items():
        cluster_info[cluster_id].append(len(documents[doc_id]))
    
    print(f"\nLength-based clusters:")
    for cluster_id in sorted(cluster_info.keys()):
        lengths = cluster_info[cluster_id]
        print(f"Cluster {cluster_id}: {len(lengths)} docs, "
              f"length range: {min(lengths)}-{max(lengths)} chars")
    
    return assignments


def hybrid_clustering(vectors: Dict[str, List[float]], 
                     documents: Dict[str, str],
                     n_clusters: int = 30) -> Dict[str, int]:
    """
    Hybrid clustering: first by length, then by similarity within length groups.
    
    Args:
        vectors: Document vectors
        documents: Original document texts  
        n_clusters: Target number of clusters
        
    Returns:
        Dictionary mapping document_id -> cluster_id
    """
    print(f"Hybrid clustering into ~{n_clusters} clusters...")
    
    # Step 1: Group by length (fewer groups)
    length_groups = 5
    length_assignments = simple_clustering_by_length(vectors, documents, length_groups)
    
    # Step 2: Sub-cluster each length group by similarity
    final_assignments = {}
    cluster_counter = 0
    
    for length_cluster_id in range(length_groups):
        # Get documents in this length cluster
        docs_in_group = [doc_id for doc_id, cluster in length_assignments.items() 
                        if cluster == length_cluster_id]
        
        if len(docs_in_group) <= 10:
            # Small group - don't subdivide
            for doc_id in docs_in_group:
                final_assignments[doc_id] = cluster_counter
            cluster_counter += 1
        else:
            # Large group - subdivide using similarity
            group_vectors = {doc_id: vectors[doc_id] for doc_id in docs_in_group}
            subclusters_needed = max(2, len(docs_in_group) // 100)  # ~100 docs per cluster
            
            subcluster_assignments = kmeans_clustering(group_vectors, subclusters_needed)
            
            # Map to global cluster IDs
            subcluster_mapping = {}
            for doc_id, subcluster in subcluster_assignments.items():
                if subcluster not in subcluster_mapping:
                    subcluster_mapping[subcluster] = cluster_counter
                    cluster_counter += 1
                final_assignments[doc_id] = subcluster_mapping[subcluster]
    
    print(f"Created {cluster_counter} final clusters")
    return final_assignments


def save_clusters(assignments: Dict[str, int], filepath: str) -> None:
    """Save cluster assignments to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(assignments, f)
    print(f"Saved cluster assignments to {filepath}")


def load_clusters(filepath: str) -> Dict[str, int]:
    """Load cluster assignments from file."""
    with open(filepath, 'rb') as f:
        assignments = pickle.load(f)
    print(f"Loaded cluster assignments from {filepath}")
    return assignments


def group_documents_by_cluster(assignments: Dict[str, int]) -> Dict[int, List[str]]:
    """Group document IDs by their cluster assignments."""
    clusters = defaultdict(list)
    for doc_id, cluster_id in assignments.items():
        clusters[cluster_id].append(doc_id)
    return dict(clusters)


def cluster_documents(vectors: Dict[str, List[float]], 
                     documents: Optional[Dict[str, str]] = None,
                     method: str = "hybrid",
                     n_clusters: int = 30) -> Dict[str, int]:
    """
    Main clustering function.
    
    Args:
        vectors: Document TF-IDF vectors
        documents: Original document texts (needed for some methods)
        method: Clustering method ("kmeans", "length", "hybrid")
        n_clusters: Number of clusters
        
    Returns:
        Dictionary mapping document_id -> cluster_id
    """
    if method == "kmeans":
        return kmeans_clustering(vectors, n_clusters)
    elif method == "length":
        if documents is None:
            raise ValueError("Documents needed for length-based clustering")
        return simple_clustering_by_length(vectors, documents, n_clusters)
    elif method == "hybrid":
        if documents is None:
            raise ValueError("Documents needed for hybrid clustering")
        return hybrid_clustering(vectors, documents, n_clusters)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
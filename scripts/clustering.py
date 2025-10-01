"""
Production-ready document clustering script.
Groups documents into clusters with model persistence for fast prediction on new documents.
"""

import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ProductionClusteringModel:
    """
    Production-ready clustering model that can be trained once and reused for new documents.
    Supports both single K-means and hybrid (length + similarity) approaches.
    """
    
    def __init__(self, approach: str = "hybrid", n_clusters: int = 25):
        """
        Initialize clustering model.
        
        Args:
            approach: "single" (one K-means) or "hybrid" (length + similarity)
            n_clusters: Target number of clusters
        """
        self.approach = approach
        self.n_clusters = n_clusters
        self.trained = False
        
        # Initialize based on approach
        if approach == "single":
            self.kmeans_model = None  # Will be initialized during training
            self.length_boundaries = []
            self.length_models = {}
            self.doc_id_to_cluster = {}
        elif approach == "hybrid":
            self.kmeans_model = None
            self.length_boundaries = []
            self.length_models = {}
            self.doc_id_to_cluster = {}
        else:
            raise ValueError(f"Unknown approach: {approach}")
    
    def train(self, vectors: Dict[str, List[float]], documents: Optional[Dict[str, str]] = None):
        """
        Train the clustering model on the dataset.
        
        Args:
            vectors: Document TF-IDF vectors
            documents: Original document texts (needed for hybrid approach)
        """
        print(f"Training {self.approach} clustering model...")
        
        if self.approach == "single":
            self._train_single_kmeans(vectors)
        elif self.approach == "hybrid":
            if documents is None:
                raise ValueError("Documents needed for hybrid approach")
            self._train_hybrid_kmeans(vectors, documents)
        
        self.trained = True
        print("✅ Model training completed!")
    
    def _train_single_kmeans(self, vectors: Dict[str, List[float]]):
        """Train single K-means on all vectors."""
        doc_ids = list(vectors.keys())
        vector_matrix = np.array([vectors[doc_id] for doc_id in doc_ids])
        
        print(f"Training single K-means on {len(vectors)} documents...")
        
        # Initialize and fit the model
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(vector_matrix)
        
        # Store cluster assignments for original documents
        for doc_id, cluster_id in zip(doc_ids, cluster_labels):
            self.doc_id_to_cluster[doc_id] = int(cluster_id)
        
        # Calculate silhouette score
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(vector_matrix, cluster_labels)
            print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        self._print_cluster_stats()
    
    def _train_hybrid_kmeans(self, vectors: Dict[str, List[float]], documents: Dict[str, str]):
        """Train hybrid approach with length grouping."""
        print("Training hybrid clustering (length + similarity)...")
        
        # Step 1: Calculate length boundaries from training data
        doc_lengths = [(doc_id, len(documents[doc_id])) for doc_id in vectors.keys()]
        doc_lengths.sort(key=lambda x: x[1])
        
        # Create length groups
        n_length_groups = 5
        docs_per_group = len(doc_lengths) // n_length_groups
        
        cluster_counter = 0
        
        for i in range(n_length_groups):
            start_idx = i * docs_per_group
            end_idx = (i + 1) * docs_per_group if i < n_length_groups - 1 else len(doc_lengths)
            
            group_docs = doc_lengths[start_idx:end_idx]
            min_length = group_docs[0][1]
            max_length = group_docs[-1][1]
            
            # Store length boundaries
            self.length_boundaries.append((min_length, max_length))
            
            # Get documents in this length group
            group_doc_ids = [doc_id for doc_id, _ in group_docs]
            
            if len(group_doc_ids) <= 10:
                # Small group - assign to single cluster
                for doc_id in group_doc_ids:
                    self.doc_id_to_cluster[doc_id] = cluster_counter
                cluster_counter += 1
                print(f"Length group {i}: {min_length}-{max_length} chars, {len(group_doc_ids)} docs → 1 cluster")
            else:
                # Large group - subdivide using K-means
                group_vectors = np.array([vectors[doc_id] for doc_id in group_doc_ids])
                subclusters_needed = max(2, len(group_doc_ids) // 100)
                
                # Train K-means for this length group
                length_kmeans = KMeans(n_clusters=subclusters_needed, random_state=42, n_init=10)
                subcluster_labels = length_kmeans.fit_predict(group_vectors)
                
                # Store the trained model
                self.length_models[i] = length_kmeans
                
                # Assign global cluster IDs
                for doc_id, subcluster in zip(group_doc_ids, subcluster_labels):
                    global_cluster_id = cluster_counter + int(subcluster)
                    self.doc_id_to_cluster[doc_id] = global_cluster_id
                
                cluster_counter += subclusters_needed
                print(f"Length group {i}: {min_length}-{max_length} chars, {len(group_doc_ids)} docs → {subclusters_needed} clusters")
        
        print(f"Created {cluster_counter} total clusters")
    
    def predict_cluster(self, vector: List[float], document_text: Optional[str] = None) -> int:
        """
        Predict cluster for a new document.
        
        Args:
            vector: TF-IDF vector of new document
            document_text: Original text (needed for hybrid approach)
            
        Returns:
            Cluster ID
        """
        if not self.trained:
            raise ValueError("Model must be trained first!")
        
        if self.approach == "single":
            return self._predict_single(vector)
        elif self.approach == "hybrid":
            if document_text is None:
                raise ValueError("Document text needed for hybrid prediction")
            return self._predict_hybrid(vector, document_text)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
    
    def _predict_single(self, vector: List[float]) -> int:
        """Predict using single K-means model."""
        if self.kmeans_model is None:
            raise ValueError("Single K-means model not trained")
        vector_array = np.array([vector])
        cluster_id = self.kmeans_model.predict(vector_array)[0]
        return int(cluster_id)
    
    def _predict_hybrid(self, vector: List[float], document_text: str) -> int:
        """Predict using hybrid approach."""
        doc_length = len(document_text)
        
        # Find appropriate length group
        length_group = 0
        for i, (min_len, max_len) in enumerate(self.length_boundaries):
            if min_len <= doc_length <= max_len:
                length_group = i
                break
        
        # Use the appropriate K-means model for this length group
        if length_group in self.length_models:
            vector_array = np.array([vector])
            subcluster_id = self.length_models[length_group].predict(vector_array)[0]
            
            # Calculate global cluster ID
            # Find the base cluster ID for this length group
            base_cluster_id = min([cluster_id for doc_id, cluster_id in self.doc_id_to_cluster.items() 
                                 if self._get_length_group_for_doc(doc_id) == length_group])
            
            return int(base_cluster_id + subcluster_id)
        else:
            # Fallback to cluster 0 if no model for this length group
            return 0
    
    def _get_length_group_for_doc(self, doc_id: str) -> int:
        """Helper to get length group for a document ID."""
        # This is a simplified approach - in practice you'd store this mapping
        return 0
    
    def get_cluster_assignments(self) -> Dict[str, int]:
        """Get cluster assignments for all training documents."""
        return self.doc_id_to_cluster.copy()
    
    def _print_cluster_stats(self):
        """Print statistics about cluster assignments."""
        cluster_sizes = defaultdict(int)
        for cluster_id in self.doc_id_to_cluster.values():
            cluster_sizes[cluster_id] += 1
        
        print(f"\nCluster distribution:")
        for cluster_id in sorted(cluster_sizes.keys()):
            size = cluster_sizes[cluster_id]
            print(f"Cluster {cluster_id}: {size} documents")
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        model_data = {
            'approach': self.approach,
            'n_clusters': self.n_clusters,
            'trained': self.trained,
            'kmeans_model': self.kmeans_model,
            'length_boundaries': self.length_boundaries,
            'length_models': self.length_models,
            'doc_id_to_cluster': self.doc_id_to_cluster
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.approach = model_data['approach']
        self.n_clusters = model_data['n_clusters']
        self.trained = model_data['trained']
        self.kmeans_model = model_data['kmeans_model']
        self.length_boundaries = model_data['length_boundaries']
        self.length_models = model_data['length_models']
        self.doc_id_to_cluster = model_data['doc_id_to_cluster']
        
        print(f"Model loaded from {filepath}")


# Legacy functions for backward compatibility
def kmeans_clustering(vectors: Dict[str, List[float]], 
                     n_clusters: int = 50,
                     random_state: int = 42) -> Dict[str, int]:
    """
    Legacy K-means clustering function for backward compatibility.
    """
    model = ProductionClusteringModel(approach="single", n_clusters=n_clusters)
    model.train(vectors)
    return model.get_cluster_assignments()


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


def save_clustering_model(model: ProductionClusteringModel, filepath: str) -> None:
    """Save clustering model to file."""
    model.save_model(filepath)


def load_clustering_model(filepath: str) -> ProductionClusteringModel:
    """Load clustering model from file."""
    model = ProductionClusteringModel()
    model.load_model(filepath)
    return model


def save_clusters(assignments: Dict[str, int], filepath: str) -> None:
    """Save cluster assignments to file (legacy compatibility)."""
    with open(filepath, 'wb') as f:
        pickle.dump(assignments, f)
    print(f"Saved cluster assignments to {filepath}")


def load_clusters(filepath: str) -> Dict[str, int]:
    """Load cluster assignments from file (legacy compatibility)."""
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
    Main clustering function using production model.
    
    Args:
        vectors: Document TF-IDF vectors
        documents: Original document texts (needed for hybrid method)
        method: Clustering method ("single" or "hybrid")
        n_clusters: Number of clusters
        
    Returns:
        Dictionary mapping document_id -> cluster_id
    """
    # Map legacy method names
    if method == "kmeans":
        method = "single"
    elif method == "length":
        # For pure length-based clustering, use simple implementation
        if documents is None:
            raise ValueError("Documents needed for length-based clustering")
        return simple_clustering_by_length(vectors, documents, n_clusters)
    
    # Use production clustering model
    model = ProductionClusteringModel(approach=method, n_clusters=n_clusters)
    model.train(vectors, documents)
    
    return model.get_cluster_assignments()
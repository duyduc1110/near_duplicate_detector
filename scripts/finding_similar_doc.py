"""
Similar document finding script.
Uses TF-IDF vectors to find near-duplicate documents using cosine similarity.
"""

import pickle
import logging
from typing import Dict, List, Tuple
from math import sqrt

# Set up logger for this module
logger = logging.getLogger("near_duplicate_detector.finding_similar_doc")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1, vec2: Normalized vectors
        
    Returns:
        Cosine similarity score (0-1)
    """
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    return max(0.0, min(1.0, dot_product))  # Clamp to [0, 1]


from typing import Optional

def find_similar_documents(vectors: Dict[str, List[float]], 
                         threshold: float = 0.7,
                         clusters: Optional[Dict[str, int]] = None) -> List[Tuple[str, str, float]]:
    """
    Find similar document pairs using cosine similarity.
    If clusters provided, only compare documents within the same cluster.
    
    Args:
        vectors: Dictionary of document_id -> tfidf_vector
        threshold: Minimum similarity threshold
        clusters: Optional dictionary of document_id -> cluster_id
        
    Returns:
        List of (doc1_id, doc2_id, similarity_score) tuples
    """
    logger.info(f"Finding similar documents with threshold {threshold}...")
    
    if clusters is None:
        # Original full comparison
        return find_similar_documents_full(vectors, threshold)
    else:
        # Clustered comparison
        return find_similar_documents_clustered(vectors, threshold, clusters)


def find_similar_documents_full(vectors: Dict[str, List[float]], 
                              threshold: float) -> List[Tuple[str, str, float]]:
    """Original full comparison method."""
    doc_ids = list(vectors.keys())
    similar_pairs = []
    total_comparisons = len(doc_ids) * (len(doc_ids) - 1) // 2
    
    logger.info(f"Comparing {total_comparisons} document pairs...")
    
    processed = 0
    for i, doc1_id in enumerate(doc_ids):
        if i % 50 == 0:
            progress = 100 * processed / total_comparisons
            logger.debug(f"Progress: {progress:.1f}% ({processed}/{total_comparisons})")
        
        vec1 = vectors[doc1_id]
        
        for j in range(i + 1, len(doc_ids)):
            doc2_id = doc_ids[j]
            vec2 = vectors[doc2_id]
            
            similarity = cosine_similarity(vec1, vec2)
            
            if similarity >= threshold:
                similar_pairs.append((doc1_id, doc2_id, similarity))
            
            processed += 1
    
    # Sort by similarity (highest first)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    logger.info(f"Found {len(similar_pairs)} similar document pairs")
    return similar_pairs


def find_similar_documents_clustered(vectors: Dict[str, List[float]], 
                                   threshold: float,
                                   clusters: Dict[str, int]) -> List[Tuple[str, str, float]]:
    """Clustered comparison - only compare documents within same cluster."""
    from collections import defaultdict
    
    # Group documents by cluster
    cluster_groups = defaultdict(list)
    for doc_id, cluster_id in clusters.items():
        if doc_id in vectors:  # Make sure document has a vector
            cluster_groups[cluster_id].append(doc_id)
    
    logger.info(f"Comparing documents within {len(cluster_groups)} clusters...")
    
    similar_pairs = []
    total_comparisons = 0
    processed = 0
    
    # Calculate total comparisons
    for cluster_docs in cluster_groups.values():
        n = len(cluster_docs)
        total_comparisons += n * (n - 1) // 2
    
    logger.info(f"Total comparisons reduced to: {total_comparisons:,} (vs {len(vectors) * (len(vectors) - 1) // 2:,} full)")
    
    # Compare within each cluster
    for cluster_id, cluster_docs in cluster_groups.items():
        if len(cluster_docs) < 2:
            continue
            
        logger.debug(f"Processing cluster {cluster_id} with {len(cluster_docs)} documents...")
        
        for i, doc1_id in enumerate(cluster_docs):
            vec1 = vectors[doc1_id]
            
            for j in range(i + 1, len(cluster_docs)):
                doc2_id = cluster_docs[j]
                vec2 = vectors[doc2_id]
                
                similarity = cosine_similarity(vec1, vec2)
                
                if similarity >= threshold:
                    similar_pairs.append((doc1_id, doc2_id, similarity))
                
                processed += 1
                
                if processed % 10000 == 0:
                    progress = 100 * processed / total_comparisons
                    logger.debug(f"Progress: {progress:.1f}% ({processed:,}/{total_comparisons:,})")
    
    # Sort by similarity (highest first)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    logger.info(f"Found {len(similar_pairs)} similar document pairs")
    return similar_pairs


def find_duplicates_for_document(doc_id: str, 
                                vectors: Dict[str, List[float]], 
                                threshold: float = 0.8) -> List[Tuple[str, float]]:
    """
    Find all documents similar to a specific document.
    
    Args:
        doc_id: Target document ID
        vectors: Dictionary of document_id -> tfidf_vector
        threshold: Minimum similarity threshold
        
    Returns:
        List of (similar_doc_id, similarity_score) tuples
    """
    if doc_id not in vectors:
        logger.warning(f"Document {doc_id} not found!")
        return []
    
    target_vector = vectors[doc_id]
    similar_docs = []
    
    for other_doc_id, other_vector in vectors.items():
        if other_doc_id != doc_id:
            similarity = cosine_similarity(target_vector, other_vector)
            if similarity >= threshold:
                similar_docs.append((other_doc_id, similarity))
    
    # Sort by similarity (highest first)
    similar_docs.sort(key=lambda x: x[1], reverse=True)
    
    return similar_docs


def get_similarity_statistics(similar_pairs: List[Tuple[str, str, float]]) -> None:
    """Print statistics about similarity scores."""
    if not similar_pairs:
        logger.info("No similar pairs found")
        return
    
    scores = [score for _, _, score in similar_pairs]
    
    logger.info("Similarity Statistics:")
    logger.info(f"Total pairs found: {len(similar_pairs)}")
    logger.info(f"Highest similarity: {max(scores):.4f}")
    logger.info(f"Lowest similarity: {min(scores):.4f}")
    logger.info(f"Average similarity: {sum(scores) / len(scores):.4f}")
    
    # Distribution
    high_sim = sum(1 for s in scores if s >= 0.9)
    med_sim = sum(1 for s in scores if 0.7 <= s < 0.9)
    low_sim = sum(1 for s in scores if s < 0.7)
    
    logger.info(f"High similarity (≥0.9): {high_sim}")
    logger.info(f"Medium similarity (0.7-0.9): {med_sim}")
    logger.info(f"Lower similarity (<0.7): {low_sim}")


def save_results(similar_pairs: List[Tuple[str, str, float]], 
                output_file: str = "similar_documents.csv") -> None:
    """
    Save similar document pairs to CSV file.
    
    Args:
        similar_pairs: List of similar document tuples
        output_file: Output filename
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Document_1,Document_2,Similarity_Score\n")
        
        for doc1, doc2, score in similar_pairs:
            f.write(f"{doc1},{doc2},{score:.6f}\n")
    
    logger.info(f"Results saved to {output_file}")


from typing import Optional

def load_vectors_and_find_duplicates(vectors_file: str, 
                                   threshold: float = 0.7,
                                   clusters_file: Optional[str] = None) -> List[Tuple[str, str, float]]:
    """
    Load vectors and find duplicate documents.
    
    Args:
        vectors_file: Path to saved vectors file
        threshold: Similarity threshold
        clusters_file: Optional path to cluster assignments file
        
    Returns:
        List of similar document pairs
    """
    # Load vectors
    logger.info(f"Loading vectors from {vectors_file}...")
    try:
        with open(vectors_file, 'rb') as f:
            vectors = pickle.load(f)
        logger.info(f"Loaded {len(vectors)} vectors")
    except FileNotFoundError:
        logger.error(f"Vectors file {vectors_file} not found!")
        return []
    
    # Load clusters if provided
    clusters = None
    if clusters_file:
        try:
            with open(clusters_file, 'rb') as f:
                clusters = pickle.load(f)
            logger.info(f"Loaded cluster assignments for {len(clusters)} documents")
        except FileNotFoundError:
            logger.warning(f"Clusters file {clusters_file} not found, using full comparison")
    
    # Find similar documents
    similar_pairs = find_similar_documents(vectors, threshold, clusters)
    
    # Show statistics
    get_similarity_statistics(similar_pairs)
    
    return similar_pairs


if __name__ == "__main__":
    # Test the script
    vectors_file = "tfidf_vectors.pkl"
    threshold = 0.7
    
    similar_pairs = load_vectors_and_find_duplicates(vectors_file, threshold)
    
    if similar_pairs:
        print(f"\nTop 10 most similar pairs:")
        for i, (doc1, doc2, sim) in enumerate(similar_pairs[:10], 1):
            print(f"{i:2d}. {doc1} <-> {doc2} (similarity: {sim:.4f})")
        
        save_results(similar_pairs)
"""
Simple TF-IDF vectorization script.
"""

import pickle
import logging
from collections import Counter
from typing import Dict, List
from math import log, sqrt

logger = logging.getLogger("near_duplicate_detector.tfidf")


def create_tfidf_vectors(documents: Dict[str, str], max_features: int = 5000) -> Dict[str, List[float]]:
    """
    Create TF-IDF vectors from documents.
    
    Args:
        documents: Dictionary of document_id -> text
        max_features: Maximum number of features
        
    Returns:
        Dictionary of document_id -> tfidf_vector
    """
    logger.info("Creating TF-IDF vectors...")
    logger.info(f"Processing {len(documents)} documents")
    
    # Build vocabulary
    word_doc_count = Counter()
    all_words = Counter()
    
    for doc_text in documents.values():
        words = set(doc_text.split())
        for word in words:
            word_doc_count[word] += 1
        
        for word in doc_text.split():
            all_words[word] += 1
    
    # Filter words (appear in 2+ docs but less than 80% of docs)
    doc_count = len(documents)
    filtered_words = {}
    for word, doc_freq in word_doc_count.items():
        if 2 <= doc_freq <= 0.8 * doc_count:
            filtered_words[word] = all_words[word]
    
    # Take top words
    top_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:max_features]
    vocabulary = {word: idx for idx, (word, _) in enumerate(top_words)}
    
    # Calculate IDF values
    idf_values = {}
    for word in vocabulary:
        doc_freq = word_doc_count[word]
        idf_values[word] = log(doc_count / doc_freq)
    
    logger.info(f"Vocabulary size: {len(vocabulary)}")
    
    # Create vectors
    vectors = {}
    doc_ids = list(documents.keys())
    
    for i, doc_id in enumerate(doc_ids):
        if i % 100 == 0:
            logger.debug(f"Vectorizing document {i+1}/{len(doc_ids)}")
        
        text = documents[doc_id]
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            vectors[doc_id] = [0.0] * len(vocabulary)
            continue
        
        # Calculate term frequencies
        tf_counter = Counter(words)
        
        # Build TF-IDF vector
        vector = []
        for word in sorted(vocabulary.keys()):
            tf = tf_counter.get(word, 0) / word_count
            idf = idf_values.get(word, 0)
            tfidf = tf * idf
            vector.append(tfidf)
        
        # Normalize vector
        magnitude = sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        vectors[doc_id] = vector
    
    logger.info(f"Created {len(vectors)} TF-IDF vectors")
    return vectors


def save_vectors(vectors: Dict[str, List[float]], filepath: str) -> None:
    """Save vectors to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(vectors, f)
    logger.info(f"Saved {len(vectors)} vectors to {filepath}")


def load_vectors(filepath: str) -> Dict[str, List[float]]:
    """Load vectors from file."""
    with open(filepath, 'rb') as f:
        vectors = pickle.load(f)
    logger.info(f"Loaded {len(vectors)} vectors from {filepath}")
    return vectors


if __name__ == "__main__":
    # Test
    from read_data import read_documents
    
    data_dir = "/Users/le.duy.duc.nguyen/Documents/Github/happeo/data/all_docs"
    documents = read_documents(data_dir)
    
    if documents:
        vectors = create_tfidf_vectors(documents)
        save_vectors(vectors, "tfidf_vectors.pkl")
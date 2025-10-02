"""
Fast TF-IDF vectorization using scikit-learn.
"""

import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger("near_duplicate_detector.tfidf")


def create_tfidf_vectors(documents: Dict[str, str], max_features: int = 10000) -> Tuple[Dict[str, List[float]], TfidfVectorizer]:
    """
    Create TF-IDF vectors from documents using scikit-learn.
    
    Args:
        documents: Dictionary of document_id -> text
        max_features: Maximum number of features
        
    Returns:
        Tuple of (Dictionary of document_id -> tfidf_vector, fitted TfidfVectorizer)
    """
    logger.info("Creating TF-IDF vectors using scikit-learn...")
    logger.info(f"Processing {len(documents)} documents")
    
    if not documents:
        logger.warning("No documents provided")
        return {}, TfidfVectorizer()
    
    # Prepare data for scikit-learn
    doc_ids = list(documents.keys())
    texts = [documents[doc_id] for doc_id in doc_ids]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        # stop_words='english',
        lowercase=True,
        strip_accents='unicode',
        token_pattern=r'\b[a-zA-Z]{2,}\b',  # Only alphabetic tokens of 2+ chars
        min_df=2,           # Word must appear in at least 2 documents
        max_df=0.8,         # Word must appear in less than 80% of documents
        ngram_range=(1, 1), # Only unigrams
        norm='l2',          # L2 normalization
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True   # Apply sublinear tf scaling
    )
    
    # Fit and transform
    logger.info("Fitting TF-IDF vectorizer...")
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Convert to dense array and ensure L2 normalization
    tfidf_dense = tfidf_matrix.toarray()  # type: ignore
    tfidf_normalized = normalize(tfidf_dense, norm='l2', axis=1)
    
    # Create result dictionary
    vectors = {}
    for i, doc_id in enumerate(doc_ids):
        vectors[doc_id] = tfidf_normalized[i].tolist()
    
    logger.info(f"Created {len(vectors)} TF-IDF vectors")
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    logger.info(f"Vector dimension: {tfidf_matrix.shape[1]}")
    
    return vectors, vectorizer


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


def save_vectorizer(vectorizer: TfidfVectorizer, filepath: str) -> None:
    """Save fitted vectorizer to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Saved vectorizer to {filepath}")


def load_vectorizer(filepath: str) -> TfidfVectorizer:
    """Load fitted vectorizer from file."""
    with open(filepath, 'rb') as f:
        vectorizer = pickle.load(f)
    logger.info(f"Loaded vectorizer from {filepath}")
    return vectorizer


def transform_new_document(vectorizer: TfidfVectorizer, doc_id: str, content: str) -> Dict[str, List[float]]:
    """
    Transform a new document using an existing fitted vectorizer.
    
    Args:
        vectorizer: Fitted TfidfVectorizer
        doc_id: Document ID
        content: Document text content
        
    Returns:
        Dictionary with doc_id -> tfidf_vector
    """
    logger.info(f"Transforming new document: {doc_id}")
    
    # Transform the new document
    tfidf_matrix = vectorizer.transform([content])
    
    # Convert to dense array and ensure L2 normalization
    tfidf_dense = tfidf_matrix.toarray()  # type: ignore
    tfidf_normalized = normalize(tfidf_dense, norm='l2', axis=1)
    
    # Return as dictionary
    return {doc_id: tfidf_normalized[0].tolist()}
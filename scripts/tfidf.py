"""
TF-IDF vectorization script.
Transforms document text into TF-IDF vectors for similarity computation.
"""

import pickle
from collections import Counter
from typing import Dict, List, Tuple
from math import log, sqrt


class SimpleTFIDF:
    """Simple TF-IDF implementation using only built-in Python libraries."""
    
    def __init__(self, max_features: int = 10000):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features to keep
        """
        self.max_features = max_features
        self.vocabulary = {}
        self.idf_values = {}
        self.document_count = 0
    
    def build_vocabulary(self, documents: Dict[str, str]) -> None:
        """Build vocabulary from documents."""
        print("Building vocabulary...")
        
        # Count word frequencies across all documents
        word_doc_count = Counter()
        all_words = Counter()
        
        for doc_text in documents.values():
            words = set(doc_text.split())
            for word in words:
                word_doc_count[word] += 1
            
            for word in doc_text.split():
                all_words[word] += 1
        
        # Select top words by frequency, but filter out too common ones
        self.document_count = len(documents)
        
        # Filter words that appear in too many documents (> 80%) or too few (< 2)
        filtered_words = {}
        for word, doc_freq in word_doc_count.items():
            if 2 <= doc_freq <= 0.8 * self.document_count:
                filtered_words[word] = all_words[word]
        
        # Take top max_features words
        top_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
        
        # Build vocabulary mapping
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(top_words)}
        
        # Calculate IDF values
        for word in self.vocabulary:
            doc_freq = word_doc_count[word]
            self.idf_values[word] = log(self.document_count / doc_freq)
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
    
    def vectorize_document(self, text: str) -> List[float]:
        """
        Convert a document to TF-IDF vector.
        
        Args:
            text: Document text
            
        Returns:
            TF-IDF vector as list of floats
        """
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return [0.0] * len(self.vocabulary)
        
        # Calculate term frequencies
        tf_counter = Counter(words)
        
        # Build TF-IDF vector
        vector = []
        for word in sorted(self.vocabulary.keys()):
            tf = tf_counter.get(word, 0) / word_count  # Term frequency
            idf = self.idf_values.get(word, 0)  # Inverse document frequency
            tfidf = tf * idf
            vector.append(tfidf)
        
        return vector
    
    def fit_transform(self, documents: Dict[str, str]) -> Dict[str, List[float]]:
        """
        Fit the vectorizer and transform documents.
        
        Args:
            documents: Dictionary of document_id -> text
            
        Returns:
            Dictionary of document_id -> tfidf_vector
        """
        self.build_vocabulary(documents)
        
        print("Transforming documents to TF-IDF vectors...")
        vectors = {}
        
        doc_ids = list(documents.keys())
        for i, doc_id in enumerate(doc_ids):
            if i % 100 == 0:
                print(f"Vectorizing document {i+1}/{len(doc_ids)}...")
            
            vectors[doc_id] = self.vectorize_document(documents[doc_id])
        
        print(f"Created {len(vectors)} TF-IDF vectors")
        return vectors


def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize vector to unit length."""
    magnitude = sqrt(sum(x * x for x in vector))
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]


def create_tfidf_vectors(documents: Dict[str, str]) -> Dict[str, List[float]]:
    """
    Create TF-IDF vectors from documents.
    
    Args:
        documents: Dictionary of document_id -> text
        
    Returns:
        Dictionary of document_id -> normalized_tfidf_vector
    """
    vectorizer = SimpleTFIDF(max_features=5000)
    vectors = vectorizer.fit_transform(documents)
    
    # Normalize vectors
    print("Normalizing vectors...")
    normalized_vectors = {}
    for doc_id, vector in vectors.items():
        normalized_vectors[doc_id] = normalize_vector(vector)
    
    return normalized_vectors


def save_vectors(vectors: Dict[str, List[float]], filepath: str) -> None:
    """Save vectors to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(vectors, f)
    print(f"Saved {len(vectors)} vectors to {filepath}")


def load_vectors(filepath: str) -> Dict[str, List[float]]:
    """Load vectors from file."""
    with open(filepath, 'rb') as f:
        vectors = pickle.load(f)
    print(f"Loaded {len(vectors)} vectors from {filepath}")
    return vectors


if __name__ == "__main__":
    # Test the script
    from read_data import read_documents
    
    data_dir = "/Users/le.duy.duc.nguyen/Documents/Github/happeo/data/all_docs"
    documents = read_documents(data_dir)
    
    if documents:
        vectors = create_tfidf_vectors(documents)
        save_vectors(vectors, "tfidf_vectors.pkl")
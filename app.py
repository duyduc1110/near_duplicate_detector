"""
Simple Near Duplicate Detection with TF-IDF and Embeddings.
"""

import os
import time
import logging
from scripts.read_data import read_documents
from scripts.tfidf import create_tfidf_vectors, save_vectors, load_vectors
from scripts.finding_similar_doc import find_similar_documents, save_results
from scripts.embedding import create_embeddings, find_similar_embeddings, save_embeddings, load_embeddings, cosine_similarity_embeddings

EMBEDDING_AVAILABLE = True

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleNearDuplicateDetector:
    """Simple duplicate detector with optional embedding support."""
    
    def __init__(self, data_dir: str, use_embeddings: bool = True):
        """
        Initialize detector.
        
        Args:
            data_dir: Directory containing documents
            use_embeddings: Whether to use embeddings for better accuracy
        """
        self.data_dir = data_dir
        self.use_embeddings = use_embeddings and EMBEDDING_AVAILABLE
        
        # File paths
        self.tfidf_file = "tfidf_vectors.pkl"
        self.embedding_file = "embeddings.pkl"
        self.tfidf_results = "tfidf_results.csv"
        self.embedding_results = "embedding_results.csv"
        
        if self.use_embeddings:
            logger.info("Embedding mode enabled")
        else:
            logger.info("TF-IDF only mode")
    
    def run_detection(self, tfidf_threshold: float = 0.7, embedding_threshold: float = 0.8):
        """
        Run duplicate detection pipeline.
        
        Args:
            tfidf_threshold: Similarity threshold for TF-IDF
            embedding_threshold: Similarity threshold for embeddings
        """
        logger.info("Starting duplicate detection...")
        start_time = time.time()
        
        # Step 1: Load documents
        logger.info("Loading documents...")
        documents = read_documents(self.data_dir)
        if not documents:
            logger.error("No documents found!")
            return
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Step 2: TF-IDF Analysis
        logger.info("Running TF-IDF analysis...")
        tfidf_results = self._run_tfidf_analysis(documents, tfidf_threshold)
        
        # Step 3: Embedding Analysis (if enabled)
        if self.use_embeddings:
            logger.info("Running embedding analysis...")
            embedding_results = self._run_embedding_analysis(documents, embedding_threshold)
            
            # Compare results
            self._compare_results(tfidf_results, embedding_results)
        
        total_time = time.time() - start_time
        logger.info(f"Detection completed in {total_time:.2f} seconds")
    
    def _run_tfidf_analysis(self, documents: dict, threshold: float):
        """Run TF-IDF duplicate detection."""
        # Create or load TF-IDF vectors
        if os.path.exists(self.tfidf_file):
            logger.info("Loading existing TF-IDF vectors...")
            vectors = load_vectors(self.tfidf_file)
        else:
            logger.info("Creating TF-IDF vectors...")
            vectors = create_tfidf_vectors(documents)
            save_vectors(vectors, self.tfidf_file)
        
        # Find similar documents
        similar_pairs = find_similar_documents(vectors, threshold)
        
        # Save results
        if similar_pairs:
            save_results(similar_pairs, self.tfidf_results)
            logger.info(f"TF-IDF found {len(similar_pairs)} similar pairs")
        else:
            logger.info("No similar pairs found with TF-IDF")
        
        return similar_pairs
    
    def _run_embedding_analysis(self, documents: dict, threshold: float):
        """Run embedding-based duplicate detection."""
        # Create or load embeddings
        if os.path.exists(self.embedding_file):
            logger.info("Loading existing embeddings...")
            embeddings = load_embeddings(self.embedding_file)
        else:
            logger.info("Creating embeddings...")
            embeddings = create_embeddings(documents)
            save_embeddings(embeddings, self.embedding_file)
        
        # Find similar documents
        similar_pairs = find_similar_embeddings(embeddings, threshold)
        
        # Save results
        if similar_pairs:
            save_results(similar_pairs, self.embedding_results)
            logger.info(f"Embeddings found {len(similar_pairs)} similar pairs")
        else:
            logger.info("No similar pairs found with embeddings")
        
        return similar_pairs
    
    def _compare_results(self, tfidf_results: list, embedding_results: list):
        """Compare TF-IDF and embedding results."""
        if not tfidf_results or not embedding_results:
            return
        
        # Convert to sets for comparison
        tfidf_pairs = {(min(r[0], r[1]), max(r[0], r[1])) for r in tfidf_results}
        embedding_pairs = {(min(r[0], r[1]), max(r[0], r[1])) for r in embedding_results}
        
        common_pairs = tfidf_pairs.intersection(embedding_pairs)
        tfidf_only = tfidf_pairs - embedding_pairs
        embedding_only = embedding_pairs - tfidf_pairs
        
        logger.info("=" * 50)
        logger.info("RESULTS COMPARISON")
        logger.info("=" * 50)
        logger.info(f"TF-IDF pairs: {len(tfidf_pairs)}")
        logger.info(f"Embedding pairs: {len(embedding_pairs)}")
        logger.info(f"Common pairs: {len(common_pairs)}")
        logger.info(f"TF-IDF only: {len(tfidf_only)}")
        logger.info(f"Embedding only: {len(embedding_only)}")
        
        if len(tfidf_pairs) > 0:
            agreement = len(common_pairs) / len(tfidf_pairs) * 100
            logger.info(f"Agreement rate: {agreement:.1f}%")
    
    def find_similar_to_document(self, doc_id: str, method: str = "both"):
        """
        Find documents similar to a specific document.
        
        Args:
            doc_id: Target document ID
            method: "tfidf", "embedding", or "both"
        """
        logger.info(f"Finding documents similar to: {doc_id}")
        
        if method in ["tfidf", "both"]:
            if os.path.exists(self.tfidf_file):
                vectors = load_vectors(self.tfidf_file)
                if doc_id in vectors:
                    from scripts.finding_similar_doc import find_duplicates_for_document
                    similar = find_duplicates_for_document(doc_id, vectors, threshold=0.7)
                    logger.info(f"TF-IDF found {len(similar)} similar documents")
                    for sim_doc, score in similar[:5]:
                        logger.info(f"  {sim_doc}: {score:.3f}")
        
        if method in ["embedding", "both"] and self.use_embeddings:
            if os.path.exists(self.embedding_file):
                embeddings = load_embeddings(self.embedding_file)
                if doc_id in embeddings:
                    similar = []
                    target_embedding = embeddings[doc_id]
                    
                    for other_id, other_embedding in embeddings.items():
                        if other_id != doc_id:
                            sim = cosine_similarity_embeddings(target_embedding, other_embedding)
                            if sim >= 0.7:
                                similar.append((other_id, sim))
                    
                    similar.sort(key=lambda x: x[1], reverse=True)
                    logger.info(f"Embeddings found {len(similar)} similar documents")
                    for sim_doc, score in similar[:5]:
                        logger.info(f"  {sim_doc}: {score:.3f}")


def main():
    """Main function."""
    data_dir = "/Users/le.duy.duc.nguyen/Documents/Github/happeo/data/all_docs"
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Initialize detector
    detector = SimpleNearDuplicateDetector(
        data_dir=data_dir,
        use_embeddings=True  # Set to False for TF-IDF only
    )
    
    # Run detection
    detector.run_detection(
        tfidf_threshold=0.7,
        embedding_threshold=0.8
    )

if __name__ == "__main__":
    main()
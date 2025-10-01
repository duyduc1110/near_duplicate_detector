"""
Main application script for Near Duplicate Detection.
Orchestrates the entire pipeline: read data -> create TF-IDF vectors -> find duplicates.
"""

import os
import sys
import time
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.append(str(scripts_dir))

from scripts.read_data import read_documents, get_document_stats
from scripts.tfidf import create_tfidf_vectors, save_vectors, load_vectors
from scripts.clustering import cluster_documents, save_clusters, load_clusters
from scripts.finding_similar_doc import load_vectors_and_find_duplicates, save_results


class NearDuplicateDetector:
    """Main application class for near duplicate detection."""
    
    def __init__(self, data_dir: str, similarity_threshold: float = 0.7, use_clustering: bool = True):
        """
        Initialize the detector.
        
        Args:
            data_dir: Directory containing text documents
            similarity_threshold: Minimum similarity to consider documents as duplicates
            use_clustering: Whether to use clustering to reduce comparisons
        """
        self.data_dir = data_dir
        self.similarity_threshold = similarity_threshold
        self.use_clustering = use_clustering
        self.vectors_file = "tfidf_vectors.pkl"
        self.clusters_file = "document_clusters.pkl"
        self.results_file = "duplicate_results.csv"
    
    def run_full_pipeline(self) -> None:
        """Run the complete duplicate detection pipeline."""
        print("=" * 60)
        print("Near Duplicate Detection Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        # Check for existing files
        vectors_exist = os.path.exists(self.vectors_file)
        clusters_exist = os.path.exists(self.clusters_file)
        
        print(f"\n🔍 Checking existing files...")
        print(f"   TF-IDF vectors: {'✅ Found' if vectors_exist else '❌ Missing'}")
        print(f"   Clusters: {'✅ Found' if clusters_exist else '❌ Missing'}")
        
        # Step 1 & 2: Load or create TF-IDF vectors
        if vectors_exist:
            print(f"\n📂 Loading existing TF-IDF vectors from {self.vectors_file}...")
            vectors = load_vectors(self.vectors_file)
            documents = None  # Don't need to load documents if we have vectors
        else:
            print("\n📁 Step 1: Reading documents...")
            documents = read_documents(self.data_dir)
            
            if not documents:
                print("❌ No documents found! Please check the data directory.")
                return
            
            get_document_stats(documents)
            
            print("\n🔢 Step 2: Creating TF-IDF vectors...")
            vectors = create_tfidf_vectors(documents)
            save_vectors(vectors, self.vectors_file)
        
        # Step 3: Load or create clusters (if enabled)
        clusters_file = None
        if self.use_clustering:
            if clusters_exist:
                print(f"\n� Loading existing clusters from {self.clusters_file}...")
                clusters = load_clusters(self.clusters_file)
                clusters_file = self.clusters_file
            else:
                print(f"\n�🗂️  Step 3: Clustering documents to reduce comparisons...")
                
                # Load documents if we don't have them yet (for hybrid clustering)
                if documents is None:
                    print("   Loading documents for clustering...")
                    documents = read_documents(self.data_dir)
                
                clusters = cluster_documents(vectors, documents, method="hybrid", n_clusters=25)
                save_clusters(clusters, self.clusters_file)
                clusters_file = self.clusters_file
            
            # Calculate comparison reduction
            total_docs = len(vectors)
            original_comparisons = total_docs * (total_docs - 1) // 2
            
            from collections import defaultdict
            cluster_groups = defaultdict(list)
            for doc_id, cluster_id in clusters.items():
                cluster_groups[cluster_id].append(doc_id)
            
            clustered_comparisons = sum(len(docs) * (len(docs) - 1) // 2 
                                      for docs in cluster_groups.values())
            
            reduction = 100 * (1 - clustered_comparisons / original_comparisons)
            print(f"   📊 Reduced comparisons from {original_comparisons:,} to {clustered_comparisons:,}")
            print(f"   🚀 Comparison reduction: {reduction:.1f}%")
        
        # Step 4: Find similar documents
        step_num = 4 if self.use_clustering else 3
        print(f"\n🔍 Step {step_num}: Finding similar documents (threshold: {self.similarity_threshold})...")
        similar_pairs = load_vectors_and_find_duplicates(self.vectors_file, self.similarity_threshold, clusters_file)
        
        # Step 5: Save and display results
        step_num += 1
        print(f"\n💾 Step {step_num}: Saving results...")
        if similar_pairs:
            save_results(similar_pairs, self.results_file)
            self.display_results(similar_pairs)
        else:
            print("No duplicate pairs found with current threshold.")
            print(f"Try lowering the threshold (currently {self.similarity_threshold})")
        
        # Cleanup
        # self.cleanup_temp_files()
        
        end_time = time.time()
        print(f"\n⏱️  Total processing time: {end_time - start_time:.2f} seconds")
        print("✅ Pipeline completed successfully!")
    
    def display_results(self, similar_pairs) -> None:
        """Display the results summary."""
        print("\n" + "=" * 60)
        print("📊 RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"Total duplicate pairs found: {len(similar_pairs)}")
        
        if similar_pairs:
            print(f"\n🏆 Top 10 most similar document pairs:")
            for i, (doc1, doc2, sim) in enumerate(similar_pairs[:10], 1):
                print(f"{i:2d}. {doc1} ↔ {doc2} (similarity: {sim:.4f})")
            
            # Group by similarity ranges
            high_sim = sum(1 for _, _, s in similar_pairs if s >= 0.9)
            med_sim = sum(1 for _, _, s in similar_pairs if 0.8 <= s < 0.9)
            low_sim = sum(1 for _, _, s in similar_pairs if s < 0.8)
            
            print(f"\n📈 Similarity distribution:")
            print(f"   Very high (≥0.9): {high_sim} pairs")
            print(f"   High (0.8-0.9):   {med_sim} pairs") 
            print(f"   Medium (<0.8):    {low_sim} pairs")
            
            print(f"\n📄 Results saved to: {self.results_file}")
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        if os.path.exists(self.vectors_file):
            os.remove(self.vectors_file)
            print(f"🧹 Cleaned up temporary file: {self.vectors_file}")
    
    def run_quick_check(self, doc_id: str) -> None:
        """Find duplicates for a specific document."""
        print(f"\n🔍 Quick check for document: {doc_id}")
        
        # Load vectors if they exist
        if not os.path.exists(self.vectors_file):
            print("❌ No vectors file found. Run full pipeline first.")
            return
        
        from scripts.finding_similar_doc import find_duplicates_for_document
        import pickle
        
        with open(self.vectors_file, 'rb') as f:
            vectors = pickle.load(f)
        
        similar_docs = find_duplicates_for_document(doc_id, vectors, self.similarity_threshold)
        
        if similar_docs:
            print(f"Found {len(similar_docs)} similar documents:")
            for i, (sim_doc, score) in enumerate(similar_docs[:5], 1):
                print(f"{i}. {sim_doc} (similarity: {score:.4f})")
        else:
            print("No similar documents found.")


def main():
    """Main execution function."""
    # Configuration
    data_directory = "/Users/le.duy.duc.nguyen/Documents/Github/happeo/data/all_docs"
    similarity_threshold = 0.7  # Adjust as needed (0.5-0.9 range)
    
    # Validate data directory
    if not os.path.exists(data_directory):
        print(f"❌ Error: Data directory '{data_directory}' not found!")
        print(f"Please ensure the directory exists and contains .txt files.")
        return
    
    # Initialize and run detector
    detector = NearDuplicateDetector(
        data_dir=data_directory,
        similarity_threshold=similarity_threshold,
        use_clustering=True  # Enable clustering for faster processing
    )
    
    # Run the pipeline
    detector.run_full_pipeline()
    
    # Optional: Quick check for specific document
    # detector.run_quick_check("0")  # Uncomment to check document "0"


if __name__ == "__main__":
    main()
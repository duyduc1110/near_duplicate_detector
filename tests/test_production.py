#!/usr/bin/env python3
"""
Test script to demonstrate the production clustering model.
Shows how to classify new documents in real-time.
"""

import os
import time
from app import NearDuplicateDetector


def test_new_document_classification():
    """Test the production model's ability to classify new documents."""
    print("=" * 60)
    print("Testing Production Clustering Model")
    print("=" * 60)
    
    # Initialize detector (make sure model exists)
    data_dir = "/Users/le.duy.duc.nguyen/Documents/Github/happeo/data/all_docs"
    detector = NearDuplicateDetector(data_dir=data_dir, use_clustering=True)
    
    # Check if model exists
    if not os.path.exists(detector.clustering_model_file):
        print("❌ Clustering model not found. Running full pipeline first...")
        detector.run_full_pipeline()
        print("\n" + "=" * 60)
        print("Now testing new document classification...")
        print("=" * 60)
    
    # Test documents (examples)
    test_documents = [
        "This is a test document about machine learning algorithms and neural networks.",
        "Financial markets and stock trading strategies for investors.",
        "Python programming tutorial for beginners with examples.",
        "Climate change and environmental protection policies worldwide.",
        "Medical research on cancer treatment and prevention methods."
    ]
    
    print(f"\n🧪 Testing classification of {len(test_documents)} new documents:")
    print("-" * 60)
    
    total_time = 0
    for i, doc_text in enumerate(test_documents, 1):
        start_time = time.time()
        
        # Classify the document
        cluster_id = detector.classify_new_document(doc_text)
        
        end_time = time.time()
        classification_time = end_time - start_time
        total_time += classification_time
        
        # Show results
        preview = doc_text[:50] + "..." if len(doc_text) > 50 else doc_text
        print(f"{i}. Document: \"{preview}\"")
        print(f"   → Cluster: {cluster_id}")
        print(f"   → Time: {classification_time:.4f}s")
        print()
    
    # Performance summary
    avg_time = total_time / len(test_documents)
    print("=" * 60)
    print("📊 Performance Summary:")
    print(f"   Total documents classified: {len(test_documents)}")
    print(f"   Total time: {total_time:.4f}s")
    print(f"   Average time per document: {avg_time:.4f}s")
    print(f"   Throughput: {len(test_documents)/total_time:.1f} docs/second")
    print("=" * 60)


def quick_demo():
    """Quick demo of the near duplicate detection system."""
    print("=" * 60)
    print("Quick Demo: Near Duplicate Detection")
    print("=" * 60)
    
    data_dir = "/Users/le.duy.duc.nguyen/Documents/Github/happeo/data/all_docs"
    detector = NearDuplicateDetector(
        data_dir=data_dir,
        similarity_threshold=0.8,  # Higher threshold for demo
        use_clustering=True
    )
    
    # Check for a specific document
    detector.run_quick_check("0")


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Test new document classification")
    print("2. Quick demo of duplicate detection")
    print("3. Run both")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        test_new_document_classification()
    elif choice == "2":
        quick_demo()
    elif choice == "3":
        test_new_document_classification()
        print("\n" + "=" * 60)
        quick_demo()
    else:
        print("Invalid choice. Running test by default...")
        test_new_document_classification()
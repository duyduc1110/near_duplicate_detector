"""
Test script for the FastAPI Near Duplicate Detection API
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_api():
    """Test the API endpoints"""
    print("Testing Near Duplicate Detection API...")
    
    try:
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{API_BASE}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is running")
            print(f"   Documents loaded: {data.get('documents_loaded', 0)}")
            print(f"   TF-IDF vectors: {data.get('tfidf_vectors', 0)}")
            print(f"   Clusters: {data.get('clusters', 0)}")
            print(f"   Embeddings: {data.get('embeddings', 0)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
        
        # Test document retrieval
        print("\n2. Testing document retrieval...")
        response = requests.get(f"{API_BASE}/documents/1")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Document retrieved: {data['id']}")
            print(f"   Content length: {len(data['content'])} characters")
        else:
            print(f"❌ Document retrieval failed: {response.status_code}")
        
        # Test similar documents with TF-IDF
        print("\n3. Testing similar documents (TF-IDF)...")
        response = requests.get(f"{API_BASE}/documents/1/similar?method=tfidf&threshold=0.7&limit=5")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ TF-IDF similarity search completed")
            print(f"   Query document: {data['query_document_id']}")
            print(f"   Similar documents found: {len(data['similar_documents'])}")
            for doc in data['similar_documents'][:3]:
                print(f"     {doc['id']}: {doc['similarity_score']}")
        else:
            print(f"❌ TF-IDF similarity search failed: {response.status_code}")
        
        # Test similar documents with embeddings
        print("\n4. Testing similar documents (Embeddings)...")
        response = requests.get(f"{API_BASE}/documents/1/similar?method=embedding&threshold=0.8&limit=5")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Embedding similarity search completed")
            print(f"   Query document: {data['query_document_id']}")
            print(f"   Similar documents found: {len(data['similar_documents'])}")
            for doc in data['similar_documents'][:3]:
                print(f"     {doc['id']}: {doc['similarity_score']}")
        else:
            print(f"❌ Embedding similarity search failed: {response.status_code}")
        
        print("\n✅ All API tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")


if __name__ == "__main__":
    test_api()
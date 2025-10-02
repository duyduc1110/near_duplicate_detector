"""
Application lifespan management for model loading
"""

from typing import Optional
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from scripts.read_data import read_documents
from scripts.tfidf import create_tfidf_vectors, save_vectors, load_vectors, save_vectorizer, load_vectorizer, transform_new_document
from scripts.clustering import create_clusters, save_clusters, load_clusters
from scripts.embedding import create_embeddings, save_embeddings, load_embeddings

logger = logging.getLogger(__name__)

# Global variables to store models and data
documents = {}
tfidf_vectors = {}
tfidf_vectorizer = None
clusters = {}
embeddings = {}
data_dir = "data/all_docs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and data on startup"""
    global documents, tfidf_vectors, tfidf_vectorizer, clusters, embeddings
    
    logger.info("Starting FastAPI application...")
    logger.info("Loading and preparing models...")
    
    # Load documents
    logger.info("Loading documents...")
    documents = read_documents(data_dir)
    if not documents:
        logger.error(f"No documents found in {data_dir}")
        yield
        return
    logger.info(f"Loaded {len(documents)} documents")
    
    # Create or load TF-IDF vectors and vectorizer
    tfidf_file = "models/tfidf_vectors.pkl"
    vectorizer_file = "models/tfidf_vectorizer.pkl"
    os.makedirs("models", exist_ok=True)
    
    if os.path.exists(tfidf_file) and os.path.exists(vectorizer_file):
        logger.info("Loading existing TF-IDF vectors and vectorizer...")
        tfidf_vectors = load_vectors(tfidf_file)
        tfidf_vectorizer = load_vectorizer(vectorizer_file)
    else:
        logger.info("Creating TF-IDF vectors...")
        tfidf_vectors, tfidf_vectorizer = create_tfidf_vectors(documents)
        save_vectors(tfidf_vectors, tfidf_file)
        save_vectorizer(tfidf_vectorizer, vectorizer_file)
    logger.info(f"TF-IDF vectors ready: {len(tfidf_vectors)}")
    
    # Create or load clusters
    clusters_file = "models/clusters.pkl"
    if os.path.exists(clusters_file):
        logger.info("Loading existing clusters...")
        clusters = load_clusters(clusters_file)
    else:
        logger.info("Creating clusters...")
        clusters = create_clusters(tfidf_vectors, documents)
        save_clusters(clusters, clusters_file)
    logger.info(f"Clusters ready: {len(clusters)} documents clustered")
    
    # Create or load embeddings
    embeddings_file = "models/embeddings.pkl"
    try:
        if os.path.exists(embeddings_file):
            logger.info("Loading existing embeddings...")
            embeddings = load_embeddings(embeddings_file)
        else:
            logger.info("Creating embeddings...")
            embeddings = create_embeddings(documents)
            save_embeddings(embeddings, embeddings_file)
        logger.info(f"Embeddings ready: {len(embeddings)}")
    except Exception as e:
        logger.warning(f"Could not load/create embeddings: {e}")
        embeddings = {}
    
    logger.info("All models ready! API is now available.")
    
    yield  # Server is running
    
    # Cleanup (if needed)
    logger.info("Shutting down...")


def get_documents():
    """Get loaded documents"""
    return documents


def get_tfidf_vectors():
    """Get TF-IDF vectors"""
    return tfidf_vectors


def get_clusters():
    """Get clusters"""
    return clusters


def get_embeddings():
    """Get embeddings"""
    return embeddings


def get_tfidf_vectorizer():
    """Get TF-IDF vectorizer"""
    return tfidf_vectorizer


def update_global_state(new_clusters: dict):
    """
    Update global clusters and save all state to disk after adding a new document.
    
    Note: The in-memory documents, tfidf_vectors, and embeddings are already updated 
    via dictionary references from the service layer. This function updates the clusters
    reference and handles persistence to disk.
    
    Args:
        new_clusters: Updated clusters dictionary
    """
    global clusters
    
    logger.info(f"Updating clusters and saving state to disk")
    
    # Update clusters reference (this is a new dict, not an update to existing dict)
    clusters = new_clusters
    
    # Save all updated models to disk
    # Note: documents don't need saving (they're just text files)
    save_vectors(tfidf_vectors, "models/tfidf_vectors.pkl")
    save_clusters(clusters, "models/clusters.pkl")
    if embeddings:
        save_embeddings(embeddings, "models/embeddings.pkl")
    
    logger.info(f"Clusters updated and state saved to disk")
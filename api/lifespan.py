"""
Application lifespan management for model loading
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from scripts.read_data import read_documents
from scripts.tfidf import create_tfidf_vectors, save_vectors, load_vectors
from scripts.clustering import create_clusters, save_clusters, load_clusters
from scripts.embedding import create_embeddings, save_embeddings, load_embeddings

logger = logging.getLogger(__name__)

# Global variables to store models and data
documents = {}
tfidf_vectors = {}
clusters = {}
embeddings = {}
data_dir = "/Users/le.duy.duc.nguyen/Documents/Github/happeo/data/all_docs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and data on startup"""
    global documents, tfidf_vectors, clusters, embeddings
    
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
    
    # Create or load TF-IDF vectors
    tfidf_file = "models/tfidf_vectors.pkl"
    os.makedirs("models", exist_ok=True)
    
    if os.path.exists(tfidf_file):
        logger.info("Loading existing TF-IDF vectors...")
        tfidf_vectors = load_vectors(tfidf_file)
    else:
        logger.info("Creating TF-IDF vectors...")
        tfidf_vectors = create_tfidf_vectors(documents)
        save_vectors(tfidf_vectors, tfidf_file)
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
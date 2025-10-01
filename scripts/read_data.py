"""
Data reading and preprocessing script.
Reads all documents from data/all_docs folder and preprocesses them.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List

# Set up logger for this module
logger = logging.getLogger("near_duplicate_detector.read_data")


def preprocess_text(text: str) -> str:
    """Clean and preprocess text."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep letters, numbers, and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    
    return text.strip()


def read_documents(data_dir: str) -> Dict[str, str]:
    """
    Read all text documents from the specified directory.
    
    Args:
        data_dir: Path to directory containing .txt files
        
    Returns:
        Dictionary mapping document ID to preprocessed text content
    """
    documents = {}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Directory {data_dir} does not exist!")
        return documents
    
    txt_files = list(data_path.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files")
    
    for i, file_path in enumerate(txt_files):
        if i % 100 == 0:
            logger.debug(f"Processing file {i+1}/{len(txt_files)}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            if content:  # Only add non-empty documents
                doc_id = file_path.stem
                processed_content = preprocess_text(content)
                documents[doc_id] = processed_content
                
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents


def get_document_stats(documents: Dict[str, str]) -> None:
    """Print statistics about the document collection."""
    if not documents:
        logger.warning("No documents to analyze")
        return
    
    lengths = [len(text) for text in documents.values()]
    word_counts = [len(text.split()) for text in documents.values()]
    
    logger.info("Document Statistics:")
    logger.info(f"Total documents: {len(documents)}")
    logger.info(f"Average length: {sum(lengths) / len(lengths):.0f} characters")
    logger.info(f"Average words: {sum(word_counts) / len(word_counts):.0f} words")
    logger.info(f"Shortest: {min(lengths)} chars, Longest: {max(lengths)} chars")


if __name__ == "__main__":
    # Test the script
    data_dir = "/Users/le.duy.duc.nguyen/Documents/Github/happeo/data/all_docs"
    docs = read_documents(data_dir)
    get_document_stats(docs)
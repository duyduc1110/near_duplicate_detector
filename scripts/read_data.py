"""
Data reading and preprocessing script.
Reads all documents from data/all_docs folder and preprocesses them.
"""

import os
import re
from pathlib import Path
from typing import Dict, List


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
        print(f"Error: Directory {data_dir} does not exist!")
        return documents
    
    txt_files = list(data_path.glob("*.txt"))
    print(f"Found {len(txt_files)} text files")
    
    for i, file_path in enumerate(txt_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(txt_files)}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            if content:  # Only add non-empty documents
                doc_id = file_path.stem
                processed_content = preprocess_text(content)
                documents[doc_id] = processed_content
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Successfully loaded {len(documents)} documents")
    return documents


def get_document_stats(documents: Dict[str, str]) -> None:
    """Print statistics about the document collection."""
    if not documents:
        print("No documents to analyze")
        return
    
    lengths = [len(text) for text in documents.values()]
    word_counts = [len(text.split()) for text in documents.values()]
    
    print(f"\nDocument Statistics:")
    print(f"Total documents: {len(documents)}")
    print(f"Average length: {sum(lengths) / len(lengths):.0f} characters")
    print(f"Average words: {sum(word_counts) / len(word_counts):.0f} words")
    print(f"Shortest: {min(lengths)} chars, Longest: {max(lengths)} chars")


if __name__ == "__main__":
    # Test the script
    data_dir = "/Users/le.duy.duc.nguyen/Documents/Github/happeo/data/all_docs"
    docs = read_documents(data_dir)
    get_document_stats(docs)
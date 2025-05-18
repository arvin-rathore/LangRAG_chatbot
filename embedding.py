"""
Embedding creation and management for PDF content.
"""

import os
import pickle
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

import config
from extraction import extract_text_from_pdfs


# Initialize the embedding model globally
embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def generate_and_store_embeddings():
    """Generate embeddings for all PDFs and store them with FAISS."""
    # Check if embeddings already exist
    if os.path.exists(config.EMBEDDINGS_FILE) and os.path.exists(config.INDEX_FILE):
        print("Embeddings already exist. Skipping generation.")
        return
    
    # Extract text from PDFs
    texts, sources = extract_text_from_pdfs()
    
    if not texts:
        print("No text extracted from PDFs. Embeddings not generated.")
        return
    
    # Generate embeddings
    embeddings = embedding_model.embed_documents(texts)
    
    # Store embeddings and metadata
    with open(config.EMBEDDINGS_FILE, "wb") as f:
        pickle.dump((embeddings, texts, sources), f)
    
    # Create and store FAISS index
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    faiss.write_index(index, config.INDEX_FILE)
    
    print(f"Generated embeddings for {len(texts)} text chunks from PDFs.")


def load_embeddings():
    """
    Load embeddings and FAISS index from files.
    
    Returns:
        tuple: (embeddings, texts, sources, faiss_index)
    """
    # Generate embeddings if they don't exist
    if not (os.path.exists(config.EMBEDDINGS_FILE) and os.path.exists(config.INDEX_FILE)):
        print("Embeddings not found. Generating new ones...")
        generate_and_store_embeddings()
    
    # Load embeddings and metadata
    with open(config.EMBEDDINGS_FILE, "rb") as f:
        embeddings, texts, sources = pickle.load(f)
    
    # Load FAISS index
    index = faiss.read_index(config.INDEX_FILE)
    
    return embeddings, texts, sources, index


def get_vectorstore():
    """
    Create a FAISS vector store for use with LangChain.
    
    Returns:
        FAISS: A LangChain FAISS vector store.
    """
    embeddings, texts, sources, _ = load_embeddings()
    documents = [
        Document(page_content=text, metadata={"source": src}) 
        for text, src in zip(texts, sources)
    ]
    return FAISS.from_documents(documents, embedding_model)
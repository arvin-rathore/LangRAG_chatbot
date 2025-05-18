"""
Configuration settings for the PDF Chatbot application.
"""

import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
INDEX_DIR = os.path.join(ROOT_DIR, "faiss_index")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Files
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
EMBEDDINGS_FILE = os.path.join(INDEX_DIR, "index.pkl")

# Model settings
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL_NAME = "gemini-1.5-flash"
GPT_MODEL_NAME = "gpt-4"  # For langchain integration

# Text processing settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval settings
TOP_K_RESULTS = 3

# API Keys - should be loaded from environment variables in production
GEMINI_API_KEY = "API_KEY_HERE"
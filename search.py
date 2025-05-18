"""
Vector search functionality using FAISS.
"""

import numpy as np
import config
from embedding import embedding_model, load_embeddings, get_vectorstore


def search_similar_texts(query, top_k=config.TOP_K_RESULTS):
    """
    Perform similarity search using FAISS.
    
    Args:
        query (str): The query text to search for.
        top_k (int): Number of results to return.
        
    Returns:
        list: A list of tuples (text, source) of the most similar chunks.
    """
    # Load embeddings and FAISS index
    embeddings, texts, sources, index = load_embeddings()
    
    # Embed the query
    query_embedding = embedding_model.embed_query(query)
    
    # Search for similar vectors
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    # Return the corresponding texts and sources
    return [(texts[i], sources[i]) for i in indices[0]]


def get_retriever():
    """
    Get a LangChain retriever based on the vector store.
    
    Returns:
        Retriever: A LangChain retriever for semantic search.
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever()
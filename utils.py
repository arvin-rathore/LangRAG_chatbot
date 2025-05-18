"""
Utility functions for the PDF Chatbot.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
import config


def chunk_text(text):
    """
    Split text into manageable chunks using RecursiveCharacterTextSplitter.
    
    Args:
        text (str): The text to be chunked.
        
    Returns:
        list: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_text(text)
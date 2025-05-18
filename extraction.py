"""
Functions for extracting text from PDF files.
"""

import os
import fitz
from concurrent.futures import ProcessPoolExecutor
from utils import chunk_text
import config


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a single PDF file and chunk it.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        list: A list of text chunks extracted from the PDF.
    """
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return chunk_text(text)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return []


def extract_text_from_pdfs(pdf_dir=config.DATA_DIR):
    """
    Extract text from all PDF files in a directory using parallel processing.
    
    Args:
        pdf_dir (str): Directory containing PDF files. Defaults to config.DATA_DIR.
        
    Returns:
        tuple: (list of text chunks, list of corresponding source files)
    """
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) 
                if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return [], []
    
    texts, sources = [], []
    
    # Use parallel processing for faster extraction
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(extract_text_from_pdf, pdf_files))
    
    for pdf, chunks in zip(pdf_files, results):
        texts.extend(chunks)
        sources.extend([pdf] * len(chunks))
    
    return texts, sources
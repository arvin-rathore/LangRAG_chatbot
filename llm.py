"""
LLM interface for generating responses to queries.
"""

import google.generativeai as genai
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import config
from search import search_similar_texts, get_retriever


# Configure Gemini API
genai.configure(api_key=config.GEMINI_API_KEY)

# Initialize Gemini model
gemini_model = genai.GenerativeModel(model_name=config.GEMINI_MODEL_NAME)


def query_with_context(query):
    """
    Query the Gemini model with relevant context from PDFs.
    
    Args:
        query (str): The query to answer.
        
    Returns:
        tuple: (response text, list of sources)
    """
    # Get relevant text chunks
    retrieved_texts = search_similar_texts(query)
    
    # Combine context
    context = "\n".join([text for text, _ in retrieved_texts])
    
    # Generate response with Gemini
    response = gemini_model.generate_content(
        f"Context: {context}\nQuestion: {query}\nAnswer:"
    )
    
    # Return response and sources
    return response.text, [src for _, src in retrieved_texts]


def get_retrieval_qa_chain():
    """
    Get a LangChain RetrievalQA chain for more complex question answering.
    
    Returns:
        RetrievalQA: A LangChain QA chain.
    """
    retriever = get_retriever()
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name=config.GPT_MODEL_NAME), 
        retriever=retriever
    )
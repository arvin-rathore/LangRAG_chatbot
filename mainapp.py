import asyncio
import os
import pickle
import faiss
import fitz 
import numpy as np
import streamlit as st
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from concurrent.futures import ProcessPoolExecutor

# Gemini API
genai.configure(api_key="API_KEY_HERE")

# Initialize Gemini model
llm = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Configuration
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDINGS_FILE = "embeddings.pkl"
INDEX_FILE = "faiss_index.idx"
PDF_DIR = "data"
CHUNK_SIZE = 512  # Token-based chunking
CHUNK_OVERLAP = 50

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)


def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)


def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return chunk_text(text)


def extract_text_from_pdfs(pdf_dir):

    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    texts, sources = [], []
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(extract_text_from_pdf, pdf_files))
    
    for pdf, chunks in zip(pdf_files, results):
        texts.extend(chunks)
        sources.extend([pdf] * len(chunks))
    
    return texts, sources


def generate_and_store_embeddings():

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE):
        print("Embeddings already exist. Skipping generation.")
        return
    
    texts, sources = extract_text_from_pdfs(PDF_DIR)
    embeddings = embedding_model.embed_documents(texts)
    
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump((embeddings, texts, sources), f)
    
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_FILE)


def load_embeddings():

    if not (os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE)):
        print("Embeddings not found. Generating new ones...")
        generate_and_store_embeddings()
    
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings, texts, sources = pickle.load(f)
    index = faiss.read_index(INDEX_FILE)
    return embeddings, texts, sources, index


def search_similar_texts(query, top_k=3):
    """Perform similarity search using FAISS."""
    embeddings, texts, sources, index = load_embeddings()
    query_embedding = embedding_model.embed_query(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [(texts[i], sources[i]) for i in indices[0]]


def get_langchain_retrieval_qa():

    embeddings, texts, sources, index = load_embeddings()
    documents = [Document(page_content=text, metadata={"source": src}) for text, src in zip(texts, sources)]
    vectorstore = FAISS.from_documents(documents, embedding_model)
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4"), retriever=retriever)


# Streamlit
st.title("📄 PDF Chatbot with LangChain & Gemini")
query_text = st.text_input("Ask a question about the PDF content:")

if st.button("Get Answer") and query_text:
    retrieved_texts = search_similar_texts(query_text)
    context = "\n".join([text for text, _ in retrieved_texts])
    
    response = llm.generate_content(f"Context: {context}\nQuestion: {query_text}\nAnswer:")
    
    st.subheader("Response:")
    st.write(response.text)
    
    st.subheader("Sources:")
    for _, src in retrieved_texts:
        st.write(f"📄 {src}")

if __name__ == "__main__":
    generate_and_store_embeddings()

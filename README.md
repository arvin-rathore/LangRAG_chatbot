# LangRAG Chatbot

LangRAG is a chatbot built using **LangChain** and **Retrieval-Augmented Generation (RAG)**. It extracts text from PDFs, generates embeddings, and retrieves relevant documents using **FAISS**. The chatbot leverages **Google Gemini** for generating responses and provides a **Streamlit** interface for easy interaction. This project is ideal for **PDF-based question answering**.

---

## Features

- **PDF Text Extraction**: Extracts text from PDF files.
- **Chunking & Embedding**: Splits text into chunks and generates embeddings using HuggingFace models.
- **Vector Search with FAISS**: Efficiently retrieves relevant text snippets based on similarity search.
- **LLM Integration**: Uses Google Gemini to generate natural language responses.
- **LangChain RetrievalQA**: Implements a retrieval-based QA system.
- **Streamlit UI**: Provides a user-friendly interface for querying PDFs.

---

## Installation

Ensure you have Python installed (>= 3.8), then install dependencies:

```sh
pip install -r requirements.txt
```

---

## Folder Structure

```
LangRAG_chatbot/
│── data/
│   └── sample_pdf/  # Sample PDF files
│── faiss_index/
│   ├── index.faiss  # FAISS index file
│   ├── index.pkl    # Pickle file for indexing
│── main.py          # Main script to run the chatbot
│── requirements.txt  # Dependencies
│── README.md        # Project documentation
```

---

## Usage

### 1. Run the chatbot

```sh
streamlit run main.py
```

### 2. Upload PDFs
Place PDFs inside the `data/sample_pdf/` folder before running the chatbot.

### 3. Ask Questions
- Enter your query in the Streamlit input box.
- Click on the **"Get Answer"** button to retrieve responses.

---

## Technical Overview

### 1. **Embedding Model**
- Uses `BAAI/bge-small-en-v1.5` for generating document embeddings.
- Embeddings are stored and retrieved using FAISS.

### 2. **PDF Processing**
- Uses `PyMuPDF` to extract text from PDFs.
- Chunks text using `RecursiveCharacterTextSplitter`.

### 3. **Retrieval-Augmented Generation (RAG)**
- FAISS retrieves relevant document chunks based on query embeddings.
- Gemini LLM generates responses using retrieved context.

### 4. **Streamlit Interface**
- Allows users to input questions.
- Displays retrieved responses and sources.

---

## Code Structure

### **Embedding Generation**
- Extracts text from PDFs.
- Converts text to embeddings and stores them in FAISS.

### **Retrieval & Response Generation**
- Retrieves similar text from FAISS.
- Feeds retrieved context to the Gemini LLM.
- Returns responses to the user.

---

## Dependencies

Required Python packages:

```
faiss-cpu
streamlit
numpy
langchain
langchain_community
langchain_huggingface
google-generativeai
fitz (PyMuPDF)
pickle
```

Install dependencies via:

```sh
pip install -r requirements.txt
```

---

## Future Improvements

- Add support for **multiple LLMs** (GPT-4, Llama-3, etc.).
- Enhance document chunking and retrieval mechanisms.
- Implement **metadata filtering** for improved accuracy.
- Extend UI for multi-document querying.

---

## License

This project is open-source and available under the MIT License.




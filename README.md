PDF Chatbot
A simple modular PDF chatbot that allows you to ask questions about your PDF documents using semantic search and LLMs.
Features

📄 Extract text from multiple PDF files
🔍 Generate embeddings for semantic search
🧠 Use FAISS for efficient vector search
💬 Leverage Google's Gemini or OpenAI's GPT for responses
🌐 Interactive Streamlit web interface

Project Structure
pdf_chatbot/
│
├── app.py                  # Main Streamlit application
├── embedding.py            # Embedding creation and management
├── extraction.py           # PDF text extraction
├── search.py               # Vector search functionality
├── llm.py                  # Gemini and LLM integration
├── utils.py                # Utility functions
├── config.py               # Configuration settings
│
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
├── .gitignore              # Git ignore file
│
└── data/                   # Directory for PDF files
    └── sample.pdf          # Sample PDF for testing
Setup Instructions

Clone the repository:
bashgit clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot

Create a virtual environment:
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
bashpip install -r requirements.txt

Configure API keys:

Open config.py
Add your Gemini API key (replace "API_KEY_HERE" with your actual API key)
Optionally, add your OpenAI API key if using GPT models


Add your PDF files:

Place your PDF files in the data/ directory


Run the application:
bashstreamlit run app.py


Usage

Navigate to the Streamlit app in your browser (usually http://localhost:8501)
Enter your question about the PDF content in the text input
Click "Get Answer" to receive a response based on the content of your PDFs
The sources used to generate the response will be displayed below the answer

Customization

Change embedding model: Edit EMBEDDING_MODEL_NAME in config.py
Adjust chunk size: Modify CHUNK_SIZE and CHUNK_OVERLAP in config.py
Switch LLM: The code supports both Gemini and OpenAI models

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
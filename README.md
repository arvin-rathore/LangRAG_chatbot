PDF Chatbot
PDF Chatbot is an intelligent document assistant that lets you have natural conversations with your PDF documents. Ask questions in plain language and get accurate, contextual answers powered by advanced language models.
Features

Multi-Document Support - Upload and chat with multiple PDFs simultaneously
Semantic Search - Advanced vector search finds relevant content beyond simple keyword matching
Smart Chunking - Optimized text segmentation for better context retention
Dual AI Support - Choose between Google's Gemini or OpenAI's GPT models
Source Citations - Answers cite their exact sources within your documents
Clean, Intuitive UI - Streamlit-powered interface that's easy to use
Built for Performance - FAISS vector database enables fast, efficient searches
Highly Customizable - Adjust chunking, models, and search parameters to suit your needs

Quick Start
Prerequisites

Python 3.7+
Pip package manager
API key for Google Gemini or OpenAI (depending on your preferred LLM)

Set up a virtual environment

bash# Create and activate virtual environment
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt

Configure your API keys

Create a .env file in the project root (or update config.py):
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional

Launch the application

bashstreamlit run app.py

Open your browser at http://localhost:8501

💬 How to Use

Upload your PDFs - Use the file uploader in the sidebar
Wait for processing - The system will extract text and generate embeddings
Ask questions - Type your question in the chat input
Get answers - Receive contextual responses based on your PDFs
View sources - See exactly which parts of your documents provided the answers

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
⚙️ Advanced Configuration
You can customize the behavior of PDF Chatbot by editing config.py:
ParameterDescriptionDefaultCHUNK_SIZEThe size of text chunks for processing1000CHUNK_OVERLAPOverlap between chunks to maintain context200EMBEDDING_MODEL_NAMEModel used for generating embeddings"all-MiniLM-L6-v2"TOP_K_RESULTSNumber of chunks to retrieve for each query5TEMPERATUREControls creativity of LLM responses0.2
🔧 Troubleshooting
Problem: Slow processing of large PDFs

Try adjusting CHUNK_SIZE to a larger value
Consider using a more powerful machine for embedding generation

Problem: Inaccurate answers

Increase TOP_K_RESULTS to provide more context to the LLM
Try a different embedding model with higher dimensionality
Adjust TEMPERATURE to get more deterministic responses

🤝 Contributing
Contributions are welcome! Here's how you can help:

Fork the repository
Create a new branch (git checkout -b feature/amazing-feature)
Make your changes
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

Please make sure to update tests as appropriate.
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgements

Streamlit for the awesome UI framework
Google Gemini and OpenAI for their powerful language models
FAISS for efficient similarity search
LangChain for inspiration on document processing techniques


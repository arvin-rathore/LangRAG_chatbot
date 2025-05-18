"""
Main Streamlit application for the PDF Chatbot.
"""

import streamlit as st
from embedding import generate_and_store_embeddings
from llm import query_with_context


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 PDF Chatbot with LangChain & Gemini")
    st.write("Ask questions about the PDFs in your data directory.")
    
    # Ensure embeddings exist
    generate_and_store_embeddings()
    
    # Query input
    query_text = st.text_input("Ask a question about the PDF content:")
    
    if st.button("Get Answer") and query_text:
        with st.spinner("Searching PDFs and generating response..."):
            # Get response and sources
            response, sources = query_with_context(query_text)
            
            # Display response
            st.subheader("Response:")
            st.write(response)
            
            # Display sources
            st.subheader("Sources:")
            for src in sources:
                st.write(f"📄 {src}")


if __name__ == "__main__":
    main()
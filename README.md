# 📑 RAG Demo with Streamlit & HuggingFace  

This project is a **Retrieval-Augmented Generation (RAG) demo** built with:  
- **Streamlit** for the user interface  
- **LangChain** for document loading, splitting, and retrieval  
- **HuggingFace Transformers** for the LLM  
- **FAISS** for vector similarity search  

The app allows users to upload a **PDF file**, process it into embeddings, and then **ask natural language questions** about the content.  

---

## 🚀 Features
- 📂 Upload any PDF document  
- 🔍 Chunk text into smaller pieces for efficient retrieval  
- 🧠 Generate embeddings with **sentence-transformers**  
- 📚 Store embeddings in **FAISS vector database**  
- 🤖 Query with a HuggingFace model (`flan-t5-small`)  
- 🎯 Get concise answers from your document 

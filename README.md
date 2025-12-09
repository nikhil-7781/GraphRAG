

# 🕸️ GraphLLM - PDF Knowledge Graph + RAG System

Transform PDFs into interactive knowledge graphs with AI-powered Q&A.

## 🚀 Features

- **📄 PDF Processing:** Extract text, tables, and images from PDFs
- **🕸️ Knowledge Graph Generation:** Build semantic graphs using Gemini AI
- **🔍 Vector Search:** FAISS-powered semantic search with sentence transformers
- **💬 RAG Chat:** Ask questions and get answers with source citations
- **🎨 Interactive Visualization:** Explore knowledge graphs in your browser

## 🛠️ Technology Stack

- **LLM:** Google Gemini (gemini-2.5-flash)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Vector Store:** FAISS with HNSW index
- **Graph:** NetworkX (in-memory)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Vanilla JS with D3.js/Cytoscape

## 📋 Setup

### Required: Gemini API Key

This app requires a Google Gemini API key:

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add it as a **Secret** in Hugging Face Spaces settings:
   - Name: `GEMINI_API_KEY`
   - Value: Your API key

### Configuration (Optional)

You can set these environment variables in Space Settings:

```bash
# LLM Settings
GEMINI_MODEL=gemini-2.5-flash     # Gemini model
LLM_TEMPERATURE=0.0               # Temperature for extraction

# Embedding Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
```

## 🎯 Usage

1. **Upload PDF:** Click "Upload PDF" and select your document
2. **Wait for Processing:** The system will:
   - Extract text chunks
   - Generate embeddings
   - Build knowledge graph with Gemini
3. **Explore Graph:** Click nodes to see details and related concepts
4. **Ask Questions:** Use the chat interface for Q&A with citations

## 📊 Graph Generation

- **Per-Page Extraction:** Max 2 concepts per page (quality over quantity)
- **Parallel Processing:** All pages processed concurrently via Gemini API
- **Strict Filtering:** Only technical/domain-specific concepts
- **Co-occurrence Relationships:** Concepts on same page are linked

## 🎨 Frontend

The frontend is a single-page application located in `/frontend/`:
- `index.html` - Main UI
- `app.js` - Graph visualization & API calls
- `styles.css` - Styling

Access it at: `http://your-space-url.hf.space/frontend/`


## 📦 Docker

This Space uses Docker for deployment:
- Base: `python:3.12-slim`
- Port: 7860 (HF Spaces default)
- Health check enabled
- Persistent data directory

## 🤝 Credits

- **LLM:** Google Gemini
- **Embeddings:** Hugging Face sentence-transformers


---


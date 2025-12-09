"""
FastAPI Backend - Main Application
Provides REST API for PDF upload, graph retrieval, chat, and node details
"""
# Suppress PyTorch JIT warnings (harmless, just noisy during import)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message="Unable to retrieve source")

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
import sys
from pathlib import Path
import os
import uuid
import pickle
from datetime import datetime
from typing import List, Dict, Any

from config import settings, ensure_directories
from models import (
    UploadResponse, GraphResponse, ChatRequest, ChatResponse,
    NodeDetailResponse, AdminStatus, SourceCitation, GraphNode, GraphEdge
)
from pdf_processor import PDFProcessor
from embedding_service import EmbeddingService
from llm_service import LLMService
from gemini_extractor import GeminiExtractor
from graph_store import GraphStore
from graph_builder import GraphBuilder
from rag_agent import RAGAgent


# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>"
)
logger.add(
    f"{settings.logs_dir}/app.log",
    rotation="500 MB",
    retention="10 days",
    level=settings.log_level
)

# Initialize services
ensure_directories()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="PDF Knowledge Graph and RAG System"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
logger.info("Initializing PDFProcessor...")
pdf_processor = PDFProcessor()

logger.info("Initializing EmbeddingService...")
embedding_service = EmbeddingService()

logger.info("Initializing LLMService...")
llm_service = LLMService()

logger.info("Initializing GeminiExtractor (direct Gemini API)...")
triplet_extractor = GeminiExtractor(llm_service)

logger.info("Initializing GraphStore...")
graph_store = GraphStore(use_neo4j=False, embedding_service=embedding_service)

logger.info("Initializing GraphBuilder...")
graph_builder = GraphBuilder(graph_store, embedding_service)

logger.info("Initializing RAGAgent (LangGraph-based)...")
rag_agent = RAGAgent(graph_store, embedding_service, llm_service)

logger.info("✓ All services initialized successfully")

# In-memory storage for PDF metadata (use database in production)
pdf_metadata_store: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Try to load existing graph
    graph_path = os.path.join(settings.data_dir, "knowledge_graph.pkl")
    if os.path.exists(graph_path):
        try:
            graph_store.load(graph_path)
            logger.info("Loaded existing knowledge graph")
        except Exception as e:
            logger.warning(f"Failed to load existing graph: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down application")

    # Save graph
    graph_path = os.path.join(settings.data_dir, "knowledge_graph.pkl")
    try:
        graph_store.save(graph_path)
        logger.info("Saved knowledge graph")
    except Exception as e:
        logger.error(f"Failed to save graph: {e}")

    # Save FAISS index
    try:
        embedding_service.save()
        logger.info("Saved FAISS index")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")


@app.get("/")
async def root():
    """Serve the frontend HTML"""
    return FileResponse("frontend/index.html")


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload a PDF and trigger ingestion pipeline

    Returns immediately with pdf_id, processes in background
    """
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_size = 0
    content = await file.read()
    file_size = len(content)

    if file_size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum of {settings.max_file_size_mb}MB"
        )

    # Generate PDF ID
    pdf_id = str(uuid.uuid4())

    # Save file
    filepath = os.path.join(settings.upload_dir, f"{pdf_id}.pdf")
    with open(filepath, 'wb') as f:
        f.write(content)

    logger.info(f"Uploaded PDF: {file.filename} (ID: {pdf_id})")

    # Store metadata with detailed progress tracking
    pdf_metadata_store[pdf_id] = {
        "filename": file.filename,
        "filepath": filepath,
        "status": "processing",
        "progress": {
            "stage": "starting",
            "message": "Upload complete, starting processing...",
            "percent": 0
        }
    }

    # Trigger background processing
    background_tasks.add_task(process_pdf_pipeline, pdf_id, filepath)

    return UploadResponse(
        pdf_id=pdf_id,
        filename=file.filename,
        status="processing",
        message="PDF uploaded successfully. Processing started in background."
    )


async def process_pdf_pipeline(pdf_id: str, filepath: str):
    """
    ⚡ OPTIMIZED: Full ingestion pipeline with progress tracking

    Steps:
    0. Clear existing graph and index (FRESH START)
    1. Extract chunks from PDF
    2. Create embeddings
    3. Add to vector index
    4. Extract triples (PARALLEL)
    5. Build knowledge graph (NO PRUNING)
    """
    def update_progress(stage: str, message: str, percent: int):
        """Update progress in metadata store"""
        if pdf_id in pdf_metadata_store:
            pdf_metadata_store[pdf_id]["progress"] = {
                "stage": stage,
                "message": message,
                "percent": percent
            }

    try:
        logger.info(f"Starting ingestion pipeline for PDF {pdf_id}")

        # Step 0: CLEAR EVERYTHING for fresh extraction
        update_progress("clearing", "Clearing previous data...", 5)
        logger.info("Step 0: Clearing existing graph and embeddings for fresh extraction")
        graph_store.clear()
        embedding_service.clear()
        logger.info("✓ Cleared all existing data")

        # Step 1: Extract chunks (with caching)
        cache_path = os.path.join(settings.data_dir, f"chunks_{pdf_id}.pkl")

        if os.path.exists(cache_path):
            # Load cached chunks (saves 2-3s on reindex)
            update_progress("extraction", "Loading cached text extraction...", 15)
            logger.info("⚡ Step 1: Loading cached chunks from previous extraction")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                refined_chunks = cache_data['chunks']
                metadata = cache_data['metadata']
            logger.info(f"✓ Loaded {len(refined_chunks)} cached chunks (skipped PDF processing)")
            update_progress("extraction", f"Loaded {len(refined_chunks)} cached chunks", 25)
        else:
            # Extract and cache chunks for future reindexing
            update_progress("extraction", "Extracting text from PDF...", 15)
            logger.info("Step 1: Extracting chunks from PDF")
            chunks, metadata = pdf_processor.process_pdf(filepath, pdf_id)
            refined_chunks = pdf_processor.chunk_text(chunks)

            # Cache for future use
            with open(cache_path, 'wb') as f:
                pickle.dump({'chunks': refined_chunks, 'metadata': metadata}, f)
            logger.info(f"✓ Cached {len(refined_chunks)} chunks for future reindexing")
            update_progress("extraction", f"Extracted {len(refined_chunks)} chunks", 25)

        # Step 2: Create embeddings
        update_progress("embeddings", f"Creating embeddings for {len(refined_chunks)} chunks...", 35)
        logger.info(f"Step 2: Creating embeddings for {len(refined_chunks)} chunks")
        embeddings = embedding_service.create_embeddings(refined_chunks)
        update_progress("embeddings", "Embeddings created", 50)

        # Step 3: Add to vector index
        update_progress("indexing", "Building vector index...", 55)
        logger.info("Step 3: Adding to vector index")
        embedding_service.add_to_index(refined_chunks, embeddings)
        embedding_service.save()
        update_progress("indexing", "Vector index complete", 60)

        # Step 4: Extract triples using Gemini (direct API - PARALLEL)
        update_progress("extraction", "Extracting concepts with AI (parallel)...", 65)
        logger.info("Step 4: Extracting triples using Gemini (PARALLEL per-page, 2 concepts max)")
        canonical_triples = await triplet_extractor.extract_from_chunks(
            refined_chunks,
            use_llm=True  # Direct Gemini API calls
        )
        update_progress("extraction", f"Extracted {len(canonical_triples)} relationships", 80)

        # Step 5: Build graph
        update_progress("graph", "Building knowledge graph...", 85)
        logger.info("Step 5: Building knowledge graph")
        num_nodes, num_edges = await graph_builder.build_graph(canonical_triples)
        update_progress("graph", f"Graph complete: {num_nodes} nodes, {num_edges} edges", 95)

        # Save graph
        update_progress("saving", "Saving graph to disk...", 98)
        graph_path = os.path.join(settings.data_dir, "knowledge_graph.pkl")
        graph_store.save(graph_path)

        # Update metadata
        update_progress("completed", f"✓ Complete! {num_nodes} nodes, {num_edges} edges", 100)
        pdf_metadata_store[pdf_id]["status"] = "completed"
        pdf_metadata_store[pdf_id]["num_chunks"] = len(refined_chunks)
        pdf_metadata_store[pdf_id]["num_nodes"] = num_nodes
        pdf_metadata_store[pdf_id]["num_edges"] = num_edges

        logger.info(f"✓ Completed ingestion for PDF {pdf_id}: {num_nodes} nodes, {num_edges} edges")

    except Exception as e:
        logger.error(f"❌ Failed to process PDF {pdf_id}: {e}", exc_info=True)
        pdf_metadata_store[pdf_id]["status"] = "failed"
        pdf_metadata_store[pdf_id]["error"] = str(e)
        update_progress("error", f"Error: {str(e)[:100]}", 0)


@app.get("/graph", response_model=GraphResponse)
async def get_graph(pdf_id: str = None):
    """
    Get the knowledge graph

    Args:
        pdf_id: Optional filter by PDF ID

    Returns:
        Graph nodes and edges
    """
    nodes = graph_store.get_all_nodes()
    edges = graph_store.get_all_edges()

    logger.info(f"Returning {len(nodes)} nodes, {len(edges)} edges")

    # Filter by PDF if specified
    if pdf_id:
        # Filter nodes and edges that belong to this PDF
        # This requires tracking PDF ID in supporting chunks
        pass

    return GraphResponse(
        nodes=nodes,
        edges=edges,
        metadata={
            "total_nodes": len(nodes),
            "total_edges": len(edges)
        }
    )


@app.get("/node/{node_id}", response_model=NodeDetailResponse)
async def get_node_details(node_id: str):
    """
    Get detailed information about a node

    Includes:
    - Node metadata
    - LLM-generated summary with citations
    - Supporting chunks
    - Related nodes
    """
    node = graph_store.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    # Check if summary is cached in node metadata
    if "cached_summary" in node.metadata:
        logger.info(f"✓ Using cached summary for node {node.label}")
        summary = node.metadata["cached_summary"]
        search_results = None  # Use node's supporting chunks for sources
    else:
        # Generate summary (first time)
        logger.info(f"⏳ Generating summary for node {node.label}...")

        # Get supporting chunks using semantic search on the node label
        # This finds chunks that are semantically similar to the concept
        search_results = embedding_service.search(
            query=node.label,
            top_k=3  # Reduced from 5 to 3 for faster processing
        )

        # Prepare chunks for LLM
        chunks_for_llm = []
        if search_results:
            chunks_for_llm = [
                {
                    "page_number": meta.get("page_number", 0),
                    "text": meta.get("text", "")
                }
                for meta, score in search_results
            ]

        # Fallback: if no chunks found, create a basic summary
        if not chunks_for_llm:
            logger.warning(f"No chunks found for node {node.label}, using basic summary")
            chunks_for_llm = [
                {
                    "page_number": chunk.page_number or 0,
                    "text": chunk.snippet or ""
                }
                for chunk in node.supporting_chunks[:3]
            ]

        # Generate summary
        summary = await llm_service.summarize_node(node.label, chunks_for_llm)

        # Cache summary in node metadata (don't cache search_results - they're not serializable)
        node.metadata["cached_summary"] = summary
        node.metadata["cache_timestamp"] = str(datetime.utcnow())

        # Update the node in the graph store
        graph_store.update_node(node)
        logger.info(f"✓ Cached summary for node {node.label}")

    # Get related nodes
    neighbors = graph_store.get_neighbors(node_id)
    related_nodes = [
        {
            "node_id": neighbor.node_id,
            "label": neighbor.label,
            "relation": edge.relation.value,
            "confidence": edge.confidence
        }
        for neighbor, edge in neighbors[:10]  # Limit to top 10
    ]

    # Build source citations
    sources = []
    if search_results is not None:
        # Use search results (freshly generated summary)
        for meta, score in search_results[:5]:
            text = meta.get("text", "")
            snippet = text[:120] + "..." if len(text) > 120 else text
            sources.append(SourceCitation(
                page_number=meta.get("page_number", 0),
                snippet=snippet,
                chunk_id=meta.get("chunk_id", ""),
                score=score
            ))
    else:
        # Use node's supporting chunks (cached summary)
        sources = [
            SourceCitation(
                page_number=chunk.page_number or 0,
                snippet=chunk.snippet or "",
                chunk_id=chunk.chunk_id,
                score=chunk.score
            )
            for chunk in node.supporting_chunks[:5]
        ]

    return NodeDetailResponse(
        node_id=node.node_id,
        label=node.label,
        type=node.type,
        summary=summary,
        sources=sources,
        related_nodes=related_nodes
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Agent-based RAG chat endpoint

    Uses LangGraph agent with multiple tools:
    - vector_search: Semantic search through chunks
    - graph_search: Find concepts in knowledge graph
    - get_node_details: Get detailed node information
    - get_related_nodes: Graph traversal for relationships
    - get_chunk_by_id: Retrieve specific chunks

    The agent intelligently decides which tools to use based on the query
    """
    logger.info(f"🤖 Agent chat request: '{request.query}'")

    # Use agent-based RAG
    response = await rag_agent.chat(
        query=request.query,
        pdf_id=request.pdf_id,
        include_citations=True
    )

    # Limit sources to requested max
    if len(response.sources) > request.max_sources:
        response.sources = response.sources[:request.max_sources]

    return response


@app.get("/status/{pdf_id}")
async def get_pdf_status(pdf_id: str):
    """Get processing status for a specific PDF"""
    if pdf_id not in pdf_metadata_store:
        raise HTTPException(status_code=404, detail="PDF not found")

    metadata = pdf_metadata_store[pdf_id]
    return {
        "pdf_id": pdf_id,
        "filename": metadata.get("filename"),
        "status": metadata.get("status"),
        "progress": metadata.get("progress", {}),
        "num_nodes": metadata.get("num_nodes", 0),
        "num_edges": metadata.get("num_edges", 0),
        "error": metadata.get("error")
    }


@app.get("/admin/status", response_model=AdminStatus)
async def admin_status():
    """Get system status and statistics"""
    faiss_stats = embedding_service.get_stats()

    return AdminStatus(
        total_pdfs=len(pdf_metadata_store),
        total_chunks=faiss_stats["num_chunks"],
        total_nodes=len(graph_store.get_all_nodes()),
        total_edges=len(graph_store.get_all_edges()),
        vector_index_size=faiss_stats["total_vectors"],
        recent_logs=[]  # Would fetch from logs in production
    )


@app.post("/admin/reindex")
async def admin_reindex(pdf_id: str):
    """Re-run ingestion for a PDF"""
    if pdf_id not in pdf_metadata_store:
        raise HTTPException(status_code=404, detail="PDF not found")

    filepath = pdf_metadata_store[pdf_id]["filepath"]

    # Clear existing data for this PDF (would need better tracking)
    # For now, just re-run the pipeline

    await process_pdf_pipeline(pdf_id, filepath)

    return {"message": "Reindexing started", "pdf_id": pdf_id}


@app.post("/admin/clear")
async def admin_clear():
    """Clear all data"""
    graph_store.clear()
    embedding_service.clear()
    pdf_metadata_store.clear()

    logger.warning("All data cleared by admin")

    return {"message": "All data cleared"}


# Mount static files for frontend
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

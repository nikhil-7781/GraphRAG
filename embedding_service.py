"""
Embedding & Vector Index Service
Handles embedding generation and FAISS vector store management
"""
# Import SentenceTransformer lazily to avoid hanging on startup
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
import pickle
import os
from models import Chunk, EmbeddingEntry
from config import settings
import json


class EmbeddingService:
    """
    Service for creating embeddings and managing FAISS vector index
    Uses lazy loading for the embedding model (loads on first use)
    """

    def __init__(self):
        logger.info(f"EmbeddingService initialized (model will load on first use)")
        self._model = None  # Lazy-loaded
        self.dimension = settings.embedding_dimension
        self.index: Optional[faiss.Index] = None
        self.chunk_metadata: Dict[int, Dict[str, Any]] = {}  # index_id -> metadata
        self._initialize_index()

    @property
    def model(self):
        """Lazy-load the embedding model on first access"""
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            # Import only when needed to avoid hanging on startup
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(settings.embedding_model)
            logger.info(f"✓ Embedding model loaded successfully")
        return self._model

    def _initialize_index(self):
        """Initialize or load FAISS index"""
        index_path = os.path.join(settings.faiss_index_path, "index.faiss")
        metadata_path = os.path.join(settings.faiss_index_path, "metadata.pkl")

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            logger.info("Loading existing FAISS index")
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.chunk_metadata = pickle.load(f)
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
        else:
            logger.info("Creating new FAISS index (optimized)")
            # Use HNSW for better performance on larger datasets
            # HNSW is ~10x faster than flat index with 99%+ accuracy
            # M=32 is good balance (higher M = more accurate but slower)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            # Set ef construction (higher = better quality, slower build)
            self.index.hnsw.efConstruction = 40
            # Set ef search (higher = better recall, slower search)
            self.index.hnsw.efSearch = 16
            self.chunk_metadata = {}
            logger.info("Using HNSW index for faster approximate search")

    def create_embeddings(self, chunks: List[Chunk]) -> List[EmbeddingEntry]:
        """
        ⚡ OPTIMIZED: Create embeddings with larger batches and parallel processing

        Args:
            chunks: List of Chunk objects

        Returns:
            List of EmbeddingEntry objects
        """
        texts = [chunk.text for chunk in chunks]
        logger.info(f"⚡ Creating embeddings for {len(texts)} chunks (batch_size={settings.embedding_batch_size})")

        import time
        start = time.time()

        # Batch encode with optimized settings
        embeddings = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,  # Disable for less overhead
            convert_to_numpy=True,
            normalize_embeddings=True  # Built-in normalization is faster
        )

        elapsed = time.time() - start
        logger.info(f"✓ Created {len(embeddings)} embeddings in {elapsed:.2f}s ({len(embeddings)/elapsed:.1f} chunks/sec)")

        # Create embedding entries
        embedding_entries = []
        for chunk, embedding in zip(chunks, embeddings):
            entry = EmbeddingEntry(
                chunk_id=chunk.chunk_id,
                embedding=embedding.tolist(),
                metadata={
                    "pdf_id": chunk.pdf_id,
                    "page_number": chunk.page_number,
                    "type": chunk.type.value,
                    "char_range": chunk.char_range
                }
            )
            embedding_entries.append(entry)

        return embedding_entries

    def add_to_index(self, chunks: List[Chunk], embeddings: List[EmbeddingEntry]):
        """
        Add chunks and their embeddings to FAISS index

        Args:
            chunks: List of chunks
            embeddings: Corresponding embeddings
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        # Convert embeddings to numpy array
        embedding_array = np.array([e.embedding for e in embeddings]).astype('float32')

        # Get current index size (starting ID for new chunks)
        start_id = self.index.ntotal

        # Add to FAISS index
        self.index.add(embedding_array)

        # Store metadata mapping
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            idx = start_id + i
            self.chunk_metadata[idx] = {
                "chunk_id": chunk.chunk_id,
                "pdf_id": chunk.pdf_id,
                "page_number": chunk.page_number,
                "type": chunk.type.value,
                "text": chunk.text,
                "char_range": chunk.char_range,
                "metadata": chunk.metadata
            }

        logger.info(f"Added {len(chunks)} chunks to index. Total: {self.index.ntotal}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_pdf_id: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar chunks

        Args:
            query: Query string
            top_k: Number of results to return
            filter_pdf_id: Optional PDF ID to filter results

        Returns:
            List of (chunk_metadata, score) tuples
        """
        # Encode and normalize query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search
        # Fetch more if we need to filter
        k = top_k * 10 if filter_pdf_id else top_k
        scores, indices = self.index.search(query_embedding, k)

        # Retrieve metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            metadata = self.chunk_metadata.get(idx)
            if metadata is None:
                continue

            # Apply filter if specified
            if filter_pdf_id and metadata.get("pdf_id") != filter_pdf_id:
                continue

            results.append((metadata, float(score)))

            if len(results) >= top_k:
                break

        return results

    def search_by_chunk_ids(self, chunk_ids: List[str], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find similar chunks to a set of chunk IDs (for node expansion)

        Args:
            chunk_ids: List of chunk IDs
            top_k: Number of similar chunks per input chunk

        Returns:
            List of (chunk_metadata, score) tuples
        """
        # Find the chunks in metadata
        chunk_indices = []
        for idx, meta in self.chunk_metadata.items():
            if meta["chunk_id"] in chunk_ids:
                chunk_indices.append(idx)

        if not chunk_indices:
            return []

        # Get embeddings for these chunks
        # Note: FAISS doesn't have a direct "get vector" API for IndexFlatIP
        # We'll search from the index using reconstruct (if supported)
        results = []
        for idx in chunk_indices:
            # Reconstruct vector (works for Flat indices)
            try:
                vector = self.index.reconstruct(idx)
                vector = vector.reshape(1, -1)
                scores, indices = self.index.search(vector, top_k + 1)  # +1 to exclude self

                for score, res_idx in zip(scores[0], indices[0]):
                    if res_idx == idx:  # Skip self
                        continue
                    if res_idx == -1:
                        continue

                    metadata = self.chunk_metadata.get(res_idx)
                    if metadata:
                        results.append((metadata, float(score)))
            except Exception as e:
                logger.warning(f"Could not reconstruct vector for index {idx}: {e}")

        # Sort by score and return top
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self):
        """Save FAISS index and metadata to disk"""
        os.makedirs(settings.faiss_index_path, exist_ok=True)

        index_path = os.path.join(settings.faiss_index_path, "index.faiss")
        metadata_path = os.path.join(settings.faiss_index_path, "metadata.pkl")

        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunk_metadata, f)

        logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")

    def clear(self):
        """Clear the index and metadata"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_metadata = {}
        logger.info("Cleared FAISS index")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "num_chunks": len(self.chunk_metadata)
        }

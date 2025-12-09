"""
Basic tests for GraphLLM components
"""
import pytest
from models import Chunk, ChunkType, GraphNode, GraphEdge, Triple, NodeType, RelationType
from config import settings


def test_chunk_creation():
    """Test chunk model creation"""
    chunk = Chunk(
        pdf_id="test-pdf",
        page_number=1,
        char_range=(0, 100),
        type=ChunkType.PARAGRAPH,
        text="This is a test chunk."
    )

    assert chunk.pdf_id == "test-pdf"
    assert chunk.page_number == 1
    assert chunk.type == ChunkType.PARAGRAPH
    assert chunk.text == "This is a test chunk."


def test_graph_node_creation():
    """Test graph node creation"""
    node = GraphNode(
        label="Test Concept",
        type=NodeType.CONCEPT,
        aliases=["test", "concept"],
        supporting_chunks=[],
        importance_score=0.75
    )

    assert node.label == "Test Concept"
    assert node.type == NodeType.CONCEPT
    assert node.importance_score == 0.75


def test_graph_edge_creation():
    """Test graph edge creation"""
    edge = GraphEdge(
        from_node="node1",
        to_node="node2",
        relation=RelationType.USES,
        confidence=0.8,
        supporting_chunks=[]
    )

    assert edge.from_node == "node1"
    assert edge.to_node == "node2"
    assert edge.relation == RelationType.USES
    assert edge.confidence == 0.8


def test_triple_creation():
    """Test triple model"""
    triple = Triple(
        subject="Machine Learning",
        predicate="uses",
        object="Neural Networks",
        confidence=0.9,
        page_number=5
    )

    assert triple.subject == "Machine Learning"
    assert triple.predicate == "uses"
    assert triple.object == "Neural Networks"
    assert triple.confidence == 0.9


def test_settings_load():
    """Test configuration loading"""
    assert settings.app_name == "GraphLLM"
    assert settings.chunk_size > 0
    assert settings.embedding_model is not None


@pytest.mark.asyncio
async def test_pdf_processor_import():
    """Test PDF processor can be imported"""
    from pdf_processor import PDFProcessor
    processor = PDFProcessor()
    assert processor is not None


@pytest.mark.asyncio
async def test_embedding_service_import():
    """Test embedding service can be imported"""
    from embedding_service import EmbeddingService
    # Note: This will load the model, may take time
    # service = EmbeddingService()
    # assert service is not None
    pass


@pytest.mark.asyncio
async def test_graph_store_import():
    """Test graph store can be imported"""
    from graph_store import GraphStore
    store = GraphStore(use_neo4j=False)
    assert store is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

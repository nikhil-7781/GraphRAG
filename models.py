"""
Data models for GraphLLM system following the manual specifications
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid


# Enums
class ChunkType(str, Enum):
    """Types of chunks extracted from PDF"""
    PARAGRAPH = "paragraph"
    CODE = "code"
    TABLE = "table"
    IMAGE = "image"
    IMAGE_TEXT = "image_text"


class NodeType(str, Enum):
    """Types of graph nodes"""
    CONCEPT = "concept"
    PERSON = "person"
    METHOD = "method"
    TERM = "term"
    CLASS = "class"
    FUNCTION = "function"
    ENTITY = "entity"


class RelationType(str, Enum):
    """Canonical relation types for edges"""
    IS_A = "is_a"
    PART_OF = "part_of"
    METHOD_OF = "method_of"
    CAUSES = "causes"
    USES = "uses"
    RELATED_TO = "related_to"
    DEFINED_AS = "defined_as"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    SIMILAR_TO = "similar_to"
    OBSERVES = "observes"
    MEASURES = "measures"
    PRODUCES = "produces"
    CONTAINS = "contains"
    AFFECTS = "affects"
    ENABLES = "enables"
    REQUIRES = "requires"
    INTERACTS_WITH = "interacts_with"
    ENRICHES = "enriches"
    ENHANCES = "enhances"
    SUPPORTS = "supports"
    DESCRIBES = "describes"
    EXPLAINS = "explains"
    REFERS_TO = "refers_to"
    ASSOCIATED_WITH = "associated_with"


# Core Data Models

class Chunk(BaseModel):
    """Individual chunk of text/content from PDF"""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pdf_id: str
    page_number: int
    char_range: tuple[int, int]
    type: ChunkType
    text: str
    table_json: Optional[Dict[str, Any]] = None
    image_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EmbeddingEntry(BaseModel):
    """Vector embedding for a chunk"""
    chunk_id: str
    embedding: List[float]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SupportingChunk(BaseModel):
    """Reference to a chunk supporting a node or edge"""
    chunk_id: str
    score: float
    page_number: Optional[int] = None
    snippet: Optional[str] = None


class GraphNode(BaseModel):
    """Node in the knowledge graph"""
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    type: NodeType
    aliases: List[str] = Field(default_factory=list)
    supporting_chunks: List[SupportingChunk] = Field(default_factory=list)
    importance_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GraphEdge(BaseModel):
    """Edge in the knowledge graph"""
    edge_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    relation: RelationType
    confidence: float
    supporting_chunks: List[SupportingChunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        # FastAPI automatically serializes enums as their string values in JSON


class Triple(BaseModel):
    """Extracted triple from text"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_chunk_id: Optional[str] = None
    page_number: Optional[int] = None
    justification: Optional[str] = None


class CanonicalTriple(BaseModel):
    """LLM-canonicalized triple"""
    subject_label: str
    object_label: str
    relation: RelationType
    confidence: float
    justification: str
    page_number: int


# API Request/Response Models

class UploadResponse(BaseModel):
    """Response from PDF upload"""
    pdf_id: str
    filename: str
    status: str
    message: str
    num_pages: Optional[int] = None
    num_chunks: Optional[int] = None


class GraphResponse(BaseModel):
    """Response containing graph data"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceCitation(BaseModel):
    """Source citation with page number and snippet"""
    page_number: int
    snippet: str
    chunk_id: str
    score: Optional[float] = None


class NodeDetailResponse(BaseModel):
    """Response for node detail request"""
    node_id: str
    label: str
    type: NodeType
    summary: str
    sources: List[SourceCitation]
    related_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    raw_chunks: Optional[List[Chunk]] = None


class ChatMessage(BaseModel):
    """Chat message"""
    role: Literal["user", "assistant", "system"]
    content: str
    sources: Optional[List[SourceCitation]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Chat request"""
    query: str
    pdf_id: str
    include_citations: bool = True
    max_sources: int = 5


class ChatResponse(BaseModel):
    """Chat response with answer and citations"""
    answer: str
    sources: List[SourceCitation]
    context_chunks: Optional[List[str]] = None


class PDFMetadata(BaseModel):
    """Metadata for uploaded PDF"""
    pdf_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    filepath: str
    num_pages: int
    file_size_bytes: int
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_status: str = "pending"
    num_chunks: int = 0
    num_nodes: int = 0
    num_edges: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionLog(BaseModel):
    """Log entry for ingestion process"""
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pdf_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    stage: str
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None


class AdminStatus(BaseModel):
    """Admin status response"""
    total_pdfs: int
    total_chunks: int
    total_nodes: int
    total_edges: int
    vector_index_size: int
    recent_logs: List[IngestionLog]

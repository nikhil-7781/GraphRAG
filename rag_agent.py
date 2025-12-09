"""
Agent-Based RAG System using LangGraph
Provides intelligent query answering with tool use and multi-hop reasoning
"""
from typing import List, Dict, Any, TypedDict, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from loguru import logger
import asyncio

from models import SourceCitation, ChatResponse
from graph_store import GraphStore
from embedding_service import EmbeddingService
from llm_service import LLMService


class AgentState(TypedDict):
    """State for the RAG agent workflow"""
    messages: List  # Conversation history
    query: str  # Current user question
    pdf_id: str  # PDF context
    tool_results: Dict[str, Any]  # Results from tool executions
    reasoning_steps: List[str]  # Agent's reasoning process
    final_answer: str  # Synthesized answer
    citations: List[SourceCitation]  # Supporting citations
    next_action: str  # What to do next


class RAGAgent:
    """
    Intelligent RAG agent that uses multiple tools to answer questions

    Tools available:
    1. vector_search - Semantic search through document chunks
    2. graph_search - Find concepts in knowledge graph
    3. get_node_details - Get detailed info about a graph node
    4. get_related_nodes - Traverse graph relationships
    5. get_chunk_by_id - Retrieve specific chunks for citations
    """

    def __init__(self,
                 graph_store: GraphStore,
                 embedding_service: EmbeddingService,
                 llm_service: LLMService):
        """Initialize the RAG agent with necessary services"""
        self.graph_store = graph_store
        self.embedding_service = embedding_service
        self.llm_service = llm_service

        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

        logger.info("✓ RAG Agent initialized with LangGraph workflow")

    def _create_tools(self):
        """Create tool functions for the agent"""

        @tool
        def vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
            """
            Search document chunks using semantic similarity.

            Args:
                query: The search query
                top_k: Number of results to return

            Returns:
                List of relevant chunks with metadata and scores
            """
            logger.info(f"🔍 Tool: vector_search('{query}', top_k={top_k})")

            try:
                results = self.embedding_service.search(
                    query=query,
                    top_k=top_k
                )

                formatted_results = []
                for metadata, score in results:
                    formatted_results.append({
                        "text": metadata.get("text", ""),
                        "page_number": metadata.get("page_number", 0),
                        "chunk_id": metadata.get("chunk_id", ""),
                        "score": float(score)
                    })

                logger.info(f"  ✓ Found {len(formatted_results)} chunks")
                return formatted_results

            except Exception as e:
                logger.error(f"  ✗ vector_search failed: {e}")
                return []

        @tool
        def graph_search(concept: str) -> Dict[str, Any]:
            """
            Find a concept node in the knowledge graph.

            Args:
                concept: The concept to search for

            Returns:
                Node information if found, None otherwise
            """
            logger.info(f"🔍 Tool: graph_search('{concept}')")

            try:
                node = self.graph_store.get_node_by_label(concept)

                if node:
                    logger.info(f"  ✓ Found node: {node.label}")
                    return {
                        "node_id": node.node_id,
                        "label": node.label,
                        "type": node.type.value if hasattr(node.type, 'value') else node.type,
                        "importance": node.importance_score
                    }
                else:
                    logger.info(f"  ✗ No node found for '{concept}'")
                    return None

            except Exception as e:
                logger.error(f"  ✗ graph_search failed: {e}")
                return None

        @tool
        def get_node_details(node_id: str) -> Dict[str, Any]:
            """
            Get detailed information about a graph node.

            Args:
                node_id: The ID of the node

            Returns:
                Detailed node information including supporting chunks
            """
            logger.info(f"🔍 Tool: get_node_details('{node_id}')")

            try:
                node = self.graph_store.get_node(node_id)

                if not node:
                    logger.info(f"  ✗ Node not found")
                    return None

                # Get supporting chunks
                chunks = []
                for chunk in node.supporting_chunks[:5]:  # Top 5
                    chunks.append({
                        "page_number": chunk.page_number,
                        "snippet": chunk.snippet,
                        "score": chunk.score
                    })

                logger.info(f"  ✓ Got details for {node.label}")
                return {
                    "label": node.label,
                    "type": node.type.value if hasattr(node.type, 'value') else node.type,
                    "importance": node.importance_score,
                    "supporting_chunks": chunks
                }

            except Exception as e:
                logger.error(f"  ✗ get_node_details failed: {e}")
                return None

        @tool
        def get_related_nodes(node_id: str, max_neighbors: int = 5) -> List[Dict[str, Any]]:
            """
            Get nodes related to a given node (graph traversal).

            Args:
                node_id: The ID of the starting node
                max_neighbors: Maximum number of related nodes to return

            Returns:
                List of related nodes with relationship information
            """
            logger.info(f"🔍 Tool: get_related_nodes('{node_id}', max={max_neighbors})")

            try:
                neighbors = self.graph_store.get_neighbors(node_id)

                related = []
                for neighbor_node, edge in neighbors[:max_neighbors]:
                    relation_value = edge.relation.value if hasattr(edge.relation, 'value') else edge.relation
                    related.append({
                        "node_id": neighbor_node.node_id,
                        "label": neighbor_node.label,
                        "relation": relation_value,
                        "confidence": edge.confidence
                    })

                logger.info(f"  ✓ Found {len(related)} related nodes")
                return related

            except Exception as e:
                logger.error(f"  ✗ get_related_nodes failed: {e}")
                return []

        @tool
        def get_chunk_by_id(chunk_id: str) -> Dict[str, Any]:
            """
            Retrieve a specific chunk by its ID (for detailed citations).

            Args:
                chunk_id: The chunk identifier

            Returns:
                Chunk content and metadata
            """
            logger.info(f"🔍 Tool: get_chunk_by_id('{chunk_id}')")

            try:
                # Search by chunk_id in metadata
                # This is a simplified version - you may need to implement proper chunk lookup
                results = self.embedding_service.search_by_chunk_ids([chunk_id], top_k=1)

                if results:
                    metadata, score = results[0]
                    logger.info(f"  ✓ Found chunk")
                    return {
                        "text": metadata.get("text", ""),
                        "page_number": metadata.get("page_number", 0),
                        "chunk_id": chunk_id
                    }
                else:
                    logger.info(f"  ✗ Chunk not found")
                    return None

            except Exception as e:
                logger.error(f"  ✗ get_chunk_by_id failed: {e}")
                return None

        return [vector_search, graph_search, get_node_details, get_related_nodes, get_chunk_by_id]

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for the agent"""

        workflow = StateGraph(AgentState)

        # Define workflow nodes
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.add_edge(START, "plan")
        workflow.add_conditional_edges(
            "plan",
            self._should_use_tools,
            {
                "tools": "execute_tools",
                "direct": "synthesize"
            }
        )
        workflow.add_edge("execute_tools", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow

    def _plan_node(self, state: AgentState) -> AgentState:
        """Agent decides which tools to use"""
        logger.info("🤖 Agent: Planning which tools to use...")

        query = state["query"]

        # Simple heuristic-based planning (can be enhanced with LLM)
        tools_to_use = []
        reasoning = []

        # Always use vector search for semantic matching
        tools_to_use.append("vector_search")
        reasoning.append("Use vector search for semantic document retrieval")

        # Check if query mentions specific concepts (use graph)
        if any(word in query.lower() for word in ["relate", "connection", "link", "between"]):
            tools_to_use.append("graph_search")
            reasoning.append("Query asks about relationships - use graph search")

        # Check if asking about a specific concept
        if any(word in query.lower() for word in ["what is", "define", "explain"]):
            tools_to_use.append("graph_search")
            reasoning.append("Query asks for concept definition - check graph")

        state["tool_results"] = {"planned_tools": tools_to_use}
        state["reasoning_steps"] = reasoning
        state["next_action"] = "tools" if tools_to_use else "direct"

        logger.info(f"  Plan: {tools_to_use}")
        return state

    def _should_use_tools(self, state: AgentState) -> str:
        """Decide if tools are needed"""
        return state.get("next_action", "direct")

    def _execute_tools_node(self, state: AgentState) -> AgentState:
        """Execute the planned tools"""
        logger.info("🔧 Agent: Executing tools...")

        query = state["query"]
        planned_tools = state["tool_results"].get("planned_tools", [])
        results = {}

        # Create tools
        tools_map = {}
        for tool in self._create_tools():
            tools_map[tool.name] = tool

        # Execute tools
        if "vector_search" in planned_tools:
            vector_tool = tools_map["vector_search"]
            results["vector_results"] = vector_tool.invoke({"query": query, "top_k": 5})

        if "graph_search" in planned_tools:
            # Extract main concept from query (simplified)
            # In production, use NER or LLM to extract concept
            words = query.lower().split()
            potential_concepts = [w for w in words if len(w) > 4 and w not in ["what", "how", "does", "relate"]]

            for concept in potential_concepts[:2]:  # Try first 2
                graph_tool = tools_map["graph_search"]
                node_result = graph_tool.invoke({"concept": concept})
                if node_result:
                    results[f"graph_node_{concept}"] = node_result

                    # Get related nodes
                    related_tool = tools_map["get_related_nodes"]
                    related = related_tool.invoke({"node_id": node_result["node_id"], "max_neighbors": 3})
                    results[f"related_{concept}"] = related
                    break

        state["tool_results"].update(results)
        logger.info(f"  ✓ Executed {len(planned_tools)} tools, got {len(results)} results")
        return state

    async def _synthesize_node(self, state: AgentState) -> AgentState:
        """Synthesize final answer from tool results"""
        logger.info("🎯 Agent: Synthesizing answer...")

        query = state["query"]
        tool_results = state["tool_results"]

        # Prepare context from tool results
        context_parts = []
        citations = []

        # Add vector search results
        if "vector_results" in tool_results:
            vector_results = tool_results["vector_results"]
            for i, result in enumerate(vector_results[:3]):  # Top 3
                context_parts.append(f"[Source {i+1}, p.{result['page_number']}]: {result['text']}")
                citations.append(SourceCitation(
                    page_number=result["page_number"],
                    snippet=result["text"][:120] + "..." if len(result["text"]) > 120 else result["text"],
                    chunk_id=result["chunk_id"],
                    score=result["score"]
                ))

        # Add graph results
        for key, value in tool_results.items():
            if key.startswith("graph_node_"):
                concept = key.replace("graph_node_", "")
                context_parts.append(f"[Graph Node]: '{value['label']}' is a {value['type']} (importance: {value['importance']:.2f})")
            elif key.startswith("related_"):
                concept = key.replace("related_", "")
                if value:
                    relations = ", ".join([f"{r['label']} ({r['relation']})" for r in value])
                    context_parts.append(f"[Related Concepts]: {relations}")

        # Create context for LLM
        context = "\n\n".join(context_parts)

        # Generate answer using Gemini
        answer = await self.llm_service.agent_synthesize(query, context)

        state["final_answer"] = answer
        state["citations"] = citations

        logger.info("  ✓ Answer synthesized")
        return state

    async def chat(self, query: str, pdf_id: str = None, include_citations: bool = True) -> ChatResponse:
        """
        Main entry point for agent-based chat

        Args:
            query: User's question
            pdf_id: Optional PDF context
            include_citations: Whether to include source citations

        Returns:
            ChatResponse with answer and citations
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🤖 Agent-Based RAG Query: '{query}'")
        logger.info(f"{'='*80}")

        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "pdf_id": pdf_id or "",
            "tool_results": {},
            "reasoning_steps": [],
            "final_answer": "",
            "citations": [],
            "next_action": ""
        }

        try:
            # Run workflow
            final_state = await self.app.ainvoke(initial_state)

            # Extract results
            answer = final_state.get("final_answer", "I couldn't generate an answer.")
            citations = final_state.get("citations", [])

            if not include_citations:
                citations = []

            logger.info(f"✓ Agent completed successfully")
            logger.info(f"  Answer length: {len(answer)} chars")
            logger.info(f"  Citations: {len(citations)}")
            logger.info(f"{'='*80}\n")

            return ChatResponse(
                answer=answer,
                sources=citations[:5]  # Top 5 citations
            )

        except Exception as e:
            logger.error(f"❌ Agent failed: {e}", exc_info=True)

            # Fallback to simple vector search
            logger.warning("Falling back to simple RAG...")
            return await self._fallback_simple_rag(query, pdf_id)

    async def _fallback_simple_rag(self, query: str, pdf_id: str = None) -> ChatResponse:
        """Fallback to simple RAG if agent fails"""
        try:
            results = self.embedding_service.search(query=query, top_k=5, filter_pdf_id=pdf_id)

            if not results:
                return ChatResponse(
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[]
                )

            # Prepare context
            context_chunks = [
                {
                    "page_number": meta.get("page_number", 0),
                    "text": meta.get("text", "")
                }
                for meta, score in results[:3]
            ]

            # Generate answer
            answer = await self.llm_service.rag_chat(query, context_chunks)

            # Format sources
            sources = []
            for meta, score in results[:5]:
                text = meta.get("text", "")
                snippet = text[:120] + "..." if len(text) > 120 else text
                sources.append(SourceCitation(
                    page_number=meta.get("page_number", 0),
                    snippet=snippet,
                    chunk_id=meta.get("chunk_id", ""),
                    score=score
                ))

            return ChatResponse(answer=answer, sources=sources)

        except Exception as e:
            logger.error(f"Fallback RAG also failed: {e}")
            return ChatResponse(
                answer="I encountered an error processing your question.",
                sources=[]
            )

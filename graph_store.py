"""
Knowledge Graph Store
Manages nodes, edges, and graph operations
Supports both NetworkX (local) and Neo4j (production)
"""
import networkx as nx
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional, Tuple, Set
from loguru import logger
from models import GraphNode, GraphEdge, CanonicalTriple, SupportingChunk, NodeType, RelationType
from config import settings
import json
import pickle
from collections import defaultdict
from embedding_service import EmbeddingService


class GraphStore:
    """
    Manages the knowledge graph with nodes and edges
    Supports multiple backends: NetworkX (default) or Neo4j
    """

    def __init__(self, use_neo4j: bool = False, embedding_service: Optional[EmbeddingService] = None):
        self.use_neo4j = use_neo4j
        self.embedding_service = embedding_service

        if use_neo4j:
            self._init_neo4j()
        else:
            self.graph = nx.MultiGraph()  # Undirected graph (no arrows)
            self.nodes_dict: Dict[str, GraphNode] = {}  # node_id -> GraphNode
            self.edges_dict: Dict[str, GraphEdge] = {}  # edge_id -> GraphEdge

        logger.info(f"Initialized GraphStore (backend: {'Neo4j' if use_neo4j else 'NetworkX'}, undirected graph)")

    def _init_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.info("Falling back to NetworkX (undirected)")
            self.use_neo4j = False
            self.graph = nx.MultiGraph()  # Undirected graph
            self.nodes_dict = {}
            self.edges_dict = {}

    def add_node(self, node: GraphNode) -> bool:
        """
        Add a node to the graph

        Args:
            node: GraphNode to add

        Returns:
            True if added, False if already exists
        """
        if self.use_neo4j:
            return self._add_node_neo4j(node)
        else:
            if node.node_id in self.nodes_dict:
                return False

            self.nodes_dict[node.node_id] = node
            # Handle both enum and string for type field
            node_type = node.type.value if hasattr(node.type, 'value') else node.type
            self.graph.add_node(
                node.node_id,
                label=node.label,
                type=node_type,
                importance=node.importance_score
            )
            return True

    def add_edge(self, edge: GraphEdge) -> bool:
        """
        Add an edge to the graph

        Args:
            edge: GraphEdge to add

        Returns:
            True if added successfully
        """
        if self.use_neo4j:
            return self._add_edge_neo4j(edge)
        else:
            self.edges_dict[edge.edge_id] = edge
            # Handle both enum and string for relation field
            relation_value = edge.relation.value if hasattr(edge.relation, 'value') else edge.relation
            self.graph.add_edge(
                edge.from_node,
                edge.to_node,
                key=edge.edge_id,
                relation=relation_value,
                confidence=edge.confidence
            )
            return True

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID"""
        if self.use_neo4j:
            return self._get_node_neo4j(node_id)
        else:
            return self.nodes_dict.get(node_id)

    def update_node(self, node: GraphNode) -> bool:
        """
        Update an existing node in the graph

        Args:
            node: GraphNode with updated data

        Returns:
            True if updated successfully, False if node doesn't exist
        """
        if node.node_id not in self.nodes_dict:
            return False

        # Update in dictionary
        self.nodes_dict[node.node_id] = node

        # Update NetworkX graph attributes
        if node.node_id in self.graph:
            node_type = node.type.value if hasattr(node.type, 'value') else node.type
            self.graph.nodes[node.node_id]['label'] = node.label
            self.graph.nodes[node.node_id]['type'] = node_type
            self.graph.nodes[node.node_id]['importance'] = node.importance_score

        return True

    def get_node_by_label(self, label: str) -> Optional[GraphNode]:
        """Get node by label (case-insensitive)"""
        label_lower = label.lower()
        for node in self.nodes_dict.values():
            if node.label.lower() == label_lower or label_lower in [a.lower() for a in node.aliases]:
                return node
        return None

    def get_neighbors(self, node_id: str) -> List[Tuple[GraphNode, GraphEdge]]:
        """
        Get neighboring nodes and connecting edges (undirected graph)

        Args:
            node_id: Node to get neighbors for

        Returns:
            List of (neighbor_node, edge) tuples
        """
        if self.use_neo4j:
            return self._get_neighbors_neo4j(node_id)
        else:
            neighbors = []
            # For undirected graph, just get all neighbors
            for neighbor_id in self.graph.neighbors(node_id):
                edges = self.graph.get_edge_data(node_id, neighbor_id)
                if edges:
                    for edge_key, edge_data in edges.items():
                        edge = self.edges_dict.get(edge_key)
                        neighbor_node = self.nodes_dict.get(neighbor_id)
                        if edge and neighbor_node:
                            neighbors.append((neighbor_node, edge))

            return neighbors

    def get_all_nodes(self) -> List[GraphNode]:
        """Get all nodes in graph"""
        if self.use_neo4j:
            return self._get_all_nodes_neo4j()
        else:
            return list(self.nodes_dict.values())

    def get_all_edges(self) -> List[GraphEdge]:
        """Get all edges in graph"""
        if self.use_neo4j:
            return self._get_all_edges_neo4j()
        else:
            return list(self.edges_dict.values())

    def remove_node(self, node_id: str):
        """Remove node and its edges"""
        if self.use_neo4j:
            self._remove_node_neo4j(node_id)
        else:
            if node_id in self.nodes_dict:
                del self.nodes_dict[node_id]
                self.graph.remove_node(node_id)

    def remove_edge(self, edge_id: str):
        """Remove edge"""
        if self.use_neo4j:
            self._remove_edge_neo4j(edge_id)
        else:
            if edge_id in self.edges_dict:
                edge = self.edges_dict[edge_id]
                del self.edges_dict[edge_id]
                if self.graph.has_edge(edge.from_node, edge.to_node, key=edge_id):
                    self.graph.remove_edge(edge.from_node, edge.to_node, key=edge_id)

    def compute_centrality(self) -> Dict[str, float]:
        """
        Compute node centrality scores (degree centrality for undirected graph)

        Returns:
            Dict mapping node_id -> centrality score
        """
        if self.use_neo4j:
            # Use Neo4j's centrality algorithm
            return self._compute_centrality_neo4j()
        else:
            try:
                # Use degree centrality for undirected graph (simpler and faster)
                centrality = nx.degree_centrality(self.graph)
                return centrality
            except Exception as e:
                logger.error(f"Failed to compute centrality: {e}")
                return {}

    def save(self, filepath: str):
        """Save graph to file (NetworkX only)"""
        if self.use_neo4j:
            logger.info("Neo4j graphs are persisted automatically")
            return

        data = {
            "nodes": [node.dict() for node in self.nodes_dict.values()],
            "edges": [edge.dict() for edge in self.edges_dict.values()],
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved graph with {len(self.nodes_dict)} nodes and {len(self.edges_dict)} edges to {filepath}")

    def load(self, filepath: str):
        """Load graph from file (NetworkX only)"""
        if self.use_neo4j:
            logger.warning("Cannot load into Neo4j from file")
            return

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Reconstruct nodes
        for node_data in data["nodes"]:
            node = GraphNode(**node_data)
            self.add_node(node)

        # Reconstruct edges
        for edge_data in data["edges"]:
            edge = GraphEdge(**edge_data)
            self.add_edge(edge)

        logger.info(f"Loaded graph with {len(self.nodes_dict)} nodes and {len(self.edges_dict)} edges")

    def clear(self):
        """Clear all nodes and edges"""
        if self.use_neo4j:
            self._clear_neo4j()
        else:
            self.graph.clear()
            self.nodes_dict.clear()
            self.edges_dict.clear()

    # Neo4j implementations (placeholders - implement as needed)

    def _add_node_neo4j(self, node: GraphNode) -> bool:
        """Add node to Neo4j"""
        with self.driver.session() as session:
            # Handle both enum and string for type field
            node_type = node.type.value if hasattr(node.type, 'value') else node.type
            result = session.run(
                """
                MERGE (n:Entity {node_id: $node_id})
                ON CREATE SET n.label = $label, n.type = $type,
                              n.importance = $importance, n.created_at = datetime()
                RETURN n
                """,
                node_id=node.node_id,
                label=node.label,
                type=node_type,
                importance=node.importance_score
            )
            return result.single() is not None

    def _add_edge_neo4j(self, edge: GraphEdge) -> bool:
        """Add edge to Neo4j"""
        with self.driver.session() as session:
            # Handle both enum and string for relation field
            relation_value = edge.relation.value if hasattr(edge.relation, 'value') else edge.relation
            session.run(
                """
                MATCH (a:Entity {node_id: $from_node})
                MATCH (b:Entity {node_id: $to_node})
                CREATE (a)-[r:RELATES {edge_id: $edge_id, relation: $relation,
                                       confidence: $confidence}]->(b)
                """,
                from_node=edge.from_node,
                to_node=edge.to_node,
                edge_id=edge.edge_id,
                relation=relation_value,
                confidence=edge.confidence
            )
            return True

    def _get_node_neo4j(self, node_id: str) -> Optional[GraphNode]:
        """Get node from Neo4j"""
        # Implementation omitted for brevity
        pass

    def _get_neighbors_neo4j(self, node_id: str) -> List[Tuple[GraphNode, GraphEdge]]:
        """Get neighbors from Neo4j"""
        # Implementation omitted for brevity
        pass

    def _get_all_nodes_neo4j(self) -> List[GraphNode]:
        """Get all nodes from Neo4j"""
        pass

    def _get_all_edges_neo4j(self) -> List[GraphEdge]:
        """Get all edges from Neo4j"""
        pass

    def _remove_node_neo4j(self, node_id: str):
        """Remove node from Neo4j"""
        pass

    def _remove_edge_neo4j(self, edge_id: str):
        """Remove edge from Neo4j"""
        pass

    def _compute_centrality_neo4j(self) -> Dict[str, float]:
        """Compute centrality in Neo4j"""
        pass

    def _clear_neo4j(self):
        """Clear Neo4j database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

"""
Graph Builder - constructs knowledge graph from canonical triples
Handles entity canonicalization, node/edge creation, and graph pruning
"""
from typing import List, Dict, Any, Set, Tuple
from loguru import logger
from models import CanonicalTriple, GraphNode, GraphEdge, SupportingChunk, NodeType
from graph_store import GraphStore
from embedding_service import EmbeddingService
from config import settings
import numpy as np
from collections import defaultdict


class GraphBuilder:
    """
    Builds and refines knowledge graph from canonical triples
    Implements entity canonicalization, deduplication, and pruning
    """

    def __init__(self, graph_store: GraphStore, embedding_service: EmbeddingService):
        self.graph_store = graph_store
        self.embedding_service = embedding_service
        self.entity_embeddings: Dict[str, np.ndarray] = {}

    async def build_graph(self, triples: List[CanonicalTriple]) -> Tuple[int, int]:
        """
        Build graph from canonical triples

        Args:
            triples: List of canonical triples

        Returns:
            Tuple of (num_nodes_added, num_edges_added)
        """
        logger.info(f"Building graph from {len(triples)} triples")

        # Step 1: Entity canonicalization - merge similar entities
        entity_map = await self._canonicalize_entities(triples)

        # Step 2: Create nodes
        nodes_created = 0
        logger.info(f"Creating nodes from {len(entity_map)} canonical entities")

        for entity_label in entity_map.keys():
            node = await self._create_node(entity_label, entity_map, triples)
            if self.graph_store.add_node(node):
                nodes_created += 1
                logger.debug(f"Created node: {node.label} (type: {node.type.value})")

        logger.info(f"✓ Successfully created {nodes_created} nodes")

        # Step 3: Create edges
        edges_created = 0
        for triple in triples:
            # Map to canonical entities
            canonical_subject = entity_map.get(triple.subject_label, triple.subject_label)
            canonical_object = entity_map.get(triple.object_label, triple.object_label)

            # Skip self-loops
            if canonical_subject == canonical_object:
                continue

            # Get node IDs
            subject_node = self.graph_store.get_node_by_label(canonical_subject)
            object_node = self.graph_store.get_node_by_label(canonical_object)

            if not subject_node or not object_node:
                continue

            # Create edge
            edge = self._create_edge(subject_node, object_node, triple)
            if self.graph_store.add_edge(edge):
                edges_created += 1

        logger.info(f"Created {nodes_created} nodes and {edges_created} edges")

        # Step 4: Compute importance scores
        self._compute_importance_scores()

        # Step 5: Prune low-importance nodes and edges
        pruned_nodes, pruned_edges = self._prune_graph()

        logger.info(f"Pruned {pruned_nodes} nodes and {pruned_edges} edges")
        logger.info(f"Final graph: {nodes_created - pruned_nodes} nodes, {edges_created - pruned_edges} edges")

        return nodes_created - pruned_nodes, edges_created - pruned_edges

    async def _canonicalize_entities(self, triples: List[CanonicalTriple]) -> Dict[str, str]:
        """
        ⚡ OPTIMIZATION: Skip expensive canonicalization (identity mapping)

        With 2 nodes per page hard cap and strict technical filtering,
        we have very few duplicates and highly specific entities.
        Embedding computation + O(n²) similarity checks not worth the cost.

        Args:
            triples: List of triples

        Returns:
            Dict mapping entity_label -> canonical_label (identity map)
        """
        # Collect all unique entities
        entities = set()
        for triple in triples:
            entities.add(triple.subject_label)
            entities.add(triple.object_label)

        # DETERMINISTIC: Sort entities for consistent ordering across runs
        entities_list = sorted(list(entities))
        logger.info(f"⚡ FAST MODE: Skipping entity canonicalization for {len(entities_list)} unique entities")
        logger.info(f"Each entity maps to itself (no merging)")

        # Return identity mapping - each entity maps to itself
        entity_map = {entity: entity for entity in entities_list}

        logger.info(f"✓ Identity mapping created (0 merges, {len(entities_list)} canonical entities)")

        return entity_map

    def _entity_to_text(self, entity: str) -> str:
        """Convert entity label to text for embedding"""
        # Simple approach: use the label as-is
        return entity

    async def _create_node(
        self,
        label: str,
        entity_map: Dict[str, str],
        triples: List[CanonicalTriple]
    ) -> GraphNode:
        """
        Create a graph node for an entity

        Args:
            label: Canonical entity label
            entity_map: Entity canonicalization map
            triples: All triples (to find supporting chunks)

        Returns:
            GraphNode
        """
        # Find all triples mentioning this entity
        supporting_chunks = []
        aliases = []

        for original_label, canonical_label in entity_map.items():
            if canonical_label == label:
                if original_label != label:
                    aliases.append(original_label)

        # Collect supporting chunks from triples
        chunk_scores = defaultdict(float)
        for triple in triples:
            canonical_subject = entity_map.get(triple.subject_label, triple.subject_label)
            canonical_object = entity_map.get(triple.object_label, triple.object_label)

            if canonical_subject == label or canonical_object == label:
                # This triple supports the node
                chunk_key = (triple.page_number, triple.justification[:100])  # Use justification as proxy
                chunk_scores[chunk_key] += triple.confidence

        # Convert to SupportingChunk objects
        for (page_number, snippet), score in chunk_scores.items():
            supporting_chunks.append(SupportingChunk(
                chunk_id=f"page_{page_number}",  # Placeholder
                score=score,
                page_number=page_number,
                snippet=snippet
            ))

        # DETERMINISTIC: Sort by score (desc) then page_number (asc) for stable ordering
        supporting_chunks.sort(key=lambda x: (-x.score, x.page_number))
        supporting_chunks = supporting_chunks[:10]

        # Infer node type (simple heuristic)
        node_type = self._infer_node_type(label)

        node = GraphNode(
            label=label,
            type=node_type,
            aliases=aliases,
            supporting_chunks=supporting_chunks,
            importance_score=0.0  # Will be computed later
        )

        return node

    def _infer_node_type(self, label: str) -> NodeType:
        """Infer node type from label (simple heuristics)"""
        label_lower = label.lower()

        # Check for common patterns
        if any(word in label_lower for word in ["function", "method", "algorithm"]):
            return NodeType.FUNCTION
        elif any(word in label_lower for word in ["class", "type", "struct"]):
            return NodeType.CLASS
        elif label[0].isupper() and " " not in label:  # Capitalized single word
            return NodeType.PERSON
        elif any(word in label_lower for word in ["definition", "term", "concept"]):
            return NodeType.TERM
        else:
            return NodeType.CONCEPT

    def _create_edge(
        self,
        from_node: GraphNode,
        to_node: GraphNode,
        triple: CanonicalTriple
    ) -> GraphEdge:
        """Create a graph edge from a triple"""
        supporting_chunk = SupportingChunk(
            chunk_id=f"page_{triple.page_number}",
            score=triple.confidence,
            page_number=triple.page_number,
            snippet=triple.justification
        )

        edge = GraphEdge(
            from_node=from_node.node_id,
            to_node=to_node.node_id,
            relation=triple.relation,
            confidence=triple.confidence,
            supporting_chunks=[supporting_chunk]
        )

        return edge

    def _compute_importance_scores(self):
        """
        ⚡ OPTIMIZATION: Simplified importance scoring (skip expensive PageRank)

        Since we're not pruning, we only need basic scores for display purposes.
        """
        logger.info("⚡ FAST MODE: Computing simplified importance scores (no PageRank)")

        # Update node importance with simple metric (just degree centrality)
        for node in self.graph_store.get_all_nodes():
            # Simple importance = number of connections (fast to compute)
            num_neighbors = len(self.graph_store.get_neighbors(node.node_id))

            # Normalize to 0-1 range (assume max 10 connections)
            importance = min(num_neighbors / 10.0, 1.0)

            node.importance_score = importance

            # Update in store (for NetworkX)
            if not self.graph_store.use_neo4j:
                self.graph_store.nodes_dict[node.node_id] = node

        logger.info(f"✓ Importance scores computed (based on degree centrality only)")

    def _prune_graph(self) -> Tuple[int, int]:
        """
        ⚡ OPTIMIZATION: Skip pruning (we already filter at extraction)

        Pruning is expensive (PageRank + multiple graph traversals).
        With strict filtering at extraction (technical concepts only, 2 per page),
        we don't need additional pruning.

        Returns:
            Tuple of (nodes_removed, edges_removed) - always (0, 0)
        """
        logger.info(f"⚡ FAST MODE: Skipping graph pruning")
        logger.info(f"Nodes already filtered at extraction with strict technical validation")
        logger.info(f"Final graph: {len(self.graph_store.get_all_nodes())} nodes, {len(self.graph_store.get_all_edges())} edges")

        return 0, 0

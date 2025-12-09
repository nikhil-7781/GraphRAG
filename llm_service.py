"""
LLM Inference Layer
Handles all LLM calls for extraction, summarization, and chat
Uses Mistral 7B with structured prompt templates
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from config import settings
import json
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from models import Triple, CanonicalTriple, RelationType


class PromptTemplates:
    """Centralized prompt templates following the manual"""

    @staticmethod
    def triplet_canonicalization(passage: str, triple: Triple) -> str:
        """Template for canonicalizing extracted triples"""
        return f"""Given the passage and an extracted triple, return a cleaned, canonical version.

Passage (from page {triple.page_number}):
{passage}

Extracted Triple:
- Subject: {triple.subject}
- Relation: {triple.predicate}
- Object: {triple.object}

CRITICAL INSTRUCTION: You MUST select the "relation" field from this EXACT list of 25 canonical relations.
Copy the exact string - do NOT create variations, synonyms, or modifications.

ALLOWED RELATIONS (choose exactly one):
1. is_a - for type/class relationships (e.g., "X is a Y")
2. part_of - for component relationships (e.g., "X is part of Y")
3. uses - for utilization (use "uses" for: utilizes, employs, applies)
4. causes - for causality (e.g., "X causes Y")
5. defined_as - for definitions (use "defined_as" for: defines, is defined as)
6. related_to - ONLY if no other relation fits
7. method_of - for methodological relationships
8. depends_on - for dependencies (e.g., "X depends on Y")
9. implements - for implementation (e.g., "X implements Y")
10. similar_to - for similarity
11. observes - for observation (use "observes" for: captures, records, detects, monitors)
12. measures - for measurement
13. produces - for production/generation (use "produces" for: makes, creates, generates, builds)
14. contains - for containment
15. affects - for influence (use "affects" for: influences, impacts, modifies, changes)
16. enables - for enablement (use "enables" for: facilitates, allows, permits)
17. requires - for requirements
18. interacts_with - for interactions
19. enriches - for enrichment
20. enhances - for enhancement (use "enhances" for: improves, optimizes, extends)
21. supports - for support (use "supports" for: contributes, helps, aids)
22. describes - for description (use "describes" for: proposes, suggests, presents, introduces)
23. explains - for explanation (use "explains" for: clarifies, demonstrates, shows, disentangles)
24. refers_to - for reference (use "refers_to" for: aims, targets, addresses, focuses on)
25. associated_with - for associations

EXAMPLES OF WHAT TO DO:
- If input has "utilizes" → use "uses"
- If input has "proposes" → use "describes"
- If input has "contributes to" → use "supports"
- If input has "aims at" → use "refers_to"

DO NOT USE: utilizes, proposes, contributes, aims, makes, captures, defines, or any other variations.
USE ONLY: The exact 25 strings listed above.

Return JSON in this exact format:
{{
  "subject_label": "cleaned subject name",
  "object_label": "cleaned object name",
  "relation": "one_of_the_25_exact_strings_above",
  "confidence": 0.85,
  "justification": "brief explanation referencing page {triple.page_number}"
}}

Output ONLY the JSON, no other text:
"""

    @staticmethod
    def node_summarization(node_label: str, chunks: List[Dict[str, Any]]) -> str:
        """Template for node summarization with citations"""
        chunks_text = "\n\n".join([
            f"[Chunk from p.{chunk['page_number']}]\n{chunk['text']}"
            for chunk in chunks
        ])

        return f"""Summarize the key facts about "{node_label}" using ONLY the following supporting chunks.

Requirements:
- Produce a concise summary (3-6 sentences)
- After any sentence that directly relies on a chunk, append (p. N) where N is the page number
- Do not invent information not present in the chunks
- Focus on the most important facts

Supporting Chunks:
{chunks_text}

Summary:
"""

    @staticmethod
    def rag_chat(user_query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Template for RAG chat with citations"""
        context_text = "\n\n".join([
            f"[Source {i+1}, p.{chunk['page_number']}]\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])

        return f"""You are an assistant that answers questions using ONLY the provided document context.

Context from document:
{context_text}

User Question: {user_query}

Instructions:
- Answer in friendly, concise language
- Include inline citations (p. N) for statements supported by chunks
- If you cannot find direct support, say "I cannot confirm this from the document"
- At the end, add a "Sources:" section listing page numbers and short snippets

Answer:
"""

    @staticmethod
    def system_message() -> str:
        """System message for chat"""
        return """You are a helpful assistant that answers questions strictly based on provided document context.
You always cite page numbers for factual statements. If information is not in the context, you say so clearly."""


class LLMService:
    """
    Service for LLM inference using Gemini API (via litellm)
    Handles generation, extraction, summarization, and agent synthesis
    """

    def __init__(self):
        # Use Gemini instead of Mistral
        self.api_key = settings.gemini_api_key
        self.model = f"gemini/{settings.gemini_model}"
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = settings.llm_timeout

        # Import litellm for Gemini
        try:
            import litellm
            self.litellm = litellm
            logger.info(f"✓ LLMService initialized with Gemini ({settings.gemini_model})")
        except ImportError:
            logger.error("litellm not installed. Install with: pip install litellm")
            raise

        if not self.api_key:
            logger.warning("No Gemini API key configured. LLM features will not work.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> str:
        """
        Call Gemini API via litellm with retry logic

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: Request JSON output

        Returns:
            Generated text
        """
        if not self.api_key:
            raise ValueError("Gemini API key not configured")

        try:
            # Use litellm for Gemini API calls
            import asyncio

            kwargs = {
                "model": self.model,
                "api_key": self.api_key,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            # litellm.completion is synchronous, wrap in asyncio.to_thread
            response = await asyncio.to_thread(
                self.litellm.completion,
                **kwargs
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise

    async def canonicalize_triple(
        self,
        triple: Triple,
        passage: str
    ) -> Optional[CanonicalTriple]:
        """
        Canonicalize a raw triple using LLM

        Args:
            triple: Raw extracted triple
            passage: Surrounding text passage

        Returns:
            CanonicalTriple or None if LLM fails
        """
        prompt = PromptTemplates.triplet_canonicalization(passage, triple)

        messages = [
            {"role": "system", "content": "You are an expert at extracting and canonicalizing knowledge graph triples. Always output valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self._call_api(messages, temperature=0.1, json_mode=True)
            data = json.loads(response)

            # Map string relation to enum
            relation_str = data.get("relation", "related_to").lower().strip()

            # Auto-correct common variations and map semantically similar verbs
            relation_corrections = {
                # Exact variations
                "defines_as": "defined_as",
                "defines": "defined_as",
                "is_part_of": "part_of",
                "used_by": "uses",
                "caused_by": "causes",
                "methods_of": "method_of",
                "depending_on": "depends_on",
                "implemented_by": "implements",
                "similar": "similar_to",
                "observed_by": "observes",
                "measured_by": "measures",
                "produced_by": "produces",
                "contained_in": "contains",
                "affected_by": "affects",
                "enabled_by": "enables",
                "required_by": "requires",
                "interact_with": "interacts_with",
                "enriched_by": "enriches",
                "enhanced_by": "enhances",
                "supported_by": "supports",
                "described_by": "describes",
                "explained_by": "explains",
                "refer_to": "refers_to",

                # Semantic mappings for common verbs
                "utilizes": "uses",
                "utilize": "uses",
                "employs": "uses",
                "applies": "uses",
                "makes": "produces",
                "creates": "produces",
                "generates": "produces",
                "builds": "produces",
                "proposes": "describes",
                "suggests": "describes",
                "presents": "describes",
                "introduces": "describes",
                "captures": "observes",
                "records": "observes",
                "detects": "observes",
                "monitors": "observes",
                "aims": "refers_to",
                "targets": "refers_to",
                "focuses_on": "refers_to",
                "addresses": "refers_to",
                "disentangles": "explains",
                "clarifies": "explains",
                "demonstrates": "explains",
                "shows": "explains",
                "contributes": "supports",
                "contributes_to": "supports",
                "helps": "supports",
                "aids": "supports",
                "facilitates": "enables",
                "allows": "enables",
                "permits": "enables",
                "improves": "enhances",
                "betters": "enhances",
                "optimizes": "enhances",
                "extends": "enhances",
                "influences": "affects",
                "impacts": "affects",
                "modifies": "affects",
                "changes": "affects",
            }

            relation_str = relation_corrections.get(relation_str, relation_str)

            try:
                relation = RelationType(relation_str)
            except ValueError:
                logger.warning(f"Invalid relation '{relation_str}', defaulting to 'related_to'")
                relation = RelationType.RELATED_TO

            return CanonicalTriple(
                subject_label=data["subject_label"],
                object_label=data["object_label"],
                relation=relation,
                confidence=data["confidence"],
                justification=data["justification"],
                page_number=triple.page_number or 0
            )
        except Exception as e:
            logger.error(f"Failed to canonicalize triple: {e}")
            return None

    async def summarize_node(
        self,
        node_label: str,
        supporting_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate summary for a graph node with citations

        Args:
            node_label: Name of the node
            supporting_chunks: List of chunk metadata dicts

        Returns:
            Summary text with inline citations
        """
        prompt = PromptTemplates.node_summarization(node_label, supporting_chunks)

        messages = [
            {"role": "system", "content": PromptTemplates.system_message()},
            {"role": "user", "content": prompt}
        ]

        try:
            # Use faster settings for node summaries
            summary = await self._call_api(
                messages,
                temperature=0.3,
                max_tokens=3072 # Shorter summaries = faster response
            )
            return summary.strip()
        except Exception as e:
            logger.error(f"Failed to summarize node: {e}")
            return f"Unable to generate summary for {node_label}."

    async def rag_chat(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Answer user query using RAG with citations

        Args:
            query: User question
            context_chunks: Retrieved context chunks

        Returns:
            Answer with citations and sources
        """
        prompt = PromptTemplates.rag_chat(query, context_chunks)

        messages = [
            {"role": "system", "content": PromptTemplates.system_message()},
            {"role": "user", "content": prompt}
        ]

        try:
            answer = await self._call_api(messages, temperature=0.3)
            return answer.strip()
        except Exception as e:
            logger.error(f"Failed to generate RAG response: {e}")
            return "I encountered an error while processing your question. Please try again."

    async def agent_synthesize(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Synthesize answer for agent-based RAG from tool results

        Args:
            query: User question
            context: Combined context from tool executions

        Returns:
            Synthesized answer with citations
        """
        prompt = f"""You are an assistant that answers questions using the provided context from multiple tools.

Context from tools:
{context}

User Question: {query}

Instructions:
- Answer in friendly, concise language
- Include inline citations (p. N) for statements supported by sources
- If you cannot find direct support, say "I cannot confirm this from the available information"
- Synthesize information from different tools (vector search, graph search, etc.) cohesively

Answer:
"""

        messages = [
            {"role": "system", "content": PromptTemplates.system_message()},
            {"role": "user", "content": prompt}
        ]

        try:
            answer = await self._call_api(messages, temperature=0.3)
            return answer.strip()
        except Exception as e:
            logger.error(f"Failed to synthesize agent response: {e}")
            return "I encountered an error while processing your question. Please try again."

    async def extract_triples_llm(
        self,
        text: str,
        page_number: int,
        chunk_id: str
    ) -> List[Triple]:
        """
        Use LLM to extract triples directly (alternative to OpenIE)

        Args:
            text: Text to extract from
            page_number: Page number
            chunk_id: Chunk identifier

        Returns:
            List of extracted triples
        """
        prompt = f"""Extract key relationships from this text as subject-predicate-object triples.
Focus on important concepts, methods, definitions, and relationships.

Text (from page {page_number}):
{text}

Return a JSON array of triples, each with:
- subject: The subject entity
- predicate: The relationship/action
- object: The object entity
- confidence: Your confidence (0-1)

Output ONLY valid JSON array:
"""

        messages = [
            {"role": "system", "content": "You are an expert at knowledge extraction. Always output valid JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self._call_api(messages, temperature=0.2, json_mode=True)
            data = json.loads(response)

            triples = []
            for item in data if isinstance(data, list) else data.get("triples", []):
                triple = Triple(
                    subject=item["subject"],
                    predicate=item["predicate"],
                    object=item["object"],
                    confidence=item.get("confidence", 0.7),
                    source_chunk_id=chunk_id,
                    page_number=page_number
                )
                triples.append(triple)

            return triples
        except Exception as e:
            logger.error(f"Failed to extract triples: {e}")
            return []

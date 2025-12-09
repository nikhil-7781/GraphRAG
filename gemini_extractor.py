"""
Gemini-based Knowledge Graph Extraction
Simple LLM-powered extraction using Google Gemini (cheapest option)
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from models import Chunk, CanonicalTriple, RelationType
from config import settings
import json
import asyncio


class GeminiExtractor:
    """
    Extract key nodes and relationships using Gemini LLM
    Simple, cost-effective approach for knowledge graph generation
    """

    def __init__(self, llm_service=None):
        """Initialize Gemini extractor"""
        logger.info("Initializing GeminiExtractor")

        # Import litellm for API calls
        try:
            import litellm
            self.litellm = litellm

            # Configure litellm for Gemini
            self.model_name = f"gemini/{settings.gemini_model}"
            self.api_key = settings.gemini_api_key

            logger.info(f"✓ GeminiExtractor initialized with model: {self.model_name}")

        except ImportError as e:
            logger.error("litellm not installed. Install with: pip install litellm")
            raise RuntimeError("litellm required for Gemini") from e

        # Comprehensive list of generic terms to REJECT
        self.generic_stopwords = {
            # Generic nouns
            'system', 'systems', 'data', 'information', 'value', 'values',
            'method', 'methods', 'approach', 'approaches', 'technique', 'techniques',
            'result', 'results', 'study', 'studies', 'paper', 'papers',
            'section', 'sections', 'figure', 'figures', 'table', 'tables',
            'example', 'examples', 'case', 'cases', 'type', 'types',
            'way', 'ways', 'thing', 'things', 'part', 'parts',
            'model', 'models', 'framework', 'frameworks',  # Too generic unless specific
            'process', 'processes', 'analysis', 'problem', 'problems',
            'solution', 'solutions', 'set', 'sets', 'group', 'groups',
            'element', 'elements', 'component', 'components',
            'feature', 'features', 'property', 'properties',
            'aspect', 'aspects', 'factor', 'factors', 'parameter', 'parameters',
            'concept', 'concepts', 'idea', 'ideas', 'theory', 'theories',
            'field', 'fields', 'area', 'areas', 'domain', 'domains',
            'task', 'tasks', 'goal', 'goals', 'objective', 'objectives',
            'input', 'inputs', 'output', 'outputs', 'function', 'functions',
            'operation', 'operations', 'step', 'steps', 'stage', 'stages',
            'phase', 'phases', 'level', 'levels', 'layer', 'layers',
            'number', 'numbers', 'amount', 'amounts', 'size', 'sizes',
            'performance', 'accuracy', 'quality', 'efficiency',
            'document', 'documents', 'text', 'texts', 'word', 'words',
            'sentence', 'sentences', 'paragraph', 'paragraphs',
            'item', 'items', 'object', 'objects', 'entity', 'entities',
            'relation', 'relations', 'relationship', 'relationships',

            # Generic verbs/actions
            'use', 'uses', 'using', 'used', 'usage',
            'apply', 'applies', 'applying', 'applied', 'application', 'applications',
            'work', 'works', 'working', 'worked',
            'provide', 'provides', 'providing', 'provided',
            'show', 'shows', 'showing', 'shown',
            'present', 'presents', 'presenting', 'presented', 'presentation',

            # Generic adjectives
            'new', 'novel', 'existing', 'current', 'previous',
            'different', 'similar', 'same', 'other', 'another',
            'various', 'several', 'multiple', 'single',
            'important', 'significant', 'main', 'key', 'major',
            'good', 'better', 'best', 'high', 'low',
            'large', 'small', 'big', 'little',

            # Research-specific generic terms
            'experiment', 'experiments', 'evaluation', 'evaluations',
            'test', 'tests', 'testing', 'validation',
            'comparison', 'comparisons', 'benchmark', 'benchmarks',
            'baseline', 'baselines', 'metric', 'metrics',
            'dataset', 'datasets', 'corpus', 'corpora',

            # Time/sequence terms
            'time', 'times', 'period', 'periods', 'year', 'years',
            'first', 'second', 'third', 'last', 'final',
            'next', 'previous', 'current', 'recent',

            # Common prepositions/articles (shouldn't appear but just in case)
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',

            # Additional generic ML/AI terms (too broad)
            'neural network', 'deep learning', 'machine learning',
            'training', 'testing', 'prediction', 'classification',
            'regression', 'clustering', 'optimization',
            'network', 'networks', 'algorithm', 'algorithms',
            'learning', 'training data', 'test data',
            'feature extraction', 'preprocessing',
            'hyperparameter', 'hyperparameters',
            'loss', 'error', 'gradient',
        }

    async def extract_from_chunks(
        self,
        chunks: List[Chunk],
        use_llm: bool = True
    ) -> List[CanonicalTriple]:
        """
        Extract knowledge graph - PER PAGE with HARD CAP of 2 concepts per page

        Args:
            chunks: List of text chunks
            use_llm: Always True for Gemini extraction

        Returns:
            List of canonical triples
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"{'GEMINI PER-PAGE EXTRACTION - 2 CONCEPTS MAX PER PAGE':^80}")
        logger.info(f"{'='*80}")

        all_triples = []

        # Filter text chunks
        text_chunks = [c for c in chunks if c.type.value in ["paragraph", "code"]]

        if not text_chunks:
            logger.warning("No text chunks to process")
            return []

        # GROUP CHUNKS BY PAGE
        from collections import defaultdict
        chunks_by_page = defaultdict(list)
        for chunk in text_chunks:
            page_num = chunk.page_number or 0
            chunks_by_page[page_num].append(chunk)

        logger.info(f"Processing {len(chunks_by_page)} pages in PARALLEL")

        # ⚡ PARALLEL PROCESSING: Create tasks for all pages
        tasks = []
        page_numbers = []
        for page_num in sorted(chunks_by_page.keys()):
            page_chunks = chunks_by_page[page_num]
            combined_text = "\n\n".join([chunk.text for chunk in page_chunks])

            logger.info(f"📄 PAGE {page_num}: {len(page_chunks)} chunks, {len(combined_text)} chars")

            # Create async task for this page
            tasks.append(self._extract_with_gemini(combined_text, page_num))
            page_numbers.append(page_num)

        # Execute all Gemini calls in parallel
        logger.info(f"\n🚀 Launching {len(tasks)} parallel Gemini API calls...")
        import time
        start_time = time.time()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time
        logger.info(f"✓ All {len(tasks)} Gemini calls completed in {elapsed:.2f}s (parallel)")
        logger.info(f"  Average: {elapsed/len(tasks):.2f}s per page (would be {elapsed*len(tasks):.2f}s sequential)")

        # Process results
        for page_num, page_triples in zip(page_numbers, results):
            if isinstance(page_triples, Exception):
                logger.error(f"  ❌ Page {page_num} failed: {page_triples}")
                continue

            if page_triples:
                all_triples.extend(page_triples)
                logger.info(f"  ✓ Page {page_num}: Extracted {len(page_triples)} triples")
                for t in page_triples:
                    relation_value = t.relation.value if hasattr(t.relation, 'value') else t.relation
                    logger.info(f"    → {t.subject_label} --[{relation_value}]--> {t.object_label}")
            else:
                logger.warning(f"  ⚠️ Page {page_num}: NO TRIPLES EXTRACTED!")

        # Summary
        unique_concepts = set()
        concepts_by_page = {}
        for triple in all_triples:
            unique_concepts.add(triple.subject_label)
            unique_concepts.add(triple.object_label)
            page = triple.page_number
            if page not in concepts_by_page:
                concepts_by_page[page] = set()
            concepts_by_page[page].add(triple.subject_label)
            concepts_by_page[page].add(triple.object_label)

        logger.info(f"\n{'='*80}")
        logger.info(f"{'EXTRACTION SUMMARY':^80}")
        logger.info(f"{'='*80}")
        logger.info(f"Pages processed: {len(chunks_by_page)}")
        logger.info(f"Total triples: {len(all_triples)}")
        logger.info(f"Unique concepts: {len(unique_concepts)} (max {len(chunks_by_page) * 2})")

        if len(all_triples) == 0:
            logger.error(f"\n❌❌❌ CRITICAL ERROR: ZERO TRIPLES EXTRACTED! ❌❌❌")
            logger.error(f"This means:")
            logger.error(f"  - Either Gemini returned no concepts")
            logger.error(f"  - Or all concepts were rejected by filters")
            logger.error(f"  - Or there was an API error")
            logger.error(f"Check the logs above for details!")
        else:
            logger.info(f"\nConcepts per page:")
            for page in sorted(concepts_by_page.keys()):
                logger.info(f"  Page {page}: {list(concepts_by_page[page])}")

        logger.info(f"{'='*80}\n")

        return all_triples

    async def _extract_with_gemini(self, text: str, page_number: int) -> List[CanonicalTriple]:
        """
        Call Gemini API to extract technical concepts (nodes) from THIS PAGE

        Args:
            text: Text from single page
            page_number: Page number

        Returns:
            List of canonical triples
        """
        # Specialized technical concept extraction prompt
        prompt = f"""You are an expert in technical information extraction and knowledge graph construction.
Your task is to identify only the most meaningful *technical concepts* from the given text.
Concepts must represent scientific, mathematical, algorithmic, or methodological entities
that could exist as standalone nodes in a knowledge graph.
Ignore generic words, section titles, variable names, and everyday terms.
Focus on high-value, domain-specific terminology relevant to the text.

Extract all important technical concepts from the following text that would form the
nodes of a knowledge graph.

⚙️ Rules:
• Each concept should represent a self-contained technical idea, model, method, metric, loss, theorem, or process
• Keep only multi-word phrases when possible ("gradient descent", "convolutional neural network", "cross-entropy loss")
• Skip single, contextless nouns ("data", "model", "value", "equation", "result")
• Merge synonymous terms (e.g., "SGD", "stochastic gradient descent" → one entry)
• Do not include equations, numeric values, figure names, or symbols
• Do not repeat concepts
• Maintain consistent naming conventions (lowercase, hyphen-separated words)
• Extract MAXIMUM 4-5 concepts from this page (quality over quantity)

Return output strictly as JSON with "nodes" key:
{{
  "nodes": [
    "gradient descent",
    "neural network",
    "cross entropy loss"
  ]
}}

PAGE {page_number} TEXT:
{text}

CRITICAL: Return ONLY the JSON. If no technical concepts found, return {{"nodes": []}}"""

        logger.info(f"  🚀 Starting Gemini extraction for page {page_number}...")
        logger.info(f"  Text length: {len(text)} characters")

        try:
            # Call Gemini via litellm
            logger.info(f"  📡 Calling Gemini API for page {page_number}...")

            response = await asyncio.to_thread(
                self.litellm.completion,
                model=self.model_name,
                api_key=self.api_key,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.0,  
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout
            )

            # Extract response text
            response_text = response.choices[0].message.content.strip()
            logger.info(f"  📥 Gemini response ({len(response_text)} chars):")
            logger.info(f"  {response_text[:500]}") 

            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            data = json.loads(response_text)

            
            if isinstance(data, dict) and "nodes" in data:
                nodes = data["nodes"]
            elif isinstance(data, list):
                # Fallback: if Gemini returned a list directly
                nodes = data
            else:
                logger.warning(f"  ❌ Gemini returned unexpected format: {type(data)}")
                return []

            if not isinstance(nodes, list):
                logger.warning(f"  ❌ Nodes is not a list, got: {type(nodes)}")
                return []

            logger.info(f"  ✓ Gemini extracted {len(nodes)} nodes from page {page_number}")
            logger.info(f"  Raw nodes: {nodes}")

            # Validate and filter nodes
            valid_nodes = []
            rejected_nodes = []

            for node in nodes:
                if not isinstance(node, str):
                    logger.warning(f"  ⚠️ Skipping non-string node: {node}")
                    continue

                node = node.strip()
                if not node:
                    continue

                logger.info(f"  Validating node: '{node}'")

                # FILTER: Validate node is a technical concept
                if not self._is_technical_concept(node):
                    rejected_nodes.append(node)
                    logger.warning(f"  ✗ REJECTED node '{node}' - not technical enough")
                    continue

                logger.info(f"  ✅ ACCEPTED node: '{node}'")
                valid_nodes.append(node.lower())

            # Summary of rejections
            if rejected_nodes:
                logger.warning(f"  📊 Rejected {len(rejected_nodes)} nodes: {rejected_nodes}")

            if not valid_nodes:
                logger.warning(f"  ⚠️ ALL {len(nodes)} NODES REJECTED for page {page_number}")
                logger.warning(f"  No valid technical concepts found. Returning empty list.")
                return []

            
            selected_nodes = valid_nodes[:2]  #
            logger.info(f"  🎯 Selected {len(selected_nodes)} nodes (hard cap = 2): {selected_nodes}")

            
            page_triples = []

            if len(selected_nodes) == 1:
                # Only one node - create self-referencing relationship or skip
                logger.info(f"  ℹ️ Only 1 node on page {page_number}, cannot create relationships")
                
                return []

            elif len(selected_nodes) == 2:
                # Use LLM to determine actual relationship between nodes
                node1, node2 = selected_nodes[0], selected_nodes[1]

                # Extract relationship using LLM with page context
                logger.info(f"  🔍 Extracting relationship between: {node1} ↔ {node2}")
                relationship_triple = await self._extract_relationship_with_gemini(
                    text=text,
                    node1=node1,
                    node2=node2,
                    page_number=page_number
                )

                if relationship_triple:
                    page_triples.append(relationship_triple)
                    logger.info(f"  ✅ Created directed edge:")
                    logger.info(f"    → {relationship_triple.subject_label} --[{relationship_triple.relation.value}]--> {relationship_triple.object_label}")
                    logger.info(f"    Justification: {relationship_triple.justification}")
                else:
                    logger.warning(f"  ⚠️ Could not extract relationship for {node1} ↔ {node2}")

            logger.info(f"  ✅ Returning {len(page_triples)} triples for page {page_number}")
            return page_triples

        except json.JSONDecodeError as e:
            logger.error(f"  ❌ JSON PARSE ERROR for page {page_number}: {e}")
            logger.error(f"  Response was: {response_text[:500]}")
            return []

        except Exception as e:
            logger.error(f"  ❌ GEMINI API FAILED for page {page_number}: {e}")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Full trace:", exc_info=True)
            return []

    async def _extract_relationship_with_gemini(self, text: str, node1: str, node2: str, page_number: int) -> Optional[CanonicalTriple]:
        """
        Use Gemini to determine the actual relationship between two nodes based on page context

        Args:
            text: Full page text for context
            node1: First node/concept
            node2: Second node/concept
            page_number: Page number

        Returns:
            CanonicalTriple with proper relationship, or None if extraction fails
        """
        # List all available relation types for the LLM
        available_relations = [r.value for r in RelationType]

        prompt = f"""You are an expert at extracting knowledge graph relationships from technical text.

Given two concepts and the text they appear in, determine the most accurate relationship between them.

**Concepts:**
- Concept A: "{node1}"
- Concept B: "{node2}"

**Context (page {page_number}):**
{text[:3000]}

**Available Relationship Types:**
{', '.join(available_relations)}

**Instructions:**
1. Analyze how these two concepts relate in the given context
2. Choose the MOST SPECIFIC relationship type from the list above
3. Determine the direction: which concept is the subject and which is the object
4. Provide a brief justification from the text

**Output Format (JSON):**
{{
  "subject": "<node1 or node2>",
  "object": "<node1 or node2>",
  "relation": "<one of the available relationship types>",
  "confidence": <0.0-1.0>,
  "justification": "<brief explanation from text>"
}}

**Rules:**
- Use the exact concept names provided
- Choose only ONE relation type from the available list
- If no clear relationship exists, use "related_to"
- Direction matters: subject performs/has the relation to the object
"""

        try:
            # Call Gemini API
            response_text = await self.litellm.acompletion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at knowledge graph relationship extraction. Always output valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                api_key=self.api_key,
                temperature=0.1,  # Low temperature for consistent relationship extraction
                response_format={"type": "json_object"}
            )

            response_content = response_text.choices[0].message.content
            data = json.loads(response_content)

            # Validate response
            subject = data.get("subject", "").strip()
            obj = data.get("object", "").strip()
            relation_str = data.get("relation", "related_to").lower().strip().replace(" ", "_")
            confidence = float(data.get("confidence", 0.7))
            justification = data.get("justification", f"Relationship extracted from page {page_number}")

            # Map relation string to enum
            try:
                relation = RelationType(relation_str)
            except ValueError:
                logger.warning(f"  ⚠️ Invalid relation '{relation_str}', defaulting to RELATED_TO")
                relation = RelationType.RELATED_TO

            # Create triple
            triple = CanonicalTriple(
                subject_label=subject,
                object_label=obj,
                relation=relation,
                confidence=confidence,
                justification=justification,
                page_number=page_number
            )

            return triple

        except json.JSONDecodeError as e:
            logger.error(f"  ❌ JSON parse error in relationship extraction: {e}")
            return None
        except Exception as e:
            logger.error(f"  ❌ Relationship extraction failed: {e}")
            return None

    def _is_technical_concept(self, concept: str) -> bool:
        """

        Args:
            concept: Concept string to validate

        Returns:
            True if highly technical/specific, False otherwise
        """
        concept_lower = concept.lower().strip()

        # RULE 1: Reject if in stopwords
        if concept_lower in self.generic_stopwords:
            logger.debug(f"Rejected '{concept}' - in stopword list")
            return False

        # RULE 2: Reject if any word is a generic stopword (stricter)
        words = concept_lower.split()
        for word in words:
            if word in self.generic_stopwords:
                # Allow if it's part of a specific multi-word technical term
                # e.g., "convolutional neural network" has "network" but is specific
                if len(words) < 2:
                    logger.debug(f"Rejected '{concept}' - contains generic word '{word}'")
                    return False

        # RULE 3: Single-word concepts must have SOME specificity (RELAXED)
        if len(words) == 1:
            # Accept if ANY of these are true:
            # - Has uppercase (BERT, Adam, PyTorch)
            # - Has numbers (VGG16, GPT3)
            # - Has special chars (t-SNE, bi-LSTM)
            # - Longish word (8+ chars like "backpropagation")
            has_uppercase = any(c.isupper() for c in concept)
            has_numbers = any(c.isdigit() for c in concept)
            has_special = '-' in concept or '_' in concept
            is_longish = len(concept) >= 8  # RELAXED from 10

            if not (has_uppercase or has_numbers or has_special or is_longish):
                logger.debug(f"Rejected '{concept}' - single word not specific enough")
                return False

        # RULE 4: Multi-word phrases - very lenient
        if len(words) >= 2:
            # Just check that it's not ALL generic words
            # At least one word should be non-generic or have caps/numbers
            has_caps = any(c.isupper() for c in concept)
            has_numbers = any(c.isdigit() for c in concept)
            has_hyphen = '-' in concept

            # Count non-generic words
            non_generic_count = sum(1 for w in words if w not in self.generic_stopwords)

            # Accept if ANY of these:
            # - Has caps/numbers/hyphen
            # - At least one word is non-generic
            # - 3+ words (likely specific enough)
            if not (has_caps or has_numbers or has_hyphen or non_generic_count > 0 or len(words) >= 3):
                logger.debug(f"Rejected '{concept}' - multi-word phrase too generic")
                return False

        # RULE 5: Reject very short terms (1-2 chars) unless they're known acronyms (all caps)
        if len(concept) <= 2 and concept.upper() != concept:
            logger.debug(f"Rejected '{concept}' - too short")
            return False

        # RULE 6: Must contain at least one alphanumeric character
        if not any(c.isalnum() for c in concept):
            logger.debug(f"Rejected '{concept}' - no alphanumeric chars")
            return False

        # RULE 7: Reject if it's just a generic category with a modifier
        # e.g., "new algorithm", "proposed method", "our model"
        generic_patterns = [
            'new ', 'novel ', 'proposed ', 'our ', 'this ', 'that ',
            'these ', 'those ', 'such ', 'other ', 'another ',
            'existing ', 'current ', 'previous ', 'standard '
        ]
        for pattern in generic_patterns:
            if concept_lower.startswith(pattern):
                logger.debug(f"Rejected '{concept}' - generic pattern")
                return False

        # Passed all strict filters
        return True

    def _map_relation(self, relation_str: str) -> RelationType:
        """Map relation string to RelationType enum"""
        relation_lower = relation_str.lower().strip()

        # Direct mapping
        mapping = {
            "uses": RelationType.USES,
            "implements": RelationType.IMPLEMENTS,
            "is_a": RelationType.IS_A,
            "is a": RelationType.IS_A,
            "part_of": RelationType.PART_OF,
            "part of": RelationType.PART_OF,
            "requires": RelationType.REQUIRES,
            "produces": RelationType.PRODUCES,
            "enables": RelationType.ENABLES,
            "improves": RelationType.IMPROVES,
            "enhances": RelationType.ENHANCES,
            "contains": RelationType.CONTAINS,
            "depends_on": RelationType.DEPENDS_ON,
            "depends on": RelationType.DEPENDS_ON,
            "related_to": RelationType.RELATED_TO,
            "related to": RelationType.RELATED_TO,
        }

        if relation_lower in mapping:
            return mapping[relation_lower]

        # Fallback
        logger.debug(f"Unknown relation '{relation_str}', using 'related_to'")
        return RelationType.RELATED_TO

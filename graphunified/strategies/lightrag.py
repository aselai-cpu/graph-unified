"""LightRAG: Dual-level knowledge graph retrieval strategy.

LightRAG is a lightweight graph-based RAG approach that intelligently combines
entity-level (local) and relationship-level (global) retrieval based on query type.

Key Features:
- Automatic query classification (local, global, or hybrid mode)
- Entity-level retrieval for specific queries
- Relationship-level retrieval for thematic queries
- Hybrid mode combining both levels for complex queries
- Lighter weight than full GraphRAG (no PPR, no community hierarchies)

Query Modes:
- LOCAL: Entity-focused retrieval (e.g., "What is GraphRAG?")
- GLOBAL: Relationship-focused retrieval (e.g., "What are the main themes?")
- HYBRID: Combined retrieval (e.g., "How does X relate to the broader context?")
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from graphunified.config.models import Chunk, Community, Entity, Relationship
from graphunified.config.settings import GraphRAGStrategyConfig
from graphunified.exceptions import IndexingError, RetrievalError
from graphunified.storage.graph_store import GraphStore
from graphunified.storage.parquet_store import ParquetStore
from graphunified.storage.vector_store import VectorStore
from graphunified.strategies.base import QueryType, RetrievalResult, RetrievalStrategy
from graphunified.utils.embedding_factory import create_embedding_client
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class QueryMode:
    """Query execution modes for LightRAG."""

    LOCAL = "local"  # Entity-level retrieval
    GLOBAL = "global"  # Relationship-level retrieval
    HYBRID = "hybrid"  # Combined retrieval


class LightRAGStrategy(RetrievalStrategy):
    """LightRAG: Intelligent dual-level knowledge graph retrieval.

    LightRAG automatically determines whether to use entity-level (local)
    or relationship-level (global) retrieval based on the query characteristics.

    Architecture:
    1. Query Classification: Determine retrieval mode (local/global/hybrid)
    2. Local Mode: Find entities → expand graph → collect chunks
    3. Global Mode: Find relationships → extract entities → collect chunks
    4. Hybrid Mode: Combine both local and global results

    Best for:
    - Mixed query workloads (both specific and exploratory)
    - Queries that need both detail and context
    - Scenarios where query type is unknown
    - Adaptive retrieval based on content

    Advantages:
    - Automatic query routing (no manual selection)
    - Combines benefits of local and global retrieval
    - More efficient than running both strategies separately
    - Graceful fallback if one mode fails
    """

    def __init__(
        self,
        config: GraphRAGStrategyConfig,
        vector_store: VectorStore,
        graph_store: GraphStore,
        parquet_store: ParquetStore,
        embedding_client: Any,
    ):
        """Initialize LightRAG strategy.

        Args:
            config: GraphRAG strategy configuration
            vector_store: Vector store for similarity search
            graph_store: Graph store for traversal
            parquet_store: Parquet store for loading data
            embedding_client: Embedding client for query encoding
        """
        super().__init__(config)
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.parquet_store = parquet_store
        self.embedding_client = embedding_client

        # Strategy config
        self.top_k = getattr(config, "top_k", 10)
        self.max_hops = getattr(config, "max_hops", 2)
        self.local_weight = getattr(config, "local_weight", 0.6)  # Weight for hybrid mode

        # Caches
        self._entity_cache: Dict[str, Entity] = {}
        self._relationship_cache: Dict[str, Relationship] = {}

        # Query classification keywords
        self._global_keywords = {
            "summarize",
            "overview",
            "themes",
            "topics",
            "trends",
            "overall",
            "general",
            "main",
            "key",
            "broad",
            "what are",
            "tell me about",
        }
        self._local_keywords = {
            "what is",
            "define",
            "explain",
            "describe",
            "how does",
            "specific",
            "exactly",
            "details",
            "who",
            "when",
            "where",
        }

        logger.info(
            f"Initialized {self.name} "
            f"(top_k={self.top_k}, max_hops={self.max_hops}, local_weight={self.local_weight})"
        )

    @classmethod
    async def from_config(
        cls,
        config: GraphRAGStrategyConfig,
        vector_store_path: Path,
        graph_store_path: Path,
        parquet_store_path: Path,
        embedding_config: Any,
    ) -> "LightRAGStrategy":
        """Create strategy from configuration.

        Args:
            config: GraphRAG strategy configuration
            vector_store_path: Path to vector store
            graph_store_path: Path to graph store
            parquet_store_path: Path to Parquet files
            embedding_config: Embedding configuration

        Returns:
            Initialized LightRAGStrategy
        """
        # Create clients
        embedding_client = create_embedding_client(embedding_config)

        # Load vector store
        from graphunified.config.settings import StorageConfig

        storage_config = StorageConfig()
        vector_store = VectorStore.from_config(
            storage_config.vector_db, vector_store_path, embedding_config.dimension
        )

        # Load graph store
        graph_store = GraphStore.from_config(storage_config, graph_store_path)
        await graph_store.load()

        # Load parquet store
        parquet_store = ParquetStore(parquet_store_path)

        return cls(config, vector_store, graph_store, parquet_store, embedding_client)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "LightRAG"

    @property
    def requires_entities(self) -> bool:
        """LightRAG requires entities."""
        return True

    @property
    def requires_relationships(self) -> bool:
        """LightRAG requires relationships."""
        return True

    @property
    def requires_communities(self) -> bool:
        """LightRAG does not require communities."""
        return False

    def supports_query_type(self, query_type: QueryType) -> bool:
        """Check if strategy supports a query type.

        LightRAG supports most query types through adaptive routing.
        """
        return query_type in (
            QueryType.FACTOID,
            QueryType.EXPLORATORY,
            QueryType.RELATIONAL,
            QueryType.THEMATIC,
            QueryType.COMPARATIVE,
        )

    async def index(
        self,
        chunks: List[Chunk],
        entities: List[Entity],
        relationships: List[Relationship],
        communities: List[Community],
    ) -> None:
        """Build strategy-specific indexes.

        For LightRAG:
        1. Cache entities and relationships
        2. Validate vector and graph stores

        Args:
            chunks: Text chunks
            entities: Extracted entities
            relationships: Extracted relationships
            communities: Not used by LightRAG
        """
        if not entities:
            raise IndexingError("No entities found for LightRAG")

        if not relationships:
            raise IndexingError("No relationships found for LightRAG")

        # Build caches
        self._entity_cache = {str(e.id): e for e in entities}
        self._relationship_cache = {str(r.id): r for r in relationships}

        logger.info(
            f"LightRAG ready: {len(self._entity_cache)} entities, "
            f"{len(self._relationship_cache)} relationships"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        mode: Optional[str] = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve relevant context using dual-level retrieval.

        Args:
            query: User query string
            top_k: Number of results to retrieve
            mode: Force specific mode (local/global/hybrid), or None for auto-detect
            **kwargs: Additional parameters

        Returns:
            RetrievalResult with chunks and metadata

        Raises:
            RetrievalError: If retrieval fails
        """
        start_time = time.time()

        try:
            # Step 1: Classify query mode
            if mode is None:
                mode = self._classify_query(query)
            logger.debug(f"Query mode: {mode}")

            # Step 2: Execute retrieval based on mode
            if mode == QueryMode.LOCAL:
                result = await self._local_retrieval(query, top_k)
            elif mode == QueryMode.GLOBAL:
                result = await self._global_retrieval(query, top_k)
            elif mode == QueryMode.HYBRID:
                result = await self._hybrid_retrieval(query, top_k)
            else:
                raise ValueError(f"Unknown query mode: {mode}")

            # Add timing and metadata
            result.retrieval_time_ms = (time.time() - start_time) * 1000
            result.metadata["query_mode"] = mode
            result.metadata["auto_classified"] = mode is None

            logger.info(
                f"LightRAG ({mode}): Retrieved {len(result.chunks)} chunks "
                f"in {result.retrieval_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"LightRAG retrieval failed: {e}", exc_info=True)
            raise RetrievalError(f"Failed to retrieve with LightRAG: {e}")

    def _classify_query(self, query: str) -> str:
        """Classify query to determine retrieval mode.

        Uses keyword matching to determine if query is:
        - LOCAL: Specific, factual questions (entity-focused)
        - GLOBAL: Broad, exploratory questions (relationship-focused)
        - HYBRID: Mixed or unclear queries

        Args:
            query: Query text

        Returns:
            Query mode (local/global/hybrid)
        """
        query_lower = query.lower()

        # Count keyword matches
        global_matches = sum(1 for kw in self._global_keywords if kw in query_lower)
        local_matches = sum(1 for kw in self._local_keywords if kw in query_lower)

        # Classify based on matches
        if global_matches > local_matches:
            return QueryMode.GLOBAL
        elif local_matches > global_matches:
            return QueryMode.LOCAL
        else:
            # Hybrid mode for ties or no matches
            return QueryMode.HYBRID

    async def _local_retrieval(
        self, query: str, top_k: int
    ) -> RetrievalResult:
        """Entity-level (local) retrieval.

        Workflow:
        1. Find relevant entities via vector search
        2. Expand to local graph (1-2 hops)
        3. Collect chunks connected to expanded entities

        Args:
            query: Query text
            top_k: Number of chunks to retrieve

        Returns:
            RetrievalResult with local context
        """
        # Step 1: Find seed entities
        query_embeddings = await self.embedding_client.embed([query])
        query_vector = query_embeddings[0]

        entity_results = await self.vector_store.search_entities(
            query_vector=query_vector, top_k=min(top_k, 10)
        )

        if not entity_results:
            logger.warning("No entities found for local retrieval")
            return RetrievalResult(
                strategy=f"{self.name} (local)",
                chunks=[],
                scores=[],
                metadata={"mode": QueryMode.LOCAL, "seed_entities": 0},
            )

        # entity_results is List[Tuple[entity_id, score, metadata]]
        seed_entity_ids = [UUID(eid) for eid, _, _ in entity_results]
        logger.debug(f"Found {len(seed_entity_ids)} seed entities")

        # Step 2: Expand to local graph
        expanded_entity_ids = await self._expand_local_graph(seed_entity_ids)
        logger.debug(f"Expanded to {len(expanded_entity_ids)} entities")

        # Step 3: Get entities and relationships
        entities = [
            self._entity_cache[str(eid)]
            for eid in expanded_entity_ids
            if str(eid) in self._entity_cache
        ]

        entity_id_set = set(expanded_entity_ids)
        relationships = [
            rel
            for rel in self._relationship_cache.values()
            if rel.source_entity_id in entity_id_set
            and rel.target_entity_id in entity_id_set
        ]

        # Step 4: Collect connected chunks
        chunks, scores = await self._collect_entity_chunks(expanded_entity_ids, top_k)

        return RetrievalResult(
            strategy=f"{self.name} (local)",
            chunks=chunks,
            scores=scores,
            entities=entities,
            relationships=relationships,
            metadata={
                "mode": QueryMode.LOCAL,
                "seed_entities": len(seed_entity_ids),
                "expanded_entities": len(expanded_entity_ids),
                "hops": self.max_hops,
            },
        )

    async def _global_retrieval(
        self, query: str, top_k: int
    ) -> RetrievalResult:
        """Relationship-level (global) retrieval (TRUE LightRAG).

        Workflow:
        1. Search relationship embeddings semantically
        2. Extract entities from top relationships
        3. Collect chunks connected to these entities

        Args:
            query: Query text
            top_k: Number of chunks to retrieve

        Returns:
            RetrievalResult with global context
        """
        # Step 1: Find relevant relationships via semantic search
        top_relationships = await self._search_relationships(query, top_k=20)

        if not top_relationships:
            logger.warning("No relationships found for global retrieval")
            return RetrievalResult(
                strategy=f"{self.name} (global)",
                chunks=[],
                scores=[],
                metadata={"mode": QueryMode.GLOBAL, "relationships_searched": 0},
            )

        logger.debug(f"Found {len(top_relationships)} relevant relationships")

        # Step 2: Extract unique entities from relationships
        entity_ids = set()
        for rel, score in top_relationships:
            entity_ids.add(rel.source_entity_id)
            entity_ids.add(rel.target_entity_id)

        logger.debug(f"Extracted {len(entity_ids)} entities from relationships")

        # Step 3: Get entities and relationships
        entities = [
            self._entity_cache[str(eid)]
            for eid in entity_ids
            if str(eid) in self._entity_cache
        ]

        relationships = [rel for rel, score in top_relationships]

        # Step 4: Collect chunks connected to these entities
        chunks, scores = await self._collect_entity_chunks(list(entity_ids), top_k)

        return RetrievalResult(
            strategy=f"{self.name} (global)",
            chunks=chunks,
            scores=scores,
            entities=entities,
            relationships=relationships,
            metadata={
                "mode": QueryMode.GLOBAL,
                "relationships_searched": len(top_relationships),
                "entities_extracted": len(entity_ids),
            },
        )

    async def _hybrid_retrieval(
        self, query: str, top_k: int
    ) -> RetrievalResult:
        """Hybrid retrieval combining local and global modes.

        Workflow:
        1. Run local retrieval (entity-focused)
        2. Run global retrieval (relationship-focused)
        3. Combine results with weighted scoring

        Args:
            query: Query text
            top_k: Number of chunks to retrieve

        Returns:
            RetrievalResult with combined context
        """
        # Run both modes
        local_result = await self._local_retrieval(query, top_k=top_k // 2)
        global_result = await self._global_retrieval(query, top_k=top_k // 2)

        # Combine chunks with weighted scores
        combined_chunks = []
        combined_scores = []

        # Add local results (weighted)
        for chunk, score in zip(local_result.chunks, local_result.scores):
            combined_chunks.append(chunk)
            combined_scores.append(score * self.local_weight)

        # Add global results (weighted)
        for chunk, score in zip(global_result.chunks, global_result.scores):
            combined_chunks.append(chunk)
            combined_scores.append(score * (1 - self.local_weight))

        # Sort by combined score and take top_k
        if combined_chunks:
            sorted_pairs = sorted(
                zip(combined_chunks, combined_scores),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
            combined_chunks, combined_scores = zip(*sorted_pairs)
            combined_chunks = list(combined_chunks)
            combined_scores = list(combined_scores)

        # Combine entities and relationships
        all_entities = local_result.entities + global_result.entities
        all_relationships = local_result.relationships + global_result.relationships

        # Deduplicate by ID
        unique_entities = {str(e.id): e for e in all_entities}.values()
        unique_relationships = {str(r.id): r for r in all_relationships}.values()

        return RetrievalResult(
            strategy=f"{self.name} (hybrid)",
            chunks=combined_chunks,
            scores=combined_scores,
            entities=list(unique_entities),
            relationships=list(unique_relationships),
            metadata={
                "mode": QueryMode.HYBRID,
                "local_weight": self.local_weight,
                "local_chunks": len(local_result.chunks),
                "global_chunks": len(global_result.chunks),
            },
        )

    async def _expand_local_graph(
        self, seed_entity_ids: List[UUID]
    ) -> List[UUID]:
        """Expand seed entities to local subgraph.

        Args:
            seed_entity_ids: Starting entity IDs

        Returns:
            List of expanded entity IDs (includes seeds)
        """
        expanded = set(seed_entity_ids)

        # BFS expansion up to max_hops
        current_level = set(seed_entity_ids)
        for hop in range(self.max_hops):
            next_level = set()
            for entity_id in current_level:
                try:
                    neighbors = await self.graph_store.get_neighbors(
                        entity_id, max_hops=1
                    )
                    neighbor_ids = [UUID(n[0]) for n in neighbors]
                    next_level.update(neighbor_ids)
                    expanded.update(neighbor_ids)
                except Exception as e:
                    logger.debug(f"Failed to get neighbors for {entity_id}: {e}")

            if not next_level:
                break

            current_level = next_level - expanded

        return list(expanded)

    async def _collect_entity_chunks(
        self, entity_ids: List[UUID], top_k: int
    ) -> Tuple[List[Chunk], List[float]]:
        """Collect chunks connected to entities.

        Args:
            entity_ids: Entity IDs to collect chunks for
            top_k: Maximum number of chunks

        Returns:
            Tuple of (chunks, scores)
        """
        entity_id_set = set(entity_ids)
        chunk_scores: Dict[str, float] = {}

        # Load all chunks and find those connected to our entities
        async for chunk in self.parquet_store.load_chunks():
            if any(eid in entity_id_set for eid in chunk.entity_ids):
                # Score by number of matching entities
                matches = sum(1 for eid in chunk.entity_ids if eid in entity_id_set)
                score = matches / len(entity_ids) if entity_ids else 0.0
                chunk_scores[str(chunk.id)] = (chunk, score)

        # Sort by score and take top_k
        if not chunk_scores:
            return [], []

        sorted_items = sorted(
            chunk_scores.values(), key=lambda x: x[1], reverse=True
        )[:top_k]

        chunks = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]

        return chunks, scores

    async def _search_relationships(
        self, query: str, top_k: int = 20
    ) -> List[Tuple[Relationship, float]]:
        """Search relationships by semantic similarity to query.

        Args:
            query: Query text
            top_k: Number of relationships to return

        Returns:
            List of (relationship, score) tuples
        """
        # Step 1: Generate query embedding
        query_embeddings = await self.embedding_client.embed([query])
        query_vector = query_embeddings[0]

        # Step 2: Search relationship vector index
        results = await self.vector_store.search_relationships(
            query_vector=query_vector,
            top_k=top_k,
        )

        # Step 3: Convert to Relationship objects with scores
        relationships = []
        for rel_id, score, metadata in results:
            if rel_id in self._relationship_cache:
                relationships.append((self._relationship_cache[rel_id], score))
            else:
                logger.warning(f"Relationship {rel_id} not found in cache")

        return relationships

    async def validate_index(self) -> bool:
        """Validate that indexes are ready.

        Returns:
            True if valid
        """
        try:
            if not self._entity_cache:
                logger.error("No entities cached")
                return False

            if not self._relationship_cache:
                logger.error("No relationships cached")
                return False

            logger.debug(
                f"LightRAG validation passed: "
                f"{len(self._entity_cache)} entities, "
                f"{len(self._relationship_cache)} relationships"
            )
            return True

        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False

"""HippoRAG: Personalized PageRank-based retrieval inspired by hippocampal memory.

HippoRAG uses Personalized PageRank (PPR) to simulate human-like memory recall
patterns, inspired by the hippocampal indexing theory in neuroscience.

Key Concepts:
- Query activates seed entities (initial memory traces)
- PPR propagates activation through knowledge graph (associative recall)
- Entities with higher PPR scores are "more memorable"
- Simulates multi-hop reasoning naturally through graph structure
- Neurologically-inspired: mirrors how hippocampus indexes memories

PPR Algorithm:
1. Find seed entities from query (vector similarity)
2. Initialize PPR scores (uniform over seeds)
3. Iteratively propagate scores through graph edges
4. Apply damping factor (probability of jumping back to seeds)
5. Converge to stationary distribution
6. Retrieve chunks connected to top-activated entities

Best for:
- Multi-hop reasoning questions
- Association-based retrieval
- Questions requiring graph traversal
- Simulating human memory patterns
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from uuid import UUID

import networkx as nx

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


class HippoRAGStrategy(RetrievalStrategy):
    """HippoRAG: Neurologically-inspired retrieval using Personalized PageRank.

    HippoRAG simulates human memory recall by:
    1. Activating seed entities from query (initial memory traces)
    2. Propagating activation through knowledge graph (associative recall)
    3. Retrieving chunks from highly-activated entities

    The PageRank algorithm naturally captures:
    - Multi-hop relationships (activation flows through graph)
    - Entity importance (central entities get higher activation)
    - Associative recall (related entities are activated)

    Advantages:
    - Natural multi-hop reasoning without explicit traversal
    - Simulates human-like memory patterns
    - Handles indirect relationships well
    - Robust to noise in seed entity selection
    - Computationally efficient (PPR is well-optimized)

    Inspired by:
    - Hippocampal indexing theory (neuroscience)
    - Complementary Learning Systems theory
    - Graph-based memory models
    """

    def __init__(
        self,
        config: GraphRAGStrategyConfig,
        vector_store: VectorStore,
        graph_store: GraphStore,
        parquet_store: ParquetStore,
        embedding_client: Any,
    ):
        """Initialize HippoRAG strategy.

        Args:
            config: GraphRAG strategy configuration
            vector_store: Vector store for entity search
            graph_store: Graph store for PPR computation
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
        self.num_seed_entities = getattr(config, "num_seed_entities", 5)
        self.damping_factor = getattr(config, "damping_factor", 0.85)  # α in PPR
        self.ppr_max_iter = getattr(config, "ppr_max_iter", 100)
        self.ppr_tolerance = getattr(config, "ppr_tolerance", 1e-6)

        # Caches
        self._entity_cache: Dict[str, Entity] = {}
        self._relationship_cache: Dict[str, Relationship] = {}

        logger.info(
            f"Initialized {self.name} "
            f"(top_k={self.top_k}, seeds={self.num_seed_entities}, "
            f"damping={self.damping_factor})"
        )

    @classmethod
    async def from_config(
        cls,
        config: GraphRAGStrategyConfig,
        vector_store_path: Path,
        graph_store_path: Path,
        parquet_store_path: Path,
        embedding_config: Any,
    ) -> "HippoRAGStrategy":
        """Create strategy from configuration.

        Args:
            config: GraphRAG strategy configuration
            vector_store_path: Path to vector store
            graph_store_path: Path to graph store
            parquet_store_path: Path to Parquet files
            embedding_config: Embedding configuration

        Returns:
            Initialized HippoRAGStrategy
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
        return "HippoRAG"

    @property
    def requires_entities(self) -> bool:
        """HippoRAG requires entities."""
        return True

    @property
    def requires_relationships(self) -> bool:
        """HippoRAG requires relationships."""
        return True

    @property
    def requires_communities(self) -> bool:
        """HippoRAG does not require communities."""
        return False

    def supports_query_type(self, query_type: QueryType) -> bool:
        """Check if strategy supports a query type.

        HippoRAG excels at relational and multi-hop reasoning queries.
        """
        return query_type in (
            QueryType.FACTOID,
            QueryType.RELATIONAL,
            QueryType.COMPARATIVE,
            QueryType.TEMPORAL,
        )

    async def index(
        self,
        chunks: List[Chunk],
        entities: List[Entity],
        relationships: List[Relationship],
        communities: List[Community],
    ) -> None:
        """Build strategy-specific indexes.

        For HippoRAG:
        1. Cache entities and relationships
        2. Validate graph structure
        3. Pre-compute any graph statistics if needed

        Args:
            chunks: Text chunks
            entities: Extracted entities
            relationships: Extracted relationships
            communities: Detected communities (unused)
        """
        if not entities:
            raise IndexingError("No entities found for HippoRAG")

        if not relationships:
            logger.warning("No relationships found - HippoRAG may not work well")

        # Build caches
        self._entity_cache = {str(e.id): e for e in entities}
        self._relationship_cache = {str(r.id): r for r in relationships}

        logger.info(
            f"HippoRAG ready: {len(self._entity_cache)} entities, "
            f"{len(self._relationship_cache)} relationships, "
            f"{self.graph_store.graph.number_of_nodes()} nodes, "
            f"{self.graph_store.graph.number_of_edges()} edges"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve relevant context using Personalized PageRank.

        Workflow:
        1. Find seed entities from query (vector similarity)
        2. Run PPR from seed entities
        3. Rank entities by PPR scores
        4. Collect chunks from top-activated entities

        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            **kwargs: Additional parameters

        Returns:
            RetrievalResult with PPR-based retrieval

        Raises:
            RetrievalError: If retrieval fails
        """
        start_time = time.time()

        try:
            # Step 1: Find seed entities
            logger.debug(f"Finding seed entities for query: {query}")
            seed_entity_ids = await self._find_seed_entities(query)

            if not seed_entity_ids:
                logger.warning("No seed entities found")
                return RetrievalResult(
                    strategy=self.name,
                    chunks=[],
                    scores=[],
                    metadata={"seed_entities": 0},
                    retrieval_time_ms=(time.time() - start_time) * 1000,
                )

            logger.debug(f"Found {len(seed_entity_ids)} seed entities")

            # Step 2: Run Personalized PageRank
            logger.debug("Running Personalized PageRank...")
            ppr_scores = await self._run_personalized_pagerank(seed_entity_ids)

            if not ppr_scores:
                logger.warning("PPR returned no scores")
                return RetrievalResult(
                    strategy=self.name,
                    chunks=[],
                    scores=[],
                    metadata={"seed_entities": len(seed_entity_ids), "ppr_scores": 0},
                    retrieval_time_ms=(time.time() - start_time) * 1000,
                )

            logger.debug(f"PPR computed scores for {len(ppr_scores)} entities")

            # Step 3: Get top-activated entities
            top_entity_ids = self._get_top_entities(ppr_scores, top_k=self.top_k)
            logger.debug(f"Selected {len(top_entity_ids)} top entities")

            # Step 4: Get entities and relationships
            entities = [
                self._entity_cache[str(eid)]
                for eid in top_entity_ids
                if str(eid) in self._entity_cache
            ]

            entity_id_set = set(top_entity_ids)
            relationships = [
                rel
                for rel in self._relationship_cache.values()
                if rel.source_entity_id in entity_id_set
                and rel.target_entity_id in entity_id_set
            ]

            # Step 5: Collect chunks connected to activated entities
            logger.debug("Collecting chunks from activated entities...")
            chunks, scores = await self._collect_activated_chunks(
                top_entity_ids, ppr_scores, top_k
            )

            retrieval_time = (time.time() - start_time) * 1000

            logger.info(
                f"HippoRAG retrieved {len(chunks)} chunks from "
                f"{len(entities)} activated entities in {retrieval_time:.2f}ms"
            )

            return RetrievalResult(
                strategy=self.name,
                chunks=chunks,
                scores=scores,
                entities=entities,
                relationships=relationships,
                retrieval_time_ms=retrieval_time,
                metadata={
                    "seed_entities": len(seed_entity_ids),
                    "activated_entities": len(top_entity_ids),
                    "ppr_iterations": self.ppr_max_iter,
                    "damping_factor": self.damping_factor,
                    "mean_ppr_score": sum(ppr_scores.values()) / len(ppr_scores)
                    if ppr_scores
                    else 0.0,
                },
            )

        except Exception as e:
            logger.error(f"HippoRAG retrieval failed: {e}", exc_info=True)
            raise RetrievalError(f"Failed to retrieve with HippoRAG: {e}")

    async def _find_seed_entities(self, query: str) -> List[UUID]:
        """Find seed entities for query using vector similarity.

        Args:
            query: Query text

        Returns:
            List of seed entity IDs
        """
        # Encode query
        query_embeddings = await self.embedding_client.embed([query])
        query_vector = query_embeddings[0]

        # Search entity index
        entity_results = await self.vector_store.search_entities(
            query_vector=query_vector, top_k=self.num_seed_entities
        )

        if not entity_results:
            return []

        # Extract entity IDs
        seed_entity_ids = [UUID(eid) for eid, _, _ in entity_results]
        return seed_entity_ids

    async def _run_personalized_pagerank(
        self, seed_entity_ids: List[UUID]
    ) -> Dict[str, float]:
        """Run Personalized PageRank from seed entities.

        PPR Formula:
            PR(v) = (1 - α) * p(v) + α * Σ(PR(u) / outdegree(u))
            where p(v) is the personalization vector (uniform over seeds)

        Args:
            seed_entity_ids: Seed entities for personalization

        Returns:
            Dictionary mapping entity_id -> PPR score
        """
        try:
            # Build personalization vector (uniform over seeds)
            seed_entity_strs = [str(eid) for eid in seed_entity_ids]
            personalization = {
                node: 1.0 / len(seed_entity_strs) if node in seed_entity_strs else 0.0
                for node in self.graph_store.graph.nodes()
            }

            # Convert to undirected graph for PageRank (if directed)
            if self.graph_store.directed:
                graph = self.graph_store.graph.to_undirected()
            else:
                graph = self.graph_store.graph

            # Run Personalized PageRank
            ppr_scores = nx.pagerank(
                graph,
                alpha=self.damping_factor,
                personalization=personalization,
                max_iter=self.ppr_max_iter,
                tol=self.ppr_tolerance,
            )

            logger.debug(
                f"PPR converged: {len(ppr_scores)} scores, "
                f"max={max(ppr_scores.values()):.6f}, "
                f"min={min(ppr_scores.values()):.6f}"
            )

            return ppr_scores

        except Exception as e:
            logger.error(f"PPR computation failed: {e}")
            return {}

    def _get_top_entities(
        self, ppr_scores: Dict[str, float], top_k: int
    ) -> List[UUID]:
        """Get top-k entities by PPR score.

        Args:
            ppr_scores: PPR scores for entities
            top_k: Number of entities to return

        Returns:
            List of top entity IDs
        """
        # Sort by PPR score (descending)
        sorted_entities = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)

        # Take top_k and convert to UUIDs
        top_entities = [UUID(entity_id) for entity_id, _ in sorted_entities[:top_k]]

        return top_entities

    async def _collect_activated_chunks(
        self, entity_ids: List[UUID], ppr_scores: Dict[str, float], top_k: int
    ) -> Tuple[List[Chunk], List[float]]:
        """Collect chunks connected to activated entities.

        Score chunks by:
        - Number of connected activated entities
        - Sum of PPR scores of connected entities
        - Hybrid scoring: combines both factors

        Args:
            entity_ids: Activated entity IDs
            ppr_scores: PPR scores for entities
            top_k: Maximum number of chunks

        Returns:
            Tuple of (chunks, scores)
        """
        entity_id_set = set(entity_ids)
        chunk_scores: Dict[str, Tuple[Chunk, float]] = {}

        # Load chunks and score by activated entities
        async for chunk in self.parquet_store.load_chunks():
            # Find connected activated entities
            connected_entities = [eid for eid in chunk.entity_ids if eid in entity_id_set]

            if connected_entities:
                # Hybrid scoring:
                # - Count of connected entities (normalized)
                # - Sum of PPR scores of connected entities
                count_score = len(connected_entities) / len(entity_ids)
                ppr_score_sum = sum(
                    ppr_scores.get(str(eid), 0.0) for eid in connected_entities
                )

                # Combined score (equal weight)
                score = 0.5 * count_score + 0.5 * ppr_score_sum

                chunk_scores[str(chunk.id)] = (chunk, score)

        if not chunk_scores:
            return [], []

        # Sort by score and take top_k
        sorted_items = sorted(
            chunk_scores.values(), key=lambda x: x[1], reverse=True
        )[:top_k]

        chunks = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]

        return chunks, scores

    async def validate_index(self) -> bool:
        """Validate that indexes are ready.

        Returns:
            True if valid
        """
        try:
            if not self._entity_cache:
                logger.error("No entities cached")
                return False

            if not self.graph_store.graph.number_of_nodes():
                logger.error("Empty knowledge graph")
                return False

            # Check graph connectivity
            if self.graph_store.directed:
                num_components = nx.number_weakly_connected_components(
                    self.graph_store.graph
                )
            else:
                num_components = nx.number_connected_components(self.graph_store.graph)

            logger.debug(
                f"HippoRAG validation passed: "
                f"{len(self._entity_cache)} entities, "
                f"{self.graph_store.graph.number_of_edges()} edges, "
                f"{num_components} connected components"
            )
            return True

        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False

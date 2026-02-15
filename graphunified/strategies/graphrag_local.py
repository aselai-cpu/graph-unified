"""GraphRAG Local: Entity-focused local search with graph traversal.

Uses entity embeddings to find relevant entities, then traverses the local
knowledge graph to gather rich context for detailed, grounded answers.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Set
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


class GraphRAGLocalStrategy(RetrievalStrategy):
    """GraphRAG Local: Entity-driven local graph search.

    Workflow:
    1. Embed query and find relevant entities (vector similarity)
    2. For each relevant entity, get local neighborhood (1-2 hops)
    3. Collect all chunks connected to entities in the local subgraph
    4. Collect relationships between entities
    5. Return chunks with entity/relationship context

    Best for:
    - Detailed questions about specific entities
    - Multi-hop reasoning ("How is X related to Y?")
    - Questions requiring entity-level context
    - Exploratory queries needing rich graph structure

    Advantages:
    - Rich context: Entities + relationships + chunks
    - Graph-aware: Understands connections between concepts
    - Multi-hop: Can traverse relationships
    - Grounded: All answers tied to source chunks
    """

    def __init__(
        self,
        config: GraphRAGStrategyConfig,
        vector_store: VectorStore,
        graph_store: GraphStore,
        parquet_store: ParquetStore,
        embedding_client: Any,
    ):
        """Initialize GraphRAG Local strategy.

        Args:
            config: GraphRAG strategy configuration
            vector_store: Vector store with entity embeddings
            graph_store: Graph store for traversal
            parquet_store: Parquet store for loading chunks
            embedding_client: Embedding client for query encoding
        """
        super().__init__(config)
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.parquet_store = parquet_store
        self.embedding_client = embedding_client

        # Strategy config
        self.top_k = getattr(config, 'top_k', 10)
        self.max_entities = 5  # Number of seed entities to find
        self.max_hops = 2  # Graph traversal depth

        # Cache for entities and relationships
        self._entity_cache: Dict[str, Entity] = {}
        self._relationship_cache: Dict[str, Relationship] = {}

        logger.info(
            f"Initialized {self.name} "
            f"(top_k={self.top_k}, max_entities={self.max_entities}, max_hops={self.max_hops})"
        )

    @classmethod
    async def from_config(
        cls,
        config: GraphRAGStrategyConfig,
        vector_store_path: Path,
        graph_store_path: Path,
        parquet_store_path: Path,
        embedding_config: Any,
    ) -> "GraphRAGLocalStrategy":
        """Create strategy from configuration.

        Args:
            config: GraphRAG strategy configuration
            vector_store_path: Path to LanceDB directory
            graph_store_path: Path to graph store
            parquet_store_path: Path to Parquet files
            embedding_config: Embedding configuration

        Returns:
            Initialized GraphRAGLocalStrategy
        """
        # Create embedding client
        embedding_client = create_embedding_client(embedding_config)

        # Load vector store
        from graphunified.config.settings import VectorDBConfig

        vector_db_config = VectorDBConfig()
        vector_store = VectorStore.from_config(
            vector_db_config,
            vector_store_path,
            embedding_config.dimension,
        )

        # Load graph store
        from graphunified.config.settings import StorageConfig

        storage_config = StorageConfig()
        graph_store = GraphStore.from_config(storage_config, graph_store_path)
        await graph_store.load()  # Load persisted graph

        # Load parquet store
        parquet_store = ParquetStore(parquet_store_path)

        return cls(config, vector_store, graph_store, parquet_store, embedding_client)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "GraphRAG Local"

    @property
    def requires_entities(self) -> bool:
        """GraphRAG Local requires entities."""
        return True

    @property
    def requires_relationships(self) -> bool:
        """GraphRAG Local requires relationships."""
        return True

    def supports_query_type(self, query_type: QueryType) -> bool:
        """Check if strategy supports a query type.

        GraphRAG Local excels at relational and exploratory queries.
        """
        return query_type in (
            QueryType.RELATIONAL,
            QueryType.EXPLORATORY,
            QueryType.FACTOID,
        )

    async def index(
        self,
        chunks: List[Chunk],
        entities: List[Entity],
        relationships: List[Relationship],
        communities: List[Community],
    ) -> None:
        """Build strategy-specific indexes.

        For GraphRAG Local, we need:
        - Entity vector index (built by Stage 5)
        - Knowledge graph (build here if not exists)

        Args:
            chunks: Text chunks with embeddings
            entities: Extracted entities with embeddings
            relationships: Extracted relationships
            communities: Detected communities (not used for local)
        """
        # Validate entities
        entities_with_embeddings = [
            e for e in entities if e.embedding and len(e.embedding) > 0
        ]

        if not entities_with_embeddings:
            raise IndexingError("No entities with embeddings found for GraphRAG Local")

        # Build or load graph
        if self.graph_store.graph.number_of_nodes() == 0:
            logger.info("Building knowledge graph...")
            await self.graph_store.build_graph(entities, relationships)
            await self.graph_store.save()
            logger.info(
                f"Built graph: {self.graph_store.graph.number_of_nodes()} nodes, "
                f"{self.graph_store.graph.number_of_edges()} edges"
            )
        else:
            logger.info(
                f"Graph already loaded: {self.graph_store.graph.number_of_nodes()} nodes"
            )

        # Cache entities and relationships for fast lookup
        await self._build_caches(entities, relationships)

        logger.info(
            f"GraphRAG Local ready: {len(entities_with_embeddings)} entities, "
            f"{len(relationships)} relationships"
        )

    async def _build_caches(
        self, entities: List[Entity], relationships: List[Relationship]
    ) -> None:
        """Build in-memory caches for entities and relationships.

        Args:
            entities: List of entities
            relationships: List of relationships
        """
        self._entity_cache = {str(e.id): e for e in entities}
        self._relationship_cache = {str(r.id): r for r in relationships}

        logger.debug(
            f"Built caches: {len(self._entity_cache)} entities, "
            f"{len(self._relationship_cache)} relationships"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve relevant context using local graph search.

        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            **kwargs: Additional parameters

        Returns:
            RetrievalResult with chunks and graph context

        Raises:
            RetrievalError: If retrieval fails
        """
        start_time = time.time()

        try:
            # Step 1: Find relevant entities
            logger.debug(f"Finding entities for: {query[:100]}...")
            seed_entities = await self._find_relevant_entities(query, self.max_entities)

            if not seed_entities:
                logger.warning("No relevant entities found")
                return RetrievalResult(
                    strategy=self.name,
                    chunks=[],
                    scores=[],
                    retrieval_time_ms=(time.time() - start_time) * 1000,
                    metadata={"entities_found": 0},
                )

            # Step 2: Expand to local neighborhood
            logger.debug(f"Expanding {len(seed_entities)} seed entities to local graph...")
            local_entity_ids = await self._expand_to_local_graph(
                [e.id for e in seed_entities],
                max_hops=self.max_hops,
            )

            # Step 3: Collect connected chunks
            logger.debug(f"Collecting chunks for {len(local_entity_ids)} entities...")
            chunks, chunk_scores = await self._collect_connected_chunks(
                local_entity_ids, top_k
            )

            # Step 4: Collect relationships
            logger.debug("Collecting relationships...")
            relationships = await self._collect_relationships(local_entity_ids)

            # Step 5: Build entities list
            entities = [self._entity_cache[str(eid)] for eid in local_entity_ids if str(eid) in self._entity_cache]

            retrieval_time = (time.time() - start_time) * 1000

            logger.info(
                f"Retrieved {len(chunks)} chunks via {len(entities)} entities "
                f"and {len(relationships)} relationships in {retrieval_time:.2f}ms"
            )

            return RetrievalResult(
                strategy=self.name,
                chunks=chunks,
                scores=chunk_scores,
                entities=entities,
                relationships=relationships,
                retrieval_time_ms=retrieval_time,
                total_chunks_searched=len(chunks),
                metadata={
                    "seed_entities": len(seed_entities),
                    "local_entities": len(local_entity_ids),
                    "relationships": len(relationships),
                    "max_hops": self.max_hops,
                },
            )

        except Exception as e:
            logger.error(f"GraphRAG Local retrieval failed: {e}", exc_info=True)
            raise RetrievalError(f"Failed to retrieve with GraphRAG Local: {e}")

    async def _find_relevant_entities(
        self, query: str, top_k: int
    ) -> List[Entity]:
        """Find entities relevant to the query using vector similarity.

        Args:
            query: Query text
            top_k: Number of entities to find

        Returns:
            List of relevant entities
        """
        # Embed query
        query_embeddings = await self.embedding_client.embed([query])
        query_vector = query_embeddings[0]

        # Search entity index
        results = await self.vector_store.search_entities(
            query_vector=query_vector,
            top_k=top_k,
        )

        # Convert to Entity objects
        entities = []
        for entity_id, distance, metadata in results:
            if entity_id in self._entity_cache:
                entities.append(self._entity_cache[entity_id])

        logger.debug(f"Found {len(entities)} relevant entities")
        return entities

    async def _expand_to_local_graph(
        self, seed_entity_ids: List[UUID], max_hops: int
    ) -> Set[UUID]:
        """Expand seed entities to local neighborhood.

        Args:
            seed_entity_ids: Starting entity IDs
            max_hops: Maximum traversal depth

        Returns:
            Set of entity IDs in local graph
        """
        local_entities: Set[UUID] = set(seed_entity_ids)

        # For each seed entity, get neighbors
        for entity_id in seed_entity_ids:
            try:
                neighbors = await self.graph_store.get_neighbors(
                    entity_id, max_hops=max_hops
                )

                # Add neighbor IDs
                for neighbor_id, _ in neighbors:
                    local_entities.add(UUID(neighbor_id))

            except Exception as e:
                logger.warning(f"Failed to get neighbors for {entity_id}: {e}")

        logger.debug(
            f"Expanded {len(seed_entity_ids)} seeds to {len(local_entities)} entities"
        )
        return local_entities

    async def _collect_connected_chunks(
        self, entity_ids: Set[UUID], top_k: int
    ) -> tuple[List[Chunk], List[float]]:
        """Collect chunks connected to entities.

        Args:
            entity_ids: Entity IDs to collect chunks for
            top_k: Maximum number of chunks

        Returns:
            Tuple of (chunks, scores)
        """
        # Load all chunks (we need to filter by entity connections)
        all_chunks = []
        async for chunk in self.parquet_store.load_chunks():
            # Check if chunk is connected to any of our entities
            if any(eid in entity_ids for eid in chunk.entity_ids):
                all_chunks.append(chunk)

        # Sort by number of entity connections (more connections = more relevant)
        scored_chunks = [
            (chunk, len(set(chunk.entity_ids) & entity_ids))
            for chunk in all_chunks
        ]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Take top-k
        chunks = [chunk for chunk, _ in scored_chunks[:top_k]]
        scores = [float(score) / len(entity_ids) for _, score in scored_chunks[:top_k]]

        return chunks, scores

    async def _collect_relationships(
        self, entity_ids: Set[UUID]
    ) -> List[Relationship]:
        """Collect relationships between entities in the local graph.

        Args:
            entity_ids: Entity IDs in local graph

        Returns:
            List of relationships
        """
        relationships = []

        # Filter relationships where both source and target are in local graph
        for relationship in self._relationship_cache.values():
            if (
                relationship.source_entity_id in entity_ids
                and relationship.target_entity_id in entity_ids
            ):
                relationships.append(relationship)

        return relationships

    async def validate_index(self) -> bool:
        """Validate that entity index and graph are ready.

        Returns:
            True if valid
        """
        try:
            # Check entity vector store
            db = await self.vector_store._get_db()
            table_names = db.table_names()

            if self.vector_store.entity_index_name not in table_names:
                logger.error("Entity vector index not found")
                return False

            # Check graph
            if self.graph_store.graph.number_of_nodes() == 0:
                logger.error("Knowledge graph is empty")
                return False

            logger.debug(
                f"GraphRAG Local validation passed: "
                f"{self.graph_store.graph.number_of_nodes()} nodes"
            )
            return True

        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False

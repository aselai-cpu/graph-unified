"""GraphRAG Global: Community-level search for thematic questions.

Uses community detection to identify clusters of related entities, generates
summaries for each community, then searches/aggregates these summaries to
answer high-level, exploratory questions.
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
from graphunified.strategies.base import QueryType, RetrievalResult, RetrievalStrategy
from graphunified.utils.embedding_factory import create_embedding_client
from graphunified.utils.llm import ClaudeClient
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class GraphRAGGlobalStrategy(RetrievalStrategy):
    """GraphRAG Global: Community-level search for thematic understanding.

    Workflow:
    1. Detect communities in the knowledge graph (Leiden/Louvain)
    2. Generate summary reports for each community (LLM)
    3. For a query, find relevant communities
    4. Aggregate community reports to answer high-level questions

    Best for:
    - Exploratory questions ("What are the main themes?")
    - Thematic questions ("What does the corpus say about X?")
    - Broad overviews ("Summarize the key topics")
    - Comparative questions across domains

    Advantages:
    - Hierarchical understanding (community structure)
    - Scalable to large corpora (summarize then search)
    - Good for "forest not trees" questions
    - Captures cross-cutting themes

    Community Report Structure:
    - Title: Community theme/topic
    - Summary: High-level description
    - Key entities: Most important entities in community
    - Key relationships: Important connections
    - Supporting facts: Evidence from chunks
    """

    def __init__(
        self,
        config: GraphRAGStrategyConfig,
        graph_store: GraphStore,
        parquet_store: ParquetStore,
        llm_client: ClaudeClient,
        embedding_client: Any,
    ):
        """Initialize GraphRAG Global strategy.

        Args:
            config: GraphRAG strategy configuration
            graph_store: Graph store for community detection
            parquet_store: Parquet store for loading data
            llm_client: LLM client for report generation
            embedding_client: Embedding client for query encoding
        """
        super().__init__(config)
        self.graph_store = graph_store
        self.parquet_store = parquet_store
        self.llm_client = llm_client
        self.embedding_client = embedding_client

        # Strategy config
        self.top_k = getattr(config, 'top_k', 5)
        self.leiden_resolution = getattr(config, 'leiden_resolution', 1.0)
        self.max_community_size = getattr(config, 'max_community_size', 50)

        # Cache for communities and reports
        self._communities: List[Community] = []
        self._community_reports: Dict[int, str] = {}  # community_id -> report text
        self._entity_cache: Dict[str, Entity] = {}

        logger.info(
            f"Initialized {self.name} "
            f"(top_k={self.top_k}, leiden_resolution={self.leiden_resolution})"
        )

    @classmethod
    async def from_config(
        cls,
        config: GraphRAGStrategyConfig,
        graph_store_path: Path,
        parquet_store_path: Path,
        llm_config: Any,
        embedding_config: Any,
    ) -> "GraphRAGGlobalStrategy":
        """Create strategy from configuration.

        Args:
            config: GraphRAG strategy configuration
            graph_store_path: Path to graph store
            parquet_store_path: Path to Parquet files
            llm_config: LLM configuration
            embedding_config: Embedding configuration

        Returns:
            Initialized GraphRAGGlobalStrategy
        """
        # Create clients
        llm_client = ClaudeClient.from_config(llm_config)
        embedding_client = create_embedding_client(embedding_config)

        # Load graph store
        from graphunified.config.settings import StorageConfig

        storage_config = StorageConfig()
        graph_store = GraphStore.from_config(storage_config, graph_store_path)
        await graph_store.load()

        # Load parquet store
        parquet_store = ParquetStore(parquet_store_path)

        return cls(config, graph_store, parquet_store, llm_client, embedding_client)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "GraphRAG Global"

    @property
    def requires_entities(self) -> bool:
        """GraphRAG Global requires entities."""
        return True

    @property
    def requires_relationships(self) -> bool:
        """GraphRAG Global requires relationships."""
        return True

    @property
    def requires_communities(self) -> bool:
        """GraphRAG Global requires communities."""
        return True

    def supports_query_type(self, query_type: QueryType) -> bool:
        """Check if strategy supports a query type.

        GraphRAG Global excels at exploratory and thematic queries.
        """
        return query_type in (
            QueryType.EXPLORATORY,
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

        For GraphRAG Global:
        1. Detect communities if not provided
        2. Generate reports for each community
        3. Cache for fast retrieval

        Args:
            chunks: Text chunks
            entities: Extracted entities
            relationships: Extracted relationships
            communities: Pre-detected communities (optional)
        """
        if not entities:
            raise IndexingError("No entities found for GraphRAG Global")

        # Build entity cache
        self._entity_cache = {str(e.id): e for e in entities}

        # Detect communities if not provided
        if not communities or len(communities) == 0:
            logger.info("Detecting communities...")
            communities = await self._detect_communities()
            logger.info(f"Detected {len(communities)} communities")

        self._communities = communities

        # Generate community reports
        logger.info("Generating community reports...")
        await self._generate_community_reports(chunks, entities, relationships)

        logger.info(
            f"GraphRAG Global ready: {len(self._communities)} communities, "
            f"{len(self._community_reports)} reports"
        )

    async def _detect_communities(self) -> List[Community]:
        """Detect communities using Louvain algorithm.

        Returns:
            List of Community objects
        """
        # Use Louvain algorithm (available in NetworkX, no extra deps needed)
        # Note: Leiden is better but requires python-igraph
        node_to_community = await self.graph_store.detect_communities_louvain(
            resolution=self.leiden_resolution
        )

        # Invert mapping: community_id -> [node_ids]
        community_to_nodes: Dict[int, List[str]] = {}
        for node_id, community_id in node_to_community.items():
            if community_id not in community_to_nodes:
                community_to_nodes[community_id] = []
            community_to_nodes[community_id].append(node_id)

        # Convert to Community objects
        communities = []
        for community_int_id, entity_id_strs in community_to_nodes.items():
            # Limit community size
            if len(entity_id_strs) > self.max_community_size:
                logger.warning(
                    f"Community {community_int_id} has {len(entity_id_strs)} entities, "
                    f"truncating to {self.max_community_size}"
                )
                entity_id_strs = entity_id_strs[:self.max_community_size]

            community = Community(
                # id will be auto-generated as UUID
                level=0,  # Single level for now (flat communities)
                entity_ids=[UUID(eid) for eid in entity_id_strs],
                size=len(entity_id_strs),  # Required field
                title=f"Community {community_int_id}",  # Placeholder, updated during report gen
                summary="",  # Generated during report generation
                metadata={
                    "resolution": self.leiden_resolution,
                    "community_number": community_int_id,  # Store original int ID
                },
            )
            communities.append(community)

        return communities

    async def _generate_community_reports(
        self,
        chunks: List[Chunk],
        entities: List[Entity],
        relationships: List[Relationship],
    ) -> None:
        """Generate summary reports for each community.

        Args:
            chunks: All text chunks
            entities: All entities
            relationships: All relationships
        """
        # Build lookup maps
        chunk_map = {str(c.id): c for c in chunks}
        rel_map = {str(r.id): r for r in relationships}

        for idx, community in enumerate(self._communities):
            try:
                # Get community int ID from metadata
                comm_int_id = community.metadata.get("community_number", idx)

                # Collect community data
                comm_entities = [
                    self._entity_cache[str(eid)]
                    for eid in community.entity_ids
                    if str(eid) in self._entity_cache
                ]

                # Get relationships within community
                entity_id_set = set(community.entity_ids)
                comm_relationships = [
                    r for r in relationships
                    if r.source_entity_id in entity_id_set
                    and r.target_entity_id in entity_id_set
                ]

                # Get chunks connected to community entities
                comm_chunks = []
                for chunk in chunks:
                    if any(eid in entity_id_set for eid in chunk.entity_ids):
                        comm_chunks.append(chunk)

                # Generate report
                report = await self._generate_single_report(
                    comm_int_id,
                    comm_entities,
                    comm_relationships,
                    comm_chunks,
                )

                self._community_reports[idx] = report

                # Update community with report summary (first 500 chars)
                community.summary = report[:500]
                community.title = self._extract_title(report, comm_int_id)

                logger.debug(
                    f"Generated report for community {idx}: "
                    f"{len(comm_entities)} entities, "
                    f"{len(comm_relationships)} relationships, "
                    f"{len(comm_chunks)} chunks"
                )

            except Exception as e:
                logger.error(f"Failed to generate report for community {idx}: {e}")
                self._community_reports[idx] = f"Error generating report: {e}"

    async def _generate_single_report(
        self,
        community_id: int,
        entities: List[Entity],
        relationships: List[Relationship],
        chunks: List[Chunk],
    ) -> str:
        """Generate a summary report for a single community.

        Args:
            community_id: Community ID
            entities: Entities in community
            relationships: Relationships in community
            chunks: Chunks connected to community

        Returns:
            Summary report text
        """
        # Build context for LLM
        entity_list = "\n".join([
            f"- {e.name} ({e.type.value}): {e.description or 'No description'}"
            for e in entities[:20]  # Limit to top 20 entities
        ])

        relationship_list = "\n".join([
            f"- {self._entity_cache.get(str(r.source_entity_id), Entity(name='Unknown', type='OTHER')).name} "
            f"--[{r.type.value}]--> "
            f"{self._entity_cache.get(str(r.target_entity_id), Entity(name='Unknown', type='OTHER')).name}"
            for r in relationships[:15]  # Limit to top 15 relationships
        ])

        chunk_text = "\n\n".join([
            f"[Chunk {i+1}]: {c.text[:300]}..."
            for i, c in enumerate(chunks[:5])  # Limit to top 5 chunks
        ])

        prompt = f"""You are analyzing a community of related entities from a knowledge graph.

Community ID: {community_id}
Number of entities: {len(entities)}
Number of relationships: {len(relationships)}
Number of supporting chunks: {len(chunks)}

Entities in this community:
{entity_list}

Key relationships:
{relationship_list}

Supporting text excerpts:
{chunk_text}

Please generate a comprehensive summary report for this community that includes:

1. TITLE: A concise title (5-10 words) that captures the main theme
2. SUMMARY: A 2-3 paragraph summary of what this community represents
3. KEY THEMES: The main topics, concepts, or themes present
4. IMPORTANCE: Why this community is significant in the overall knowledge graph

Format your response as:

TITLE: [Title here]

SUMMARY:
[Summary paragraphs]

KEY THEMES:
- [Theme 1]
- [Theme 2]
- [Theme 3]

IMPORTANCE:
[Why this matters]
"""

        # Generate with LLM
        try:
            response, _, _ = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,  # Slightly creative for summaries
                max_tokens=1500,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback: simple concatenation
            return f"Community {community_id}\n\nEntities: {', '.join([e.name for e in entities[:10]])}\n\n{chunk_text[:500]}"

    def _extract_title(self, report: str, community_id: int) -> str:
        """Extract title from report.

        Args:
            report: Full report text
            community_id: Community ID (fallback)

        Returns:
            Extracted title
        """
        lines = report.split("\n")
        for line in lines:
            if line.strip().startswith("TITLE:"):
                return line.replace("TITLE:", "").strip()

        # Fallback
        return f"Community {community_id}"

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve relevant context using global community search.

        Args:
            query: User query string
            top_k: Number of communities to retrieve
            **kwargs: Additional parameters

        Returns:
            RetrievalResult with community reports

        Raises:
            RetrievalError: If retrieval fails
        """
        start_time = time.time()

        try:
            if not self._community_reports:
                logger.warning("No community reports available")
                return RetrievalResult(
                    strategy=self.name,
                    chunks=[],
                    scores=[],
                    communities=[],
                    retrieval_time_ms=(time.time() - start_time) * 1000,
                    metadata={"communities_found": 0},
                )

            # Step 1: Rank communities by relevance to query
            logger.debug(f"Ranking {len(self._communities)} communities for query...")
            ranked_communities = await self._rank_communities(query, top_k)

            # Step 2: Collect reports and create synthetic "chunks" from them
            logger.debug(f"Collecting {len(ranked_communities)} community reports...")
            chunks = []
            scores = []
            communities = []

            for idx, (community, score) in enumerate(ranked_communities):
                # Get report using index
                comm_idx = self._communities.index(community)
                report = self._community_reports.get(comm_idx, "")

                # Create a pseudo-chunk from the report
                chunk = Chunk(
                    id=community.id,  # Use community UUID
                    document_id=community.id,  # Use community UUID
                    chunk_index=0,
                    text=report,
                    start_char=0,
                    end_char=len(report),
                    token_count=len(report.split()),
                    metadata={
                        "community_id": str(community.id),
                        "community_title": community.title or "Untitled",
                        "community_size": len(community.entity_ids),
                    },
                )
                chunks.append(chunk)
                scores.append(score)
                communities.append(community)

            retrieval_time = (time.time() - start_time) * 1000

            logger.info(
                f"Retrieved {len(communities)} communities in {retrieval_time:.2f}ms"
            )

            return RetrievalResult(
                strategy=self.name,
                chunks=chunks,  # Community reports as "chunks"
                scores=scores,
                communities=communities,
                retrieval_time_ms=retrieval_time,
                total_chunks_searched=len(self._communities),
                metadata={
                    "total_communities": len(self._communities),
                    "communities_retrieved": len(communities),
                },
            )

        except Exception as e:
            logger.error(f"GraphRAG Global retrieval failed: {e}", exc_info=True)
            raise RetrievalError(f"Failed to retrieve with GraphRAG Global: {e}")

    async def _rank_communities(
        self, query: str, top_k: int
    ) -> List[tuple[Community, float]]:
        """Rank communities by relevance to query.

        Uses simple keyword matching on community reports for now.
        Could be enhanced with embedding-based similarity.

        Args:
            query: Query text
            top_k: Number of communities to return

        Returns:
            List of (community, score) tuples
        """
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        ranked = []
        for idx, community in enumerate(self._communities):
            report = self._community_reports.get(idx, "")
            report_lower = report.lower()

            # Simple scoring: count query token matches
            matches = sum(1 for token in query_tokens if token in report_lower)
            score = matches / len(query_tokens) if query_tokens else 0.0

            ranked.append((community, score))

        # Sort by score (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked[:top_k]

    async def validate_index(self) -> bool:
        """Validate that communities and reports are ready.

        Returns:
            True if valid
        """
        try:
            if not self._communities:
                logger.error("No communities detected")
                return False

            if not self._community_reports:
                logger.error("No community reports generated")
                return False

            logger.debug(
                f"GraphRAG Global validation passed: "
                f"{len(self._communities)} communities"
            )
            return True

        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False

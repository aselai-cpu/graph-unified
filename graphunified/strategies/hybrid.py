"""Hybrid RAG: Vector similarity + BM25 keyword search with RRF fusion.

Combines dense vector retrieval with sparse keyword matching for robust
retrieval that handles both semantic similarity and exact keyword matches.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from graphunified.config.models import Chunk, Community, Entity, Relationship
from graphunified.config.settings import HybridStrategyConfig
from graphunified.exceptions import IndexingError, RetrievalError
from graphunified.index.stages.index import BM25Index
from graphunified.storage.vector_store import VectorStore
from graphunified.strategies.base import QueryType, RetrievalResult, RetrievalStrategy
from graphunified.utils.embedding_factory import create_embedding_client
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class HybridRAGStrategy(RetrievalStrategy):
    """Hybrid RAG: Dense vector + sparse keyword search with RRF fusion.

    Workflow:
    1. Vector search: Embed query and find similar chunks (semantic)
    2. BM25 search: Keyword matching on chunks (lexical)
    3. RRF fusion: Combine and rerank results from both methods
    4. Return top-k fused results

    Best for:
    - Queries requiring both semantic understanding and exact terms
    - Domain-specific queries with technical terminology
    - Questions where keywords matter (names, dates, specific concepts)

    Advantages over Naive RAG:
    - Better recall (finds chunks missed by vector-only search)
    - Handles out-of-vocabulary terms
    - More robust to embedding model limitations

    RRF Formula:
    score(chunk) = Σ 1 / (k + rank_i)
    where rank_i is the rank from each retrieval method
    """

    def __init__(
        self,
        config: HybridStrategyConfig,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedding_client: Any,
    ):
        """Initialize Hybrid RAG strategy.

        Args:
            config: Hybrid strategy configuration
            vector_store: Vector store with chunk index
            bm25_index: BM25 text index for keyword search
            embedding_client: Embedding client for query encoding
        """
        super().__init__(config)
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_client = embedding_client

        # Strategy config
        self.top_k = getattr(config, 'top_k', 10)
        self.alpha = getattr(config, 'alpha', 0.5)  # Weight: 1.0=vector only, 0.0=BM25 only
        self.rrf_k = 60  # RRF constant (standard value)

        logger.info(
            f"Initialized {self.name} (top_k={self.top_k}, alpha={self.alpha}, rrf_k={self.rrf_k})"
        )

    @classmethod
    async def from_config(
        cls,
        config: HybridStrategyConfig,
        vector_store_path: Path,
        bm25_index: BM25Index,
        embedding_config: Any,
    ) -> "HybridRAGStrategy":
        """Create strategy from configuration.

        Args:
            config: Hybrid strategy configuration
            vector_store_path: Path to LanceDB directory
            bm25_index: Pre-built BM25 index
            embedding_config: Embedding configuration

        Returns:
            Initialized HybridRAGStrategy
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

        return cls(config, vector_store, bm25_index, embedding_client)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "Hybrid RAG"

    def supports_query_type(self, query_type: QueryType) -> bool:
        """Check if strategy supports a query type.

        Hybrid RAG works well for factoid and comparative queries.
        """
        return query_type in (QueryType.FACTOID, QueryType.COMPARATIVE)

    async def index(
        self,
        chunks: List[Chunk],
        entities: List[Entity],
        relationships: List[Relationship],
        communities: List[Community],
    ) -> None:
        """Build strategy-specific indexes.

        For Hybrid RAG, both vector and BM25 indexes are built by Stage 5.
        This just validates they're ready.

        Args:
            chunks: Text chunks with embeddings
            entities: Extracted entities (not used)
            relationships: Extracted relationships (not used)
            communities: Detected communities (not used)
        """
        # Validate chunks
        chunks_with_embeddings = [c for c in chunks if c.embedding and len(c.embedding) > 0]

        if not chunks_with_embeddings:
            raise IndexingError("No chunks with embeddings found for Hybrid RAG")

        # Validate BM25 index
        if self.bm25_index.num_docs == 0:
            raise IndexingError("BM25 index is empty")

        logger.info(
            f"Hybrid RAG ready: {len(chunks_with_embeddings)} chunks (vector + BM25)"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve relevant chunks using hybrid search.

        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            **kwargs: Additional parameters
                - alpha: Weight for vector vs BM25 (overrides config)

        Returns:
            RetrievalResult with fused chunks and scores

        Raises:
            RetrievalError: If retrieval fails
        """
        start_time = time.time()

        try:
            # Override alpha if provided
            alpha = kwargs.get("alpha", self.alpha)

            # Step 1: Vector similarity search
            logger.debug(f"Vector search for: {query[:100]}...")
            vector_results = await self._vector_search(query, top_k * 2)  # Get more for fusion

            # Step 2: BM25 keyword search
            logger.debug(f"BM25 search for: {query[:100]}...")
            bm25_results = self._bm25_search(query, top_k * 2)

            # Step 3: Reciprocal Rank Fusion
            logger.debug("Fusing results with RRF...")
            fused_results = self._reciprocal_rank_fusion(
                vector_results,
                bm25_results,
                alpha=alpha,
                top_k=top_k,
            )

            # Step 4: Build result
            chunks = []
            scores = []

            for chunk_id, score, chunk_data in fused_results:
                # Reconstruct Chunk object
                chunk = self._reconstruct_chunk(chunk_id, chunk_data)
                chunks.append(chunk)
                scores.append(score)

            retrieval_time = (time.time() - start_time) * 1000  # Convert to ms

            logger.info(
                f"Retrieved {len(chunks)} chunks in {retrieval_time:.2f}ms "
                f"(mean_score={sum(scores)/len(scores) if scores else 0:.3f}, "
                f"vector={len(vector_results)}, bm25={len(bm25_results)})"
            )

            return RetrievalResult(
                strategy=self.name,
                chunks=chunks,
                scores=scores,
                retrieval_time_ms=retrieval_time,
                total_chunks_searched=len(vector_results) + len(bm25_results),
                metadata={
                    "top_k": top_k,
                    "alpha": alpha,
                    "rrf_k": self.rrf_k,
                    "vector_candidates": len(vector_results),
                    "bm25_candidates": len(bm25_results),
                },
            )

        except Exception as e:
            logger.error(f"Hybrid RAG retrieval failed: {e}", exc_info=True)
            raise RetrievalError(f"Failed to retrieve with Hybrid RAG: {e}")

    async def _vector_search(
        self, query: str, top_k: int
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform vector similarity search.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of (chunk_id, score, data) tuples
        """
        # Embed query
        query_embeddings = await self.embedding_client.embed([query])
        query_vector = query_embeddings[0]

        # Search vector store
        results = await self.vector_store.search_chunks(
            query_vector=query_vector,
            top_k=top_k,
        )

        # Convert to common format: (chunk_id, score, data)
        # LanceDB returns (chunk_id, distance, {text, metadata})
        # Convert distance to similarity: score = 1 / (1 + distance)
        formatted = []
        for chunk_id, distance, data in results:
            score = 1.0 / (1.0 + distance)
            formatted.append((chunk_id, score, data))

        return formatted

    def _bm25_search(
        self, query: str, top_k: int
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Perform BM25 keyword search.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of (chunk_id, score, data) tuples
        """
        # BM25 search
        results = self.bm25_index.search(query, top_k=top_k)

        # Convert to common format: (chunk_id, score, data)
        # BM25 returns (doc_id, score)
        formatted = []
        for doc_id, score in results:
            # Get document text from BM25 index
            text = self.bm25_index.documents.get(doc_id, "")
            data = {"text": text, "metadata": {}}
            formatted.append((doc_id, score, data))

        return formatted

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float, Dict[str, Any]]],
        bm25_results: List[Tuple[str, float, Dict[str, Any]]],
        alpha: float = 0.5,
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Fuse vector and BM25 results using Reciprocal Rank Fusion.

        RRF formula: score(d) = α * (1/(k + rank_vector)) + (1-α) * (1/(k + rank_bm25))

        Args:
            vector_results: Vector search results
            bm25_results: BM25 search results
            alpha: Weight for vector (0.0 to 1.0, default 0.5 = equal weight)
            top_k: Number of results to return

        Returns:
            Fused and reranked results
        """
        # Build rank maps
        vector_ranks = {chunk_id: rank for rank, (chunk_id, _, _) in enumerate(vector_results)}
        bm25_ranks = {chunk_id: rank for rank, (chunk_id, _, _) in enumerate(bm25_results)}

        # Collect all unique chunk IDs
        all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}
        chunk_data_map: Dict[str, Dict[str, Any]] = {}

        for chunk_id in all_chunk_ids:
            # Get ranks (if not present, use large rank to penalize)
            v_rank = vector_ranks.get(chunk_id, len(vector_results) + 100)
            b_rank = bm25_ranks.get(chunk_id, len(bm25_results) + 100)

            # RRF formula with alpha weighting
            v_score = 1.0 / (self.rrf_k + v_rank)
            b_score = 1.0 / (self.rrf_k + b_rank)
            rrf_score = alpha * v_score + (1.0 - alpha) * b_score

            rrf_scores[chunk_id] = rrf_score

            # Store chunk data (prefer vector results if available)
            if chunk_id in vector_ranks:
                chunk_data_map[chunk_id] = vector_results[vector_ranks[chunk_id]][2]
            elif chunk_id in bm25_ranks:
                chunk_data_map[chunk_id] = bm25_results[bm25_ranks[chunk_id]][2]

        # Sort by RRF score (descending)
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-k with data
        results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            data = chunk_data_map[chunk_id]
            results.append((chunk_id, score, data))

        return results

    def _reconstruct_chunk(
        self, chunk_id: str, chunk_data: Dict[str, Any]
    ) -> Chunk:
        """Reconstruct Chunk object from search result.

        Args:
            chunk_id: Chunk ID
            chunk_data: Chunk data dict with text and metadata

        Returns:
            Chunk object
        """
        meta = chunk_data.get("metadata", {})
        text = chunk_data.get("text", "")

        # Extract fields with safe defaults
        document_id = meta.get("document_id", chunk_id)
        chunk_index = meta.get("chunk_index", 0)
        start_char = meta.get("start_char", 0)
        end_char = meta.get("end_char", len(text))
        token_count = meta.get("token_count", 0)

        # Ensure end_char > start_char
        if end_char <= start_char:
            end_char = start_char + len(text)

        return Chunk(
            id=chunk_id,
            text=text,
            document_id=document_id,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            token_count=token_count,
            embedding=None,
            metadata=meta,
        )

    async def validate_index(self) -> bool:
        """Validate that both vector and BM25 indexes are ready.

        Returns:
            True if both indexes are valid
        """
        try:
            # Check vector store
            db = await self.vector_store._get_db()
            table_names = db.table_names()

            if self.vector_store.chunk_index_name not in table_names:
                logger.error(f"Chunks vector index not found")
                return False

            # Check BM25 index
            if self.bm25_index.num_docs == 0:
                logger.error("BM25 index is empty")
                return False

            logger.debug(f"Hybrid RAG validation passed: {self.bm25_index.num_docs} docs")
            return True

        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False

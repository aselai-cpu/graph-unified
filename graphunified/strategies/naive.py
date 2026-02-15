"""Naive RAG: Simple vector similarity search on chunks.

The baseline retrieval strategy that uses pure dense vector similarity
to find relevant chunks. Fast and simple, works well for factoid queries.
"""

import time
from pathlib import Path
from typing import Any, List

from graphunified.config.models import Chunk, Community, Entity, Relationship
from graphunified.config.settings import NaiveStrategyConfig
from graphunified.exceptions import IndexingError, RetrievalError
from graphunified.storage.vector_store import VectorStore
from graphunified.strategies.base import QueryType, RetrievalResult, RetrievalStrategy
from graphunified.utils.embedding_factory import create_embedding_client
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class NaiveRAGStrategy(RetrievalStrategy):
    """Naive RAG: Pure vector similarity search on text chunks.

    Workflow:
    1. Embed the user query
    2. Search chunks vector index for most similar embeddings (L2 distance)
    3. Return top-k chunks

    Best for:
    - Factoid questions
    - Simple Q&A
    - When semantic similarity is sufficient

    Limitations:
    - No keyword matching (missing exact terms)
    - No graph traversal (missing connections)
    - No aggregation (can't answer broad questions)
    """

    def __init__(
        self,
        config: NaiveStrategyConfig,
        vector_store: VectorStore,
        embedding_client: Any,
    ):
        """Initialize Naive RAG strategy.

        Args:
            config: Naive strategy configuration
            vector_store: Vector store with chunk index
            embedding_client: Embedding client for query encoding
        """
        super().__init__(config)
        self.vector_store = vector_store
        self.embedding_client = embedding_client

        # Strategy config
        self.top_k = config.top_k if hasattr(config, 'top_k') else 10

        logger.info(f"Initialized {self.name} (top_k={self.top_k})")

    @classmethod
    async def from_config(
        cls,
        config: NaiveStrategyConfig,
        vector_store_path: Path,
        embedding_config: Any,
    ) -> "NaiveRAGStrategy":
        """Create strategy from configuration.

        Args:
            config: Naive strategy configuration
            vector_store_path: Path to LanceDB directory
            embedding_config: Embedding configuration

        Returns:
            Initialized NaiveRAGStrategy
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

        return cls(config, vector_store, embedding_client)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "Naive RAG"

    def supports_query_type(self, query_type: QueryType) -> bool:
        """Check if strategy supports a query type.

        Naive RAG works best for factoid queries.
        """
        return query_type == QueryType.FACTOID

    async def index(
        self,
        chunks: List[Chunk],
        entities: List[Entity],
        relationships: List[Relationship],
        communities: List[Community],
    ) -> None:
        """Build strategy-specific indexes.

        For Naive RAG, indexing is already done by Stage 5 (IndexPipeline).
        The chunks vector index is ready to use.

        Args:
            chunks: Text chunks with embeddings
            entities: Extracted entities (not used)
            relationships: Extracted relationships (not used)
            communities: Detected communities (not used)
        """
        # Validate that chunks have embeddings
        chunks_with_embeddings = [c for c in chunks if c.embedding and len(c.embedding) > 0]

        if not chunks_with_embeddings:
            raise IndexingError("No chunks with embeddings found for Naive RAG")

        logger.info(f"Naive RAG ready: {len(chunks_with_embeddings)} chunks indexed")

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve relevant chunks using vector similarity.

        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            **kwargs: Additional parameters (ignored)

        Returns:
            RetrievalResult with retrieved chunks and scores

        Raises:
            RetrievalError: If retrieval fails
        """
        start_time = time.time()

        try:
            # Step 1: Embed query
            logger.debug(f"Embedding query: {query[:100]}...")
            query_embeddings = await self.embedding_client.embed([query])
            query_vector = query_embeddings[0]

            # Step 2: Vector similarity search
            logger.debug(f"Searching chunks index (top_k={top_k})")
            results = await self.vector_store.search_chunks(
                query_vector=query_vector,
                top_k=top_k,
            )

            # Step 3: Build result
            chunks = []
            scores = []

            for chunk_id, distance, metadata_dict in results:
                # Reconstruct Chunk object from search result
                # search_chunks returns: (chunk_id, distance, {text, metadata})
                meta = metadata_dict.get("metadata", {})

                # Extract required fields with safe defaults
                # Note: LanceDB doesn't store all fields, use what's available
                text = metadata_dict.get("text", "")
                document_id = meta.get("document_id", chunk_id)  # Fallback to chunk_id
                chunk_index = meta.get("chunk_index", 0)
                start_char = meta.get("start_char", 0)
                end_char = meta.get("end_char", len(text))  # Fallback to text length
                token_count = meta.get("token_count", 0)

                # Ensure end_char > start_char for validation
                if end_char <= start_char:
                    end_char = start_char + len(text)

                chunk = Chunk(
                    id=chunk_id,
                    text=text,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=token_count,
                    embedding=None,  # Don't return embeddings to save memory
                    metadata=meta,
                )
                chunks.append(chunk)

                # LanceDB returns L2 distance, convert to similarity score
                # Lower distance = higher similarity
                # Score = 1 / (1 + distance)
                score = 1.0 / (1.0 + distance)
                scores.append(score)

            retrieval_time = (time.time() - start_time) * 1000  # Convert to ms

            logger.info(
                f"Retrieved {len(chunks)} chunks in {retrieval_time:.2f}ms "
                f"(mean_score={sum(scores)/len(scores) if scores else 0:.3f})"
            )

            return RetrievalResult(
                strategy=self.name,
                chunks=chunks,
                scores=scores,
                retrieval_time_ms=retrieval_time,
                total_chunks_searched=0,  # LanceDB doesn't report this
                metadata={
                    "top_k": top_k,
                    "query_length": len(query),
                },
            )

        except Exception as e:
            logger.error(f"Naive RAG retrieval failed: {e}", exc_info=True)
            raise RetrievalError(f"Failed to retrieve with Naive RAG: {e}")

    async def validate_index(self) -> bool:
        """Validate that chunks index is ready.

        Returns:
            True if index is valid
        """
        try:
            # Try to connect to vector store and check if chunks table exists
            db = await self.vector_store._get_db()
            table_names = db.table_names()

            if self.vector_store.chunk_index_name not in table_names:
                logger.error(f"Chunks index '{self.vector_store.chunk_index_name}' not found")
                return False

            logger.debug("Naive RAG index validation passed")
            return True

        except Exception as e:
            logger.error(f"Index validation failed: {e}")
            return False

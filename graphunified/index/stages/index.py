"""Index building stage for vector and text indexes."""

import asyncio
import time
from pathlib import Path
from typing import List, Optional

from graphunified.config.models import Chunk, Entity, Relationship
from graphunified.index.stages.base import PipelineStage, ProgressCallback, StageResult, StageStatus
from graphunified.storage.vector_store import VectorStore
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class BM25Index:
    """Simple BM25 text index for keyword search.

    Uses in-memory inverted index for fast keyword-based retrieval.
    Good for small to medium datasets (<100K chunks).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b

        # Inverted index: term -> [(doc_id, freq), ...]
        self.inverted_index: dict[str, list[tuple[str, int]]] = {}

        # Document lengths
        self.doc_lengths: dict[str, int] = {}

        # Document metadata
        self.documents: dict[str, str] = {}  # doc_id -> text

        # Statistics
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index.

        Args:
            doc_id: Document ID
            text: Document text
        """
        # Simple tokenization (lowercase, split on whitespace/punctuation)
        tokens = self._tokenize(text)

        # Count term frequencies
        term_freqs: dict[str, int] = {}
        for token in tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1

        # Add to inverted index
        for term, freq in term_freqs.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = []
            self.inverted_index[term].append((doc_id, freq))

        # Store document info
        self.documents[doc_id] = text
        self.doc_lengths[doc_id] = len(tokens)
        self.num_docs += 1

    def finalize(self) -> None:
        """Finalize index (compute statistics)."""
        if self.num_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.num_docs

        logger.info(f"BM25 index built: {self.num_docs} docs, {len(self.inverted_index)} terms")

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search the index.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        # Tokenize query
        query_tokens = self._tokenize(query)

        # Compute BM25 scores
        scores: dict[str, float] = {}

        for token in set(query_tokens):  # Unique query terms
            if token not in self.inverted_index:
                continue

            # Document frequency
            df = len(self.inverted_index[token])

            # IDF (inverse document frequency)
            idf = self._compute_idf(df)

            # Score each document containing this term
            for doc_id, term_freq in self.inverted_index[token]:
                doc_length = self.doc_lengths[doc_id]

                # BM25 formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )

                score = idf * (numerator / denominator)
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        # Sort by score and return top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        import re

        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)

        # Remove very short tokens (< 2 chars)
        tokens = [t for t in tokens if len(t) >= 2]

        return tokens

    def _compute_idf(self, df: int) -> float:
        """Compute IDF score.

        Args:
            df: Document frequency

        Returns:
            IDF score
        """
        import math

        # Standard IDF formula
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)


class IndexStage(PipelineStage):
    """Stage 5: Build searchable indexes from embeddings.

    Builds:
    - LanceDB vector indexes (chunks, entities, relationships)
    - BM25 text index (chunks)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        build_text_index: bool = True,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """Initialize index building stage.

        Args:
            vector_store: Vector store for LanceDB indexes
            build_text_index: Whether to build BM25 text index
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            progress_callback: Optional progress callback
        """
        super().__init__("index", progress_callback)

        self.vector_store = vector_store
        self.build_text_index = build_text_index
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        # BM25 index (created during execution)
        self.bm25_index: Optional[BM25Index] = None

    async def execute(self, data: dict) -> StageResult:
        """Build indexes from extracted data.

        Args:
            data: Dict with 'chunks', 'entities', 'relationships' keys

        Returns:
            StageResult with index statistics
        """
        start_time = time.time()

        try:
            chunks: List[Chunk] = data.get("chunks", [])
            entities: List[Entity] = data.get("entities", [])
            relationships: List[Relationship] = data.get("relationships", [])

            logger.info(f"Building indexes for {len(chunks)} chunks, {len(entities)} entities, {len(relationships)} relationships")

            # Build vector indexes
            await self._build_vector_indexes(chunks, entities, relationships)

            # Build text index
            if self.build_text_index and chunks:
                await self._build_text_index(chunks)

            duration = time.time() - start_time

            metadata = {
                "chunks_indexed": len(chunks),
                "entities_indexed": len(entities),
                "relationships_indexed": len(relationships),
                "text_index_built": self.build_text_index,
                "duration": duration,
            }

            logger.info(f"Index building completed in {duration:.2f}s")

            return StageResult(
                status=StageStatus.COMPLETED,
                data=data,  # Pass through
                metadata=metadata,
                duration=duration,
            )

        except Exception as e:
            logger.error(f"Index building failed: {e}", exc_info=True)
            duration = time.time() - start_time
            return StageResult(
                status=StageStatus.FAILED,
                data=data,
                metadata={"error": str(e)},
                duration=duration,
            )

    async def _build_vector_indexes(
        self,
        chunks: List[Chunk],
        entities: List[Entity],
        relationships: List[Relationship],
    ) -> None:
        """Build LanceDB vector indexes.

        Args:
            chunks: List of chunks with embeddings
            entities: List of entities with embeddings
            relationships: List of relationships with embeddings
        """
        # Index chunks
        if chunks:
            chunks_with_embeddings = [c for c in chunks if c.embedding and len(c.embedding) > 0]

            if chunks_with_embeddings:
                await self.vector_store.index_chunks(
                    chunk_ids=[c.id for c in chunks_with_embeddings],
                    embeddings=[c.embedding for c in chunks_with_embeddings],
                    texts=[c.text for c in chunks_with_embeddings],
                    metadata=[c.metadata or {} for c in chunks_with_embeddings],
                )
                logger.info(f"Indexed {len(chunks_with_embeddings)} chunks in LanceDB")

        # Index entities
        if entities:
            entities_with_embeddings = [e for e in entities if e.embedding and len(e.embedding) > 0]

            if entities_with_embeddings:
                await self.vector_store.index_entities(
                    entity_ids=[e.id for e in entities_with_embeddings],
                    embeddings=[e.embedding for e in entities_with_embeddings],
                    names=[e.name for e in entities_with_embeddings],
                    types=[e.type.value for e in entities_with_embeddings],
                    descriptions=[e.description or "" for e in entities_with_embeddings],
                )
                logger.info(f"Indexed {len(entities_with_embeddings)} entities in LanceDB")

        # Index relationships
        if relationships:
            relationships_with_embeddings = [
                r for r in relationships if r.embedding and len(r.embedding) > 0
            ]

            if relationships_with_embeddings:
                await self.vector_store.index_relationships(
                    relationship_ids=[r.id for r in relationships_with_embeddings],
                    embeddings=[r.embedding for r in relationships_with_embeddings],
                    descriptions=[r.description or "" for r in relationships_with_embeddings],
                    types=[r.type.value for r in relationships_with_embeddings],
                )
                logger.info(f"Indexed {len(relationships_with_embeddings)} relationships in LanceDB")

    async def _build_text_index(self, chunks: List[Chunk]) -> None:
        """Build BM25 text index.

        Args:
            chunks: List of chunks
        """
        # Create BM25 index
        self.bm25_index = BM25Index(k1=self.bm25_k1, b=self.bm25_b)

        # Add all chunks
        for chunk in chunks:
            self.bm25_index.add_document(chunk.id, chunk.text)

        # Finalize
        self.bm25_index.finalize()

        logger.info(f"Built BM25 text index for {len(chunks)} chunks")

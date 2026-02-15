"""Embedding generation stage for chunks and entities."""

import time
from typing import Dict, List, Optional

from graphunified.config.models import Chunk, Entity
from graphunified.index.stages.base import PipelineStage, ProgressCallback, StageResult, StageStatus
from graphunified.utils.embedding import EmbeddingClient
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class EmbedStage(PipelineStage):
    """Generate embeddings for chunks and entities."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        embed_chunks: bool = True,
        embed_entities: bool = True,
        embed_relationships: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """Initialize embedding stage.

        Args:
            embedding_client: Embedding client for API calls
            embed_chunks: Whether to embed chunks
            embed_entities: Whether to embed entities
            embed_relationships: Whether to embed relationships
            progress_callback: Optional progress callback
        """
        super().__init__("embed", progress_callback)
        self.embedding_client = embedding_client
        self.embed_chunks = embed_chunks
        self.embed_entities = embed_entities
        self.embed_relationships = embed_relationships

    async def execute(self, input_data: Dict) -> StageResult:
        """Generate embeddings for chunks and entities.

        Args:
            input_data: Dict with 'chunks' and 'entities' keys

        Returns:
            StageResult containing updated chunks and entities with embeddings
        """
        start_time = time.time()

        # Handle both dict input (from pipeline) and list input (for testing)
        if isinstance(input_data, dict):
            chunks = input_data.get("chunks", [])
            entities = input_data.get("entities", [])
            relationships = input_data.get("relationships", [])
            entity_map = input_data.get("entity_map", {})
        else:
            # Backwards compatibility for testing
            chunks = input_data
            entities = []
            relationships = []
            entity_map = {}

        try:
            embedded_chunks = chunks
            embedded_entities = entities
            embedded_relationships = relationships

            # Embed chunks
            if self.embed_chunks and chunks:
                logger.info(f"Generating embeddings for {len(chunks)} chunks...")
                embedded_chunks = await self._embed_chunks(chunks)
                logger.info(f"Embedded {len(embedded_chunks)} chunks")

            # Embed entities
            if self.embed_entities and entities:
                logger.info(f"Generating embeddings for {len(entities)} entities...")
                embedded_entities = await self._embed_entities(entities)
                logger.info(f"Embedded {len(embedded_entities)} entities")

            # Embed relationships
            if self.embed_relationships and relationships:
                logger.info(f"Generating embeddings for {len(relationships)} relationships...")
                embedded_relationships = await self._embed_relationships(relationships, embedded_entities)
                logger.info(f"Embedded {len(embedded_relationships)} relationships")

            return StageResult(
                status=StageStatus.COMPLETED,
                data={
                    "chunks": embedded_chunks,
                    "entities": embedded_entities,
                    "relationships": embedded_relationships,
                    "entity_map": entity_map,
                },
                metadata={
                    "chunk_count": len(embedded_chunks),
                    "entity_count": len(embedded_entities),
                    "relationship_count": len(embedded_relationships),
                    "chunks_embedded": sum(1 for c in embedded_chunks if c.embedding is not None),
                    "entities_embedded": sum(1 for e in embedded_entities if e.embedding is not None),
                    "relationships_embedded": sum(1 for r in embedded_relationships if r.embedding is not None),
                },
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Embed stage failed: {e}")
            return StageResult(
                status=StageStatus.FAILED,
                data=None,
                metadata={"error": str(e)},
                duration=time.time() - start_time,
            )

    async def _embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate embeddings for chunks.

        Args:
            chunks: List of chunks

        Returns:
            List of chunks with embeddings attached
        """
        if not chunks:
            return []

        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batches
        embeddings = await self.embedding_client.embed(texts)

        # Attach embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            # Create new chunk with embedding
            embedded_chunk = Chunk(
                id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                token_count=chunk.token_count,
                embedding=embedding,
                embedding_model=self.embedding_client.model,
                entity_ids=chunk.entity_ids,
                relationship_ids=chunk.relationship_ids,
                metadata=chunk.metadata,
            )
            embedded_chunks.append(embedded_chunk)

        # Report progress (50% after chunk embedding)
        self._report_progress(0.5)

        return embedded_chunks

    async def _embed_entities(self, entities: List[Entity]) -> List[Entity]:
        """Generate embeddings for entities.

        Args:
            entities: List of entities

        Returns:
            List of entities with embeddings attached
        """
        if not entities:
            return []

        # Extract entity texts (name + description)
        texts = []
        for entity in entities:
            text = entity.name
            if entity.description:
                text += f": {entity.description}"
            texts.append(text)

        # Generate embeddings in batches
        embeddings = await self.embedding_client.embed(texts)

        # Attach embeddings to entities
        embedded_entities = []
        for entity, embedding in zip(entities, embeddings):
            # Create new entity with embedding
            embedded_entity = Entity(
                id=entity.id,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                source_chunks=entity.source_chunks,
                extraction_confidence=entity.extraction_confidence,
                embedding=embedding,
                embedding_model=self.embedding_client.model,
                aliases=entity.aliases,
                metadata=entity.metadata,
            )
            embedded_entities.append(embedded_entity)

        # Report progress (75% after entity embedding)
        self._report_progress(0.75)

        return embedded_entities

    async def _embed_relationships(
        self, relationships: List[Relationship], entities: List[Entity]
    ) -> List[Relationship]:
        """Generate embeddings for relationships.

        Args:
            relationships: List of relationships
            entities: List of entities (for name lookup)

        Returns:
            List of relationships with embeddings attached
        """
        if not relationships:
            return []

        # Build entity lookup map
        entity_map = {e.id: e for e in entities}

        # Extract relationship texts (source + type + target + description)
        texts = []
        for rel in relationships:
            source = entity_map.get(rel.source_entity_id)
            target = entity_map.get(rel.target_entity_id)

            if source and target:
                # Format: "Source RELATIONSHIP_TYPE Target: description"
                text = f"{source.name} {rel.type.value} {target.name}"
                if rel.description:
                    text += f": {rel.description}"
                texts.append(text)
            else:
                # Fallback if entity lookup fails
                text = rel.description or f"{rel.type.value}"
                texts.append(text)
                logger.warning(
                    f"Relationship {rel.id} missing entity reference "
                    f"(source={rel.source_entity_id}, target={rel.target_entity_id})"
                )

        # Generate embeddings in batches
        embeddings = await self.embedding_client.embed(texts)

        # Attach embeddings to relationships
        embedded_relationships = []
        for rel, embedding in zip(relationships, embeddings):
            # Create new relationship with embedding
            embedded_rel = Relationship(
                id=rel.id,
                source_entity_id=rel.source_entity_id,
                target_entity_id=rel.target_entity_id,
                type=rel.type,
                description=rel.description,
                source_chunks=rel.source_chunks,
                extraction_confidence=rel.extraction_confidence,
                weight=rel.weight,
                embedding=embedding,
                embedding_model=self.embedding_client.model,
                metadata=rel.metadata,
            )
            embedded_relationships.append(embedded_rel)

        # Report progress (100% after relationship embedding)
        self._report_progress(1.0)

        return embedded_relationships

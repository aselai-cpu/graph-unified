"""Entity and relationship extraction stage using Claude."""

import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from fuzzywuzzy import fuzz

from graphunified.config.models import Chunk, Entity, EntityType, Relationship, RelationshipType
from graphunified.exceptions import APIError
from graphunified.index.stages.base import PipelineStage, ProgressCallback, StageResult, StageStatus
from graphunified.prompts.extraction import ENTITY_EXTRACTION_PROMPT, RELATIONSHIP_EXTRACTION_PROMPT
from graphunified.utils.llm import ClaudeClient
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class ExtractStage(PipelineStage):
    """Extract entities and relationships from chunks using Claude."""

    def __init__(
        self,
        llm_client: ClaudeClient,
        batch_size: int = 10,
        dedup_threshold: int = 90,
        max_retries: int = 1,
        max_concurrent: int = 10,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """Initialize extraction stage.

        Args:
            llm_client: Claude client for API calls
            batch_size: Number of chunks to process per LLM call
            dedup_threshold: Fuzzy matching threshold for entity deduplication (0-100)
            max_retries: Maximum retries for failed extractions
            max_concurrent: Maximum concurrent LLM calls
            progress_callback: Optional progress callback
        """
        super().__init__("extract", progress_callback)
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.dedup_threshold = dedup_threshold
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent

    async def execute(self, input_data: List[Chunk]) -> StageResult:
        """Extract entities and relationships from chunks.

        Args:
            input_data: List of chunks from chunk stage

        Returns:
            StageResult containing dict with entities, relationships, and entity_map
        """
        start_time = time.time()

        if not input_data:
            logger.warning("No chunks to process")
            return StageResult(
                status=StageStatus.COMPLETED,
                data={"entities": [], "relationships": [], "entity_map": {}},
                metadata={"chunk_count": 0, "entity_count": 0, "relationship_count": 0},
                duration=time.time() - start_time,
            )

        try:
            # Step 1: Extract entities from all chunks
            logger.info(f"Extracting entities from {len(input_data)} chunks...")
            raw_entities = await self._extract_entities(input_data)
            logger.info(f"Extracted {len(raw_entities)} raw entities")

            # Step 2: Deduplicate entities
            logger.info("Deduplicating entities...")
            entities, entity_map = self._deduplicate_entities(raw_entities)
            logger.info(f"Deduplicated to {len(entities)} unique entities ({len(raw_entities) - len(entities)} removed)")

            # Step 3: Extract relationships
            logger.info("Extracting relationships...")
            relationships = await self._extract_relationships(input_data, entities)
            logger.info(f"Extracted {len(relationships)} relationships")

            # Step 4: Resolve relationship entity references using dedup map
            relationships = self._resolve_relationships(relationships, entity_map)

            # Step 5: Populate bidirectional chunk-entity/relationship links
            logger.info("Populating bidirectional chunk-entity links...")
            chunks_with_links = self._populate_chunk_links(input_data, entities, relationships)
            logger.info(f"Updated {len(chunks_with_links)} chunks with entity/relationship links")

            return StageResult(
                status=StageStatus.COMPLETED,
                data={
                    "entities": entities,
                    "relationships": relationships,
                    "entity_map": entity_map,
                    "chunks": chunks_with_links,
                },
                metadata={
                    "chunk_count": len(input_data),
                    "entity_count": len(entities),
                    "relationship_count": len(relationships),
                    "deduplication_ratio": (len(raw_entities) - len(entities)) / len(raw_entities)
                    if raw_entities
                    else 0,
                },
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Extract stage failed: {e}")
            return StageResult(
                status=StageStatus.FAILED,
                data=None,
                metadata={"error": str(e)},
                duration=time.time() - start_time,
            )

    async def _extract_entities(self, chunks: List[Chunk]) -> List[Entity]:
        """Extract entities from chunks using Claude with concurrent batches.

        Args:
            chunks: List of chunks

        Returns:
            List of extracted entities
        """
        import asyncio

        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def extract_batch(batch_idx: int, batch: List[Chunk]) -> List[Entity]:
            """Extract entities from a single batch."""
            async with semaphore:
                batch_num = batch_idx + 1

                # Format chunk texts
                chunk_texts = self._format_chunks(batch)

                # Create prompt
                prompt = ENTITY_EXTRACTION_PROMPT.format(chunk_texts=chunk_texts)

                # Call Claude
                try:
                    response_text, _, _ = await self.llm_client.generate(
                        prompt=prompt, temperature=0.0, max_tokens=4096
                    )

                    # Parse JSON response
                    entities = self._parse_entity_response(response_text, batch)

                    logger.debug(f"Batch {batch_num}/{total_batches}: extracted {len(entities)} entities")

                    # Report progress
                    progress = batch_num / total_batches
                    self._report_progress(progress * 0.4)  # 0-40% for entity extraction

                    return entities

                except Exception as e:
                    logger.error(f"Entity extraction failed for batch {batch_num}: {e}")
                    return []

        # Create tasks for all batches
        tasks = []
        for batch_idx in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_idx : batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size
            tasks.append(extract_batch(batch_num, batch))

        # Execute all batches concurrently
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        all_entities = []
        for entities in batch_results:
            all_entities.extend(entities)

        return all_entities

    async def _extract_relationships(self, chunks: List[Chunk], entities: List[Entity]) -> List[Relationship]:
        """Extract relationships from chunks using Claude with concurrent batches.

        Args:
            chunks: List of chunks
            entities: List of extracted entities

        Returns:
            List of extracted relationships
        """
        import asyncio

        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        # Create entity name list for the prompt
        entity_names = "\n".join([f"- {entity.name} ({entity.type.value})" for entity in entities])

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def extract_batch(batch_idx: int, batch: List[Chunk]) -> List[Relationship]:
            """Extract relationships from a single batch."""
            async with semaphore:
                batch_num = batch_idx + 1

                # Format chunk texts
                chunk_texts = self._format_chunks(batch)

                # Create prompt
                prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
                    entity_names=entity_names, chunk_texts=chunk_texts
                )

                # Call Claude
                try:
                    response_text, _, _ = await self.llm_client.generate(
                        prompt=prompt, temperature=0.0, max_tokens=4096
                    )

                    # Parse JSON response
                    relationships = self._parse_relationship_response(response_text, batch, entities)

                    logger.debug(f"Batch {batch_num}/{total_batches}: extracted {len(relationships)} relationships")

                    # Report progress
                    progress = batch_num / total_batches
                    self._report_progress(0.4 + progress * 0.4)  # 40-80% for relationship extraction

                    return relationships

                except Exception as e:
                    logger.error(f"Relationship extraction failed for batch {batch_num}: {e}")
                    return []

        # Create tasks for all batches
        tasks = []
        for batch_idx in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_idx : batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size
            tasks.append(extract_batch(batch_num, batch))

        # Execute all batches concurrently
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        all_relationships = []
        for relationships in batch_results:
            all_relationships.extend(relationships)

        return all_relationships

    def _format_chunks(self, chunks: List[Chunk]) -> str:
        """Format chunks for prompt.

        Args:
            chunks: List of chunks

        Returns:
            Formatted string
        """
        formatted = []
        for i, chunk in enumerate(chunks):
            formatted.append(f"---CHUNK {i + 1}---\n{chunk.text}")
        return "\n\n".join(formatted)

    def _parse_entity_response(self, response_text: str, chunks: List[Chunk]) -> List[Entity]:
        """Parse entity extraction response.

        Args:
            response_text: JSON response from Claude
            chunks: Source chunks

        Returns:
            List of entities
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = self._extract_json(response_text)
            data = json.loads(json_text)

            entities = []
            for entity_data in data.get("entities", []):
                try:
                    entity = Entity(
                        id=uuid4(),
                        name=entity_data["name"],
                        type=EntityType(entity_data["type"]),
                        description=entity_data.get("description"),
                        source_chunks=[chunk.id for chunk in chunks],
                        extraction_confidence=entity_data.get("confidence", 1.0),
                    )
                    entities.append(entity)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid entity: {e}")
                    continue

            return entities

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entity JSON: {e}")
            return []

    def _parse_relationship_response(
        self, response_text: str, chunks: List[Chunk], entities: List[Entity]
    ) -> List[Relationship]:
        """Parse relationship extraction response.

        Args:
            response_text: JSON response from Claude
            chunks: Source chunks
            entities: List of entities for name matching

        Returns:
            List of relationships
        """
        try:
            # Extract JSON from response
            json_text = self._extract_json(response_text)
            data = json.loads(json_text)

            # Build entity name to ID mapping
            entity_name_to_id = {entity.name: entity.id for entity in entities}

            relationships = []
            for rel_data in data.get("relationships", []):
                try:
                    source_name = rel_data["source"]
                    target_name = rel_data["target"]

                    # Look up entity IDs
                    source_id = entity_name_to_id.get(source_name)
                    target_id = entity_name_to_id.get(target_name)

                    if not source_id or not target_id:
                        logger.debug(
                            f"Skipping relationship with unknown entities: {source_name} -> {target_name}"
                        )
                        continue

                    relationship = Relationship(
                        id=uuid4(),
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        type=RelationshipType(rel_data["type"]),
                        description=rel_data.get("description"),
                        source_chunks=[chunk.id for chunk in chunks],
                        extraction_confidence=rel_data.get("confidence", 1.0),
                    )
                    relationships.append(relationship)

                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid relationship: {e}")
                    continue

            return relationships

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse relationship JSON: {e}")
            return []

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text (handle markdown code blocks).

        Args:
            text: Response text

        Returns:
            JSON string
        """
        text = text.strip()

        # Remove markdown code block markers
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        return text

    def _deduplicate_entities(self, entities: List[Entity]) -> Tuple[List[Entity], Dict[UUID, UUID]]:
        """Deduplicate entities using fuzzy matching.

        Args:
            entities: List of entities

        Returns:
            Tuple of (deduplicated entities, mapping from old ID to new ID)
        """
        if not entities:
            return [], {}

        # Group entities by type
        entities_by_type: Dict[EntityType, List[Entity]] = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.type].append(entity)

        deduplicated = []
        entity_map = {}  # old_id -> new_id

        # Process each type separately
        for entity_type, type_entities in entities_by_type.items():
            clusters = self._cluster_entities(type_entities)

            for cluster in clusters:
                # Merge cluster into single entity
                merged = self._merge_cluster(cluster)
                deduplicated.append(merged)

                # Map all old IDs to new merged ID
                for entity in cluster:
                    entity_map[entity.id] = merged.id

        self._report_progress(0.9)  # 90% after deduplication

        return deduplicated, entity_map

    def _cluster_entities(self, entities: List[Entity]) -> List[List[Entity]]:
        """Cluster entities by name similarity.

        Args:
            entities: List of entities of the same type

        Returns:
            List of entity clusters
        """
        if not entities:
            return []

        # Start with each entity in its own cluster
        clusters = [[entity] for entity in entities]
        merged = True

        # Iteratively merge similar clusters
        while merged:
            merged = False
            new_clusters = []
            used = set()

            for i, cluster_i in enumerate(clusters):
                if i in used:
                    continue

                # Try to merge with other clusters
                for j in range(i + 1, len(clusters)):
                    if j in used:
                        continue

                    cluster_j = clusters[j]

                    # Check if any pair of entities is similar
                    should_merge = False
                    for entity_i in cluster_i:
                        for entity_j in cluster_j:
                            similarity = fuzz.ratio(entity_i.name.lower(), entity_j.name.lower())
                            if similarity >= self.dedup_threshold:
                                should_merge = True
                                break
                        if should_merge:
                            break

                    if should_merge:
                        # Merge clusters
                        cluster_i.extend(cluster_j)
                        used.add(j)
                        merged = True

                new_clusters.append(cluster_i)
                used.add(i)

            clusters = new_clusters

        return clusters

    def _merge_cluster(self, cluster: List[Entity]) -> Entity:
        """Merge a cluster of entities into a single entity.

        Args:
            cluster: List of entities to merge

        Returns:
            Merged entity
        """
        # Choose most common name (or first if tie)
        names = [entity.name for entity in cluster]
        name = max(set(names), key=names.count)

        # Collect all unique names as aliases
        aliases = list(set(names) - {name})

        # Use highest confidence
        confidence = max(entity.extraction_confidence for entity in cluster)

        # Merge descriptions (take longest)
        descriptions = [e.description for e in cluster if e.description]
        description = max(descriptions, key=len) if descriptions else None

        # Merge source chunks
        source_chunks = []
        for entity in cluster:
            source_chunks.extend(entity.source_chunks)
        source_chunks = list(set(source_chunks))

        return Entity(
            id=uuid4(),
            name=name,
            type=cluster[0].type,
            description=description,
            source_chunks=source_chunks,
            extraction_confidence=confidence,
            aliases=aliases,
        )

    def _resolve_relationships(
        self, relationships: List[Relationship], entity_map: Dict[UUID, UUID]
    ) -> List[Relationship]:
        """Resolve relationship entity IDs using deduplication map.

        Args:
            relationships: List of relationships
            entity_map: Mapping from old entity IDs to new deduplicated IDs

        Returns:
            List of relationships with resolved IDs
        """
        resolved = []

        for rel in relationships:
            # Update source and target IDs
            source_id = entity_map.get(rel.source_entity_id, rel.source_entity_id)
            target_id = entity_map.get(rel.target_entity_id, rel.target_entity_id)

            # Skip self-loops after deduplication
            if source_id == target_id:
                logger.debug("Skipping self-loop after deduplication")
                continue

            resolved_rel = Relationship(
                id=rel.id,
                source_entity_id=source_id,
                target_entity_id=target_id,
                type=rel.type,
                description=rel.description,
                source_chunks=rel.source_chunks,
                extraction_confidence=rel.extraction_confidence,
            )
            resolved.append(resolved_rel)

        self._report_progress(1.0)  # 100% complete

        return resolved

    def _populate_chunk_links(
        self, chunks: List[Chunk], entities: List[Entity], relationships: List[Relationship]
    ) -> List[Chunk]:
        """Populate bidirectional links from chunks to entities and relationships.

        Args:
            chunks: List of chunks
            entities: List of extracted entities (with source_chunks)
            relationships: List of extracted relationships (with source_chunks)

        Returns:
            List of chunks with entity_ids and relationship_ids populated
        """
        from collections import defaultdict

        # Build reverse mappings: chunk_id -> list of entity/relationship IDs
        chunk_to_entities = defaultdict(list)
        chunk_to_relationships = defaultdict(list)

        # Map entities to their source chunks
        for entity in entities:
            for chunk_id in entity.source_chunks:
                chunk_to_entities[chunk_id].append(entity.id)

        # Map relationships to their source chunks
        for relationship in relationships:
            for chunk_id in relationship.source_chunks:
                chunk_to_relationships[chunk_id].append(relationship.id)

        # Update chunks with links
        updated_chunks = []
        for chunk in chunks:
            # Create new chunk with populated links
            updated_chunk = Chunk(
                id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                token_count=chunk.token_count,
                embedding=chunk.embedding,
                embedding_model=chunk.embedding_model,
                entity_ids=chunk_to_entities[chunk.id],
                relationship_ids=chunk_to_relationships[chunk.id],
                metadata=chunk.metadata,
            )
            updated_chunks.append(updated_chunk)

        return updated_chunks

"""Parquet storage backend with async batch operations."""

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Dict, List, Any
from uuid import UUID

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from graphunified.config.models import (
    Chunk,
    Community,
    CommunityReport,
    Document,
    Entity,
    Relationship,
)
from graphunified.exceptions import StorageError
from graphunified.storage.base import StorageBackend
from graphunified.storage.schemas import (
    CHUNK_SCHEMA,
    COMMUNITY_REPORT_SCHEMA,
    COMMUNITY_SCHEMA,
    DOCUMENT_SCHEMA,
    ENTITY_SCHEMA,
    RELATIONSHIP_SCHEMA,
)
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class ParquetStore(StorageBackend):
    """Parquet-based storage with batched writes."""

    def __init__(
        self,
        root_dir: Path,
        batch_size: int = 1000,
        compression: str = "snappy",
    ):
        """Initialize Parquet storage.

        Args:
            root_dir: Root directory for storage files
            batch_size: Buffer size before flushing to disk
            compression: Compression codec (snappy, gzip, brotli, none)
        """
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.compression = compression

        # Create storage directory
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # Buffers for batched writes
        self.document_buffer: List[Document] = []
        self.chunk_buffer: List[Chunk] = []
        self.entity_buffer: List[Entity] = []
        self.relationship_buffer: List[Relationship] = []
        self.community_buffer: List[Community] = []
        self.community_report_buffer: List[CommunityReport] = []

        # Partition directories (for efficient append)
        self.documents_dir = self.root_dir / "documents"
        self.chunks_dir = self.root_dir / "chunks"
        self.entities_dir = self.root_dir / "entities"
        self.relationships_dir = self.root_dir / "relationships"
        self.communities_dir = self.root_dir / "communities"
        self.community_reports_dir = self.root_dir / "community_reports"
        self.facts_dir = self.root_dir / "facts"
        self.entity_chunk_edges_dir = self.root_dir / "entity_chunk_edges"

        # Create partition directories
        for directory in [
            self.documents_dir,
            self.chunks_dir,
            self.entities_dir,
            self.relationships_dir,
            self.communities_dir,
            self.community_reports_dir,
            self.facts_dir,
            self.entity_chunk_edges_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Partition counters
        self._partition_counters = {
            "documents": 0,
            "chunks": 0,
            "entities": 0,
            "relationships": 0,
            "communities": 0,
            "community_reports": 0,
            "facts": 0,
            "entity_chunk_edges": 0,
        }

        # Load existing partition counts
        self._load_partition_counts()

    async def save_documents(self, documents: List[Document]) -> None:
        """Save documents to buffer, flushing if full."""
        self.document_buffer.extend(documents)

        if len(self.document_buffer) >= self.batch_size:
            await self._flush_documents()

    async def save_chunks(self, chunks: List[Chunk]) -> None:
        """Save chunks to buffer, flushing if full."""
        self.chunk_buffer.extend(chunks)

        if len(self.chunk_buffer) >= self.batch_size:
            await self._flush_chunks()

    async def save_entities(self, entities: List[Entity]) -> None:
        """Save entities to buffer, flushing if full."""
        self.entity_buffer.extend(entities)

        if len(self.entity_buffer) >= self.batch_size:
            await self._flush_entities()

    async def save_relationships(self, relationships: List[Relationship]) -> None:
        """Save relationships to buffer, flushing if full."""
        self.relationship_buffer.extend(relationships)

        if len(self.relationship_buffer) >= self.batch_size:
            await self._flush_relationships()

    async def save_communities(self, communities: List[Community]) -> None:
        """Save communities to buffer, flushing if full."""
        self.community_buffer.extend(communities)

        if len(self.community_buffer) >= self.batch_size:
            await self._flush_communities()

    async def save_community_reports(self, reports: List[CommunityReport]) -> None:
        """Save community reports to buffer, flushing if full."""
        self.community_report_buffer.extend(reports)

        if len(self.community_report_buffer) >= self.batch_size:
            await self._flush_community_reports()

    async def flush(self) -> None:
        """Flush all buffers to disk."""
        await asyncio.gather(
            self._flush_documents(),
            self._flush_chunks(),
            self._flush_entities(),
            self._flush_relationships(),
            self._flush_communities(),
            self._flush_community_reports(),
        )

    async def _flush_documents(self) -> None:
        """Flush document buffer to Parquet."""
        if not self.document_buffer:
            return

        await asyncio.to_thread(
            self._write_parquet,
            self.document_buffer,
            self.documents_dir,
            "documents",
            DOCUMENT_SCHEMA,
            self._document_to_dict,
        )

        logger.debug(f"Flushed {len(self.document_buffer)} documents")
        self.document_buffer.clear()

    async def _flush_chunks(self) -> None:
        """Flush chunk buffer to Parquet."""
        if not self.chunk_buffer:
            return

        await asyncio.to_thread(
            self._write_parquet,
            self.chunk_buffer,
            self.chunks_dir,
            "chunks",
            CHUNK_SCHEMA,
            self._chunk_to_dict,
        )

        logger.debug(f"Flushed {len(self.chunk_buffer)} chunks")
        self.chunk_buffer.clear()

    async def _flush_entities(self) -> None:
        """Flush entity buffer to Parquet."""
        if not self.entity_buffer:
            return

        await asyncio.to_thread(
            self._write_parquet,
            self.entity_buffer,
            self.entities_dir,
            "entities",
            ENTITY_SCHEMA,
            self._entity_to_dict,
        )

        logger.debug(f"Flushed {len(self.entity_buffer)} entities")
        self.entity_buffer.clear()

    async def _flush_relationships(self) -> None:
        """Flush relationship buffer to Parquet."""
        if not self.relationship_buffer:
            return

        await asyncio.to_thread(
            self._write_parquet,
            self.relationship_buffer,
            self.relationships_dir,
            "relationships",
            RELATIONSHIP_SCHEMA,
            self._relationship_to_dict,
        )

        logger.debug(f"Flushed {len(self.relationship_buffer)} relationships")
        self.relationship_buffer.clear()

    async def _flush_communities(self) -> None:
        """Flush community buffer to Parquet."""
        if not self.community_buffer:
            return

        await asyncio.to_thread(
            self._write_parquet,
            self.community_buffer,
            self.communities_dir,
            "communities",
            COMMUNITY_SCHEMA,
            self._community_to_dict,
        )

        logger.debug(f"Flushed {len(self.community_buffer)} communities")
        self.community_buffer.clear()

    async def _flush_community_reports(self) -> None:
        """Flush community report buffer to Parquet."""
        if not self.community_report_buffer:
            return

        await asyncio.to_thread(
            self._write_parquet,
            self.community_report_buffer,
            self.community_reports_dir,
            "community_reports",
            COMMUNITY_REPORT_SCHEMA,
            self._community_report_to_dict,
        )

        logger.debug(f"Flushed {len(self.community_report_buffer)} community reports")
        self.community_report_buffer.clear()

    def _load_partition_counts(self) -> None:
        """Load partition counts from existing files."""
        for name, directory in [
            ("documents", self.documents_dir),
            ("chunks", self.chunks_dir),
            ("entities", self.entities_dir),
            ("relationships", self.relationships_dir),
            ("communities", self.communities_dir),
            ("community_reports", self.community_reports_dir),
            ("facts", self.facts_dir),
            ("entity_chunk_edges", self.entity_chunk_edges_dir),
        ]:
            # Find highest partition number
            partitions = list(directory.glob("part_*.parquet"))
            if partitions:
                max_partition = max(
                    int(p.stem.split("_")[1]) for p in partitions
                )
                self._partition_counters[name] = max_partition + 1

    def _write_parquet(
        self,
        items: List[Any],
        partition_dir: Path,
        partition_name: str,
        schema: pa.Schema,
        converter: callable,
    ) -> None:
        """Write items to Parquet partition (blocking operation).

        Args:
            items: Items to write
            partition_dir: Directory for partitions
            partition_name: Name for partition counter
            schema: PyArrow schema
            converter: Function to convert item to dict
        """
        try:
            # Convert items to dicts
            records = [converter(item) for item in items]

            # Create PyArrow table
            table = pa.Table.from_pylist(records, schema=schema)

            # Write to new partition file
            partition_num = self._partition_counters[partition_name]
            partition_file = partition_dir / f"part_{partition_num:06d}.parquet"

            pq.write_table(table, partition_file, compression=self.compression)

            # Increment partition counter
            self._partition_counters[partition_name] += 1

            logger.debug(f"Wrote {len(items)} items to partition {partition_file.name}")

        except Exception as e:
            raise StorageError(f"Failed to write Parquet partition: {e}")

    # Converter methods
    @staticmethod
    def _document_to_dict(doc: Document) -> Dict[str, Any]:
        return {
            "id": str(doc.id),
            "filename": doc.filename,
            "text": doc.text,
            "metadata": json.dumps(doc.metadata),
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
            "char_count": doc.char_count,
            "token_count": doc.token_count,
        }

    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> Dict[str, Any]:
        return {
            "id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "token_count": chunk.token_count,
            "entity_ids": [str(eid) for eid in chunk.entity_ids],
            "relationship_ids": [str(rid) for rid in chunk.relationship_ids],
            "metadata": json.dumps(chunk.metadata),
        }

    @staticmethod
    def _entity_to_dict(entity: Entity) -> Dict[str, Any]:
        return {
            "id": str(entity.id),
            "name": entity.name,
            "type": entity.type.value,
            "description": entity.description or "",
            "source_chunks": [str(cid) for cid in entity.source_chunks],
            "extraction_confidence": entity.extraction_confidence,
            "aliases": entity.aliases,
            "metadata": json.dumps(entity.metadata),
        }

    @staticmethod
    def _relationship_to_dict(rel: Relationship) -> Dict[str, Any]:
        return {
            "id": str(rel.id),
            "source_entity_id": str(rel.source_entity_id),
            "target_entity_id": str(rel.target_entity_id),
            "type": rel.type.value,
            "description": rel.description or "",
            "source_chunks": [str(cid) for cid in rel.source_chunks],
            "extraction_confidence": rel.extraction_confidence,
            "weight": rel.weight,
            "metadata": json.dumps(rel.metadata),
        }

    @staticmethod
    def _community_to_dict(comm: Community) -> Dict[str, Any]:
        return {
            "id": str(comm.id),
            "level": comm.level,
            "entity_ids": [str(eid) for eid in comm.entity_ids],
            "parent_community_id": str(comm.parent_community_id) if comm.parent_community_id else "",
            "child_community_ids": [str(cid) for cid in comm.child_community_ids],
            "relationship_ids": [str(rid) for rid in comm.relationship_ids],
            "size": comm.size,
            "density": comm.density,
            "title": comm.title or "",
            "summary": comm.summary or "",
            "findings": comm.findings,
            "metadata": json.dumps(comm.metadata),
        }

    @staticmethod
    def _community_report_to_dict(report: CommunityReport) -> Dict[str, Any]:
        return {
            "id": str(report.id),
            "community_id": str(report.community_id),
            "title": report.title,
            "summary": report.summary,
            "full_content": report.full_content,
            "findings": report.findings,
            "token_count": report.token_count,
            "rank": report.rank,
        }

    # Load methods (lazy loading with AsyncIterator from partitions)
    async def load_documents(self) -> AsyncIterator[Document]:
        """Load documents lazily from Parquet partitions."""
        partitions = list(self.documents_dir.glob("part_*.parquet"))
        if not partitions:
            return

        # Read all partitions using PyArrow dataset API
        table = await asyncio.to_thread(pq.read_table, str(self.documents_dir))
        df = table.to_pandas()

        for _, row in df.iterrows():
            yield Document(
                id=UUID(row["id"]),
                filename=row["filename"],
                text=row["text"],
                metadata=json.loads(row["metadata"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                char_count=row["char_count"],
                token_count=row["token_count"],
            )

    async def load_chunks(self) -> AsyncIterator[Chunk]:
        """Load chunks lazily from Parquet partitions."""
        partitions = list(self.chunks_dir.glob("part_*.parquet"))
        if not partitions:
            return

        table = await asyncio.to_thread(pq.read_table, str(self.chunks_dir))
        df = table.to_pandas()

        for _, row in df.iterrows():
            yield Chunk(
                id=UUID(row["id"]),
                document_id=UUID(row["document_id"]),
                chunk_index=row["chunk_index"],
                text=row["text"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                token_count=row["token_count"],
                entity_ids=[UUID(eid) for eid in row["entity_ids"]],
                relationship_ids=[UUID(rid) for rid in row["relationship_ids"]],
                metadata=json.loads(row["metadata"]),
            )

    async def load_entities(self) -> AsyncIterator[Entity]:
        """Load entities lazily from Parquet partitions."""
        partitions = list(self.entities_dir.glob("part_*.parquet"))
        if not partitions:
            return

        table = await asyncio.to_thread(pq.read_table, str(self.entities_dir))
        df = table.to_pandas()

        for _, row in df.iterrows():
            yield Entity(
                id=UUID(row["id"]),
                name=row["name"],
                type=row["type"],
                description=row["description"] if row["description"] else None,
                source_chunks=[UUID(cid) for cid in row["source_chunks"]],
                extraction_confidence=row["extraction_confidence"],
                aliases=row["aliases"],
                metadata=json.loads(row["metadata"]),
            )

    async def load_relationships(self) -> AsyncIterator[Relationship]:
        """Load relationships lazily from Parquet partitions."""
        partitions = list(self.relationships_dir.glob("part_*.parquet"))
        if not partitions:
            return

        table = await asyncio.to_thread(pq.read_table, str(self.relationships_dir))
        df = table.to_pandas()

        for _, row in df.iterrows():
            yield Relationship(
                id=UUID(row["id"]),
                source_entity_id=UUID(row["source_entity_id"]),
                target_entity_id=UUID(row["target_entity_id"]),
                type=row["type"],
                description=row["description"] if row["description"] else None,
                source_chunks=[UUID(cid) for cid in row["source_chunks"]],
                extraction_confidence=row["extraction_confidence"],
                weight=row["weight"],
                metadata=json.loads(row["metadata"]),
            )

    async def load_communities(self) -> AsyncIterator[Community]:
        """Load communities lazily from Parquet partitions."""
        partitions = list(self.communities_dir.glob("part_*.parquet"))
        if not partitions:
            return

        table = await asyncio.to_thread(pq.read_table, str(self.communities_dir))
        df = table.to_pandas()

        for _, row in df.iterrows():
            yield Community(
                id=UUID(row["id"]),
                level=row["level"],
                entity_ids=[UUID(eid) for eid in row["entity_ids"]],
                parent_community_id=UUID(row["parent_community_id"]) if row["parent_community_id"] else None,
                child_community_ids=[UUID(cid) for cid in row["child_community_ids"]],
                relationship_ids=[UUID(rid) for rid in row["relationship_ids"]],
                size=row["size"],
                density=row["density"],
                title=row["title"] if row["title"] else None,
                summary=row["summary"] if row["summary"] else None,
                findings=row["findings"],
                metadata=json.loads(row["metadata"]),
            )

    async def load_community_reports(self) -> AsyncIterator[CommunityReport]:
        """Load community reports lazily from Parquet partitions."""
        partitions = list(self.community_reports_dir.glob("part_*.parquet"))
        if not partitions:
            return

        table = await asyncio.to_thread(pq.read_table, str(self.community_reports_dir))
        df = table.to_pandas()

        for _, row in df.iterrows():
            yield CommunityReport(
                id=UUID(row["id"]),
                community_id=UUID(row["community_id"]),
                title=row["title"],
                summary=row["summary"],
                full_content=row["full_content"],
                findings=row["findings"],
                token_count=row["token_count"],
                rank=row["rank"],
            )

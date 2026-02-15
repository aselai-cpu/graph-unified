"""Vector database storage for embeddings and similarity search."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lancedb
import numpy as np
from lancedb.pydantic import LanceModel, Vector

from graphunified.config.settings import VectorDBConfig
from graphunified.exceptions import StorageError
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """LanceDB-based vector storage for multiple indexes.

    Supports separate indexes for:
    - Chunks (for Naive, Hybrid RAG)
    - Entities (for GraphRAG, LightRAG, HippoRAG)
    - Relationships (for LightRAG global search)
    - Facts (for HippoRAG Stage 1)
    - Communities (for GraphRAG Global search)
    """

    def __init__(
        self,
        root_dir: Path,
        dimension: int = 1024,
        chunk_index_name: str = "chunks",
        entity_index_name: str = "entities",
        relationship_index_name: str = "relationships",
        fact_index_name: str = "facts",
        community_index_name: str = "communities",
    ):
        """Initialize vector store.

        Args:
            root_dir: Root directory for vector database
            dimension: Embedding dimension (default: 1024 for voyage-3)
            chunk_index_name: Name for chunk index
            entity_index_name: Name for entity index
            relationship_index_name: Name for relationship index
            fact_index_name: Name for fact index
            community_index_name: Name for community index
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.dimension = dimension

        # Index names
        self.chunk_index_name = chunk_index_name
        self.entity_index_name = entity_index_name
        self.relationship_index_name = relationship_index_name
        self.fact_index_name = fact_index_name
        self.community_index_name = community_index_name

        # LanceDB connection (lazy initialization)
        self._db = None
        self._tables: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: VectorDBConfig, root_dir: Path, dimension: int) -> "VectorStore":
        """Create vector store from configuration.

        Args:
            config: Vector DB configuration
            root_dir: Root directory for storage
            dimension: Embedding dimension

        Returns:
            VectorStore instance
        """
        return cls(
            root_dir=root_dir,
            dimension=dimension,
            chunk_index_name=config.chunk_index_name,
            entity_index_name=config.entity_index_name,
            relationship_index_name=config.relationship_index_name,
            fact_index_name=config.fact_index_name,
            community_index_name=config.community_index_name,
        )

    async def _get_db(self) -> lancedb.DBConnection:
        """Get or create LanceDB connection."""
        if self._db is None:
            self._db = await asyncio.to_thread(lancedb.connect, str(self.root_dir))
        return self._db

    async def _get_or_create_table(
        self,
        table_name: str,
        schema: Optional[LanceModel] = None,
        data: Optional[List[Dict]] = None,
    ) -> Any:
        """Get or create a table in the database.

        Args:
            table_name: Name of the table
            schema: Optional Pydantic schema for the table
            data: Optional data to create table with (schema will be inferred)

        Returns:
            LanceDB table
        """
        if table_name in self._tables:
            return self._tables[table_name]

        db = await self._get_db()

        # Check if table exists
        existing_tables = await asyncio.to_thread(db.table_names)

        if table_name in existing_tables:
            table = await asyncio.to_thread(db.open_table, table_name)
        else:
            if schema is not None:
                # Create new table with schema
                table = await asyncio.to_thread(db.create_table, table_name, schema=schema)
            elif data is not None:
                # Create new table with data (schema inferred)
                table = await asyncio.to_thread(db.create_table, table_name, data=data)
            else:
                raise StorageError(f"Schema or data required to create new table: {table_name}")

        self._tables[table_name] = table
        return table

    async def index_chunks(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Index chunk embeddings.

        Args:
            chunk_ids: List of chunk IDs
            embeddings: List of embedding vectors
            texts: List of chunk texts
            metadata: Optional metadata per chunk

        Raises:
            StorageError: If indexing fails
        """
        if len(chunk_ids) != len(embeddings) != len(texts):
            raise ValueError("chunk_ids, embeddings, and texts must have same length")

        try:
            # Prepare data
            data = []
            for i, (chunk_id, embedding, text) in enumerate(zip(chunk_ids, embeddings, texts)):
                record = {
                    "id": str(chunk_id),  # Convert UUID to string
                    "vector": embedding,
                    "text": text,
                    "metadata": metadata[i] if metadata else {},
                }
                data.append(record)

            # Check if table exists
            db = await self._get_db()
            existing_tables = await asyncio.to_thread(db.table_names)
            table_exists = self.chunk_index_name in existing_tables

            if table_exists:
                # Table exists, open it and add data
                table = await self._get_or_create_table(self.chunk_index_name)
                await asyncio.to_thread(table.add, data)
            else:
                # Table doesn't exist, create it with data
                table = await self._get_or_create_table(
                    self.chunk_index_name,
                    data=data,
                )

            logger.info(f"Indexed {len(chunk_ids)} chunks")

        except Exception as e:
            raise StorageError(f"Failed to index chunks: {e}")

    async def index_entities(
        self,
        entity_ids: List[str],
        embeddings: List[List[float]],
        names: List[str],
        types: List[str],
        descriptions: Optional[List[str]] = None,
    ) -> None:
        """Index entity embeddings.

        Args:
            entity_ids: List of entity IDs
            embeddings: List of embedding vectors
            names: List of entity names
            types: List of entity types
            descriptions: Optional entity descriptions

        Raises:
            StorageError: If indexing fails
        """
        if len(entity_ids) != len(embeddings) != len(names):
            raise ValueError("entity_ids, embeddings, and names must have same length")

        try:
            data = []
            for i, (entity_id, embedding, name, etype) in enumerate(
                zip(entity_ids, embeddings, names, types)
            ):
                record = {
                    "id": str(entity_id),  # Convert UUID to string
                    "vector": embedding,
                    "name": name,
                    "type": etype,
                    "description": descriptions[i] if descriptions else "",
                }
                data.append(record)

            # Check if table exists
            db = await self._get_db()
            existing_tables = await asyncio.to_thread(db.table_names)
            table_exists = self.entity_index_name in existing_tables

            if table_exists:
                table = await self._get_or_create_table(self.entity_index_name)
                await asyncio.to_thread(table.add, data)
            else:
                table = await self._get_or_create_table(
                    self.entity_index_name,
                    data=data,
                )

            logger.info(f"Indexed {len(entity_ids)} entities")

        except Exception as e:
            raise StorageError(f"Failed to index entities: {e}")

    async def index_relationships(
        self,
        relationship_ids: List[str],
        embeddings: List[List[float]],
        descriptions: List[str],
        types: List[str],
    ) -> None:
        """Index relationship embeddings (for LightRAG global search).

        Args:
            relationship_ids: List of relationship IDs
            embeddings: List of embedding vectors
            descriptions: List of relationship descriptions
            types: List of relationship types

        Raises:
            StorageError: If indexing fails
        """
        if len(relationship_ids) != len(embeddings) != len(descriptions):
            raise ValueError("All input lists must have same length")

        try:
            data = []
            for rel_id, embedding, desc, rtype in zip(
                relationship_ids, embeddings, descriptions, types
            ):
                record = {
                    "id": str(rel_id),  # Convert UUID to string
                    "vector": embedding,
                    "description": desc,
                    "type": rtype,
                }
                data.append(record)

            # Check if table exists
            db = await self._get_db()
            existing_tables = await asyncio.to_thread(db.table_names)
            table_exists = self.relationship_index_name in existing_tables

            if table_exists:
                table = await self._get_or_create_table(self.relationship_index_name)
                await asyncio.to_thread(table.add, data)
            else:
                table = await self._get_or_create_table(
                    self.relationship_index_name,
                    data=data,
                )

            logger.info(f"Indexed {len(relationship_ids)} relationships")

        except Exception as e:
            raise StorageError(f"Failed to index relationships: {e}")

    async def index_facts(
        self,
        fact_ids: List[str],
        embeddings: List[List[float]],
        subjects: List[str],
        predicates: List[str],
        objects: List[str],
    ) -> None:
        """Index fact embeddings (for HippoRAG Stage 1).

        Args:
            fact_ids: List of fact IDs
            embeddings: List of embedding vectors
            subjects: List of fact subjects
            predicates: List of fact predicates
            objects: List of fact objects

        Raises:
            StorageError: If indexing fails
        """
        if len(fact_ids) != len(embeddings) != len(subjects):
            raise ValueError("All input lists must have same length")

        try:
            data = []
            for fact_id, embedding, subj, pred, obj in zip(
                fact_ids, embeddings, subjects, predicates, objects
            ):
                record = {
                    "id": fact_id,
                    "vector": embedding,
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                }
                data.append(record)

            table = await self._get_or_create_table(self.fact_index_name, schema=None)
            await asyncio.to_thread(table.add, data)

            logger.info(f"Indexed {len(fact_ids)} facts")

        except Exception as e:
            raise StorageError(f"Failed to index facts: {e}")

    async def search_chunks(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar chunks.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of (chunk_id, score, metadata) tuples

        Raises:
            StorageError: If search fails
        """
        try:
            table = await self._get_or_create_table(self.chunk_index_name, schema=None)

            # Perform search
            # LanceDB table.search() returns a Query object, call .limit() then .to_pandas()
            search_query = table.search(query_vector).limit(top_k)
            results = await asyncio.to_thread(search_query.to_pandas)

            # Format results
            output = []
            for _, row in results.iterrows():
                output.append((
                    row["id"],
                    row["_distance"],  # LanceDB returns distance (lower is better)
                    {
                        "text": row["text"],
                        "metadata": row.get("metadata", {}),
                    },
                ))

            return output

        except Exception as e:
            raise StorageError(f"Failed to search chunks: {e}")

    async def search_entities(
        self,
        query_vector: List[float],
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar entities.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (entity_id, score, metadata) tuples

        Raises:
            StorageError: If search fails
        """
        try:
            table = await self._get_or_create_table(self.entity_index_name, schema=None)

            # LanceDB table.search() returns a Query object, call .limit() then .to_pandas()
            search_query = table.search(query_vector).limit(top_k)
            results = await asyncio.to_thread(search_query.to_pandas)

            output = []
            for _, row in results.iterrows():
                output.append((
                    row["id"],
                    row["_distance"],
                    {
                        "name": row["name"],
                        "type": row["type"],
                        "description": row.get("description", ""),
                    },
                ))

            return output

        except Exception as e:
            raise StorageError(f"Failed to search entities: {e}")

    async def search_relationships(
        self,
        query_vector: List[float],
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar relationships (LightRAG global search).

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (relationship_id, score, metadata) tuples

        Raises:
            StorageError: If search fails
        """
        try:
            table = await self._get_or_create_table(self.relationship_index_name, schema=None)

            # LanceDB table.search() returns a Query object, call .limit() then .to_pandas()
            search_query = table.search(query_vector).limit(top_k)
            results = await asyncio.to_thread(search_query.to_pandas)

            output = []
            for _, row in results.iterrows():
                output.append((
                    row["id"],
                    row["_distance"],
                    {
                        "description": row["description"],
                        "type": row["type"],
                    },
                ))

            return output

        except Exception as e:
            raise StorageError(f"Failed to search relationships: {e}")

    async def search_facts(
        self,
        query_vector: List[float],
        top_k: int = 20,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar facts (HippoRAG Stage 1).

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (fact_id, score, metadata) tuples

        Raises:
            StorageError: If search fails
        """
        try:
            table = await self._get_or_create_table(self.fact_index_name, schema=None)

            # LanceDB table.search() returns a Query object, call .limit() then .to_pandas()
            search_query = table.search(query_vector).limit(top_k)
            results = await asyncio.to_thread(search_query.to_pandas)

            output = []
            for _, row in results.iterrows():
                output.append((
                    row["id"],
                    row["_distance"],
                    {
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                    },
                ))

            return output

        except Exception as e:
            raise StorageError(f"Failed to search facts: {e}")

    async def index_communities(
        self,
        community_ids: List[str],
        embeddings: List[List[float]],
        summaries: List[str],
    ) -> None:
        """Index community embeddings (for GraphRAG Global search).

        Args:
            community_ids: List of community IDs
            embeddings: List of embedding vectors
            summaries: List of community summaries

        Raises:
            StorageError: If indexing fails
        """
        if len(community_ids) != len(embeddings) != len(summaries):
            raise ValueError("All input lists must have same length")

        try:
            data = []
            for comm_id, embedding, summary in zip(community_ids, embeddings, summaries):
                record = {
                    "id": str(comm_id),  # Convert UUID to string
                    "vector": embedding,
                    "summary": summary,
                }
                data.append(record)

            # Check if table exists
            db = await self._get_db()
            existing_tables = await asyncio.to_thread(db.table_names)
            table_exists = self.community_index_name in existing_tables

            if table_exists:
                table = await self._get_or_create_table(self.community_index_name)
                await asyncio.to_thread(table.add, data)
            else:
                table = await self._get_or_create_table(
                    self.community_index_name,
                    data=data,
                )

            logger.info(f"Indexed {len(community_ids)} communities")

        except Exception as e:
            raise StorageError(f"Failed to index communities: {e}")

    async def search_communities(
        self,
        query_vector: List[float],
        top_k: int = 10,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar communities (GraphRAG Global search).

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (community_id, score, metadata) tuples

        Raises:
            StorageError: If search fails
        """
        try:
            table = await self._get_or_create_table(self.community_index_name, schema=None)

            # LanceDB table.search() returns a Query object, call .limit() then .to_pandas()
            search_query = table.search(query_vector).limit(top_k)
            results = await asyncio.to_thread(search_query.to_pandas)

            output = []
            for _, row in results.iterrows():
                output.append((
                    row["id"],
                    row["_distance"],
                    {
                        "summary": row["summary"],
                    },
                ))

            return output

        except Exception as e:
            raise StorageError(f"Failed to search communities: {e}")

    async def count(self, index_name: str) -> int:
        """Count vectors in an index.

        Args:
            index_name: Name of index to count

        Returns:
            Number of vectors in index

        Raises:
            StorageError: If count fails
        """
        try:
            table = await self._get_or_create_table(index_name, schema=None)
            count = await asyncio.to_thread(table.count_rows)
            return count
        except Exception as e:
            raise StorageError(f"Failed to count vectors in {index_name}: {e}")

    async def index_entity_chunk_mappings(
        self,
        chunks: List[Any],  # Chunk objects with entity_ids
        embeddings_dict: Dict[str, List[float]],  # chunk_id -> embedding
    ) -> None:
        """Build reverse index: entity_id -> chunks containing that entity.

        This enables O(log n) lookup instead of O(n) scan when finding chunks
        by entity IDs.

        Args:
            chunks: List of Chunk objects with populated entity_ids
            embeddings_dict: Mapping of chunk_id -> embedding vector

        Raises:
            StorageError: If indexing fails
        """
        try:
            # Build entity->chunk mappings
            mappings = []
            for chunk in chunks:
                chunk_embedding = embeddings_dict.get(str(chunk.id))
                if not chunk_embedding:
                    continue

                for entity_id in chunk.entity_ids:
                    mappings.append({
                        "id": f"{entity_id}_{chunk.id}",  # Composite key
                        "entity_id": str(entity_id),
                        "chunk_id": str(chunk.id),
                        "vector": chunk_embedding,  # Store chunk embedding for fast retrieval
                        "text_preview": chunk.text[:200],  # For display
                    })

            if not mappings:
                logger.warning("No entity-chunk mappings to index")
                return

            # Check if table exists
            db = await self._get_db()
            existing_tables = await asyncio.to_thread(db.table_names)
            table_exists = "entity_chunks" in existing_tables

            if table_exists:
                # Table exists, open it and add data
                table = await self._get_or_create_table("entity_chunks")
                await asyncio.to_thread(table.add, mappings)
            else:
                # Table doesn't exist, create it with data
                table = await self._get_or_create_table("entity_chunks", data=mappings)

            logger.info(f"Indexed {len(mappings)} entity-chunk mappings")

        except Exception as e:
            raise StorageError(f"Failed to index entity-chunk mappings: {e}")

    async def get_chunks_by_entities(
        self,
        entity_ids: List[str],  # List of entity UUIDs as strings
        parquet_store: Any,  # ParquetStore for loading full chunks
    ) -> List[Any]:  # Returns Chunk objects
        """Get all chunks connected to given entities (reverse lookup).

        Fast O(log n) operation using entity-chunk index instead of O(n) scan.

        Args:
            entity_ids: List of entity IDs to find chunks for
            parquet_store: ParquetStore to load full chunk data

        Returns:
            List of Chunk objects connected to any of the entities

        Raises:
            StorageError: If lookup fails
        """
        if not entity_ids:
            return []

        try:
            table = await self._get_or_create_table("entity_chunks", schema=None)

            # Query for all entity IDs
            # LanceDB doesn't have great multi-value support, so query each entity
            all_chunk_ids = set()

            for entity_id in entity_ids:
                results = await asyncio.to_thread(
                    lambda eid=entity_id: table.search()
                    .where(f"entity_id = '{eid}'")
                    .limit(10000)  # Set reasonable limit
                    .to_pandas()
                )
                if not results.empty:
                    all_chunk_ids.update(results["chunk_id"].tolist())

            if not all_chunk_ids:
                logger.debug(f"No chunks found for {len(entity_ids)} entities")
                return []

            # Load full chunks from Parquet (much faster than scanning all)
            chunks = []
            async for chunk in parquet_store.load_chunks():
                if str(chunk.id) in all_chunk_ids:
                    chunks.append(chunk)
                    # Early exit if we found all
                    if len(chunks) == len(all_chunk_ids):
                        break

            logger.debug(f"Found {len(chunks)} chunks for {len(entity_ids)} entities")
            return chunks

        except Exception as e:
            logger.error(f"Failed to get chunks by entities: {e}")
            # Gracefully return empty list on error rather than raising
            return []

    async def close(self) -> None:
        """Close database connection."""
        self._db = None
        self._tables.clear()

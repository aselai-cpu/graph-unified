"""Indexing pipeline orchestrator with async DAG execution."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from graphunified.config.models import Chunk, Entity, Relationship
from graphunified.config.settings import Settings
from graphunified.index.stages.base import ProgressCallback, StageResult, StageStatus
from graphunified.index.stages.chunk import ChunkStage
from graphunified.index.stages.embed import EmbedStage
from graphunified.index.stages.extract import ExtractStage
from graphunified.index.stages.index import IndexStage
from graphunified.index.stages.load import LoadStage
from graphunified.storage.parquet_store import ParquetStore
from graphunified.storage.vector_store import VectorStore
from graphunified.utils.embedding_factory import create_embedding_client
from graphunified.utils.llm import ClaudeClient
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class IndexPipeline:
    """Async pipeline orchestrator for document indexing."""

    def __init__(
        self,
        settings: Settings,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """Initialize indexing pipeline.

        Args:
            settings: Application settings
            input_dir: Input directory containing documents
            output_dir: Output directory for Parquet files
            progress_callback: Optional progress callback
        """
        self.settings = settings
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.progress_callback = progress_callback

        # Initialize clients
        self.llm_client = ClaudeClient.from_config(settings.llm)
        self.embedding_client = create_embedding_client(settings.embedding)

        # Initialize storage
        self.storage = ParquetStore.from_config(settings.storage, output_dir)

        # Initialize vector store
        self.vector_store = VectorStore.from_config(
            settings.storage.vector_db,
            output_dir / "lancedb",
            settings.embedding.dimension
        )

        # Initialize stages
        self.load_stage = LoadStage(input_dir, progress_callback=self._make_stage_callback("load"))
        self.chunk_stage = ChunkStage(
            chunk_size=settings.indexing.chunk_size,
            chunk_overlap=settings.indexing.chunk_overlap,
            progress_callback=self._make_stage_callback("chunk"),
        )
        self.extract_stage = ExtractStage(
            llm_client=self.llm_client,
            batch_size=settings.indexing.extraction_batch_size,
            dedup_threshold=settings.indexing.dedup_threshold,
            max_concurrent=settings.indexing.max_concurrent,
            progress_callback=self._make_stage_callback("extract"),
        )
        self.embed_stage = EmbedStage(
            embedding_client=self.embedding_client,
            embed_chunks=True,
            embed_entities=True,
            progress_callback=self._make_stage_callback("embed"),
        )
        self.index_stage = IndexStage(
            vector_store=self.vector_store,
            build_text_index=True,
            progress_callback=self._make_stage_callback("index"),
        )

        # Pipeline state
        self.checkpoint_file = output_dir / "checkpoint.json"
        self.checkpoint_interval = 1000  # Save every N documents

    def _make_stage_callback(self, stage_name: str) -> ProgressCallback:
        """Create a progress callback for a stage.

        Args:
            stage_name: Name of the stage

        Returns:
            Progress callback function
        """
        if not self.progress_callback:
            return None

        def callback(name: str, progress: float) -> None:
            # Combine stage name with substage name
            full_name = f"{stage_name}.{name}" if name != stage_name else stage_name
            self.progress_callback(full_name, progress)

        return callback

    async def run(self, skip_extraction: bool = False, skip_embedding: bool = False) -> Dict[str, Any]:
        """Run the full indexing pipeline.

        Args:
            skip_extraction: Skip extraction stage (for testing)
            skip_embedding: Skip embedding stage (for testing)

        Returns:
            Dict with pipeline results and metadata
        """
        logger.info("Starting indexing pipeline")
        start_time = time.time()

        try:
            # Stage 1: Load documents
            logger.info("Stage 1/5: Loading documents")
            load_result = await self.load_stage.execute()
            if load_result.status == StageStatus.FAILED:
                raise RuntimeError(f"Load stage failed: {load_result.metadata.get('error')}")

            documents = load_result.data
            logger.info(f"Loaded {len(documents)} documents")

            # Save documents
            if documents:
                await self.storage.save_documents(documents)
                logger.info("Saved documents to Parquet")

            # Stage 2: Chunk documents
            logger.info("Stage 2/5: Chunking documents")
            chunk_result = await self.chunk_stage.execute(documents)
            if chunk_result.status == StageStatus.FAILED:
                raise RuntimeError(f"Chunk stage failed: {chunk_result.metadata.get('error')}")

            chunks = chunk_result.data
            logger.info(f"Created {len(chunks)} chunks")

            # Initialize extraction results
            entities = []
            relationships = []
            entity_map = {}

            # Stage 3: Extract entities and relationships (optional)
            if not skip_extraction and chunks:
                logger.info("Stage 3/5: Extracting entities and relationships")
                extract_result = await self.extract_stage.execute(chunks)
                if extract_result.status == StageStatus.FAILED:
                    logger.warning(f"Extract stage failed: {extract_result.metadata.get('error')}")
                else:
                    entities = extract_result.data["entities"]
                    relationships = extract_result.data["relationships"]
                    entity_map = extract_result.data["entity_map"]
                    # Update chunks with populated bidirectional links
                    chunks = extract_result.data.get("chunks", chunks)
                    logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
            else:
                logger.info("Stage 3/5: Skipping extraction (skip_extraction=True)")

            # Stage 4: Generate embeddings (optional)
            if not skip_embedding:
                logger.info("Stage 4/5: Generating embeddings")

                # Prepare input for embed stage
                embed_input = {
                    "chunks": chunks,
                    "entities": entities,
                    "relationships": relationships,
                    "entity_map": entity_map,
                }

                embed_result = await self.embed_stage.execute(embed_input)
                if embed_result.status == StageStatus.FAILED:
                    logger.warning(f"Embed stage failed: {embed_result.metadata.get('error')}")
                else:
                    chunks = embed_result.data["chunks"]
                    entities = embed_result.data["entities"]
                    relationships = embed_result.data["relationships"]
                    logger.info(
                        f"Embedded {embed_result.metadata['chunks_embedded']} chunks, "
                        f"{embed_result.metadata['entities_embedded']} entities, "
                        f"and {embed_result.metadata['relationships_embedded']} relationships"
                    )
            else:
                logger.info("Stage 4/5: Skipping embedding (skip_embedding=True)")

            # Save all results to Parquet
            logger.info("Saving results to Parquet")
            if chunks:
                await self.storage.save_chunks(chunks)
                logger.info(f"Saved {len(chunks)} chunks")

            if entities:
                await self.storage.save_entities(entities)
                logger.info(f"Saved {len(entities)} entities")

            if relationships:
                await self.storage.save_relationships(relationships)
                logger.info(f"Saved {len(relationships)} relationships")

            # Flush storage
            await self.storage.flush()
            logger.info("Flushed storage buffers")

            # Stage 5: Build searchable indexes
            if not skip_embedding:  # Only build indexes if we have embeddings
                logger.info("Stage 5/5: Building searchable indexes")

                index_input = {
                    "chunks": chunks,
                    "entities": entities,
                    "relationships": relationships,
                }

                index_result = await self.index_stage.execute(index_input)
                if index_result.status == StageStatus.FAILED:
                    logger.warning(f"Index stage failed: {index_result.metadata.get('error')}")
                else:
                    logger.info(
                        f"Built indexes: {index_result.metadata['chunks_indexed']} chunks, "
                        f"{index_result.metadata['entities_indexed']} entities, "
                        f"{index_result.metadata['relationships_indexed']} relationships"
                    )
            else:
                logger.info("Stage 5/5: Skipping index building (no embeddings)")

            # Calculate final metrics
            duration = time.time() - start_time
            metrics = {
                "duration_seconds": duration,
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "chunks_with_embeddings": sum(1 for c in chunks if c.embedding is not None),
                "entities_with_embeddings": sum(1 for e in entities if e.embedding is not None),
                "relationships_with_embeddings": sum(1 for r in relationships if r.embedding is not None),
                "output_dir": str(self.output_dir),
            }

            logger.info(f"Pipeline completed in {duration:.2f}s")
            logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

            return {
                "status": "success",
                "documents": documents,
                "chunks": chunks,
                "entities": entities,
                "relationships": relationships,
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "metrics": {"duration_seconds": time.time() - start_time},
            }

    async def run_incremental(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """Run pipeline with checkpointing support.

        Args:
            checkpoint_id: Optional checkpoint ID to resume from

        Returns:
            Dict with pipeline results
        """
        # Load checkpoint if exists
        if checkpoint_id and self.checkpoint_file.exists():
            checkpoint = self._load_checkpoint()
            logger.info(f"Resuming from checkpoint: {checkpoint}")
            # TODO: Implement resume logic
        else:
            checkpoint = None

        # Run pipeline
        result = await self.run()

        # Save checkpoint if successful
        if result["status"] == "success":
            self._save_checkpoint(result["metrics"])

        return result

    def _save_checkpoint(self, metrics: Dict[str, Any]) -> None:
        """Save checkpoint state.

        Args:
            metrics: Pipeline metrics to save
        """
        try:
            checkpoint = {
                "timestamp": time.time(),
                "metrics": metrics,
            }
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)
            logger.debug(f"Saved checkpoint to {self.checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint state.

        Returns:
            Checkpoint data or None if load fails
        """
        try:
            with open(self.checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            logger.debug(f"Loaded checkpoint from {self.checkpoint_file}")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    @classmethod
    def from_config(
        cls,
        settings: Settings,
        input_dir: Path,
        output_dir: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> "IndexPipeline":
        """Create pipeline from settings.

        Args:
            settings: Application settings
            input_dir: Input directory
            output_dir: Output directory
            progress_callback: Optional progress callback

        Returns:
            IndexPipeline instance
        """
        return cls(settings, input_dir, output_dir, progress_callback)

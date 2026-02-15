"""Document chunking stage with token-based overlapping windows."""

import time
from typing import List, Optional
from uuid import uuid4

from graphunified.config.models import Chunk, Document
from graphunified.index.stages.base import PipelineStage, ProgressCallback, StageResult, StageStatus
from graphunified.utils.logging import get_logger
from graphunified.utils.tokenizer import decode_tokens, get_encoding, tokenize

logger = get_logger(__name__)


class ChunkStage(PipelineStage):
    """Chunk documents into overlapping token windows."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        encoding_name: str = "cl100k_base",
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """Initialize chunk stage.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap size in tokens
            encoding_name: Tokenizer encoding to use
            progress_callback: Optional progress callback
        """
        super().__init__("chunk", progress_callback)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    async def execute(self, input_data: List[Document]) -> StageResult:
        """Chunk all documents into overlapping windows.

        Args:
            input_data: List of documents from load stage

        Returns:
            StageResult containing list of Chunk objects
        """
        start_time = time.time()

        if not input_data:
            logger.warning("No documents to chunk")
            return StageResult(
                status=StageStatus.COMPLETED,
                data=[],
                metadata={"document_count": 0, "chunk_count": 0},
                duration=time.time() - start_time,
            )

        try:
            all_chunks = []
            total_docs = len(input_data)

            for idx, document in enumerate(input_data):
                chunks = self._chunk_document(document)
                all_chunks.extend(chunks)

                # Report progress
                progress = (idx + 1) / total_docs
                self._report_progress(progress)

            logger.info(f"Created {len(all_chunks)} chunks from {total_docs} documents")

            return StageResult(
                status=StageStatus.COMPLETED,
                data=all_chunks,
                metadata={
                    "document_count": total_docs,
                    "chunk_count": len(all_chunks),
                    "avg_chunks_per_doc": len(all_chunks) / total_docs if total_docs > 0 else 0,
                },
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Chunk stage failed: {e}")
            return StageResult(
                status=StageStatus.FAILED,
                data=None,
                metadata={"error": str(e)},
                duration=time.time() - start_time,
            )

    def _chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document into overlapping windows.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        # Tokenize the full document
        encoding = get_encoding(self.encoding_name)
        tokens = tokenize(document.text, self.encoding_name)

        if not tokens:
            logger.warning(f"Document {document.filename} has no tokens")
            return []

        chunks = []
        chunk_index = 0

        # Calculate stride (how much to advance each window)
        stride = self.chunk_size - self.chunk_overlap

        # Create overlapping windows
        for start_idx in range(0, len(tokens), stride):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode tokens back to text
            chunk_text = decode_tokens(chunk_tokens, self.encoding_name)

            # Find character positions in original text
            # Decode tokens up to start position
            start_text = decode_tokens(tokens[:start_idx], self.encoding_name) if start_idx > 0 else ""
            start_char = len(start_text)
            end_char = start_char + len(chunk_text)

            # Create chunk
            chunk = Chunk(
                id=uuid4(),
                document_id=document.id,
                chunk_index=chunk_index,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                token_count=len(chunk_tokens),
                metadata={
                    "document_filename": document.filename,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            )

            chunks.append(chunk)
            chunk_index += 1

            # Stop if we've reached the end
            if end_idx >= len(tokens):
                break

        logger.debug(f"Chunked {document.filename}: {len(chunks)} chunks from {len(tokens)} tokens")

        return chunks

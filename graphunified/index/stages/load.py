"""Document loading stage with async file I/O."""

import asyncio
import hashlib
import time
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import aiofiles

from graphunified.config.models import Document
from graphunified.exceptions import StorageError
from graphunified.index.stages.base import PipelineStage, ProgressCallback, StageResult, StageStatus
from graphunified.utils.logging import get_logger
from graphunified.utils.tokenizer import count_tokens

logger = get_logger(__name__)


class LoadStage(PipelineStage):
    """Load documents from a directory with async file I/O."""

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def __init__(
        self,
        input_dir: Path,
        max_concurrent: int = 10,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """Initialize load stage.

        Args:
            input_dir: Directory containing documents
            max_concurrent: Maximum concurrent file reads
            progress_callback: Optional progress callback
        """
        super().__init__("load", progress_callback)
        self.input_dir = Path(input_dir)
        self.max_concurrent = max_concurrent

        if not self.input_dir.exists():
            raise StorageError(f"Input directory does not exist: {input_dir}")
        if not self.input_dir.is_dir():
            raise StorageError(f"Input path is not a directory: {input_dir}")

    async def execute(self, input_data: None = None) -> StageResult:
        """Load all supported documents from input directory.

        Args:
            input_data: Not used (initial stage)

        Returns:
            StageResult containing list of Document objects
        """
        start_time = time.time()

        try:
            # Find all supported files
            file_paths = self._find_files()
            logger.info(f"Found {len(file_paths)} files to load")

            if not file_paths:
                logger.warning("No files found in input directory")
                return StageResult(
                    status=StageStatus.COMPLETED,
                    data=[],
                    metadata={"file_count": 0},
                    duration=time.time() - start_time,
                )

            # Load files concurrently with semaphore
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [self._load_document(fp, semaphore, i, len(file_paths)) for i, fp in enumerate(file_paths)]
            documents = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out errors and None values
            valid_documents = []
            error_count = 0
            for doc in documents:
                if isinstance(doc, Exception):
                    logger.error(f"Error loading document: {doc}")
                    error_count += 1
                elif doc is not None:
                    valid_documents.append(doc)

            logger.info(f"Loaded {len(valid_documents)} documents successfully ({error_count} errors)")

            return StageResult(
                status=StageStatus.COMPLETED,
                data=valid_documents,
                metadata={
                    "file_count": len(file_paths),
                    "success_count": len(valid_documents),
                    "error_count": error_count,
                },
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Load stage failed: {e}")
            return StageResult(
                status=StageStatus.FAILED,
                data=None,
                metadata={"error": str(e)},
                duration=time.time() - start_time,
            )

    def _find_files(self) -> List[Path]:
        """Find all supported files in input directory.

        Returns:
            List of file paths
        """
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.input_dir.glob(f"**/*{ext}"))
        return sorted(files)  # Sort for deterministic ordering

    async def _load_document(
        self, filepath: Path, semaphore: asyncio.Semaphore, index: int, total: int
    ) -> Optional[Document]:
        """Load a single document from file.

        Args:
            filepath: Path to file
            semaphore: Semaphore for concurrency control
            index: Current file index
            total: Total number of files

        Returns:
            Document object or None if loading fails
        """
        async with semaphore:
            try:
                # Read file content with encoding fallbacks
                text = await self._read_file(filepath)

                if not text or not text.strip():
                    logger.warning(f"Skipping empty file: {filepath}")
                    return None

                # Generate document hash for change detection
                doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

                # Count tokens
                token_count = count_tokens(text)

                # Create document
                document = Document(
                    id=uuid4(),
                    filename=filepath.name,
                    text=text,
                    char_count=len(text),
                    token_count=token_count,
                    metadata={
                        "source_path": str(filepath),
                        "file_extension": filepath.suffix,
                        "hash": doc_hash,
                    },
                )

                # Report progress
                progress = (index + 1) / total
                self._report_progress(progress)

                logger.debug(f"Loaded {filepath.name}: {len(text)} chars, {token_count} tokens")

                return document

            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
                return None

    async def _read_file(self, filepath: Path) -> str:
        """Read file content with encoding fallbacks.

        Args:
            filepath: Path to file

        Returns:
            File content as string

        Raises:
            StorageError: If file cannot be read
        """
        encodings = ["utf-8", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                async with aiofiles.open(filepath, "r", encoding=encoding) as f:
                    return await f.read()
            except UnicodeDecodeError:
                if encoding == encodings[-1]:
                    raise StorageError(f"Could not decode file {filepath} with any encoding")
                continue
            except Exception as e:
                raise StorageError(f"Failed to read file {filepath}: {e}")

        raise StorageError(f"Failed to read file {filepath}")

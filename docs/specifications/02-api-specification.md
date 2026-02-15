# API & Interface Specifications

**Version:** 1.0
**Last Updated:** 2026-02-15
**Status:** Technical Specification

## Overview

This document specifies the Python API for programmatic usage of Graph-Unified. It defines protocols, interfaces, public methods, error handling, and usage examples for developers integrating Graph-Unified into applications.

## Design Principles

1. **Protocol-Oriented:** Use Python protocols (PEP 544) for interfaces
2. **Type-Safe:** Full type hints for all public APIs
3. **Async-First:** Support asyncio for I/O-bound operations
4. **Composable:** Components can be used independently or together
5. **Testable:** Dependency injection for easy mocking

---

## Core Protocols

### Retriever Protocol

All retrieval strategies implement this protocol.

```python
from typing import Protocol, List, Dict, Any
from graphunified.config.models import Chunk

class Retriever(Protocol):
    """Protocol for retrieval strategies."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant contexts for a query.

        Args:
            query: User query string
            top_k: Number of contexts to retrieve
            **kwargs: Strategy-specific parameters

        Returns:
            List of context dictionaries with keys:
                - text (str): Context text
                - score (float): Relevance score [0.0, 1.0]
                - source (str): Source identifier (chunk_id, entity_id, etc.)
                - metadata (dict): Additional metadata

        Raises:
            ValueError: Invalid query or parameters
            RuntimeError: Retrieval failed
        """
        ...

    def get_name(self) -> str:
        """Return strategy name (e.g., 'naive', 'graphrag_local')."""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Return strategy configuration."""
        ...
```

**Implementation Requirements:**
- `retrieve()` must be async
- Scores must be normalized to [0.0, 1.0]
- Empty query should raise `ValueError`
- Missing indexes should raise `RuntimeError`

---

### PipelineStage Protocol

Indexing pipeline stages implement this protocol.

```python
from typing import Protocol, Any, Optional
from dataclasses import dataclass

@dataclass
class StageResult:
    """Result from a pipeline stage."""
    status: str  # "success" | "failed" | "skipped"
    data: Any
    metadata: Dict[str, Any]
    error: Optional[Exception] = None

class PipelineStage(Protocol):
    """Protocol for indexing pipeline stages."""

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any]
    ) -> StageResult:
        """
        Execute the pipeline stage.

        Args:
            input_data: Output from previous stage
            context: Shared pipeline context (config, storage, etc.)

        Returns:
            StageResult with output data and metadata

        Raises:
            RuntimeError: Stage execution failed
        """
        ...

    def get_name(self) -> str:
        """Return stage name (e.g., 'chunk', 'extract')."""
        ...

    def requires(self) -> List[str]:
        """Return list of required prerequisite stage names."""
        ...

    def provides(self) -> str:
        """Return name of data this stage provides."""
        ...
```

---

### Storage Interfaces

#### ChunkStore

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Iterator
from uuid import UUID
from graphunified.config.models import Chunk

class ChunkStore(ABC):
    """Interface for chunk storage."""

    @abstractmethod
    async def save(self, chunks: List[Chunk]) -> None:
        """Save chunks to storage."""
        ...

    @abstractmethod
    async def get(self, chunk_id: UUID) -> Optional[Chunk]:
        """Retrieve a chunk by ID."""
        ...

    @abstractmethod
    async def get_by_document(self, document_id: UUID) -> List[Chunk]:
        """Retrieve all chunks for a document."""
        ...

    @abstractmethod
    async def delete(self, chunk_id: UUID) -> None:
        """Delete a chunk."""
        ...

    @abstractmethod
    def iter_all(self) -> Iterator[Chunk]:
        """Iterate over all chunks."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Count total chunks."""
        ...
```

#### EntityStore

```python
from graphunified.config.models import Entity

class EntityStore(ABC):
    """Interface for entity storage."""

    @abstractmethod
    async def save(self, entities: List[Entity]) -> None:
        """Save entities to storage."""
        ...

    @abstractmethod
    async def get(self, entity_id: UUID) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        ...

    @abstractmethod
    async def get_by_name(
        self,
        name: str,
        entity_type: Optional[str] = None
    ) -> Optional[Entity]:
        """Retrieve entity by name and optional type."""
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Entity]:
        """Search entities by text similarity."""
        ...

    @abstractmethod
    def iter_all(self) -> Iterator[Entity]:
        """Iterate over all entities."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Count total entities."""
        ...
```

#### VectorStore

```python
import numpy as np
from typing import List, Tuple

class VectorStore(ABC):
    """Interface for vector storage and search."""

    @abstractmethod
    async def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add vectors to the index.

        Args:
            ids: Unique identifiers for vectors
            embeddings: Array of shape (n, dimension)
            metadata: Optional metadata for each vector
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector of shape (dimension,)
            top_k: Number of results
            filter: Optional metadata filter

        Returns:
            List of (id, score) tuples, sorted by descending score
        """
        ...

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Count total vectors in index."""
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        ...
```

#### GraphStore

```python
import networkx as nx
from graphunified.config.models import Relationship

class GraphStore(ABC):
    """Interface for graph storage."""

    @abstractmethod
    async def add_edges(self, relationships: List[Relationship]) -> None:
        """Add relationships as graph edges."""
        ...

    @abstractmethod
    async def get_graph(self) -> nx.Graph:
        """Return full graph (or subgraph)."""
        ...

    @abstractmethod
    async def get_neighbors(
        self,
        entity_id: UUID,
        max_hops: int = 1
    ) -> List[UUID]:
        """Get neighboring entity IDs within max_hops."""
        ...

    @abstractmethod
    async def get_subgraph(
        self,
        entity_ids: List[UUID]
    ) -> nx.Graph:
        """Extract subgraph containing specified entities."""
        ...

    @abstractmethod
    async def save(self, filepath: str) -> None:
        """Save graph to file (GraphML format)."""
        ...

    @abstractmethod
    async def load(self, filepath: str) -> None:
        """Load graph from file."""
        ...
```

---

## Public API

### Indexing API

```python
from pathlib import Path
from typing import Optional, Callable
from graphunified.config.settings import Settings

class Indexer:
    """High-level indexing API."""

    def __init__(self, settings: Settings):
        """
        Initialize indexer.

        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self._pipeline = None

    async def index(
        self,
        input_path: Path,
        strategies: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Index documents and build retrieval indexes.

        Args:
            input_path: Directory containing documents
            strategies: List of strategies to build indexes for.
                       If None, builds for all configured strategies.
            progress_callback: Optional callback(stage_name, progress_pct)

        Returns:
            Dictionary with indexing statistics:
                - document_count (int)
                - chunk_count (int)
                - entity_count (int)
                - relationship_count (int)
                - duration_seconds (float)
                - cost_usd (float)

        Raises:
            FileNotFoundError: input_path does not exist
            ValueError: Invalid strategies specified
            RuntimeError: Indexing failed

        Example:
            ```python
            settings = Settings.load("settings.yaml")
            indexer = Indexer(settings)

            stats = await indexer.index(
                Path("./documents"),
                strategies=["naive", "graphrag_local"],
                progress_callback=lambda stage, pct: print(f"{stage}: {pct:.1f}%")
            )

            print(f"Indexed {stats['document_count']} documents")
            print(f"Extracted {stats['entity_count']} entities")
            ```
        """
        ...

    async def update(
        self,
        document_ids: Optional[List[UUID]] = None,
        incremental: bool = True
    ) -> Dict[str, Any]:
        """
        Update existing index with new or modified documents.

        Args:
            document_ids: Specific documents to update. If None, scans for changes.
            incremental: If True, only reprocess changed documents

        Returns:
            Update statistics

        Raises:
            RuntimeError: Update failed
        """
        ...

    async def delete_documents(self, document_ids: List[UUID]) -> None:
        """
        Delete documents and associated data.

        Args:
            document_ids: Documents to delete

        Raises:
            RuntimeError: Deletion failed
        """
        ...
```

---

### Query API

```python
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class QueryResult:
    """Result from a query."""
    response: str
    contexts: List[Dict[str, Any]]
    strategy: str
    latency_ms: int
    tokens_used: int
    cost_usd: float

class Querier:
    """High-level query API."""

    def __init__(self, settings: Settings):
        """
        Initialize querier.

        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self._retrievers = {}

    async def query(
        self,
        query: str,
        strategy: Optional[str] = None,
        top_k: int = 10,
        generate: bool = True,
        **kwargs: Any
    ) -> QueryResult:
        """
        Query the indexed knowledge base.

        Args:
            query: User query string
            strategy: Retrieval strategy name. If None, uses router.
            top_k: Number of contexts to retrieve
            generate: If True, generates response. If False, returns contexts only.
            **kwargs: Strategy-specific parameters

        Returns:
            QueryResult with response and metadata

        Raises:
            ValueError: Invalid query or strategy
            RuntimeError: Query failed

        Example:
            ```python
            querier = Querier(settings)

            # Auto-route to best strategy
            result = await querier.query("What causes climate change?")
            print(result.response)
            print(f"Used {result.strategy} strategy")

            # Explicit strategy
            result = await querier.query(
                "What causes climate change?",
                strategy="graphrag_local",
                top_k=5
            )

            # Retrieval only (no generation)
            result = await querier.query(
                "climate change",
                generate=False
            )
            for ctx in result.contexts:
                print(f"[{ctx['score']:.2f}] {ctx['text'][:100]}...")
            ```
        """
        ...

    async def compare_strategies(
        self,
        query: str,
        strategies: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, QueryResult]:
        """
        Compare query results across multiple strategies.

        Args:
            query: User query string
            strategies: List of strategies to compare. If None, uses all available.
            top_k: Number of contexts to retrieve per strategy

        Returns:
            Dictionary mapping strategy name to QueryResult

        Example:
            ```python
            results = await querier.compare_strategies(
                "What causes climate change?",
                strategies=["naive", "graphrag_local", "lightrag"]
            )

            for strategy, result in results.items():
                print(f"\n{strategy}:")
                print(f"  Latency: {result.latency_ms}ms")
                print(f"  Cost: ${result.cost_usd:.4f}")
                print(f"  Response: {result.response[:100]}...")
            ```
        """
        ...

    def get_available_strategies(self) -> List[str]:
        """Return list of available strategies (indexes built)."""
        ...
```

---

### Prompt Tuning API

```python
from pathlib import Path
from typing import List, Dict

@dataclass
class EvaluationResult:
    """Prompt evaluation metrics."""
    precision: float
    recall: float
    f1: float
    entity_count: int
    relationship_count: int

class PromptTuner:
    """Prompt tuning API."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def tune(
        self,
        sample_documents: List[Path],
        ground_truth: Dict[str, Any],
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Auto-tune extraction prompts for domain.

        Args:
            sample_documents: Representative sample documents
            ground_truth: Manual entity/relationship labels
            iterations: Number of tuning iterations

        Returns:
            Tuned prompt templates and evaluation metrics

        Example:
            ```python
            tuner = PromptTuner(settings)

            results = await tuner.tune(
                sample_documents=[Path("sample1.txt"), Path("sample2.txt")],
                ground_truth={
                    "sample1.txt": {
                        "entities": [
                            {"name": "IPCC", "type": "ORGANIZATION"},
                            ...
                        ],
                        "relationships": [
                            {"source": "IPCC", "target": "Climate Report", "type": "PUBLISHED"}
                        ]
                    }
                },
                iterations=3
            )

            print(f"Baseline F1: {results['baseline_f1']:.2f}")
            print(f"Tuned F1: {results['tuned_f1']:.2f}")
            print(f"Improvement: {results['improvement_pct']:.1f}%")
            ```
        """
        ...

    async def evaluate(
        self,
        sample_documents: List[Path],
        ground_truth: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate current extraction prompts.

        Args:
            sample_documents: Documents to evaluate on
            ground_truth: Manual labels

        Returns:
            Evaluation metrics
        """
        ...
```

---

## Error Handling

### Exception Hierarchy

```python
class GraphUnifiedError(Exception):
    """Base exception for all Graph-Unified errors."""
    pass

class ConfigurationError(GraphUnifiedError):
    """Configuration is invalid or missing."""
    pass

class IndexingError(GraphUnifiedError):
    """Error during indexing pipeline."""
    pass

class QueryError(GraphUnifiedError):
    """Error during query execution."""
    pass

class StorageError(GraphUnifiedError):
    """Error in storage layer."""
    pass

class APIError(GraphUnifiedError):
    """Error calling external API (LLM, embedding)."""
    pass

class ValidationError(GraphUnifiedError):
    """Data validation failed."""
    pass
```

### Error Handling Pattern

```python
from graphunified.exceptions import GraphUnifiedError, IndexingError

try:
    indexer = Indexer(settings)
    stats = await indexer.index(Path("./docs"))
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Fix settings.yaml
except APIError as e:
    print(f"API error: {e}")
    # Check API keys, rate limits
except IndexingError as e:
    print(f"Indexing failed: {e}")
    # Check input documents, storage permissions
except GraphUnifiedError as e:
    print(f"Unexpected error: {e}")
    # Generic error handling
```

---

## Usage Examples

### Complete Indexing Workflow

```python
import asyncio
from pathlib import Path
from graphunified.config.settings import Settings
from graphunified.index.indexer import Indexer
from graphunified.query.querier import Querier

async def main():
    # Load configuration
    settings = Settings.load("settings.yaml")

    # Index documents
    indexer = Indexer(settings)

    print("Starting indexing...")
    stats = await indexer.index(
        input_path=Path("./documents"),
        strategies=["naive", "hybrid", "graphrag_local"],
        progress_callback=lambda stage, pct: print(f"{stage}: {pct:.0f}%")
    )

    print(f"\nIndexing complete!")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Chunks: {stats['chunk_count']}")
    print(f"  Entities: {stats['entity_count']}")
    print(f"  Relationships: {stats['relationship_count']}")
    print(f"  Duration: {stats['duration_seconds']:.1f}s")
    print(f"  Cost: ${stats['cost_usd']:.2f}")

    # Query with different strategies
    querier = Querier(settings)

    queries = [
        "What causes climate change?",
        "Who are the key organizations?",
        "Summarize the climate policy landscape"
    ]

    for query_text in queries:
        print(f"\n\nQuery: {query_text}")

        # Compare strategies
        results = await querier.compare_strategies(
            query_text,
            strategies=["naive", "graphrag_local"]
        )

        for strategy, result in results.items():
            print(f"\n{strategy.upper()}:")
            print(f"  Latency: {result.latency_ms}ms")
            print(f"  Response: {result.response[:200]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Retriever Implementation

```python
from typing import List, Dict, Any
from graphunified.query.retrievers.base import Retriever
from graphunified.storage.vector_store import VectorStore

class CustomRetriever:
    """Example custom retriever."""

    def __init__(self, vector_store: VectorStore, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.config = config

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Custom retrieval logic."""

        # Embed query
        query_embedding = await self._embed_query(query)

        # Search vectors
        results = await self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        # Format results
        contexts = []
        for chunk_id, score in results:
            chunk = await self._load_chunk(chunk_id)
            contexts.append({
                "text": chunk.text,
                "score": score,
                "source": str(chunk.id),
                "metadata": chunk.metadata
            })

        return contexts

    def get_name(self) -> str:
        return "custom"

    def get_config(self) -> Dict[str, Any]:
        return self.config
```

---

## Type Stubs

For better IDE support, provide type stubs:

```python
# graphunified/__init__.pyi
from .config.settings import Settings as Settings
from .index.indexer import Indexer as Indexer
from .query.querier import Querier as Querier, QueryResult as QueryResult
from .exceptions import (
    GraphUnifiedError as GraphUnifiedError,
    ConfigurationError as ConfigurationError,
    IndexingError as IndexingError,
    QueryError as QueryError,
)

__version__: str
```

---

## API Versioning

Version the API using semantic versioning:

```python
# graphunified/__init__.py
__version__ = "1.0.0"

# Breaking changes increment major version
# New features increment minor version
# Bug fixes increment patch version
```

---

## Summary

This specification defines:

- **3 core protocols:** Retriever, PipelineStage, StorageInterface
- **4 storage interfaces:** ChunkStore, EntityStore, VectorStore, GraphStore
- **3 high-level APIs:** Indexer, Querier, PromptTuner
- **Exception hierarchy** for error handling
- **Complete usage examples** for common workflows
- **Type hints** for all public methods
- **Async-first design** for I/O operations

**Implementation Files:**
- `graphunified/protocols.py` - Protocol definitions
- `graphunified/index/indexer.py` - Indexer API
- `graphunified/query/querier.py` - Querier API
- `graphunified/prompt_tune/tuner.py` - PromptTuner API
- `graphunified/storage/*.py` - Storage interfaces
- `graphunified/exceptions.py` - Exception classes

# Module Architecture

**Version:** 1.0
**Last Updated:** 2026-02-15
**Status:** Technical Specification

## Overview

This document specifies the internal module structure, dependencies, class responsibilities, and interaction patterns for Graph-Unified. It provides the blueprint for organizing code and managing complexity.

## Package Structure

```
graphunified/
├── __init__.py                    # Public API exports
├── __version__.py                 # Version string
├── cli.py                         # Click CLI entry point
│
├── config/                        # Configuration management
│   ├── __init__.py
│   ├── models.py                  # Pydantic data models
│   ├── settings.py                # Settings schema and loading
│   ├── defaults.py                # Default configurations
│   └── validation.py              # Config validation logic
│
├── storage/                       # Storage layer
│   ├── __init__.py
│   ├── base.py                    # Storage interfaces (ABC)
│   ├── parquet_store.py           # Parquet operations
│   ├── vector_store.py            # Vector DB wrapper (LanceDB/FAISS)
│   ├── graph_store.py             # Graph storage (NetworkX)
│   ├── schemas.py                 # PyArrow schemas
│   └── migrations.py              # Schema migration logic
│
├── index/                         # Indexing pipeline
│   ├── __init__.py
│   ├── indexer.py                 # High-level Indexer API
│   ├── pipeline.py                # DAG orchestrator
│   ├── stages/                    # Pipeline stages
│   │   ├── __init__.py
│   │   ├── base.py                # PipelineStage protocol
│   │   ├── load.py                # Document loading
│   │   ├── chunk.py               # Chunking stage
│   │   ├── extract.py             # Entity/relationship extraction
│   │   ├── embed.py               # Embedding generation
│   │   └── build.py               # Index building dispatcher
│   └── strategies/                # Strategy-specific index builders
│       ├── __init__.py
│       ├── naive.py               # Naive: vector index only
│       ├── hybrid.py              # Hybrid: vector + BM25
│       ├── graphrag.py            # GraphRAG: communities + summaries
│       ├── lightrag.py            # LightRAG: dual entity/relation indexes
│       └── hipporag.py            # HippoRAG: associative graph
│
├── query/                         # Query pipeline
│   ├── __init__.py
│   ├── querier.py                 # High-level Querier API
│   ├── router.py                  # Query routing logic
│   ├── generator.py               # Response generation (Claude)
│   └── retrievers/                # Retrieval strategies
│       ├── __init__.py
│       ├── base.py                # Retriever protocol
│       ├── naive.py               # Naive retriever
│       ├── hybrid.py              # Hybrid retriever
│       ├── graphrag_local.py      # GraphRAG local search
│       ├── graphrag_global.py     # GraphRAG global search
│       ├── lightrag.py            # LightRAG retriever
│       └── hipporag.py            # HippoRAG retriever
│
├── prompt_tune/                   # Prompt tuning
│   ├── __init__.py
│   ├── tuner.py                   # PromptTuner API
│   ├── templates.py               # Base prompt templates
│   ├── generator.py               # Prompt generation logic
│   └── evaluator.py               # Extraction evaluation
│
├── utils/                         # Shared utilities
│   ├── __init__.py
│   ├── llm.py                     # Claude API client
│   ├── embedding.py               # Embedding model wrapper
│   ├── tokenizer.py               # Tokenization utilities
│   ├── graph_utils.py             # Graph algorithms
│   ├── metrics.py                 # Evaluation metrics
│   ├── logging.py                 # Logging configuration
│   └── progress.py                # Progress tracking
│
├── protocols.py                   # Shared protocol definitions
└── exceptions.py                  # Exception hierarchy
```

---

## Module Dependencies

### Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                         │
└───────────┬─────────────────────────┬───────────────────────┘
            │                         │
    ┌───────▼────────┐       ┌────────▼─────────┐
    │  Indexer API   │       │   Querier API    │
    │ (index/indexer)│       │ (query/querier)  │
    └───────┬────────┘       └────────┬─────────┘
            │                         │
    ┌───────▼────────┐       ┌────────▼─────────┐
    │   Pipeline     │       │  Router          │
    │ (index/pipeline)│       │ (query/router)   │
    └───────┬────────┘       └────────┬─────────┘
            │                         │
    ┌───────▼────────┐       ┌────────▼─────────┐
    │  Stages        │       │  Retrievers      │
    │ (index/stages/)│       │ (query/retrievers)│
    └───────┬────────┘       └────────┬─────────┘
            │                         │
            │        ┌────────────────┴─────────────┐
            │        │                              │
    ┌───────▼────────▼─────┐              ┌────────▼─────────┐
    │  Storage Layer       │              │  Utils           │
    │ (storage/)           │              │ (utils/)         │
    └───────┬──────────────┘              └──────────────────┘
            │
    ┌───────▼────────┐
    │  Config        │
    │ (config/)      │
    └────────────────┘
```

### Dependency Rules

1. **Layered Architecture:**
   - CLI → API → Core → Storage → Config
   - Higher layers depend on lower layers
   - Lower layers never import from higher layers

2. **No Circular Dependencies:**
   - Use dependency injection to break cycles
   - Use protocols for interface-based dependencies

3. **Shared Modules:**
   - `config/` - No dependencies (except Pydantic)
   - `storage/` - Depends on `config/`
   - `utils/` - Depends on `config/`, no dependency on `index/` or `query/`
   - `protocols.py` - No dependencies
   - `exceptions.py` - No dependencies

---

## Core Classes

### Configuration Layer

#### Settings

```python
# config/settings.py
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_validator
import yaml

class Settings(BaseModel):
    """Root configuration."""

    version: str
    llm: LLMConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    extraction: ExtractionConfig
    strategies: StrategiesConfig
    storage: StorageConfig
    query: QueryConfig
    performance: PerformanceConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load settings from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save settings to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f)

    def validate_completeness(self) -> None:
        """Validate that all required fields are present."""
        ...
```

**Responsibilities:**
- Load/save configuration
- Validate configuration schema
- Provide defaults for optional fields
- Environment variable substitution

**Dependencies:**
- Pydantic for validation
- PyYAML for serialization

---

### Storage Layer

#### ParquetStore

```python
# storage/parquet_store.py
from pathlib import Path
from typing import List, Iterator, Optional
import pyarrow.parquet as pq
from graphunified.config.models import Chunk, Entity, Relationship

class ParquetStore:
    """Parquet-based storage for structured data."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    async def save_chunks(self, chunks: List[Chunk]) -> None:
        """Save chunks to parquet."""
        ...

    async def load_chunks(self) -> Iterator[Chunk]:
        """Load all chunks."""
        ...

    async def save_entities(self, entities: List[Entity]) -> None:
        """Save entities to parquet."""
        ...

    async def load_entities(self) -> Iterator[Entity]:
        """Load all entities."""
        ...

    # Similar for relationships, communities, etc.
```

**Responsibilities:**
- Serialize/deserialize Pydantic models to Parquet
- Batch writing for efficiency
- Incremental appends
- Schema validation

**Dependencies:**
- PyArrow for Parquet I/O
- `config.models` for data models

---

#### VectorStore

```python
# storage/vector_store.py
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import lancedb

class VectorStore:
    """LanceDB-based vector storage."""

    def __init__(self, db_path: str, table_name: str, dimension: int):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.dimension = dimension
        self._table = None

    async def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add vectors to index."""
        ...

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        ...

    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs."""
        ...
```

**Responsibilities:**
- Vector indexing and search
- Metadata filtering
- Batch operations
- Index persistence

**Dependencies:**
- LanceDB for vector operations
- NumPy for arrays

---

#### GraphStore

```python
# storage/graph_store.py
from typing import List
from uuid import UUID
import networkx as nx
from graphunified.config.models import Entity, Relationship

class GraphStore:
    """NetworkX-based graph storage."""

    def __init__(self):
        self.graph = nx.DiGraph()

    async def add_entities(self, entities: List[Entity]) -> None:
        """Add entities as nodes."""
        for entity in entities:
            self.graph.add_node(
                str(entity.id),
                name=entity.name,
                type=entity.type,
                description=entity.description
            )

    async def add_relationships(self, relationships: List[Relationship]) -> None:
        """Add relationships as edges."""
        for rel in relationships:
            self.graph.add_edge(
                str(rel.source_entity_id),
                str(rel.target_entity_id),
                type=rel.type,
                weight=rel.weight,
                description=rel.description
            )

    async def get_neighbors(self, entity_id: UUID, max_hops: int = 1) -> List[UUID]:
        """Get neighboring entities within max_hops."""
        ...

    async def save(self, filepath: str) -> None:
        """Save graph to GraphML."""
        nx.write_graphml(self.graph, filepath)

    async def load(self, filepath: str) -> None:
        """Load graph from GraphML."""
        self.graph = nx.read_graphml(filepath)
```

**Responsibilities:**
- Graph construction from entities/relationships
- Graph traversal operations
- Community detection
- Persistence to GraphML

**Dependencies:**
- NetworkX for graph algorithms
- `config.models` for data models

---

### Indexing Layer

#### Pipeline

```python
# index/pipeline.py
from typing import List, Dict, Any, Callable
import asyncio
from graphunified.index.stages.base import PipelineStage, StageResult

class Pipeline:
    """Async DAG-based pipeline orchestrator."""

    def __init__(self, stages: List[PipelineStage], config: Dict[str, Any]):
        self.stages = stages
        self.config = config
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        """Ensure all stage dependencies are satisfied."""
        ...

    async def execute(
        self,
        input_data: Any,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, StageResult]:
        """Execute pipeline stages in dependency order."""
        results = {}
        context = {"config": self.config}

        for stage in self._topological_sort():
            stage_name = stage.get_name()

            # Get input from previous stages
            stage_input = self._prepare_input(stage, results)

            # Execute stage
            if progress_callback:
                progress_callback(stage_name, 0.0)

            result = await stage.execute(stage_input, context)
            results[stage_name] = result

            if progress_callback:
                progress_callback(stage_name, 100.0)

        return results

    def _topological_sort(self) -> List[PipelineStage]:
        """Sort stages by dependencies."""
        ...

    def _prepare_input(
        self,
        stage: PipelineStage,
        results: Dict[str, StageResult]
    ) -> Any:
        """Prepare input data for stage from previous results."""
        ...
```

**Responsibilities:**
- Orchestrate pipeline execution
- Resolve stage dependencies
- Parallel execution where possible
- Error handling and recovery
- Progress tracking

**Dependencies:**
- `index.stages.base` for PipelineStage protocol
- asyncio for concurrency

---

#### ChunkStage

```python
# index/stages/chunk.py
from typing import List
from graphunified.config.models import Document, Chunk
from graphunified.index.stages.base import PipelineStage, StageResult
from graphunified.utils.tokenizer import Tokenizer

class ChunkStage:
    """Document chunking stage."""

    def __init__(self, chunk_size: int, overlap: int, tokenizer: Tokenizer):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer

    async def execute(self, input_data: List[Document], context: dict) -> StageResult:
        """Chunk documents into overlapping chunks."""
        chunks = []
        for doc in input_data:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)

        return StageResult(
            status="success",
            data=chunks,
            metadata={"chunk_count": len(chunks)}
        )

    def _chunk_document(self, doc: Document) -> List[Chunk]:
        """Chunk a single document."""
        ...

    def get_name(self) -> str:
        return "chunk"

    def requires(self) -> List[str]:
        return ["load"]

    def provides(self) -> str:
        return "chunks"
```

**Responsibilities:**
- Tokenize documents
- Create overlapping chunks
- Track character positions
- Preserve chunk metadata

**Dependencies:**
- `utils.tokenizer` for tokenization
- `config.models` for Document/Chunk

---

#### ExtractStage

```python
# index/stages/extract.py
from typing import List, Tuple
from graphunified.config.models import Chunk, Entity, Relationship
from graphunified.index.stages.base import PipelineStage, StageResult
from graphunified.utils.llm import LLMClient

class ExtractStage:
    """Entity and relationship extraction stage."""

    def __init__(self, llm_client: LLMClient, prompt_template: str):
        self.llm = llm_client
        self.prompt_template = prompt_template

    async def execute(self, input_data: List[Chunk], context: dict) -> StageResult:
        """Extract entities and relationships from chunks."""
        all_entities = []
        all_relationships = []

        # Batch process chunks
        batches = self._create_batches(input_data, batch_size=10)

        for batch in batches:
            entities, relationships = await self._extract_batch(batch)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # Deduplicate entities
        deduped_entities = self._deduplicate_entities(all_entities)

        return StageResult(
            status="success",
            data=(deduped_entities, all_relationships),
            metadata={
                "entity_count": len(deduped_entities),
                "relationship_count": len(all_relationships)
            }
        )

    async def _extract_batch(
        self,
        chunks: List[Chunk]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract from batch of chunks."""
        ...

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge duplicate entities."""
        ...

    def get_name(self) -> str:
        return "extract"

    def requires(self) -> List[str]:
        return ["chunk"]

    def provides(self) -> str:
        return "entities_relationships"
```

**Responsibilities:**
- Prompt LLM for extraction
- Parse JSON responses
- Batch processing
- Entity deduplication
- Error handling for malformed outputs

**Dependencies:**
- `utils.llm` for Claude API
- `config.models` for Entity/Relationship

---

### Query Layer

#### QueryRouter

```python
# query/router.py
from typing import Optional
from graphunified.utils.llm import LLMClient

class QueryRouter:
    """Route queries to optimal retrieval strategy."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    async def route(self, query: str) -> str:
        """
        Determine optimal strategy for query.

        Returns strategy name: 'naive', 'hybrid', 'graphrag_local', etc.
        """
        # Rule-based routing
        if self._is_factual_lookup(query):
            return "naive"

        if self._has_entity_focus(query):
            return "graphrag_local"

        if self._is_summarization(query):
            return "graphrag_global"

        if self._needs_multi_hop(query):
            return "lightrag"

        # Default fallback
        return "hybrid"

    def _is_factual_lookup(self, query: str) -> bool:
        """Detect simple factual queries."""
        ...

    def _has_entity_focus(self, query: str) -> bool:
        """Detect entity-centric queries."""
        ...

    def _is_summarization(self, query: str) -> bool:
        """Detect summarization queries."""
        ...

    def _needs_multi_hop(self, query: str) -> bool:
        """Detect multi-hop reasoning queries."""
        ...
```

**Responsibilities:**
- Query type detection
- Strategy selection
- Heuristic-based routing
- Optional LLM-based routing

**Dependencies:**
- `utils.llm` for LLM-based routing (optional)

---

#### NaiveRetriever

```python
# query/retrievers/naive.py
from typing import List, Dict, Any
import numpy as np
from graphunified.storage.vector_store import VectorStore
from graphunified.storage.parquet_store import ParquetStore
from graphunified.utils.embedding import EmbeddingModel

class NaiveRetriever:
    """Naive vector similarity retrieval."""

    def __init__(
        self,
        vector_store: VectorStore,
        chunk_store: ParquetStore,
        embedding_model: EmbeddingModel
    ):
        self.vector_store = vector_store
        self.chunk_store = chunk_store
        self.embedding_model = embedding_model

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks by vector similarity."""
        # Embed query
        query_embedding = await self.embedding_model.embed([query])

        # Search vector store
        results = await self.vector_store.search(
            query_embedding[0],
            top_k=top_k
        )

        # Load chunk details
        contexts = []
        for chunk_id, score in results:
            chunk = await self.chunk_store.get_chunk(chunk_id)
            contexts.append({
                "text": chunk.text,
                "score": float(score),
                "source": str(chunk.id),
                "metadata": chunk.metadata
            })

        return contexts

    def get_name(self) -> str:
        return "naive"

    def get_config(self) -> Dict[str, Any]:
        return {"type": "naive", "vector_only": True}
```

**Responsibilities:**
- Embed query
- Vector similarity search
- Load chunk details
- Format results

**Dependencies:**
- `storage.vector_store` for vector search
- `storage.parquet_store` for chunk loading
- `utils.embedding` for query embedding

---

## Interaction Patterns

### Dependency Injection

Use constructor injection for dependencies:

```python
# Good: Dependencies injected
class ExtractStage:
    def __init__(self, llm_client: LLMClient, config: ExtractionConfig):
        self.llm = llm_client
        self.config = config

# Bad: Hard-coded dependencies
class ExtractStage:
    def __init__(self):
        self.llm = LLMClient(api_key="hardcoded")  # ❌
```

### Factory Pattern

Use factories to construct complex objects:

```python
# index/factory.py
from graphunified.config.settings import Settings
from graphunified.index.indexer import Indexer
from graphunified.storage.parquet_store import ParquetStore
from graphunified.storage.vector_store import VectorStore
from graphunified.utils.llm import LLMClient

class IndexerFactory:
    """Factory for creating Indexer instances."""

    @staticmethod
    def create(settings: Settings) -> Indexer:
        """Create fully-configured Indexer."""
        # Create storage
        parquet_store = ParquetStore(settings.storage.root_dir)
        vector_store = VectorStore(
            settings.storage.vector_db_path,
            "chunks",
            settings.embedding.dimension
        )

        # Create LLM client
        llm_client = LLMClient(
            api_key=settings.llm.api_key,
            model=settings.llm.model,
            rate_limit=settings.llm.rate_limit
        )

        # Create stages
        stages = [
            LoadStage(settings.storage.root_dir),
            ChunkStage(
                settings.chunking.chunk_size,
                settings.chunking.overlap,
                Tokenizer(settings.chunking.encoding_name)
            ),
            ExtractStage(llm_client, settings.extraction),
            # ... more stages
        ]

        # Create pipeline
        pipeline = Pipeline(stages, settings.model_dump())

        return Indexer(settings, pipeline, parquet_store)
```

---

## Configuration Management Flow

```
User calls CLI
    ↓
CLI loads settings.yaml
    ↓
Settings.load() validates config
    ↓
Factory creates components with config
    ↓
Components use config for behavior
```

**Example:**

```python
# cli.py
@click.command()
@click.option("--config", default="settings.yaml")
def index(config: str):
    # Load settings
    settings = Settings.load(Path(config))

    # Create indexer via factory
    indexer = IndexerFactory.create(settings)

    # Run indexing
    asyncio.run(indexer.index(Path("./docs")))
```

---

## Testing Strategy

### Unit Tests

Test individual classes in isolation:

```python
# tests/storage/test_parquet_store.py
import pytest
from graphunified.storage.parquet_store import ParquetStore
from graphunified.config.models import Chunk

@pytest.mark.asyncio
async def test_save_and_load_chunks(tmp_path):
    store = ParquetStore(tmp_path)

    chunks = [Chunk(...), Chunk(...)]
    await store.save_chunks(chunks)

    loaded = list(await store.load_chunks())
    assert len(loaded) == len(chunks)
```

### Integration Tests

Test interactions between components:

```python
# tests/integration/test_pipeline.py
import pytest
from graphunified.index.pipeline import Pipeline
from graphunified.index.stages import LoadStage, ChunkStage

@pytest.mark.asyncio
async def test_pipeline_execution():
    stages = [LoadStage(...), ChunkStage(...)]
    pipeline = Pipeline(stages, {})

    results = await pipeline.execute(input_path)

    assert "load" in results
    assert "chunk" in results
    assert results["chunk"].status == "success"
```

---

## Summary

This specification defines:

- **Package structure** with clear module organization
- **Dependency graph** showing import relationships
- **Core class responsibilities** for each module
- **Interaction patterns** (DI, factories)
- **Configuration flow** from CLI to components
- **Testing strategy** for unit and integration tests

**Key Principles:**
- Layered architecture (CLI → API → Core → Storage → Config)
- No circular dependencies
- Dependency injection for testability
- Factory pattern for construction
- Protocol-based interfaces

**Implementation Order:**
1. `config/` - Configuration and data models
2. `storage/` - Storage layer
3. `utils/` - Shared utilities
4. `index/stages/` - Individual stages
5. `index/pipeline.py` - Pipeline orchestrator
6. `query/retrievers/` - Retrievers
7. `index/indexer.py`, `query/querier.py` - High-level APIs
8. `cli.py` - CLI interface

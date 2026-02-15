"""Configuration settings schema with environment variable substitution."""

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from graphunified.config import defaults
from graphunified.config.validation import substitute_env_vars, validate_api_key
from graphunified.exceptions import ConfigurationError


class RateLimitConfig(BaseModel):
    """Rate limiting configuration for API calls."""

    requests_per_minute: int = Field(defaults.DEFAULT_LLM_RPM, ge=1, le=1000)
    tokens_per_minute: int = Field(defaults.DEFAULT_LLM_TPM, ge=1000)


class RetryConfig(BaseModel):
    """Retry configuration for failed API calls."""

    max_attempts: int = Field(defaults.DEFAULT_RETRY_MAX_ATTEMPTS, ge=1, le=10)
    backoff_factor: float = Field(defaults.DEFAULT_RETRY_BACKOFF, ge=1.0, le=10.0)


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: Literal["anthropic", "openai", "azure"] = defaults.DEFAULT_LLM_PROVIDER
    model: str = defaults.DEFAULT_LLM_MODEL
    api_key: str = Field(..., min_length=1)
    temperature: float = Field(defaults.DEFAULT_LLM_TEMPERATURE, ge=0.0, le=2.0)
    max_tokens: int = Field(defaults.DEFAULT_LLM_MAX_TOKENS, ge=100, le=200000)
    timeout: int = Field(defaults.DEFAULT_LLM_TIMEOUT, ge=10, le=600)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @field_validator("api_key")
    @classmethod
    def validate_api_key_field(cls, v: str) -> str:
        validate_api_key(v, "LLM")
        return v


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""

    provider: Literal["voyage", "openai", "cohere", "local"] = defaults.DEFAULT_EMBEDDING_PROVIDER
    model: str = defaults.DEFAULT_EMBEDDING_MODEL
    api_key: str = Field(default="")  # Optional for local embeddings
    dimension: int = Field(defaults.DEFAULT_EMBEDDING_DIMENSION, ge=384, le=4096)
    batch_size: int = Field(defaults.DEFAULT_EMBEDDING_BATCH_SIZE, ge=1, le=512)
    normalize: bool = defaults.DEFAULT_EMBEDDING_NORMALIZE

    @field_validator("api_key")
    @classmethod
    def validate_api_key_field(cls, v: str, info: any) -> str:
        # Skip validation for local embeddings (no API key needed)
        if "provider" in info.data and info.data["provider"] in ["local"]:
            return v
        validate_api_key(v, "Embedding")
        return v


class ChunkingConfig(BaseModel):
    """Chunking configuration."""

    strategy: Literal["fixed", "sentence", "paragraph", "semantic"] = (
        defaults.DEFAULT_CHUNKING_STRATEGY
    )
    chunk_size: int = Field(defaults.DEFAULT_CHUNK_SIZE, ge=128, le=4096)
    chunk_overlap: int = Field(defaults.DEFAULT_CHUNK_OVERLAP, ge=0, le=512)
    respect_boundaries: bool = defaults.DEFAULT_RESPECT_BOUNDARIES
    encoding_name: str = defaults.DEFAULT_ENCODING_NAME

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info: any) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be < chunk_size")
        return v


class ExtractionConfig(BaseModel):
    """Entity and relationship extraction configuration."""

    entity_types: List[str] = Field(default_factory=lambda: defaults.DEFAULT_ENTITY_TYPES.copy())
    relationship_types: List[str] = Field(
        default_factory=lambda: defaults.DEFAULT_RELATIONSHIP_TYPES.copy()
    )
    max_gleanings: int = Field(defaults.DEFAULT_MAX_GLEANINGS, ge=0, le=3)
    min_confidence: float = Field(defaults.DEFAULT_MIN_CONFIDENCE, ge=0.0, le=1.0)
    enable_coreference: bool = defaults.DEFAULT_ENABLE_COREFERENCE


class IndexingConfig(BaseModel):
    """Indexing pipeline configuration."""

    chunk_size: int = Field(defaults.DEFAULT_INDEXING_CHUNK_SIZE, ge=128, le=4096)
    chunk_overlap: int = Field(defaults.DEFAULT_INDEXING_CHUNK_OVERLAP, ge=0, le=512)
    extraction_batch_size: int = Field(defaults.DEFAULT_INDEXING_EXTRACTION_BATCH_SIZE, ge=1, le=50)
    dedup_threshold: int = Field(defaults.DEFAULT_INDEXING_DEDUP_THRESHOLD, ge=50, le=100)
    max_concurrent: int = Field(defaults.DEFAULT_INDEXING_MAX_CONCURRENT, ge=1, le=50)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info: any) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be < chunk_size")
        return v


class NaiveStrategyConfig(BaseModel):
    """Naive RAG strategy configuration."""

    enabled: bool = True


class HybridStrategyConfig(BaseModel):
    """Hybrid RAG strategy configuration."""

    enabled: bool = True
    alpha: float = Field(defaults.DEFAULT_HYBRID_ALPHA, ge=0.0, le=1.0)
    bm25_k1: float = Field(defaults.DEFAULT_BM25_K1, ge=0.5, le=3.0)
    bm25_b: float = Field(defaults.DEFAULT_BM25_B, ge=0.0, le=1.0)


class GraphRAGStrategyConfig(BaseModel):
    """GraphRAG strategy configuration."""

    enabled: bool = True
    leiden_resolution: float = Field(defaults.DEFAULT_LEIDEN_RESOLUTION, ge=0.1, le=2.0)
    max_community_size: int = Field(defaults.DEFAULT_MAX_COMMUNITY_SIZE, ge=5, le=100)
    generate_reports: bool = defaults.DEFAULT_GENERATE_REPORTS


class LightRAGStrategyConfig(BaseModel):
    """LightRAG strategy configuration."""

    enabled: bool = True
    entity_weight: float = Field(defaults.DEFAULT_LIGHTRAG_ENTITY_WEIGHT, ge=0.0, le=1.0)


class HippoRAGStrategyConfig(BaseModel):
    """HippoRAG strategy configuration."""

    enabled: bool = True
    ppr_alpha: float = Field(defaults.DEFAULT_HIPPORAG_PPR_ALPHA, ge=0.5, le=0.95)


class StrategiesConfig(BaseModel):
    """Configuration for all retrieval strategies."""

    naive: NaiveStrategyConfig = Field(default_factory=NaiveStrategyConfig)
    hybrid: HybridStrategyConfig = Field(default_factory=HybridStrategyConfig)
    graphrag: GraphRAGStrategyConfig = Field(default_factory=GraphRAGStrategyConfig)
    lightrag: LightRAGStrategyConfig = Field(default_factory=LightRAGStrategyConfig)
    hipporag: HippoRAGStrategyConfig = Field(default_factory=HippoRAGStrategyConfig)


class VectorDBConfig(BaseModel):
    """Vector database configuration."""

    backend: Literal["lancedb", "faiss", "qdrant"] = defaults.DEFAULT_VECTOR_DB_BACKEND
    index_type: Literal["IVF_FLAT", "IVF_PQ", "HNSW"] = defaults.DEFAULT_VECTOR_INDEX_TYPE
    chunk_index_name: str = defaults.DEFAULT_CHUNK_INDEX_NAME
    entity_index_name: str = defaults.DEFAULT_ENTITY_INDEX_NAME
    relationship_index_name: str = defaults.DEFAULT_RELATIONSHIP_INDEX_NAME
    fact_index_name: str = defaults.DEFAULT_FACT_INDEX_NAME
    community_index_name: str = defaults.DEFAULT_COMMUNITY_INDEX_NAME


class StorageConfig(BaseModel):
    """Storage configuration."""

    root_dir: Path = Field(default=Path(defaults.DEFAULT_STORAGE_ROOT))
    parquet_compression: Literal["snappy", "gzip", "brotli", "none"] = (
        defaults.DEFAULT_PARQUET_COMPRESSION
    )
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    graph_format: Literal["graphml", "gexf", "pickle"] = defaults.DEFAULT_GRAPH_FORMAT


class GenerationConfig(BaseModel):
    """Response generation configuration."""

    enabled: bool = defaults.DEFAULT_GENERATION_ENABLED
    temperature: float = Field(defaults.DEFAULT_GENERATION_TEMPERATURE, ge=0.0, le=1.0)
    max_tokens: int = defaults.DEFAULT_GENERATION_MAX_TOKENS


class RoutingConfig(BaseModel):
    """Query routing configuration."""

    enabled: bool = defaults.DEFAULT_ROUTING_ENABLED
    strategy: Literal["rule_based", "llm_based"] = defaults.DEFAULT_ROUTING_STRATEGY


class QueryConfig(BaseModel):
    """Query configuration."""

    default_strategy: Literal[
        "naive", "hybrid", "graphrag_local", "graphrag_global", "lightrag", "hipporag", "auto"
    ] = defaults.DEFAULT_QUERY_STRATEGY
    top_k: int = Field(defaults.DEFAULT_TOP_K, ge=1, le=100)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)


class PerformanceConfig(BaseModel):
    """Performance tuning configuration."""

    workers: int = Field(defaults.DEFAULT_WORKERS, ge=1, le=32)
    batch_size: int = Field(defaults.DEFAULT_BATCH_SIZE, ge=1, le=100)
    cache_embeddings: bool = defaults.DEFAULT_CACHE_EMBEDDINGS
    enable_profiling: bool = defaults.DEFAULT_ENABLE_PROFILING


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = defaults.DEFAULT_LOG_LEVEL
    format: Literal["text", "json"] = defaults.DEFAULT_LOG_FORMAT
    output: Literal["stdout", "file", "both"] = defaults.DEFAULT_LOG_OUTPUT
    file_path: Path = Field(default=Path(defaults.DEFAULT_LOG_FILE_PATH))


class Settings(BaseModel):
    """Root configuration settings."""

    version: str = Field("1.0", pattern=r"^\d+\.\d+$")
    llm: LLMConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, config_path: Path) -> "Settings":
        """Load settings from YAML file with environment variable substitution.

        Automatically loads environment variables from .env file if present.
        Search order for .env file:
        1. Same directory as config file
        2. Current working directory
        3. Project root (parent directories up to 3 levels)

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Settings instance with environment variables resolved

        Raises:
            ConfigurationError: If file not found or invalid
        """
        # Load .env file if present (searches multiple locations)
        cls._load_env_file(config_path)

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path) as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")

        if not isinstance(raw_config, dict):
            raise ConfigurationError(f"Configuration must be a YAML object, got {type(raw_config)}")

        # Substitute environment variables
        try:
            resolved_config = substitute_env_vars(raw_config)
        except ConfigurationError as e:
            raise ConfigurationError(f"Environment variable substitution failed: {e}")

        # Validate and construct Settings
        try:
            return cls(**resolved_config)
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")

    @staticmethod
    def _load_env_file(config_path: Path) -> None:
        """Load .env file from multiple possible locations.

        Search order:
        1. Same directory as config file
        2. Current working directory
        3. Parent directories (up to 3 levels)

        Args:
            config_path: Path to configuration file
        """
        from pathlib import Path

        search_paths = [
            # Same directory as config file
            config_path.parent / ".env",
            # Current working directory
            Path.cwd() / ".env",
        ]

        # Add parent directories (up to 3 levels up)
        current = Path.cwd()
        for _ in range(3):
            parent = current.parent
            if parent != current:  # Not at filesystem root
                search_paths.append(parent / ".env")
                current = parent

        # Load first .env file found
        for env_path in search_paths:
            if env_path.exists():
                load_dotenv(env_path, override=False)  # Don't override existing env vars
                break

    def validate_completeness(self) -> None:
        """Validate that all required fields are present and valid.

        Raises:
            ConfigurationError: If configuration is incomplete or invalid
        """
        # Check API keys are not placeholders
        if self.llm.api_key.startswith("${"):
            raise ConfigurationError("LLM API key not resolved from environment")

        # Skip embedding API key validation for local provider
        if self.embedding.provider != "local" and self.embedding.api_key.startswith("${"):
            raise ConfigurationError("Embedding API key not resolved from environment")

        # Validate storage directory
        if not self.storage.root_dir:
            raise ConfigurationError("Storage root_dir cannot be empty")

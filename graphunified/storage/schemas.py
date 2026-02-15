"""PyArrow schemas for Parquet storage."""

import pyarrow as pa

# Document Schema
DOCUMENT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),  # UUID as string
        pa.field("filename", pa.string()),
        pa.field("text", pa.string()),
        pa.field("metadata", pa.string()),  # JSON string
        pa.field("created_at", pa.timestamp("us")),
        pa.field("updated_at", pa.timestamp("us")),
        pa.field("char_count", pa.int32()),
        pa.field("token_count", pa.int32()),
    ]
)

# Chunk Schema
CHUNK_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("document_id", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("start_char", pa.int32()),
        pa.field("end_char", pa.int32()),
        pa.field("token_count", pa.int32()),
        pa.field("embedding", pa.list_(pa.float32())),  # Dense vector (1024d for Voyage)
        pa.field("embedding_model", pa.string()),  # Model identifier
        pa.field("entity_ids", pa.list_(pa.string())),  # Bidirectional graph links
        pa.field("relationship_ids", pa.list_(pa.string())),
        pa.field("metadata", pa.string()),  # JSON string
    ]
)

# Entity Schema
ENTITY_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("type", pa.string()),
        pa.field("description", pa.string()),
        pa.field("source_chunks", pa.list_(pa.string())),
        pa.field("extraction_confidence", pa.float32()),
        pa.field("embedding", pa.list_(pa.float32())),  # Dense vector
        pa.field("embedding_model", pa.string()),  # Model identifier
        pa.field("aliases", pa.list_(pa.string())),
        pa.field("metadata", pa.string()),  # JSON string
    ]
)

# Relationship Schema
RELATIONSHIP_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("source_entity_id", pa.string()),
        pa.field("target_entity_id", pa.string()),
        pa.field("type", pa.string()),
        pa.field("description", pa.string()),
        pa.field("source_chunks", pa.list_(pa.string())),
        pa.field("extraction_confidence", pa.float32()),
        pa.field("weight", pa.float32()),
        pa.field("embedding", pa.list_(pa.float32())),  # Dense vector for LightRAG global search
        pa.field("embedding_model", pa.string()),  # Model identifier
        pa.field("metadata", pa.string()),
    ]
)

# Community Schema
COMMUNITY_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("level", pa.int32()),
        pa.field("entity_ids", pa.list_(pa.string())),
        pa.field("parent_community_id", pa.string()),  # Hierarchical structure
        pa.field("child_community_ids", pa.list_(pa.string())),
        pa.field("relationship_ids", pa.list_(pa.string())),
        pa.field("size", pa.int32()),
        pa.field("density", pa.float32()),
        pa.field("title", pa.string()),
        pa.field("summary", pa.string()),
        pa.field("findings", pa.list_(pa.string())),
        pa.field("metadata", pa.string()),
    ]
)

# Community Report Schema
COMMUNITY_REPORT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("community_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("summary", pa.string()),
        pa.field("full_content", pa.string()),
        pa.field("findings", pa.list_(pa.string())),
        pa.field("token_count", pa.int32()),
        pa.field("rank", pa.float32()),
        # Note: community report embeddings stored in vector DB for GraphRAG global search
    ]
)

# Fact Schema (HippoRAG)
FACT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("subject", pa.string()),
        pa.field("predicate", pa.string()),
        pa.field("object", pa.string()),
        pa.field("source_chunk", pa.string()),
        pa.field("extraction_confidence", pa.float32()),
        pa.field("entity_ids", pa.list_(pa.string())),
        pa.field("metadata", pa.string()),  # JSON string
        # Note: fact embeddings stored in vector DB for HippoRAG Stage 1
    ]
)

# Entity-Chunk Edge Schema (HippoRAG bipartite graph)
ENTITY_CHUNK_EDGE_SCHEMA = pa.schema(
    [
        pa.field("entity_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("weight", pa.float32()),
        pa.field("mention_count", pa.int32()),
        pa.field("first_position", pa.int32()),
        pa.field("metadata", pa.string()),  # JSON string
    ]
)

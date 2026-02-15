# GraphRAG Troubleshooting Guide

## Common Issues & Solutions

### 1. Entity Extraction Quality Issues

#### Problem: Too few entities extracted
**Symptoms:**
- Low entity count in entities.parquet
- Sparse knowledge graph
- Poor local search results

**Solutions:**
```yaml
# Increase extraction attempts
entity_extraction:
  max_gleanings: 2  # or 3 for maximum recall

# Use more capable model
llm:
  model: gpt-4-turbo-preview  # instead of gpt-3.5-turbo

# Tune prompts for domain
graphrag prompt-tune --root ./data --output ./prompts
```

**Validation:**
```python
import pandas as pd
entities = pd.read_parquet("./output/latest/artifacts/entities.parquet")
print(f"Total entities: {len(entities)}")
print(f"Entities per type:\n{entities['type'].value_counts()}")
print(f"Avg entities per text unit: {entities['text_unit_ids'].apply(len).mean()}")
```

#### Problem: Too many low-quality entities
**Symptoms:**
- Excessive entity count
- Many generic or irrelevant entities
- Graph noise obscures signal

**Solutions:**
1. Refine entity types in prompt
2. Add negative examples to prompts
3. Post-process entities:
```python
# Filter entities by frequency or connectivity
entities = pd.read_parquet("entities.parquet")
relationships = pd.read_parquet("relationships.parquet")

# Remove entities with no relationships
connected_entities = set(relationships['source']) | set(relationships['target'])
entities_filtered = entities[entities['title'].isin(connected_entities)]

# Remove very common entities (likely noise)
entity_counts = entities['title'].value_counts()
rare_entities = entity_counts[entity_counts > 2].index
entities_filtered = entities[entities['title'].isin(rare_entities)]
```

### 2. Graph Construction Problems

#### Problem: Fragmented communities
**Symptoms:**
- Many small communities (< 5 entities)
- Hierarchical levels collapse (only 1-2 levels)
- Global search lacks coherence

**Solutions:**
```yaml
graph:
  max_cluster_size: 30  # Increase from default 20
  leiden_config:
    max_cluster_size: 30
    seed: 42
```

**Diagnosis:**
```python
communities = pd.read_parquet("communities.parquet")
print(f"Total communities: {len(communities)}")
print(f"Level distribution:\n{communities['level'].value_counts()}")

# Check community sizes
communities['entity_count'] = communities['entity_ids'].apply(len)
print(f"Avg entities per community: {communities['entity_count'].mean()}")
print(f"Min/Max: {communities['entity_count'].min()} / {communities['entity_count'].max()}")
```

#### Problem: Weak relationships
**Symptoms:**
- Low relationship count
- Poor multi-hop search results
- Disconnected graph components

**Solutions:**
1. Increase chunk overlap to capture cross-chunk relationships
```yaml
chunks:
  overlap: 150  # Increase from 100
```

2. Refine relationship extraction prompt
3. Use larger chunks to capture more context
```yaml
chunks:
  size: 1000  # Increase from 800
```

### 3. Query Performance Issues

#### Problem: Local search too slow
**Symptoms:**
- Queries take > 5 seconds
- High token usage per query

**Solutions:**
```python
# Reduce search scope
local_search = LocalSearch(
    top_k_entities=10,      # Reduce from 20
    top_k_relationships=5,  # Reduce from 10
    max_tokens=3000         # Reduce context size
)

# Add caching
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_local_search(query: str):
    return local_search.search(query)
```

#### Problem: Global search incomplete or shallow
**Symptoms:**
- Missing themes in results
- Answers lack depth

**Solutions:**
```python
# Increase map-reduce context
global_search = GlobalSearch(
    map_max_tokens=3000,    # Increase per-community context
    reduce_max_tokens=6000, # Increase synthesis context
    max_tokens=10000        # Increase final response
)
```

### 4. Indexing Failures

#### Problem: Out of memory during indexing
**Symptoms:**
- Process killed during entity extraction
- Memory errors in logs

**Solutions:**
1. Process documents in batches
```yaml
chunks:
  size: 500  # Reduce chunk size

async:
  max_concurrency: 5  # Reduce parallelism
```

2. Use streaming or incremental processing
3. Increase system memory or use distributed processing

#### Problem: Rate limit errors from LLM API
**Symptoms:**
- HTTP 429 errors
- Indexing stalls or fails

**Solutions:**
```yaml
llm:
  max_retries: 5          # Increase retries
  retry_delay: 5.0        # Longer delay
  concurrent_requests: 3  # Reduce parallelism

# Or use exponential backoff
llm:
  retry_strategy: exponential
  retry_delay: 2.0
  max_retry_delay: 60.0
```

#### Problem: Prompt token limit exceeded
**Symptoms:**
- Errors: "This model's maximum context length is..."
- Failed entity extraction on long chunks

**Solutions:**
```yaml
chunks:
  size: 600  # Reduce chunk size

entity_extraction:
  max_input_length: 3000  # Truncate long inputs

community_reports:
  max_input_length: 6000  # For report generation
```

### 5. Storage Issues

#### Problem: Parquet files corrupted or missing
**Symptoms:**
- Missing artifacts after indexing
- Errors reading Parquet files

**Solutions:**
1. Check storage permissions
2. Ensure sufficient disk space
3. Validate output directory
```bash
ls -lh ./output/*/artifacts/*.parquet
```

4. Re-run indexing with verbose logging
```bash
graphrag index --root ./data --verbose
```

#### Problem: Version compatibility issues
**Symptoms:**
- Can't read Parquet files from different GraphRAG version
- Schema mismatches

**Solutions:**
1. Check schema version:
```python
import pandas as pd
df = pd.read_parquet("entities.parquet")
print(df.columns.tolist())
```

2. Migrate schema if needed
3. Re-index with current GraphRAG version

### 6. Cost Control Issues

#### Problem: Unexpected high LLM costs
**Symptoms:**
- API bills higher than expected
- Token usage exceeds estimates

**Diagnosis:**
```bash
# Check logs for token usage
grep "tokens" ./output/*/logs/*.log

# Estimate costs
# entities.parquet rows * avg_tokens_per_extraction * cost_per_1k_tokens
```

**Solutions:**
```yaml
# Use cheaper models
llm:
  model: gpt-4o-mini  # Instead of gpt-4-turbo

# Reduce extraction attempts
entity_extraction:
  max_gleanings: 1  # Instead of 2-3

# Smaller chunks (fewer LLM calls)
chunks:
  size: 500

# Shorter community reports
community_reports:
  max_length: 1500
```

### 7. Query Result Quality Issues

#### Problem: Local search misses relevant information
**Symptoms:**
- Known entities not found
- Relationships ignored

**Solutions:**
1. Check entity coverage:
```python
entities = pd.read_parquet("entities.parquet")
query_entities = ["entity1", "entity2"]
found = entities[entities['title'].isin(query_entities)]
print(f"Found: {len(found)}/{len(query_entities)}")
```

2. Increase search scope:
```python
local_search = LocalSearch(
    top_k_entities=30,       # Increase
    top_k_relationships=20,  # Increase
    include_entity_rank=True # Use graph centrality
)
```

3. Check embeddings quality:
```python
# Verify embeddings exist and have correct dimensions
entities = pd.read_parquet("entities.parquet")
embedding_sample = entities['description_embedding'].iloc[0]
print(f"Embedding dimension: {len(embedding_sample)}")
```

#### Problem: Global search too generic
**Symptoms:**
- Answers lack specificity
- Missing important details

**Solutions:**
1. Improve community reports:
```yaml
community_reports:
  max_length: 3000  # Increase detail
  prompt: "./prompts/detailed_community_report.txt"
```

2. Adjust community granularity:
```yaml
graph:
  max_cluster_size: 15  # Smaller, more focused communities
```

### 8. Integration Issues

#### Problem: Can't use GraphRAG as library
**Symptoms:**
- Import errors
- API not working as expected

**Solutions:**
```python
# Correct imports
from graphrag.index import create_pipeline_config, run_pipeline_with_config
from graphrag.query.structured_search.local_search import LocalSearch
from graphrag.query.structured_search.global_search import GlobalSearch
from graphrag.config import GraphRagConfig

# Load config
config = GraphRagConfig.from_file("./settings.yaml")

# Run indexing programmatically
import asyncio
pipeline_config = create_pipeline_config(config)
asyncio.run(run_pipeline_with_config(pipeline_config))
```

#### Problem: Vector store integration
**Symptoms:**
- Can't load embeddings
- Dimension mismatches

**Solutions:**
1. Check embedding dimensions:
```python
entities = pd.read_parquet("entities.parquet")
embedding_dim = len(entities['description_embedding'].iloc[0])
print(f"Entity embedding dimension: {embedding_dim}")

reports = pd.read_parquet("community_reports.parquet")
report_embedding_dim = len(reports['summary_embedding'].iloc[0])
print(f"Report embedding dimension: {report_embedding_dim}")
```

2. Ensure vector store matches:
```python
# For FAISS
import faiss
index = faiss.IndexFlatL2(embedding_dim)  # Must match

# For Lance
import lancedb
db = lancedb.connect("./lancedb")
# Schema must match embedding dimension
```

## Debugging Workflow

### Step 1: Enable Verbose Logging
```bash
export GRAPHRAG_LOG_LEVEL=DEBUG
graphrag index --root ./data --verbose
```

### Step 2: Check Each Pipeline Stage

**After chunking:**
```python
text_units = pd.read_parquet("./output/latest/artifacts/text_units.parquet")
print(f"Total chunks: {len(text_units)}")
print(f"Avg chunk length: {text_units['text'].str.len().mean()}")
```

**After entity extraction:**
```python
entities = pd.read_parquet("./output/latest/artifacts/entities.parquet")
print(f"Total entities: {len(entities)}")
print(f"Entity types: {entities['type'].unique()}")
print(f"Entities with embeddings: {entities['description_embedding'].notna().sum()}")
```

**After graph construction:**
```python
relationships = pd.read_parquet("./output/latest/artifacts/relationships.parquet")
print(f"Total relationships: {len(relationships)}")
print(f"Avg weight: {relationships['weight'].mean()}")

# Check connectivity
import networkx as nx
G = nx.Graph()
for _, row in relationships.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])
print(f"Graph components: {nx.number_connected_components(G)}")
print(f"Largest component size: {len(max(nx.connected_components(G), key=len))}")
```

**After community detection:**
```python
communities = pd.read_parquet("./output/latest/artifacts/communities.parquet")
print(f"Total communities: {len(communities)}")
print(f"Hierarchy levels: {communities['level'].unique()}")
print(f"Avg entities per community: {communities['entity_ids'].apply(len).mean()}")
```

### Step 3: Validate Query Setup

```python
# Test local search
from graphrag.query.structured_search.local_search import LocalSearch

local_search = LocalSearch(...)
result = local_search.search("test query")
print(f"Context used: {result.context_data}")
print(f"Entities found: {len(result.context_data['entities'])}")
print(f"Relationships found: {len(result.context_data['relationships'])}")
```

## Performance Profiling

### Indexing Performance
```python
import time

stages = {}

start = time.time()
# ... chunking ...
stages['chunking'] = time.time() - start

start = time.time()
# ... entity extraction ...
stages['extraction'] = time.time() - start

start = time.time()
# ... graph construction ...
stages['graph'] = time.time() - start

print("Pipeline timing:")
for stage, duration in stages.items():
    print(f"  {stage}: {duration:.2f}s")
```

### Query Performance
```python
import time

query = "test query"

start = time.time()
result = local_search.search(query)
duration = time.time() - start

print(f"Query time: {duration:.2f}s")
print(f"Context size: {len(result.context_text)} chars")
print(f"Token estimate: {len(result.context_text) / 4}")
```

## Common Error Messages

### "No entities found"
- Check entity extraction prompts
- Verify LLM is responding correctly
- Ensure entity types match your data domain

### "Community detection failed"
- Insufficient relationships in graph
- Try smaller max_cluster_size
- Check relationship weights (should be > 0)

### "Vector dimension mismatch"
- Embedding model changed between indexing and querying
- Re-index with consistent embedding model
- Check embedding model configuration

### "Context length exceeded"
- Reduce chunk size
- Reduce max_tokens in search
- Use summarization for long contexts

### "Rate limit exceeded"
- Reduce concurrent_requests
- Increase retry_delay
- Use rate-limited API tier or increase limits

## Emergency Recovery

### Partial Indexing Failure
```bash
# Resume from last successful stage
graphrag index --root ./data --resume
```

### Corrupted Parquet Files
```python
# Identify corrupted files
import pandas as pd
artifacts = ["documents", "text_units", "entities", "relationships", "communities", "community_reports"]
for artifact in artifacts:
    try:
        df = pd.read_parquet(f"./output/latest/artifacts/{artifact}.parquet")
        print(f"{artifact}: OK ({len(df)} rows)")
    except Exception as e:
        print(f"{artifact}: CORRUPTED - {e}")

# Re-run indexing if needed
```

### Reset and Start Over
```bash
# Backup existing output
mv ./output ./output.backup

# Clear all artifacts
rm -rf ./output/*

# Re-run indexing
graphrag index --root ./data --verbose
```

# LightRAG Relationship Indexing

## Core Concept

LightRAG's global search capability depends on generating **semantic relationship descriptions** that go beyond simple "Entity A relates to Entity B" statements. These descriptions capture the meaning, context, and significance of relationships, enabling thematic and conceptual queries.

## Relationship Description Generation

### What Makes a Good Relationship Description

**Bad (Too Generic):**
```
"Python relates to Django"
"API connects to Database"
"User interacts with System"
```

**Good (Semantically Rich):**
```
"Python serves as the primary implementation language for Django, providing dynamic typing and extensive standard libraries that enable rapid web development"

"API authenticates requests and executes parameterized queries against Database through connection pooling to prevent SQL injection and optimize performance"

"User submits form data through System's React frontend, which validates inputs client-side before sending authenticated POST requests to REST endpoints"
```

### Key Properties of Good Descriptions:

1. **Semantic significance:** Why does this relationship matter?
2. **Contextual detail:** Under what circumstances does this relationship occur?
3. **Mechanism clarity:** How does the relationship manifest?
4. **Domain specificity:** Uses domain-appropriate terminology
5. **Retrievability:** Contains keywords likely to appear in relevant queries

## Prompt Engineering for Relationship Descriptions

### Generic Prompt (Baseline):
```
Given these entities and their context, describe their relationships:
Entities: {entity_list}
Context: {chunk_text}

For each relationship, provide a concise description.
```

### Improved Prompt (Domain-Aware):
```
You are analyzing a {domain} knowledge base. Given the following entities extracted from a document chunk, generate semantically rich relationship descriptions.

Entities: {entity_list}
Context: {chunk_text}

For each significant relationship between entities:
1. Describe HOW they relate (mechanism/process)
2. Explain WHY this relationship matters (significance)
3. Specify WHEN/WHERE this relationship applies (context)
4. Use domain-specific terminology

Format: "Entity1 [relationship] Entity2: [detailed description]"

Focus on relationships that would help answer thematic or conceptual questions about {domain}.
```

### Domain-Specific Examples:

**Software Engineering Domain:**
```
Prompt addition: "Focus on architectural patterns, data flows, dependencies,
implementation choices, and design decisions."

Example output:
"React [implements] Component Lifecycle: React manages component state through
lifecycle methods (mount, update, unmount), enabling developers to control
rendering behavior and side effects at specific stages of component existence,
which is crucial for performance optimization and resource management."
```

**Medical Domain:**
```
Prompt addition: "Focus on causal relationships, treatment mechanisms,
contraindications, and clinical significance."

Example output:
"Aspirin [inhibits] COX-2 Enzyme: Aspirin irreversibly acetylates cyclooxygenase-2
enzyme, blocking prostaglandin synthesis and reducing inflammation, which makes
it effective for pain relief but also increases bleeding risk in surgical contexts."
```

**Business Domain:**
```
Prompt addition: "Focus on value flows, strategic impacts, dependencies,
and business rationale."

Example output:
"Marketing Campaign [drives] Revenue Growth: Targeted social media campaigns
increased brand awareness by 40% among 25-35 demographic, resulting in 15%
lift in online sales through direct attribution tracking via UTM parameters."
```

## Implementation Patterns

### Basic Relationship Generation (LightRAG Style):

```python
async def generate_relationship_descriptions(entities, chunk_text, domain="general"):
    """Generate semantic relationship descriptions for entity pairs."""

    prompt = f"""You are analyzing a {domain} knowledge base.

Entities: {', '.join(entities)}
Context: {chunk_text}

Generate detailed relationship descriptions that capture:
- HOW entities relate (mechanism)
- WHY the relationship matters (significance)
- WHEN/WHERE it applies (context)

Format each as: "Entity1 [relationship_type] Entity2: [detailed_description]"
"""

    response = await llm.generate(prompt)
    return parse_relationships(response)


def parse_relationships(llm_output):
    """Parse LLM output into structured relationship data."""
    relationships = []

    for line in llm_output.strip().split('\n'):
        if '[' in line and ']' in line:
            # Parse: "Entity1 [type] Entity2: description"
            match = re.match(r'(.*?)\[(.*?)\](.*?):(.*)', line)
            if match:
                relationships.append({
                    'source': match.group(1).strip(),
                    'relation_type': match.group(2).strip(),
                    'target': match.group(3).strip(),
                    'description': match.group(4).strip()
                })

    return relationships
```

### Optimized: Batch Processing to Reduce LLM Calls

```python
async def generate_relationships_batch(chunks_with_entities, domain="general"):
    """Process multiple chunks in parallel to reduce latency."""

    tasks = []
    for chunk_id, (entities, text) in chunks_with_entities.items():
        task = generate_relationship_descriptions(entities, text, domain)
        tasks.append(task)

    # Process in parallel with rate limiting
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        chunk_id: result
        for chunk_id, result in zip(chunks_with_entities.keys(), results)
        if not isinstance(result, Exception)
    }
```

### Quality Filtering

```python
def filter_significant_relationships(relationships, min_description_length=50):
    """Filter out trivial or low-quality relationship descriptions."""

    significant = []
    trivial_patterns = [
        r'^Entity\d+ relates to Entity\d+$',
        r'^simple connection$',
        r'^associated with$',
    ]

    for rel in relationships:
        desc = rel['description']

        # Skip if too short (likely generic)
        if len(desc) < min_description_length:
            continue

        # Skip if matches trivial patterns
        if any(re.search(pattern, desc, re.I) for pattern in trivial_patterns):
            continue

        # Skip if missing key semantic elements
        if not has_semantic_depth(desc):
            continue

        significant.append(rel)

    return significant


def has_semantic_depth(description):
    """Check if description contains semantic depth indicators."""
    depth_indicators = [
        'through', 'by', 'via',  # mechanism
        'enables', 'causes', 'prevents',  # causality
        'because', 'since', 'due to',  # rationale
        'when', 'while', 'during',  # temporal context
        'in order to', 'for', 'to',  # purpose
    ]

    return any(indicator in description.lower() for indicator in depth_indicators)
```

## Relationship Index Storage

### JSON Format (vdb_relationships.json):

```json
{
  "relationships": [
    {
      "id": "rel_001",
      "source_entity": "Django",
      "target_entity": "Python",
      "relation_type": "implemented_in",
      "description": "Django is implemented in Python, leveraging dynamic typing...",
      "description_embedding": [0.123, -0.456, ...],
      "source_chunks": ["chunk_42", "chunk_103"],
      "metadata": {
        "domain": "software_engineering",
        "extraction_date": "2026-02-15",
        "confidence": 0.89
      }
    }
  ]
}
```

### Key Fields:

- **id**: Unique relationship identifier
- **source_entity / target_entity**: Entity pair (bidirectional for undirected relationships)
- **relation_type**: Semantic relation type (implements, causes, enables, etc.)
- **description**: Rich semantic description (this is embedded and searched)
- **description_embedding**: Vector embedding of the description
- **source_chunks**: Chunks where this relationship was found
- **metadata**: Domain, confidence, timestamps, etc.

## Global Search Process

### Query → Relationship Matching:

```python
async def global_search(query, top_k=10):
    """Search for relationships matching query semantics."""

    # Embed query
    query_embedding = await embed(query)

    # Vector search over relationship descriptions
    matched_relationships = vector_db.search(
        collection="relationships",
        query_vector=query_embedding,
        top_k=top_k
    )

    # Expand to include entities and source chunks
    results = []
    for rel in matched_relationships:
        results.append({
            'relationship': rel['description'],
            'entities': [rel['source_entity'], rel['target_entity']],
            'source_chunks': get_chunks(rel['source_chunks']),
            'relevance_score': rel['score']
        })

    return results
```

### Example Query Flow:

**Query:** "What security mechanisms protect user data?"

1. Embed query → [0.23, -0.41, 0.56, ...]
2. Search relationship embeddings → Top matches:
   - "Encryption [protects] User Data: AES-256 encryption secures data at rest..."
   - "Authentication [validates] User Access: JWT tokens verify user identity..."
   - "Firewall [filters] Network Traffic: Stateful inspection blocks malicious requests..."
3. Return relationship descriptions + involved entities + source chunks
4. LLM synthesizes final answer from matched relationships

## Performance Considerations

### Computational Cost:

**Per Document Chunk:**
- Entity extraction: 1 LLM call (shared with other strategies)
- Relationship generation: 1 additional LLM call
- Embedding relationships: 1 embedding call per relationship (~5-10 relationships/chunk)

**Optimization Strategies:**
1. Batch relationship generation across chunks
2. Cache frequent relationship patterns
3. Use smaller/faster models for relationship generation (Claude Haiku, GPT-4o-mini)
4. Generate relationships only for significant entity pairs (filter low-importance entities)

### Storage Overhead:

**Approximate sizes per 1000 documents:**
- Entities: ~5MB (entity vectors + metadata)
- Relationships: ~10MB (relationship vectors + descriptions)
- Graph: ~2MB (GraphML structure)
- **Total: ~17MB** (2-3x vs vector-only RAG)

### Indexing Time:

**For 1000 documents (~500K tokens):**
- Entity extraction: ~10 min (parallel batching)
- Relationship generation: ~15 min (additional LLM calls)
- Embedding relationships: ~5 min (batch embedding)
- **Total: ~30 min** (vs ~15 min for entity-only indexing)

## Common Issues & Solutions

### Issue 1: Generic Relationship Descriptions

**Symptom:** Relationships like "X relates to Y" without semantic depth

**Solution:**
- Improve prompt with domain context and examples
- Add post-processing filter to reject generic descriptions
- Use few-shot examples in prompt showing good vs bad descriptions

### Issue 2: Too Many Trivial Relationships

**Symptom:** Every entity connected to every other entity in chunk

**Solution:**
- Filter entities by importance (use entity ranking)
- Instruct LLM to generate only "significant" relationships
- Set relationship count limit per chunk (e.g., top 5 most important)

### Issue 3: Relationship Description Redundancy

**Symptom:** Same relationship described differently across chunks

**Solution:**
- Implement relationship deduplication with semantic similarity
- Merge similar relationships with aggregated descriptions
- Use canonical entity names to improve matching

### Issue 4: High LLM Cost

**Symptom:** Relationship generation doubles indexing cost

**Solution:**
- Use cheaper models (Haiku/4o-mini) for relationship generation
- Generate relationships only for high-importance entity pairs
- Cache common relationship patterns for reuse
- Implement smart batching to maximize tokens per call

## Best Practices

1. **Domain-Specific Prompts:** Tailor relationship generation prompts to your domain
2. **Quality Over Quantity:** Better to have 5 rich relationships than 20 generic ones
3. **Iterative Refinement:** Test queries → Review retrieved relationships → Refine prompts
4. **Semantic Validation:** Ensure descriptions contain mechanisms, context, and significance
5. **Deduplication:** Merge semantically similar relationships to reduce index size
6. **Metadata Tracking:** Store confidence scores, domains, timestamps for quality monitoring
7. **A/B Testing:** Compare global search results with different relationship generation prompts

## Integration with Unified System

### Workflow in Multi-Strategy RAG:

```
Document Ingestion:
1. Chunk documents (shared)
2. Extract entities (shared with GraphRAG, HippoRAG)
3. Generate relationship descriptions (LightRAG-specific)
4. Build LightRAG dual indexes (entities + relationships)
5. Store graph structure

Query Processing:
1. Query router determines strategy
2. If LightRAG selected:
   - Local search → Entity index
   - Global search → Relationship index
   - Hybrid → Both indexes
3. Return matched relationships + entities + chunks
4. LLM synthesizes answer
```

### Avoiding Duplicate Work:

- Share entity extraction output across strategies
- Use same chunk IDs for cross-strategy referencing
- Store relationships separately from GraphRAG communities
- Cache embeddings for entities used in multiple strategies

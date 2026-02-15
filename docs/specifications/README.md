# Technical Specifications Index

**Project:** Graph-Unified Multi-Strategy RAG System
**Version:** 1.0
**Last Updated:** 2026-02-15
**Status:** Complete

## Overview

This directory contains comprehensive technical specifications for implementing Graph-Unified, a unified multi-strategy RAG system supporting six retrieval strategies through a shared chunking and entity extraction pipeline.

## Specification Documents

### [01. Data Schemas & Models](01-data-schemas.md)

**Purpose:** Define all data models, schemas, and validation rules.

**Contents:**
- 6 core data models (Document, Chunk, Entity, Relationship, Community, CommunityReport)
- Pydantic schemas with validation
- PyArrow schemas for Parquet storage
- JSON schemas for API communication
- Configuration schema for settings.yaml
- Validation rules and constraints

**Key Outputs:**
- Python Pydantic models
- Parquet storage schemas
- API contract definitions

**Implementation Priority:** ⭐⭐⭐ Critical (Foundation)

---

### [02. API & Interface Specifications](02-api-specification.md)

**Purpose:** Define Python APIs for programmatic usage.

**Contents:**
- 3 core protocols (Retriever, PipelineStage, StorageInterface)
- 4 storage interfaces (ChunkStore, EntityStore, VectorStore, GraphStore)
- 3 high-level APIs (Indexer, Querier, PromptTuner)
- Exception hierarchy
- Complete usage examples
- Type hints for all methods

**Key Outputs:**
- Protocol definitions
- Public API classes
- Error handling patterns

**Implementation Priority:** ⭐⭐⭐ Critical (Public Interface)

---

### [03. Module Architecture](03-module-architecture.md)

**Purpose:** Specify internal module structure and dependencies.

**Contents:**
- Package structure (9 modules)
- Dependency graph
- Core class responsibilities
- Interaction patterns (DI, factories)
- Configuration management flow
- Testing strategy

**Key Outputs:**
- Module organization
- Dependency injection patterns
- Factory implementations

**Implementation Priority:** ⭐⭐⭐ Critical (Structure)

---

### [04. Storage Format Specifications](04-storage-formats.md)

**Purpose:** Define all storage formats and file structures.

**Contents:**
- Directory structure
- 7 Parquet schemas (documents, chunks, entities, relationships, communities, reports)
- 2 LanceDB vector indexes
- 2 GraphML graph files
- 3 auxiliary indexes (BM25, LightRAG, metadata)
- Migration strategy
- Backup procedures

**Key Outputs:**
- Parquet I/O operations
- Vector DB integration
- Graph storage handling

**Storage Size:** ~706 MB for 10K documents

**Implementation Priority:** ⭐⭐⭐ Critical (Persistence)

---

### [05. Indexing Pipeline Specification](05-indexing-pipeline-spec.md)

**Purpose:** Specify the complete indexing pipeline.

**Contents:**
- 6 pipeline stages (Load, Chunk, Extract, Embed, Build)
- Detailed algorithms for each stage
- Prompt templates for extraction
- Error handling and recovery
- Progress tracking
- Performance benchmarks

**Key Outputs:**
- Stage implementations
- Pipeline orchestrator
- Strategy builders

**Performance:** ~26 minutes for 10K documents
**Cost:** ~$45.67 for 10K documents

**Implementation Priority:** ⭐⭐⭐ Critical (Core Processing)

---

### [06. Retrieval Pipeline Specification](06-retrieval-pipeline-spec.md)

**Purpose:** Specify all retrieval strategies and query processing.

**Contents:**
- 6 retrieval strategies with complete algorithms:
  1. Naive RAG (vector similarity)
  2. Hybrid RAG (dense + sparse)
  3. GraphRAG Local (entity neighborhood)
  4. GraphRAG Global (community summaries)
  5. LightRAG (dual-index)
  6. HippoRAG (Personalized PageRank)
- Query router logic
- Response generation
- Cost tracking

**Key Outputs:**
- Retriever implementations
- Query router
- Generation wrapper

**Latency:** 100ms - 5s depending on strategy

**Implementation Priority:** ⭐⭐⭐ Critical (Query Processing)

---

### [07. Testing Specifications](07-testing-spec.md)

**Purpose:** Define comprehensive testing requirements.

**Contents:**
- 200+ unit tests
- 60+ integration tests
- 20 end-to-end scenarios
- Performance benchmarks
- Evaluation metrics (Precision@K, Recall@K, F1, MRR)
- Test data requirements
- CI/CD integration

**Key Outputs:**
- Unit test suites
- Integration tests
- E2E scenarios
- Performance tests

**Coverage Target:** >80% overall, >90% critical paths

**Implementation Priority:** ⭐⭐ High (Quality Assurance)

---

### [08. Configuration Specification](08-configuration-spec.md)

**Purpose:** Complete specification for settings.yaml.

**Contents:**
- 60+ configuration fields with types, ranges, defaults
- Field-by-field documentation
- Validation rules
- Configuration profiles (dev, production, research)
- Environment variable substitution
- Examples for common scenarios

**Key Outputs:**
- Settings schema
- Validation logic
- Profile templates
- Config generator

**Implementation Priority:** ⭐⭐⭐ Critical (Configuration)

---

### [09. CLI Specification](09-cli-spec.md)

**Purpose:** Define complete command-line interface.

**Contents:**
- 10 CLI commands with full syntax:
  1. `init` - Generate configuration
  2. `index` - Index documents
  3. `query` - Query knowledge base
  4. `prompt-tune` - Tune extraction prompts
  5. `evaluate` - Evaluate quality
  6. `inspect` - Inspect indexed data
  7. `visualize` - Generate visualizations
  8. `delete` - Delete data
  9. `export` - Export data
  10. `migrate` - Migrate from standalone tools
- 50+ command options
- Output formats
- Exit codes
- Shell completion

**Key Outputs:**
- CLI implementation
- Progress indicators
- Output formatters

**Implementation Priority:** ⭐⭐ High (User Interface)

---

### [10. Prompt Templates Specification](10-prompt-templates-spec.md)

**Purpose:** Define all LLM prompts with parameters.

**Contents:**
- 5 core prompts (entity extraction, relationship extraction, community reports, query generation, relationship descriptions)
- 3 prompt tuning prompts
- Few-shot examples (medical, legal, financial domains)
- Output validation schemas
- Prompt metrics and A/B testing
- Temperature and token settings

**Key Outputs:**
- Prompt templates
- Few-shot example library
- Validation logic
- Tuning framework

**Implementation Priority:** ⭐⭐⭐ Critical (LLM Integration)

---

## Specification Statistics

| Specification | Lines | Key Sections | Implementation Priority |
|---------------|-------|--------------|------------------------|
| Data Schemas | ~600 | 6 models, 7 Parquet schemas | Critical |
| API | ~700 | 3 protocols, 3 APIs | Critical |
| Module Architecture | ~800 | 9 modules, dependency graph | Critical |
| Storage Formats | ~700 | 7 Parquet files, 2 vector indexes | Critical |
| Indexing Pipeline | ~900 | 6 stages, algorithms | Critical |
| Retrieval Pipeline | ~950 | 6 strategies, algorithms | Critical |
| Testing | ~800 | 280 tests, benchmarks | High |
| Configuration | ~850 | 60+ fields, 3 profiles | Critical |
| CLI | ~750 | 10 commands, 50+ options | High |
| Prompt Templates | ~900 | 8 prompts, examples | Critical |
| **Total** | **~7,950 lines** | | |

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Priority:** Critical
**Specifications:** 01, 02, 03, 08

**Tasks:**
1. Implement data models (01-data-schemas.md)
2. Create storage interfaces (02-api-specification.md)
3. Set up module structure (03-module-architecture.md)
4. Build configuration system (08-configuration-spec.md)

**Deliverables:**
- Working Pydantic models
- Storage interface definitions
- Configuration loading
- Project structure

---

### Phase 2: Storage Layer (Week 3)

**Priority:** Critical
**Specifications:** 04

**Tasks:**
1. Implement Parquet I/O (04-storage-formats.md)
2. Integrate LanceDB (04-storage-formats.md)
3. Build GraphStore (04-storage-formats.md)
4. Create migration framework

**Deliverables:**
- Parquet read/write operations
- Vector index integration
- Graph storage
- Storage tests

---

### Phase 3: Indexing Pipeline (Weeks 4-5)

**Priority:** Critical
**Specifications:** 05, 10

**Tasks:**
1. Implement pipeline stages (05-indexing-pipeline-spec.md)
2. Build stage orchestrator (05-indexing-pipeline-spec.md)
3. Integrate prompts (10-prompt-templates-spec.md)
4. Add error handling

**Deliverables:**
- Load, Chunk, Extract, Embed stages
- Pipeline orchestrator
- Prompt templates
- Progress tracking

---

### Phase 4: Retrieval Strategies (Weeks 6-8)

**Priority:** Critical
**Specifications:** 06

**Tasks:**
1. Implement Naive + Hybrid retrievers (06-retrieval-pipeline-spec.md)
2. Implement GraphRAG Local + Global (06-retrieval-pipeline-spec.md)
3. Implement LightRAG (06-retrieval-pipeline-spec.md)
4. Implement HippoRAG (06-retrieval-pipeline-spec.md)

**Deliverables:**
- 6 retriever implementations
- Query router
- Response generation
- Retrieval tests

---

### Phase 5: CLI & Testing (Weeks 9-10)

**Priority:** High
**Specifications:** 07, 09

**Tasks:**
1. Build CLI commands (09-cli-spec.md)
2. Add progress indicators (09-cli-spec.md)
3. Write unit tests (07-testing-spec.md)
4. Write integration tests (07-testing-spec.md)

**Deliverables:**
- CLI implementation
- 200+ unit tests
- 60+ integration tests
- CI/CD pipeline

---

### Phase 6: Optimization & Evaluation (Weeks 11-12)

**Priority:** High
**Specifications:** 07, 10

**Tasks:**
1. Implement prompt tuning (10-prompt-templates-spec.md)
2. Build evaluation framework (07-testing-spec.md)
3. Performance optimization
4. Documentation

**Deliverables:**
- Prompt tuner
- Evaluation metrics
- Performance benchmarks
- User documentation

---

## Key Design Decisions

### 1. Single Extraction Pass

**Decision:** Share extraction across all strategies
**Rationale:** Extraction is 90% of cost
**Tradeoff:** Lose 5-10% strategy-specific optimization, gain 60-70% cost savings
**Documented in:** 05-indexing-pipeline-spec.md, 06-retrieval-pipeline-spec.md

### 2. Parquet for Canonical Storage

**Decision:** Use Parquet as source of truth
**Rationale:** Columnar format, compression, wide adoption
**Tradeoff:** No ACID transactions, append-only updates
**Documented in:** 04-storage-formats.md

### 3. Async-First API

**Decision:** All I/O operations async
**Rationale:** Better concurrency, scalable
**Tradeoff:** More complex than sync code
**Documented in:** 02-api-specification.md

### 4. Protocol-Based Interfaces

**Decision:** Use Python protocols (PEP 544)
**Rationale:** Type-safe, flexible, testable
**Tradeoff:** Requires Python 3.8+
**Documented in:** 02-api-specification.md

### 5. LanceDB for Vectors

**Decision:** LanceDB over FAISS/Qdrant
**Rationale:** Columnar, disk-based, good for <10M vectors
**Tradeoff:** Less mature than FAISS
**Documented in:** 04-storage-formats.md

---

## Usage Examples

### Reading Specifications

**For implementers:**
1. Start with 03-module-architecture.md (understand structure)
2. Read 01-data-schemas.md (understand data models)
3. Read 02-api-specification.md (understand public API)
4. Read specific pipeline specs as needed (05, 06)

**For architects:**
1. Read all "Overview" sections
2. Focus on design decisions
3. Review API boundaries (02)
4. Check performance benchmarks (05, 06, 07)

**For contributors:**
1. Read 03-module-architecture.md (find module to work on)
2. Read relevant spec for that module
3. Read 07-testing-spec.md (write tests)
4. Follow implementation roadmap

---

## Validation Checklist

Before starting implementation:

- [ ] All 10 specifications reviewed
- [ ] Design decisions understood
- [ ] Dependencies identified
- [ ] Module structure clear
- [ ] API contracts defined
- [ ] Storage formats specified
- [ ] Prompt templates prepared
- [ ] Testing strategy defined
- [ ] Performance targets set

---

## Maintenance

### Updating Specifications

When making changes:

1. Update relevant specification document
2. Update this README if structure changes
3. Update version and "Last Updated" date
4. Note breaking changes clearly
5. Update dependent specifications

### Versioning

Specifications use semantic versioning:
- **Major version (e.g., 2.0):** Breaking changes
- **Minor version (e.g., 1.1):** Backward-compatible additions
- **Current version:** 1.0 (initial release)

---

## Summary

These technical specifications provide:

- **~8,000 lines** of detailed documentation
- **10 comprehensive documents** covering all aspects
- **Complete implementation blueprint** for Graph-Unified
- **Precise specifications** for data models, APIs, algorithms, storage, testing
- **Performance benchmarks** and cost estimates
- **6-phase implementation roadmap** (12 weeks)

**Ready for Implementation:** ✅

All specifications are complete, consistent, and provide sufficient detail for unambiguous implementation.

**Next Steps:**

1. Review specifications with team
2. Set up project repository
3. Follow Phase 1 implementation roadmap
4. Begin with data models and configuration system

---

**Questions or Clarifications:**

For questions about these specifications:
- Check cross-references in relevant spec documents
- Review design decisions section
- Consult implementation examples in each spec

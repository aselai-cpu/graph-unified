# Prompt Templates Specification

**Version:** 1.0
**Last Updated:** 2026-02-15
**Status:** Technical Specification

## Overview

This document specifies all LLM prompts used in Graph-Unified, including entity extraction, relationship extraction, community summarization, query generation, and prompt tuning templates. Each prompt includes parameter placeholders, few-shot examples, output format requirements, and prompt engineering guidelines.

## Prompt Engineering Guidelines

### Core Principles

1. **Clear task description:** Explicitly state what the LLM should do
2. **Output format specification:** Use JSON schemas or structured examples
3. **Few-shot examples:** Provide 1-3 examples for complex tasks
4. **Constraints and edge cases:** Address ambiguities and edge cases
5. **Temperature settings:** 0.0 for extraction, 0.1-0.3 for generation

### Best Practices

- Use XML tags or delimiters for sections
- Specify confidence scores for uncertain outputs
- Include "I don't know" instructions when appropriate
- Keep prompts under 1000 tokens when possible
- Test with diverse inputs

---

## Entity Extraction Prompt

### Purpose

Extract named entities from text chunks with type classification and descriptions.

### Template

```python
ENTITY_EXTRACTION_PROMPT = """
You are an expert at extracting structured information from text. Your task is to identify and extract named entities from the provided text chunks.

# Entity Types

Extract entities of the following types:
{entity_types}

# Instructions

1. Identify all entities of the specified types
2. For each entity, provide:
   - **name**: The canonical name of the entity (use full names, not abbreviations when possible)
   - **type**: One of the specified entity types
   - **description**: A brief (1-2 sentence) description of the entity
   - **confidence**: Your confidence in this extraction (0.0-1.0)

3. Guidelines:
   - Use specific names: "United Nations" not "UN organization"
   - Avoid overly generic entities: "climate change" is good, "problem" is too generic
   - If an entity appears multiple times with variations (e.g., "IPCC" and "Intergovernmental Panel on Climate Change"), use the full name
   - Include confidence < 1.0 for ambiguous entities

# Text Chunks

{chunk_texts}

# Output Format

Return a JSON object with this exact structure:

{{
  "entities": [
    {{
      "name": "entity name",
      "type": "ENTITY_TYPE",
      "description": "brief description of the entity",
      "confidence": 0.95
    }}
  ]
}}

# Example

Input:
"The IPCC published its Sixth Assessment Report in 2021. The report shows that human activities are unequivocally causing climate change."

Output:
{{
  "entities": [
    {{
      "name": "Intergovernmental Panel on Climate Change",
      "type": "ORGANIZATION",
      "description": "UN body for assessing the science related to climate change",
      "confidence": 1.0
    }},
    {{
      "name": "Sixth Assessment Report",
      "type": "EVENT",
      "description": "IPCC's major scientific assessment report published in 2021",
      "confidence": 0.95
    }},
    {{
      "name": "climate change",
      "type": "CONCEPT",
      "description": "Long-term changes in temperature and weather patterns",
      "confidence": 1.0
    }}
  ]
}}

Now extract entities from the text chunks above. Return only valid JSON.
"""
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `entity_types` | str | Comma-separated list of entity types |
| `chunk_texts` | str | Text chunks to extract from (separated by "---") |

### Expected Output

```json
{
  "entities": [
    {
      "name": "string",
      "type": "ENTITY_TYPE",
      "description": "string",
      "confidence": 0.95
    }
  ]
}
```

### Tuning Parameters

```python
EntityExtractionConfig(
    temperature=0.0,      # Deterministic
    max_tokens=2000,      # Sufficient for 50-100 entities
    top_p=1.0
)
```

---

## Relationship Extraction Prompt

### Purpose

Extract relationships between entities with type classification and descriptions.

### Template

```python
RELATIONSHIP_EXTRACTION_PROMPT = """
You are an expert at identifying relationships between entities in text.

# Relationship Types

Extract relationships of the following types:
{relationship_types}

# Entities Extracted

The following entities were identified in the text:
{entity_list}

# Instructions

1. Identify relationships between the entities listed above
2. For each relationship, provide:
   - **source**: Name of the source entity (must match an entity from the list above)
   - **target**: Name of the target entity (must match an entity from the list above)
   - **type**: One of the specified relationship types
   - **description**: A brief description of how they are related
   - **confidence**: Your confidence in this relationship (0.0-1.0)

3. Guidelines:
   - Only create relationships between entities in the provided list
   - Relationships are directed (source → target)
   - If the relationship is bidirectional, create two relationships
   - Prefer specific relationship types over generic "RELATED_TO"
   - Include confidence < 1.0 for inferred relationships

# Text Chunks

{chunk_texts}

# Output Format

Return a JSON object with this exact structure:

{{
  "relationships": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "type": "RELATIONSHIP_TYPE",
      "description": "description of the relationship",
      "confidence": 0.90
    }}
  ]
}}

# Example

Entities:
- IPCC (ORGANIZATION)
- Sixth Assessment Report (EVENT)
- climate change (CONCEPT)

Input:
"The IPCC published its Sixth Assessment Report in 2021. The report shows that human activities are unequivocally causing climate change."

Output:
{{
  "relationships": [
    {{
      "source": "IPCC",
      "target": "Sixth Assessment Report",
      "type": "PUBLISHED",
      "description": "IPCC published the Sixth Assessment Report in 2021",
      "confidence": 1.0
    }},
    {{
      "source": "Sixth Assessment Report",
      "target": "climate change",
      "type": "STUDIES",
      "description": "The report analyzes human causes of climate change",
      "confidence": 0.95
    }}
  ]
}}

Now extract relationships from the text chunks above. Return only valid JSON.
"""
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `relationship_types` | str | Comma-separated relationship types |
| `entity_list` | str | Formatted list of extracted entities |
| `chunk_texts` | str | Text chunks (same as entity extraction) |

### Expected Output

```json
{
  "relationships": [
    {
      "source": "string",
      "target": "string",
      "type": "RELATIONSHIP_TYPE",
      "description": "string",
      "confidence": 0.90
    }
  ]
}
```

---

## Community Report Generation Prompt

### Purpose

Generate comprehensive summaries of entity communities for GraphRAG global search.

### Template

```python
COMMUNITY_REPORT_PROMPT = """
You are an expert analyst creating a comprehensive report about a group of related entities.

# Task

Analyze the entities and relationships below and create a structured report.

# Entities in Community

{entity_descriptions}

# Relationships

{relationship_descriptions}

# Report Structure

Create a report with the following sections:

1. **Title**: A concise, descriptive title (max 10 words)
2. **Summary**: A 2-3 sentence executive summary
3. **Key Findings**: 3-7 bullet points highlighting the most important insights
4. **Analysis**: 2-3 paragraphs of detailed analysis

# Guidelines

- Focus on the overall theme or purpose of this community
- Identify key actors, concepts, or events
- Highlight important relationships and connections
- Note any patterns or trends
- Be concise but informative

# Output Format

Format your response as follows:

# TITLE: [Your Title Here]

## Summary
[2-3 sentence summary]

## Key Findings
- [Finding 1]
- [Finding 2]
- [Finding 3]
...

## Analysis

[Detailed analysis paragraph 1]

[Detailed analysis paragraph 2]

# Example

Entities:
- IPCC (ORGANIZATION): UN body for assessing climate science
- UNFCCC (ORGANIZATION): UN Framework Convention on Climate Change
- Paris Agreement (EVENT): International climate treaty
- Climate Change (CONCEPT): Global warming phenomenon

Relationships:
- IPCC → UNFCCC: Provides scientific assessment to
- UNFCCC → Paris Agreement: Established
- Paris Agreement → Climate Change: Addresses

Output:

# TITLE: International Climate Policy Ecosystem

## Summary
This community represents the core international institutions and agreements addressing climate change. The IPCC provides scientific foundations that inform the UNFCCC's policy framework, which established the Paris Agreement as a global response mechanism.

## Key Findings
- IPCC serves as the scientific authority, providing evidence-based assessments
- UNFCCC acts as the institutional framework for international climate negotiations
- Paris Agreement represents the operational treaty for coordinated climate action
- Strong interconnection between science, policy, and implementation mechanisms

## Analysis

The International Climate Policy Ecosystem demonstrates a sophisticated multi-layered approach to addressing climate change. At its foundation, the IPCC provides rigorous scientific assessments that establish the evidence base for policy action. These assessments directly inform the UNFCCC, which serves as the primary institutional mechanism for international climate negotiations and coordination.

The Paris Agreement represents the culmination of this ecosystem, translating scientific understanding and policy frameworks into concrete commitments and action plans. The tight integration between these entities reflects a well-structured approach where science informs policy, and policy drives coordinated global action on climate change.

Now create a report for the community above.
"""
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `entity_descriptions` | str | Formatted entity descriptions |
| `relationship_descriptions` | str | Formatted relationship descriptions |

### Expected Output

Markdown-formatted report with title, summary, findings, and analysis.

### Tuning Parameters

```python
CommunityReportConfig(
    temperature=0.1,      # Slight creativity
    max_tokens=2000,      # Sufficient for detailed report
)
```

---

## Query Generation Prompt

### Purpose

Generate responses to user queries based on retrieved contexts.

### Template

```python
QUERY_GENERATION_PROMPT = """
You are a helpful assistant answering questions based on provided context.

# Context

The following information is relevant to the user's query:

{contexts}

# Query

{query}

# Instructions

1. Answer the query based **only** on the information in the context above
2. If the context does not contain sufficient information, say: "I don't have enough information to answer this question fully based on the provided context."
3. Be concise and accurate
4. Cite specific information from the context when possible
5. If the context is contradictory, acknowledge the contradiction

# Guidelines

- Use clear, direct language
- Organize your answer logically (use bullet points or paragraphs as appropriate)
- Focus on answering the specific question asked
- Don't add information not present in the context
- If asked for a summary, be comprehensive but concise

Now answer the query based on the context provided.
"""
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | str | User's query |
| `contexts` | str | Retrieved contexts (formatted with scores) |

### Context Formatting

```python
def format_contexts(contexts: List[Dict]) -> str:
    """Format contexts for prompt."""
    formatted = []
    for i, ctx in enumerate(contexts, 1):
        formatted.append(
            f"[Context {i}] (Relevance: {ctx['score']:.2f})\n{ctx['text']}"
        )
    return "\n\n".join(formatted)
```

### Tuning Parameters

```python
QueryGenerationConfig(
    temperature=0.3,      # Balanced creativity
    max_tokens=1000,      # Sufficient for detailed answers
)
```

---

## Relationship Description Prompt (LightRAG)

### Purpose

Generate natural language descriptions for relationships (LightRAG-specific).

### Template

```python
RELATIONSHIP_DESCRIPTION_PROMPT = """
Describe the relationship between two entities in one concise sentence.

# Source Entity
Name: {source_name}
Type: {source_type}
Description: {source_description}

# Target Entity
Name: {target_name}
Type: {target_type}
Description: {target_description}

# Relationship Type
{relationship_type}

# Context
{context_text}

Generate a single sentence that naturally describes how the source entity relates to the target entity. Be specific and informative.

Example:
Source: IPCC (ORGANIZATION)
Target: Climate Report 2024 (EVENT)
Type: PUBLISHED
Context: "The IPCC published its comprehensive assessment..."

Output: "IPCC published the Climate Report 2024, a comprehensive assessment of climate science."

Now generate the relationship description.
"""
```

---

## Prompt Tuning Templates

### Domain Analysis Prompt

```python
DOMAIN_ANALYSIS_PROMPT = """
Analyze the following sample documents to identify domain-specific terminology and entity patterns.

# Sample Documents

{sample_texts}

# Task

Identify:
1. **Domain-specific entity types** not covered by standard types (PERSON, ORGANIZATION, etc.)
2. **Key terminology** that should be recognized
3. **Common entity patterns** (e.g., citation formats, product codes)
4. **Relationship patterns** specific to this domain

Provide:
- List of custom entity types with definitions
- Key terms to recognize
- Example entities for each type
- Suggested relationship types

Format your response as JSON.
"""
```

### Error Analysis Prompt

```python
ERROR_ANALYSIS_PROMPT = """
Analyze extraction errors to improve the prompt.

# Ground Truth Entities
{ground_truth}

# Extracted Entities
{extracted}

# Errors
{errors}

# Task

Analyze why the extraction failed:
1. Missing entities (in ground truth but not extracted)
2. Incorrect entities (extracted but not in ground truth)
3. Wrong types (entity name correct but type wrong)

For each error category, suggest prompt improvements:
- Additional instructions
- Examples to add
- Clarifications needed

Format your response as structured recommendations.
"""
```

### Improved Prompt Generation

```python
IMPROVED_PROMPT_GENERATION = """
Generate an improved entity extraction prompt based on error analysis.

# Current Prompt
{current_prompt}

# Domain Characteristics
{domain_analysis}

# Common Errors
{error_patterns}

# Task

Create an improved prompt that:
1. Addresses the identified errors
2. Includes domain-specific instructions
3. Adds relevant few-shot examples
4. Clarifies ambiguous cases

Maintain the same output format but improve instructions and examples.
"""
```

---

## Few-Shot Examples Library

### Medical Domain Example

```json
{
  "entities": [
    {
      "name": "Type 2 Diabetes",
      "type": "DISEASE",
      "description": "Metabolic disorder characterized by high blood sugar",
      "confidence": 1.0
    },
    {
      "name": "Metformin",
      "type": "DRUG",
      "description": "First-line medication for Type 2 Diabetes",
      "confidence": 1.0
    },
    {
      "name": "HbA1c",
      "type": "BIOMARKER",
      "description": "Glycated hemoglobin test measuring average blood sugar",
      "confidence": 0.95
    }
  ],
  "relationships": [
    {
      "source": "Metformin",
      "target": "Type 2 Diabetes",
      "type": "TREATS",
      "description": "Metformin is first-line treatment for Type 2 Diabetes",
      "confidence": 1.0
    },
    {
      "source": "HbA1c",
      "target": "Type 2 Diabetes",
      "type": "DIAGNOSES",
      "description": "HbA1c test used to diagnose and monitor Type 2 Diabetes",
      "confidence": 0.95
    }
  ]
}
```

### Legal Domain Example

```json
{
  "entities": [
    {
      "name": "Brown v. Board of Education",
      "type": "LEGAL_CASE",
      "description": "Landmark 1954 Supreme Court case on school segregation",
      "confidence": 1.0
    },
    {
      "name": "Equal Protection Clause",
      "type": "LEGAL_DOCTRINE",
      "description": "14th Amendment clause requiring equal treatment under law",
      "confidence": 1.0
    },
    {
      "name": "Earl Warren",
      "type": "PERSON",
      "description": "Chief Justice of the United States (1953-1969)",
      "confidence": 1.0
    }
  ],
  "relationships": [
    {
      "source": "Brown v. Board of Education",
      "target": "Equal Protection Clause",
      "type": "CITES",
      "description": "Case relied on Equal Protection Clause to rule segregation unconstitutional",
      "confidence": 1.0
    },
    {
      "source": "Earl Warren",
      "target": "Brown v. Board of Education",
      "type": "PRESIDED_OVER",
      "description": "Chief Justice Warren wrote the unanimous opinion",
      "confidence": 1.0
    }
  ]
}
```

### Financial Domain Example

```json
{
  "entities": [
    {
      "name": "Federal Reserve",
      "type": "ORGANIZATION",
      "description": "Central banking system of the United States",
      "confidence": 1.0
    },
    {
      "name": "Federal Funds Rate",
      "type": "FINANCIAL_INSTRUMENT",
      "description": "Interest rate at which banks lend reserves overnight",
      "confidence": 1.0
    },
    {
      "name": "Quantitative Easing",
      "type": "POLICY",
      "description": "Monetary policy of purchasing securities to increase money supply",
      "confidence": 0.95
    }
  ],
  "relationships": [
    {
      "source": "Federal Reserve",
      "target": "Federal Funds Rate",
      "type": "SETS",
      "description": "Federal Reserve sets the target Federal Funds Rate",
      "confidence": 1.0
    },
    {
      "source": "Federal Reserve",
      "target": "Quantitative Easing",
      "type": "IMPLEMENTS",
      "description": "Federal Reserve implemented Quantitative Easing during financial crisis",
      "confidence": 1.0
    }
  ]
}
```

---

## Prompt Validation

### Output Format Validation

```python
def validate_extraction_output(response: str) -> bool:
    """Validate LLM response matches expected format."""
    try:
        data = json.loads(response)

        # Check required keys
        if "entities" not in data:
            return False

        # Validate entity structure
        for entity in data["entities"]:
            required = {"name", "type", "description", "confidence"}
            if not required.issubset(entity.keys()):
                return False

            # Validate types
            if not isinstance(entity["name"], str):
                return False
            if not isinstance(entity["confidence"], (int, float)):
                return False
            if not (0.0 <= entity["confidence"] <= 1.0):
                return False

        return True

    except json.JSONDecodeError:
        return False
```

### Retry with Clarification

```python
CLARIFICATION_PROMPT = """
The previous response was not in valid JSON format. Please return your response as valid JSON matching this structure:

{{
  "entities": [
    {{"name": "...", "type": "...", "description": "...", "confidence": 0.95}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "description": "...", "confidence": 0.90}}
  ]
}}

Previous response:
{previous_response}

Please provide the corrected JSON response.
"""
```

---

## Prompt Metrics

### Quality Metrics

Track prompt performance:

```python
@dataclass
class PromptMetrics:
    """Metrics for prompt evaluation."""
    success_rate: float          # Valid JSON responses
    avg_confidence: float         # Average extraction confidence
    entity_precision: float       # Against ground truth
    entity_recall: float
    relationship_precision: float
    relationship_recall: float
    avg_response_length: int      # Tokens
    parse_error_rate: float
```

### A/B Testing Framework

```python
class PromptABTest:
    """A/B test different prompts."""

    def __init__(self, prompt_a: str, prompt_b: str):
        self.prompt_a = prompt_a
        self.prompt_b = prompt_b

    async def compare(
        self,
        test_cases: List[str],
        ground_truth: Dict
    ) -> Dict[str, PromptMetrics]:
        """Compare prompts on test cases."""
        metrics_a = await self.evaluate(self.prompt_a, test_cases, ground_truth)
        metrics_b = await self.evaluate(self.prompt_b, test_cases, ground_truth)

        return {
            "prompt_a": metrics_a,
            "prompt_b": metrics_b,
            "improvement": self.compute_improvement(metrics_a, metrics_b)
        }
```

---

## Summary

This specification defines:

- **5 core prompts** for extraction, summarization, generation
- **3 prompt tuning prompts** for domain adaptation
- **3 few-shot example sets** for medical, legal, financial domains
- **Output validation** schemas and retry logic
- **Prompt metrics** for evaluation
- **A/B testing framework** for prompt optimization

**Prompt Characteristics:**

| Prompt | Temperature | Max Tokens | Use Case |
|--------|-------------|------------|----------|
| Entity Extraction | 0.0 | 2000 | Deterministic extraction |
| Relationship Extraction | 0.0 | 2000 | Deterministic extraction |
| Community Report | 0.1 | 2000 | Slight creativity for summaries |
| Query Generation | 0.3 | 1000 | Balanced for natural responses |
| Prompt Tuning | 0.2 | 1500 | Creative but controlled |

**Next Steps:**

- Implement prompts in `prompt_tune/templates.py`
- Create few-shot example library in `prompts/examples/`
- Build prompt validation in `utils/prompt_validation.py`
- Add domain-specific prompt sets
- Implement A/B testing framework

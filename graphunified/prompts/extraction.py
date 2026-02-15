"""Prompt templates for entity and relationship extraction."""

ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting structured information from text. Your task is to extract entities from the provided text chunks.

Extract the following entity types:
- PERSON: Individual people (e.g., "Albert Einstein", "Marie Curie")
- ORGANIZATION: Companies, institutions, organizations (e.g., "NASA", "United Nations", "Apple Inc.")
- LOCATION: Geographic locations, places (e.g., "New York City", "Mount Everest", "Pacific Ocean")
- CONCEPT: Abstract concepts, theories, ideas (e.g., "machine learning", "democracy", "evolution")
- EVENT: Named events, incidents, occurrences (e.g., "World War II", "Apollo 11 landing", "COVID-19 pandemic")

For each entity you extract, provide:
- name: The entity name exactly as it appears in the text (preserve capitalization)
- type: One of the entity types listed above
- description: A brief 1-2 sentence description of what this entity is or represents
- confidence: A score from 0.0 to 1.0 indicating how confident you are in this extraction

Guidelines:
1. Extract only entities that are clearly mentioned in the text
2. Do not infer or hallucinate entities not present in the text
3. Use the most specific entity name mentioned (e.g., "Dr. Jane Smith" not just "Smith")
4. If an entity appears multiple times with different forms, use the most complete form
5. Assign lower confidence scores (0.5-0.7) if you're uncertain about the entity type or extraction

Return your results as a JSON object with this exact structure:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "ENTITY_TYPE",
      "description": "Brief description of the entity",
      "confidence": 0.95
    }}
  ]
}}

Text chunks to process:
{chunk_texts}

Return only the JSON object, no additional text or explanation."""

RELATIONSHIP_EXTRACTION_PROMPT = """You are an expert at extracting relationships between entities. Your task is to identify and extract relationships between the entities found in the previous extraction step.

Extract relationships of these types:
- RELATED_TO: General association or connection between entities
- PART_OF: Component, member, or subset relationship (e.g., "NASA is PART_OF US Government")
- LOCATED_IN: Geographic containment (e.g., "Paris is LOCATED_IN France")
- WORKS_FOR: Employment or affiliation relationship (e.g., "Jane Smith WORKS_FOR NASA")
- CAUSES: Causal relationship (e.g., "Global warming CAUSES sea level rise")

For each relationship you extract, provide:
- source: Source entity name (must EXACTLY match an entity name from the list below)
- target: Target entity name (must EXACTLY match an entity name from the list below)
- type: One of the relationship types listed above
- description: A brief 1 sentence description of the relationship
- confidence: A score from 0.0 to 1.0 indicating confidence in this relationship

Guidelines:
1. Only extract relationships between entities that appear in the entity list below
2. Source and target names must EXACTLY match entity names (case-sensitive)
3. Extract only relationships that are explicitly stated or strongly implied in the text
4. Do not infer distant or speculative relationships
5. Relationships are directed (source → target), so order matters
6. Assign lower confidence (0.5-0.7) if the relationship is only weakly implied

Available entities:
{entity_names}

Text chunks to process:
{chunk_texts}

Return your results as a JSON object with this exact structure:
{{
  "relationships": [
    {{
      "source": "Source Entity Name",
      "target": "Target Entity Name",
      "type": "RELATIONSHIP_TYPE",
      "description": "Brief description of the relationship",
      "confidence": 0.90
    }}
  ]
}}

Return only the JSON object, no additional text or explanation."""

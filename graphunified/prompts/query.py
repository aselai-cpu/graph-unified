"""Prompt templates for query classification and response synthesis."""

QUERY_CLASSIFICATION_PROMPT = """You are a query classifier for a knowledge graph retrieval system. Your task is to classify the user's query into one of the predefined query types to determine the best retrieval strategy.

Query Types:

1. FACTOID - Specific fact-based questions with short, definitive answers
   Examples: "What is GraphRAG?", "Who invented the telephone?", "When was Python created?"

2. EXPLORATORY - Broad, open-ended questions requiring comprehensive overview
   Examples: "Tell me about machine learning", "Summarize the key themes", "What are the main concepts?"

3. RELATIONAL - Questions about relationships, connections, or interactions between entities
   Examples: "How does X relate to Y?", "What is the connection between A and B?", "What are the dependencies?"

4. THEMATIC - Questions about themes, trends, patterns, or abstract concepts across the knowledge
   Examples: "What are the emerging trends?", "What themes connect these topics?", "What patterns exist?"

5. COMPARATIVE - Comparison questions asking about similarities or differences
   Examples: "Compare X and Y", "What's the difference between A and B?", "How do these approaches differ?"

6. TEMPORAL - Time-based questions about sequences, timelines, or historical progression
   Examples: "What happened after X?", "When did this occur?", "What is the timeline of events?"

User Query: {query}

Analyze the query and return a JSON object with the following structure:
{{
  "query_type": "FACTOID|EXPLORATORY|RELATIONAL|THEMATIC|COMPARATIVE|TEMPORAL",
  "confidence": 0.95,
  "reasoning": "Brief explanation of why this classification was chosen"
}}

Return ONLY the JSON object, no additional text."""

RESPONSE_SYNTHESIS_PROMPT = """You are an AI assistant synthesizing an answer to a user's question based on retrieved information from a knowledge graph.

User Question: {query}
Query Type: {query_type}

Retrieved Context:
{context}

Your Task:
1. Answer the user's question directly and comprehensively using the retrieved context
2. Integrate information from multiple sources coherently
3. Cite sources using [Source N] notation where N is the source number
4. Be concise but thorough (aim for 2-4 paragraphs)
5. If the context lacks sufficient information to fully answer the question, acknowledge this clearly
6. Maintain factual accuracy - do not add information not present in the context

Guidelines:
- For FACTOID queries: Provide a concise, direct answer
- For EXPLORATORY queries: Give a comprehensive overview covering key aspects
- For RELATIONAL queries: Focus on connections and relationships
- For THEMATIC queries: Identify and discuss patterns and themes
- For COMPARATIVE queries: Highlight similarities and differences systematically
- For TEMPORAL queries: Present information in chronological order

Answer:"""

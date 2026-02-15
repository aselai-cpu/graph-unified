# How to Tune Extraction Prompts for Your Domain

## Goal

Optimize entity and relationship extraction for your specific domain (legal, medical, financial, etc.) to improve extraction quality by 15-30%.

## Prerequisites

- Graph-Unified installed and configured
- Sample documents from your domain (50-100 docs recommended)
- 2-3 hours for tuning and evaluation

## Time Required

2-3 hours (most time spent on prompt iteration)

## Overview

**Default extraction prompts are generic.** They work reasonably well across domains but miss domain-specific nuances:

- **Medical:** Miss disease-symptom relationships, drug interactions
- **Legal:** Miss statute citations, case precedents
- **Financial:** Miss company-metric relationships, SEC filings

**Prompt tuning** adapts extraction to your domain by:
1. Analyzing your corpus for domain-specific patterns
2. Generating domain-specific instructions and examples
3. Testing extraction quality improvements

**Expected improvement:** +15-30% F1 score on domain-specific entities and relationships.

---

## Step 1: Prepare Evaluation Dataset

Before tuning, create ground truth for measuring improvement.

### 1.1 Sample Documents

Select 10-20 representative documents:

```bash
# Create evaluation directory
mkdir -p tuning/eval_docs

# Copy diverse samples
cp input/doc_001.txt tuning/eval_docs/
cp input/doc_042.txt tuning/eval_docs/
# ... (10-20 total)
```

**Selection criteria:**
- Representative of corpus diversity
- Mix of simple and complex documents
- Cover key domain concepts

### 1.2 Manually Label Entities

Create ground truth for entities:

```bash
cat > tuning/eval_docs/doc_001_entities.json << 'EOF'
{
  "document": "doc_001.txt",
  "entities": [
    {
      "text": "Metformin",
      "type": "DRUG",
      "description": "Oral antidiabetic medication"
    },
    {
      "text": "Type 2 diabetes",
      "type": "DISEASE",
      "description": "Metabolic disorder affecting blood sugar"
    },
    {
      "text": "Hypoglycemia",
      "type": "SIDE_EFFECT",
      "description": "Low blood sugar condition"
    }
  ]
}
EOF
```

**Labeling tips:**
- Label all entities of interest (don't skip)
- Use consistent type names
- Write clear descriptions

### 1.3 Manually Label Relationships

Create ground truth for relationships:

```bash
cat > tuning/eval_docs/doc_001_relationships.json << 'EOF'
{
  "document": "doc_001.txt",
  "relationships": [
    {
      "source": "Metformin",
      "target": "Type 2 diabetes",
      "type": "TREATS",
      "description": "Metformin is first-line treatment for type 2 diabetes"
    },
    {
      "source": "Metformin",
      "target": "Hypoglycemia",
      "type": "CAUSES",
      "description": "Metformin can cause hypoglycemia in rare cases"
    }
  ]
}
EOF
```

**Time estimate:** 15-20 minutes per document.

---

## Step 2: Run Baseline Extraction

Extract entities using default (generic) prompts:

```bash
graph-unified index \
  --input tuning/eval_docs \
  --output tuning/baseline_output \
  --config settings.yaml
```

### Evaluate Baseline

```bash
graph-unified evaluate \
  --predictions tuning/baseline_output/entities.parquet \
  --ground-truth tuning/eval_docs/*_entities.json \
  --output tuning/baseline_metrics.json
```

**Example output:**
```json
{
  "entity_metrics": {
    "precision": 0.72,
    "recall": 0.68,
    "f1": 0.70
  },
  "relationship_metrics": {
    "precision": 0.65,
    "recall": 0.58,
    "f1": 0.61
  },
  "errors": {
    "missed_entities": 32,
    "false_positive_entities": 18,
    "missed_relationships": 21
  }
}
```

**Baseline is your target to beat.** Aim for +0.10 F1 improvement.

---

## Step 3: Generate Domain-Specific Prompts

Graph-Unified includes automatic prompt tuning:

```bash
graph-unified prompt-tune \
  --input input/ \
  --domain "medical" \
  --output prompts/medical_extraction.yaml
```

**What this does:**
1. Analyzes your corpus (100 random documents)
2. Identifies domain-specific entity types
3. Extracts example entities for few-shot learning
4. Generates domain-specific instructions
5. Creates tuned prompt template

**Example output (`prompts/medical_extraction.yaml`):**

```yaml
# Auto-generated prompt template for medical domain

entity_extraction_prompt: |
  You are an expert medical information extractor. Extract entities from the
  following medical text.

  # Domain-Specific Entity Types
  - DRUG: Medications, pharmaceuticals, treatments
  - DISEASE: Conditions, disorders, illnesses
  - SYMPTOM: Patient symptoms, clinical presentations
  - PROCEDURE: Medical procedures, surgeries, tests
  - SIDE_EFFECT: Adverse effects, complications

  # Domain-Specific Instructions
  - For drugs, extract both generic and brand names
  - Link symptoms to diseases when mentioned together
  - Include dosage information in drug descriptions
  - Distinguish between diseases and symptoms (e.g., "fever" is symptom, "malaria" is disease)

  # Examples
  Input: "The patient was prescribed Metformin 500mg for type 2 diabetes."
  Entities:
  - Metformin (DRUG): "Oral antidiabetic medication, 500mg dose"
  - Type 2 diabetes (DISEASE): "Metabolic disorder affecting blood sugar regulation"

  Input: "Common side effects include nausea and headache."
  Entities:
  - Nausea (SIDE_EFFECT): "Gastrointestinal adverse effect"
  - Headache (SIDE_EFFECT): "Neurological adverse effect"

  # Task
  Extract entities from the following text:

  {chunk_text}

  Output format:
  entity_name<|>entity_type<|>entity_description
  ##

relationship_extraction_prompt: |
  You are an expert medical relationship extractor. Identify relationships
  between entities in medical text.

  # Domain-Specific Relationship Types
  - TREATS: Drug treats disease
  - CAUSES: Entity causes another (disease causes symptom, drug causes side effect)
  - DIAGNOSES: Procedure diagnoses disease
  - INDICATES: Symptom indicates disease
  - CONTRAINDICATES: Drug contraindicated for disease

  # Examples
  Input: "Aspirin treats headaches but can cause stomach ulcers."
  Relationships:
  - Aspirin<|>TREATS<|>Headaches<|>"Pain relief mechanism"
  - Aspirin<|>CAUSES<|>Stomach ulcers<|>"Gastrointestinal side effect"

  # Task
  Extract relationships from the following text:

  {chunk_text}

  Entities present: {entities}

  Output format:
  source_entity<|>relationship_type<|>target_entity<|>description
  ##
```

**Key components:**
- **Domain-specific entity types:** Replaces generic types with medical ones
- **Domain-specific instructions:** Rules for extraction (e.g., dosage info)
- **Few-shot examples:** Demonstrations of correct extraction
- **Format specification:** Output structure

---

## Step 4: Manual Prompt Refinement (Optional)

Auto-generated prompts are good starting points. Refine for better results.

### 4.1 Add Domain Terminology

**Problem:** Generic prompt misses domain-specific abbreviations.

**Fix:** Add terminology section:

```yaml
entity_extraction_prompt: |
  ...

  # Domain Terminology
  - "MI" = Myocardial Infarction (heart attack) → DISEASE
  - "BP" = Blood Pressure → MEASUREMENT
  - "ACE inhibitor" = Angiotensin-Converting Enzyme inhibitor → DRUG
  - "CBC" = Complete Blood Count → PROCEDURE
  - "Rx" = Prescription → DRUG

  ...
```

### 4.2 Add Disambiguation Rules

**Problem:** Ambiguous terms extracted incorrectly.

**Fix:** Add disambiguation instructions:

```yaml
entity_extraction_prompt: |
  ...

  # Disambiguation Rules
  - "Cold" can be disease (common cold) or symptom (cold extremities). Use context.
  - "High" often refers to measurement (high blood pressure), not entity.
  - "Treatment" is not an entity; the specific drug/procedure is.

  ...
```

### 4.3 Add More Examples

**Problem:** Auto-generated examples don't cover edge cases.

**Fix:** Add examples from your ground truth:

```yaml
entity_extraction_prompt: |
  ...

  # Additional Examples
  Input: "Patient presents with acute MI, administered 300mg aspirin."
  Entities:
  - Acute MI (DISEASE): "Myocardial infarction (heart attack)"
  - Aspirin (DRUG): "Antiplatelet medication, 300mg dose"

  ...
```

**Best practices:**
- 3-5 examples per entity type
- Cover common and edge cases
- Show correct handling of ambiguity

---

## Step 5: Test Tuned Prompts

Re-run extraction with tuned prompts:

```bash
# Update settings to use tuned prompts
graph-unified index \
  --input tuning/eval_docs \
  --output tuning/tuned_output \
  --config settings.yaml \
  --prompts prompts/medical_extraction.yaml
```

### Evaluate Improvement

```bash
graph-unified evaluate \
  --predictions tuning/tuned_output/entities.parquet \
  --ground-truth tuning/eval_docs/*_entities.json \
  --output tuning/tuned_metrics.json
```

**Example output:**
```json
{
  "entity_metrics": {
    "precision": 0.84,
    "recall": 0.82,
    "f1": 0.83
  },
  "relationship_metrics": {
    "precision": 0.78,
    "recall": 0.72,
    "f1": 0.75
  },
  "improvement_over_baseline": {
    "entity_f1": +0.13,
    "relationship_f1": +0.14
  }
}
```

**Success criteria:**
- Entity F1 improvement: +0.10 or more
- Relationship F1 improvement: +0.08 or more
- Precision doesn't drop (avoid false positives)

**If improvement < 0.10:**
- Add more examples to prompt
- Refine disambiguation rules
- Check if entity types are correct for domain

---

## Step 6: Iterate (If Needed)

If improvement is insufficient, analyze errors and iterate.

### 6.1 Error Analysis

```bash
graph-unified evaluate \
  --predictions tuning/tuned_output/entities.parquet \
  --ground-truth tuning/eval_docs/*_entities.json \
  --output tuning/tuned_metrics.json \
  --show-errors
```

**Output shows specific errors:**
```
Missed Entities (Recall Errors):
1. "Ibuprofen" (DRUG) in doc_003.txt
   - Reason: Not extracted
   - Context: "...administered ibuprofen 400mg..."

2. "Chronic kidney disease" (DISEASE) in doc_007.txt
   - Reason: Extracted as two separate entities ("Chronic" + "kidney disease")

False Positive Entities (Precision Errors):
1. "Healthy" (STATUS) in doc_005.txt
   - Reason: Not a valid entity type
   - Context: "...patient appears healthy..."
```

### 6.2 Fix Identified Issues

**Issue:** Missed "Ibuprofen"

**Fix:** Add example with similar drugs:

```yaml
# Additional example
Input: "Administered ibuprofen 400mg and paracetamol 500mg for pain."
Entities:
- Ibuprofen (DRUG): "NSAID pain reliever, 400mg dose"
- Paracetamol (DRUG): "Analgesic, 500mg dose"
```

**Issue:** "Chronic kidney disease" split

**Fix:** Add instruction:

```yaml
# Instructions
- Extract multi-word entities as single entities (e.g., "chronic kidney disease", not separate words)
- Use context to determine entity boundaries
```

**Issue:** False positive "Healthy"

**Fix:** Add negative example:

```yaml
# Non-Entities (Do Not Extract)
- Adjectives without specific conditions ("healthy", "ill", "sick")
- Generic terms ("treatment", "medication" without specifics)
- Measurements alone ("120/80", "5.4" without context)
```

### 6.3 Re-Test

```bash
# Update prompts with fixes
# Re-run extraction and evaluation
graph-unified index --input tuning/eval_docs --output tuning/tuned_v2_output ...
graph-unified evaluate --predictions tuning/tuned_v2_output/entities.parquet ...
```

**Iterate until:**
- F1 improvement plateaus (diminishing returns)
- or F1 > 0.85 (excellent quality)

---

## Step 7: Deploy Tuned Prompts

Once satisfied with tuned prompts, use them for full corpus indexing.

### Update Configuration

Edit `settings.yaml`:

```yaml
# Add prompt configuration
prompts:
  extraction_template: "prompts/medical_extraction.yaml"
```

### Re-Index Full Corpus

```bash
graph-unified index \
  --input input/ \
  --output output/ \
  --config settings.yaml
```

**This re-extracts all entities using tuned prompts.** The improvement will apply to the entire corpus.

---

## Domain-Specific Examples

### Medical Domain

**Custom entity types:**
```yaml
entity_types:
  - DRUG
  - DISEASE
  - SYMPTOM
  - PROCEDURE
  - SIDE_EFFECT
  - ANATOMICAL_STRUCTURE
```

**Custom relationship types:**
```yaml
relationship_types:
  - TREATS
  - CAUSES
  - DIAGNOSES
  - INDICATES
  - CONTRAINDICATES
  - LOCATED_IN
```

### Legal Domain

**Custom entity types:**
```yaml
entity_types:
  - STATUTE
  - CASE
  - COURT
  - LEGAL_PERSON
  - LEGAL_CONCEPT
  - JURISDICTION
```

**Custom relationship types:**
```yaml
relationship_types:
  - CITES
  - OVERRULES
  - DISTINGUISHES
  - APPLIES
  - VIOLATES
```

**Domain-specific instructions:**
```yaml
entity_extraction_prompt: |
  ...
  # Legal-specific instructions
  - Extract full case names (e.g., "Roe v. Wade", not just "Roe")
  - Include statute numbers (e.g., "18 U.S.C. § 1001")
  - Identify jurisdiction for courts (e.g., "9th Circuit Court of Appeals")
  ...
```

### Financial Domain

**Custom entity types:**
```yaml
entity_types:
  - COMPANY
  - METRIC
  - FINANCIAL_INSTRUMENT
  - REGULATORY_FILING
  - PERSON
  - EVENT
```

**Custom relationship types:**
```yaml
relationship_types:
  - HAS_METRIC
  - REPORTS
  - ACQUIRES
  - COMPETES_WITH
  - REGULATES
```

**Domain-specific instructions:**
```yaml
entity_extraction_prompt: |
  ...
  # Financial-specific instructions
  - Extract ticker symbols with company names (e.g., "Apple Inc. (AAPL)")
  - Include time periods for metrics (e.g., "Q4 2024 revenue")
  - Extract both amounts and percentages (e.g., "$1.2B", "15% growth")
  ...
```

---

## Common Pitfalls

### Pitfall 1: Over-Specification

**Problem:** Prompt becomes too long and prescriptive, confusing the model.

**Symptom:** Tuned prompt performs worse than baseline.

**Fix:**
- Keep instructions concise (10-15 rules max)
- Focus on high-impact disambiguation
- Remove redundant examples

### Pitfall 2: Insufficient Examples

**Problem:** Too few examples, model doesn't generalize.

**Symptom:** Extracts entities from examples but misses similar entities.

**Fix:**
- Provide 3-5 examples per entity type
- Cover common patterns and edge cases
- Use diverse phrasing

### Pitfall 3: Ignoring Precision

**Problem:** Focus on recall (finding all entities) at expense of precision.

**Symptom:** F1 increases but precision drops below 0.75.

**Fix:**
- Add negative examples (what NOT to extract)
- Add specificity rules ("medication" alone is not DRUG)
- Post-filter low-confidence extractions

### Pitfall 4: Not Measuring Improvement

**Problem:** Tuning without evaluation, assuming prompts are better.

**Symptom:** Unquantified "feels better" improvements.

**Fix:**
- Always create ground truth evaluation set
- Measure baseline before tuning
- Require +0.10 F1 improvement to deploy

---

## Checklist

**Before starting:**
- [ ] Created evaluation dataset (10-20 labeled docs)
- [ ] Measured baseline performance
- [ ] Identified domain-specific entity/relationship types

**During tuning:**
- [ ] Generated auto-tuned prompts
- [ ] Added domain terminology
- [ ] Added disambiguation rules
- [ ] Added 3-5 examples per entity type
- [ ] Tested and measured improvement
- [ ] Iterated on errors (if needed)

**Before deployment:**
- [ ] Achieved +0.10 F1 improvement over baseline
- [ ] Precision remains > 0.75
- [ ] Tested on held-out documents (not used in tuning)
- [ ] Updated settings.yaml with tuned prompts
- [ ] Documented tuning decisions for future reference

---

## Success Metrics

| Metric | Baseline | Target | Excellent |
|--------|----------|--------|-----------|
| Entity F1 | 0.70 | 0.80 | 0.90+ |
| Relationship F1 | 0.60 | 0.70 | 0.80+ |
| Precision | 0.72 | 0.78 | 0.85+ |
| Recall | 0.68 | 0.78 | 0.85+ |

**Improvement over baseline:**
- Good: +0.10 F1
- Excellent: +0.15 F1
- Outstanding: +0.20 F1

---

## Next Steps

- **Test on full corpus:** Re-index with tuned prompts, measure retrieval quality
- **Optimize further:** Try few-shot vs. zero-shot, different model tiers (Haiku/Sonnet)
- **Share prompts:** Contribute domain-specific prompts to Graph-Unified community
- **Monitor drift:** Extraction quality may change as corpus evolves; re-tune periodically

**Related guides:**
- [Evaluating Retrieval Quality](04-evaluate-quality.md)
- [Choosing Entity Types](../reference/entity-types.md)
- [Prompt Engineering Best Practices](../explanation/prompt-engineering.md)


# Environment Variables Setup Guide

## Quick Start with .env Files

GraphRAG Unified supports `.env` files for easy API key management. This is the **recommended approach** for local development and production deployments.

### 1. Create .env File

```bash
# Copy the example template
cp .env.example .env

# Edit with your API keys
nano .env  # or use your favorite editor
```

### 2. Add Your API Keys

**Minimal setup (with Voyage AI embeddings):**
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
VOYAGE_API_KEY=pa-your-actual-voyage-key-here
```

**Or use FREE local embeddings (recommended):**
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
# No VOYAGE_API_KEY needed!
```

### 3. Run Pipeline

```bash
# The .env file is automatically loaded
python -m graphunified.cli index \
  --input-dir ./corpus \
  --output-dir ./output \
  --config settings.yaml  # or settings-local-embeddings.yaml
```

**That's it!** No need to manually export environment variables.

---

## How .env Files Work

### Automatic Loading

The system **automatically searches** for `.env` files in this order:

1. **Same directory as config file** (e.g., if config is in `./configs/settings.yaml`, looks for `./configs/.env`)
2. **Current working directory** (`./env`)
3. **Parent directories** (up to 3 levels up)

### Example Directory Structure

```
my-project/
├── .env                    # ← Loaded automatically
├── settings.yaml
├── corpus/
│   └── documents/
└── output/
```

### Environment Variable Syntax in YAML

Use `${VAR_NAME}` to reference environment variables:

```yaml
# settings.yaml
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: ${ANTHROPIC_API_KEY}  # ← Reads from .env

embedding:
  provider: voyage
  model: voyage-3
  api_key: ${VOYAGE_API_KEY}  # ← Reads from .env
```

### Optional Variables with Defaults

Use `${VAR_NAME:-default_value}` syntax:

```yaml
storage:
  root_dir: ${STORAGE_ROOT_DIR:-./output}  # ← Defaults to ./output if not set
```

---

## Configuration Examples

### Example 1: Voyage AI Embeddings (Paid)

**.env:**
```bash
ANTHROPIC_API_KEY=sk-ant-api03-abc123...
VOYAGE_API_KEY=pa-xyz789...
```

**settings.yaml:**
```yaml
llm:
  api_key: ${ANTHROPIC_API_KEY}
  provider: anthropic

embedding:
  api_key: ${VOYAGE_API_KEY}
  provider: voyage
  model: voyage-3
```

**Cost:** ~$3.25 per 100 documents (LLM + embeddings)

### Example 2: Local Embeddings (FREE)

**.env:**
```bash
ANTHROPIC_API_KEY=sk-ant-api03-abc123...
# No VOYAGE_API_KEY needed!
```

**settings-local-embeddings.yaml:**
```yaml
llm:
  api_key: ${ANTHROPIC_API_KEY}
  provider: anthropic

embedding:
  provider: local  # ← Use sentence-transformers
  model: BAAI/bge-large-en-v1.5
  api_key: ""  # ← Not needed
```

**Cost:** ~$3.00 per 100 documents (LLM only, embeddings FREE!)

---

## Security Best Practices

### ✅ DO:
- ✅ Use `.env` files for local development
- ✅ Use `.env.example` as a template (safe to commit)
- ✅ Add `.env` to `.gitignore` (already done)
- ✅ Use different `.env` files for dev/staging/prod
- ✅ Rotate API keys regularly
- ✅ Use minimal permissions for API keys

### ❌ DON'T:
- ❌ Commit `.env` files to git (contains secrets!)
- ❌ Share `.env` files via email/Slack
- ❌ Hardcode API keys in YAML files
- ❌ Use production keys in development

### Production Deployments

For production, consider using:
- **Docker secrets** (Docker Swarm/Kubernetes)
- **Cloud provider secrets** (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault)
- **Environment variables** set by CI/CD pipeline
- **Vault** (HashiCorp Vault)

The system works with any method - `.env` files are just one option.

---

## Troubleshooting

### Issue: "Environment variable ${ANTHROPIC_API_KEY} is required but not set"

**Cause:** API key not found in environment or `.env` file.

**Solutions:**

1. **Check .env file exists:**
   ```bash
   ls -la .env  # Should show the file
   ```

2. **Check .env file format:**
   ```bash
   cat .env
   # Should show: ANTHROPIC_API_KEY=sk-ant-...
   # NOT: export ANTHROPIC_API_KEY=...  (wrong)
   ```

3. **Check variable name matches:**
   ```bash
   # .env must use EXACT name from YAML
   # If YAML says ${ANTHROPIC_API_KEY}, .env must use ANTHROPIC_API_KEY
   ```

4. **Manual override (testing only):**
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   python -m graphunified.cli index ...
   ```

### Issue: "Configuration validation failed: api_key: field required"

**Cause:** API key is empty string or missing.

**Solution:** Add valid API key to `.env`:
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

### Issue: Changes to .env not taking effect

**Cause:** Environment variables from shell session take precedence.

**Solution:**
```bash
# Unset any shell env vars
unset ANTHROPIC_API_KEY
unset VOYAGE_API_KEY

# Run again
python -m graphunified.cli index ...
```

### Issue: .env file not found

**Cause:** `.env` file is not in a searchable location.

**Solution:** Put `.env` in one of these locations:
- Same directory as your config file
- Current working directory
- Project root directory

---

## Advanced: Multiple Environments

### Approach 1: Multiple .env Files

```bash
# Development
cp .env.example .env.dev
# Edit .env.dev with dev keys

# Production
cp .env.example .env.prod
# Edit .env.prod with prod keys

# Load specific file
export $(cat .env.dev | xargs)  # Load dev env
python -m graphunified.cli index ...

export $(cat .env.prod | xargs)  # Load prod env
python -m graphunified.cli index ...
```

### Approach 2: Multiple Config Files

```bash
# configs/dev.yaml - uses ${ANTHROPIC_API_KEY}
# configs/prod.yaml - uses ${ANTHROPIC_API_KEY_PROD}

# .env
ANTHROPIC_API_KEY=sk-dev-key...
ANTHROPIC_API_KEY_PROD=sk-prod-key...

# Run with specific config
python -m graphunified.cli index --config configs/dev.yaml
python -m graphunified.cli index --config configs/prod.yaml
```

---

## Verification

### Test .env Loading

```python
# test_env.py
from pathlib import Path
from graphunified.config.settings import Settings

# Load config (automatically loads .env)
settings = Settings.load(Path("settings.yaml"))

# Should not raise error if .env is configured correctly
print(f"LLM provider: {settings.llm.provider}")
print(f"LLM API key: {settings.llm.api_key[:10]}...")  # Print first 10 chars
print(f"Embedding provider: {settings.embedding.provider}")
```

```bash
python test_env.py
# Should output:
# LLM provider: anthropic
# LLM API key: sk-ant-api...
# Embedding provider: voyage (or local)
```

---

## Quick Reference

| Method | Setup Effort | Security | Best For |
|--------|-------------|----------|----------|
| **`.env` file** | ⭐ Easy | ⭐⭐⭐ Good | Local dev, testing |
| **Shell export** | ⭐⭐ Medium | ⭐⭐ OK | Quick tests |
| **CI/CD env vars** | ⭐⭐⭐ Complex | ⭐⭐⭐⭐⭐ Excellent | Production |
| **Cloud secrets** | ⭐⭐⭐⭐ Complex | ⭐⭐⭐⭐⭐ Excellent | Production |

**Recommendation:** Use `.env` files for local development, cloud secrets for production.

---

## Getting API Keys

### Anthropic (Claude) - Required

1. Go to: https://console.anthropic.com/
2. Sign up / Log in
3. Navigate to **API Keys**
4. Click **Create Key**
5. Copy key (starts with `sk-ant-api03-`)

**Pricing:** ~$3 per 100 documents (extraction only)

### Voyage AI - Optional (or use FREE local embeddings)

1. Go to: https://www.voyageai.com/
2. Sign up / Log in
3. Navigate to **API Keys**
4. Click **Create API Key**
5. Copy key (starts with `pa-`)

**Pricing:** ~$0.25 per 100 documents (embeddings)

**Alternative:** Use **FREE local embeddings** instead! See `LOCAL_EMBEDDINGS_GUIDE.md`

---

## Summary

✅ **Recommended Setup:**
```bash
# 1. Copy template
cp .env.example .env

# 2. Edit .env with your keys
nano .env

# 3. Use local embeddings (free!)
python -m graphunified.cli index \
  --input-dir ./corpus \
  --output-dir ./output \
  --config settings-local-embeddings.yaml
```

**Cost:** ~$3 per 100 documents (FREE embeddings!)

---

For more information:
- **Local Embeddings:** See `LOCAL_EMBEDDINGS_GUIDE.md`
- **Configuration:** See `CONFIGURATION.md` (if exists)
- **Issues:** https://github.com/your-repo/issues

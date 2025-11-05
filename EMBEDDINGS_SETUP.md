# EMBEDDINGS_SETUP.md

## Vector Search Embeddings - Setup Guide

Your app currently tries to use OpenAI embeddings, but you only have an OpenRouter API key. **OpenRouter does NOT provide an embeddings endpoint** - it only does chat completions.

### Quick Fix Options:

#### Option 1: Disable Vector Search (Simplest)
Your app will work with just Cypher keyword search (already implemented as fallback).

**Already done in the code** - the app now automatically disables vector RAG if no valid `OPENAI_API_KEY` is set.

No changes needed! Just set your Streamlit secrets without `OPENAI_API_KEY` and the app will use Cypher search only.

---

#### Option 2: Use Free HuggingFace Embeddings (Recommended)
No API key needed, runs locally or on server.

**Steps:**

1. Add to `requirements.txt`:
```
sentence-transformers
langchain-huggingface
```

2. In `streamlit_app.py`, change line 46-48 from:
```python
try:
	from KG.VectorRAG import query_vector_rag
except Exception:
	query_vector_rag = None
```
to:
```python
try:
	from KG.VectorRAG_HuggingFace import query_vector_rag
except Exception:
	query_vector_rag = None
```

3. Commit and push:
```powershell
git add requirements.txt streamlit_app.py KG/VectorRAG_HuggingFace.py
git commit -m "Use HuggingFace embeddings instead of OpenAI"
git push origin main
```

**Pros:** Free, no API keys, works anywhere
**Cons:** Slightly slower first run (downloads model ~80MB), embeddings quality slightly lower than OpenAI's

---

#### Option 3: Get an OpenAI API Key
Sign up at https://platform.openai.com/ and get a real OpenAI key.

Then set in Streamlit secrets:
```toml
OPENAI_API_KEY = "sk-proj-..."  # Real OpenAI key, not OpenRouter
```

**Pros:** Best embedding quality
**Cons:** Costs money (~$0.0001 per 1K tokens)

---

### Current Streamlit Secrets You Need:

```toml
# Neo4j Aura
NEO4J_URI = "neo4j+s://049a7bfd.databases.neo4j.io:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_password_here"
NEO4J_DATABASE = "neo4j"

# For chat completions (already working with OpenRouter)
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

# Optional - only if using Option 3 (OpenAI embeddings)
# OPENAI_API_KEY = "sk-proj-..."

# Vector config (optional)
VECTOR_INDEX_NAME = "default_index"
VECTOR_NODE_LABEL = "Document"
VECTOR_SOURCE_PROPERTY = "text"
VECTOR_EMBEDDING_PROPERTY = "embedding"
VECTOR_TOP_K = "3"
```

---

### What I've Already Fixed:

✅ Fixed Cypher syntax (`exists()` → `IS NOT NULL`)
✅ Made vector RAG optional (auto-disables if no OpenAI key)
✅ Created HuggingFace alternative (`VectorRAG_HuggingFace.py`)
✅ Fixed `requirements.txt` typo

### What You Need To Do:

1. **Push the code fixes:**
```powershell
git add .
git commit -m "Fix embeddings and Neo4j syntax for Streamlit Cloud"
git push origin main
```

2. **Set Streamlit Cloud secrets** (see above TOML)

3. **Choose an embeddings option** (1, 2, or 3 above)

The app will work with Option 1 (no vector search) immediately after you push and set secrets!

# EMBEDDINGS_SETUP.md

## Vector Search Embeddings - Setup Guide

✅ **UPDATE: Now using HuggingFace embeddings by default!**

Your app now automatically uses **free HuggingFace embeddings** instead of requiring OpenAI API keys. No configuration needed!

### What Changed:

The app now supports **multiple embeddings providers** and automatically picks the best one:

1. **HuggingFace** (free, no API key) - **DEFAULT** ⭐
2. **OpenAI** (paid, requires OPENAI_API_KEY) - fallback if HuggingFace not installed

---

### Current Setup (Automatic):

```python
# Auto-detection in KG/VectorRAG.py:
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # Try free option first
except ImportError:
    from langchain_openai import OpenAIEmbeddings  # Fall back to OpenAI
```

**Model used**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Supports Thai language ✅
- Lightweight (~80MB)
- Good quality embeddings
- No API key required
- Runs on server CPU

---

### Quick Comparison:

| Provider | Cost | Thai Support | API Key | Quality |
|----------|------|--------------|---------|---------|
| **HuggingFace** (default) | Free | ✅ Yes | ❌ None | Good |
| OpenAI | ~$0.0001/1K tokens | ✅ Yes | ✅ Required | Best |
| OpenRouter | N/A | N/A | ❌ No embeddings | N/A |

---

### Installation (Already in requirements.txt):

```
langchain-huggingface>=0.1.0
sentence-transformers>=2.2.0
```

Streamlit Cloud will automatically install these on deploy.

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

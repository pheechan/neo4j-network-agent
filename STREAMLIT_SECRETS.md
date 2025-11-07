# Streamlit Cloud Secrets Configuration

## Important: Update Your Streamlit Cloud Secrets

After deploying, you MUST update the secrets in Streamlit Cloud:

1. Go to https://share.streamlit.io/
2. Click on your app: **neo4j-network-agent**
3. Click **Settings** (⚙️) → **Secrets**
4. Replace the secrets with:

```toml
# Neo4j AuraDB
NEO4J_URI = "neo4j+s://049a7bfd.databases.neo4j.io:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "hR4OIDJi0_KXJlqSTWvSnVvNBklXK_uJeoCPffmgNx0"
NEO4J_DATABASE = "neo4j"

# OpenRouter API - NEW KEY with Gemini 2.0 Flash (FREE!)
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://api.openrouter.ai"
OPENROUTER_MODEL = "google/gemini-2.0-flash-exp:free"
```

5. Click **Save**
6. The app will automatically restart with the new API key

## Model Information

**Google Gemini 2.0 Flash Experimental**
- Model ID: `google/gemini-2.0-flash-exp:free`
- Cost: **FREE** (no charges)
- Context: 1.05M tokens
- Output: Up to 8.2K tokens
- Speed: Very fast (1.05s latency)
- Thai Support: Excellent
- Providers: Google AI Studio + Vertex AI

## Benefits
✅ Completely free - no usage charges
✅ Faster than Gemini 1.5
✅ Better at complex instructions
✅ Excellent Thai language support
✅ 1M+ context window (handles large queries)

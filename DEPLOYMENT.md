# Streamlit Cloud Deployment Guide

This guide walks you through deploying the Neo4j Network Agent to Streamlit Cloud.

## Prerequisites

1. **GitHub Account**: Your code should be on GitHub
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io) using your GitHub account
3. **Neo4j Aura Database**: Set up a free Neo4j Aura database
4. **OpenRouter API Key**: Get a free API key from [openrouter.ai](https://openrouter.ai)

## Step 1: Prepare Your Secrets

**IMPORTANT**: Never commit API keys or passwords to git!

Create a `.streamlit/secrets.toml` file locally (this is already in `.gitignore`):

```toml
# Neo4j AuraDB Connection
NEO4J_URI = "neo4j+s://YOUR_AURA_HOST.databases.neo4j.io:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "YOUR_NEO4J_PASSWORD"
NEO4J_DATABASE = "neo4j"

# OpenRouter API (DeepSeek - Free Tier)
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OPENROUTER_API_BASE = "https://api.openrouter.ai"
OPENROUTER_MODEL = "deepseek/deepseek-chat"

# Optional: Chat UI authentication token
CHAT_UI_TOKEN = "YOUR_SECURE_TOKEN"
```

## Step 2: Push Code to GitHub

Make sure your latest code is pushed:

```bash
git push origin main
```

## Step 3: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select:
   - **Repository**: Your GitHub repo
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Click **"Advanced settings"** (optional):
   - **Python version**: 3.9 or higher
5. Click **"Deploy!"**

## Step 4: Configure Secrets in Streamlit Cloud

After deployment starts, go to **Settings** (⚙️) > **Secrets**

Copy the content from your local `.streamlit/secrets.toml` file and paste it there.

Click **"Save"** - the app will automatically restart.

## Step 5: Verify Deployment

1. Wait for deployment to complete (usually 2-5 minutes)
2. Your app will be available at: `https://<your-app-name>.streamlit.app`
3. Test by asking a question in Thai
4. Check that:
   - ✅ Neo4j connection works (no connection errors)
   - ✅ Vector search returns results
   - ✅ LLM responds correctly
   - ✅ Follow-up questions appear
   - ✅ Cache toggle works

## Features Included in v2.2.0

✅ **Working Features**:
- DeepSeek Chat via OpenRouter (free tier)
- Neo4j Aura vector search across all node types
- Anti-hallucination rules (RULE #0, RULE #1)
- Optimal connection path finding
- "Connect by" field display
- Streaming responses (ChatGPT-like)
- Cache control (toggle, clear, regenerate bypass)
- Analytics tracking
- Follow-up question suggestions
- Thai language optimization

## Troubleshooting

### Issue: "Connection refused" or DNS errors
**Solution**: Check Neo4j credentials in Secrets. Make sure NEO4J_URI uses `neo4j+s://` (secure).

### Issue: "OpenRouter API error 401"
**Solution**: Verify OPENROUTER_API_KEY in Secrets is correct.

### Issue: "No results found"
**Solution**: 
1. Check if vector indexes exist in Neo4j
2. Run `create_vector_index.py` locally to rebuild indexes
3. Verify embedding model is accessible

### Issue: App shows "Rerunning..." continuously
**Solution**: Check Streamlit Cloud logs for Python errors. Common causes:
- Missing dependencies in `requirements.txt`
- Import errors
- Configuration issues

## Security Best Practices

⚠️ **CRITICAL**:
1. **Never commit credentials to git**
2. Use `.gitignore` for `.env` and `secrets.toml`
3. Rotate API keys and passwords regularly
4. Use different credentials for dev/prod
5. Check git history for exposed secrets (use `git log -p`)
6. If credentials are exposed in git history, rotate them immediately

## Monitoring & Maintenance

### Check Logs
- Go to your app > Click "Manage app" > "Logs"
- Look for errors in red

### Update Configuration
- Settings > Secrets > Edit
- App restarts automatically after saving

### Redeploy
- Push changes to GitHub `main` branch
- Streamlit Cloud auto-deploys on push

### Resource Limits (Free Tier)
- **CPU**: 1 vCPU
- **RAM**: 1 GB
- **Storage**: Limited
- **Uptime**: Apps sleep after inactivity (wake on access)

## Local Development

To run locally with the same configuration:

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run Streamlit
python -m streamlit run streamlit_app.py
```

Configuration is loaded from `.env` file or `.streamlit/secrets.toml`.

## Support

- **Streamlit Docs**: https://docs.streamlit.io
- **Community Forum**: https://discuss.streamlit.io
- **GitHub Issues**: Create issue in your repo

---

**Current Version**: v2.2.0  
**Last Updated**: November 21, 2025  
**Status**: ✅ Production Ready

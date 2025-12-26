Using Ollama with this project

This project supports running a local Ollama model and will use it when you set:

- LOCAL_LLM=ollama
- LOCAL_LLM_MODEL=scb10x/typhoon2.1-gemma3-4b  (or whichever Ollama model you prefer)
- OLLAMA_URL (optional, default: http://localhost:11434)

Steps to run Ollama locally (recommended):
1. Install Ollama following https://ollama.com/docs (macOS / Linux / Windows installers).
2. Pull the model you want, for example:

   ollama pull scb10x/typhoon2.1-gemma3-4b

3. Start the Ollama daemon (if not running already):

   ollama daemon

4. Configure env vars for the backend (PowerShell example):

```powershell
$env:LOCAL_LLM = 'ollama'
$env:LOCAL_LLM_MODEL = 'scb10x/typhoon2.1-gemma3-4b'
$env:OLLAMA_URL = 'http://localhost:11434'
```

5. Start the backend (uvicorn) and the frontend dev server as documented in the main README.

Notes & troubleshooting
- Ollama must be running locally and the model must be pulled before the backend will successfully call it.
- If the backend times out or throws errors, check the Ollama daemon logs and ensure the model is available.
- For streaming token-level behavior, Ollama supports streaming endpoints. I can wire streaming forwarding to the frontend if you want full token SSE forwarding (next step).

Docker: this repo includes a `docker-compose.dev.yml` and a `DOCKER_TUTORIAL.md` with instructions to run the backend, frontend, Neo4j and an Ollama container for local testing.

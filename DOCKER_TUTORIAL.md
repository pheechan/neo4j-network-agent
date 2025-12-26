Docker setup and Ollama + app quickstart

This tutorial shows how to run the backend, frontend, Neo4j and (optionally) Ollama in Docker for local development.

Prerequisites
- Docker Desktop (Windows) or Docker Engine
- docker-compose (Docker Desktop includes this)

Quick steps (development)
1) Build and start services

```powershell
# From the repository root
docker compose -f docker-compose.dev.yml up --build
```

This will build the backend and frontend images, and start services on these ports by default:
- Backend FastAPI: http://localhost:8000
- Frontend Vite dev server: http://localhost:5173
- Neo4j browser: http://localhost:7474 (bolt:7687)
- Ollama API (if enabled in compose): http://localhost:11434

2) Pull Ollama model (if using Ollama service in compose)

If you use the provided Ollama container, you need to pull the model into the Ollama daemon. Open a second terminal and run:

```powershell
# Exec into the running ollama container
docker compose -f docker-compose.dev.yml exec ollama ollama pull scb10x/typhoon2.1-gemma3-4b
```

If you run Ollama on the host (recommended for GPU access), run on your host:

```powershell
# On host with Ollama installed
ollama pull scb10x/typhoon2.1-gemma3-4b
ollama daemon
```

3) Configure environment

Copy `.env.template` (if you have one) or set the following environment variables for the backend (examples):

```
LOCAL_LLM=ollama
LOCAL_LLM_MODEL=scb10x/typhoon2.1-gemma3-4b
OLLAMA_URL=http://localhost:11434
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=letmein
```

You can add these to your PowerShell session or a `.env` file and Docker Compose will pick them up.

4) Access the app
- Open the frontend at http://localhost:5173 and chat. The frontend will call the backend streaming endpoint at `/api/stream-chat`.

Notes and troubleshooting
- Ollama and large models often require GPU support. Running the Ollama container on a GPU host may need additional docker runtime flags (nvidia runtime). You may prefer to install Ollama on the host and run the daemon there.
- If the Ollama container fails to start or lacks models, run the `ollama pull` command inside the container (see step 2).
- To view logs:

```powershell
# Tail all services
docker compose -f docker-compose.dev.yml logs -f
```

Cleaning up

```powershell
# Stop and remove containers, networks
docker compose -f docker-compose.dev.yml down
```

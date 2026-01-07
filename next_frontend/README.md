# Next.js Neo4j Aura Mask

Minimal Next.js frontend to act as a mask/proxy for Neo4j Aura. It sends user-provided Aura URI and credentials to the backend `/api/aura/query` endpoint and shows results.

How to run (dev):

1. Start the backend (FastAPI) as you normally do (it already exposes `/api/aura/query`).
2. In `next_frontend/` run:

```bash
npm install
npm run dev
```

3. Open http://localhost:5173 and enter your Aura connection details.

Security notes:
- Credentials are posted to the backend for ephemeral queries only. Do not commit secrets.
- Use HTTPS when exposing this app publicly.

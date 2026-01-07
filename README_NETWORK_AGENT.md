# 🌐 STelligence Network Agent

**A beautiful, intelligent chat interface for exploring professional networks using Neo4j knowledge graphs**

![React](https://img.shields.io/badge/React-18-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![Neo4j](https://img.shields.io/badge/Neo4j-5.6-blue)
![Ollama](https://img.shields.io/badge/Ollama-Thai%20LLM-orange)

---

## ✨ Features

### 🤖 Smart Network Intelligence
The AI agent understands **5 types of network queries**:

1. **📍 Shortest Path** - "How can John reach Sarah?"
   - Finds the shortest connection path between two people
   - Shows relationship chain and distance

2. **🤝 Mutual Connections** - "Who do John and Sarah both know?"
   - Discovers common connections between people
   - Shows relationship types

3. **🌐 Personal Network** - "Show me John's network"
   - Analyzes someone's complete network
   - Breaks down by relationship types (REPORTS_TO, COLLABORATES_WITH, etc.)

4. **👤 Best Introducer** - "Who can introduce me to the CEO?"
   - Finds the optimal person to make introductions
   - Considers network size and relationship strength

5. **💬 General Chat** - Natural conversation with context awareness

### 🎨 Beautiful Modern UI
- **Gradient-based design** - Professional blue gradients
- **Smooth animations** - Slide-in effects and hover states  
- **Responsive layout** - Works on desktop and mobile
- **Real-time streaming** - SSE streaming for instant responses
- **Message history** - localStorage persistence

### 🚀 Thai Language Support
- Uses **Ollama** with `scb10x/typhoon2.1-gemma3-4b` model
- Native Thai language understanding
- Bilingual responses (Thai/English)

---

## 🏗️ Architecture

```
┌─────────────────┐
│  React Frontend │  (Port 5173)
│   Vite + Hooks  │
└────────┬────────┘
         │ HTTP/SSE
         ▼
┌─────────────────┐
│  FastAPI Backend│  (Port 8000)
│  /api/chat      │
│  /api/stream-chat│
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌────────┐
│ Neo4j  │  │ Ollama │
│ (7687) │  │(11434) │
└────────┘  └────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB RAM minimum (for Ollama model)
- Ports available: 5173, 8000, 7474, 7687, 11434

### 1. Clone & Setup
```bash
git clone https://github.com/pheechan/neo4j-network-agent.git
cd neo4j-network-agent
```

### 2. Start All Services
```bash
docker compose -f docker-compose.dev.yml up -d
```

This starts:
- **Frontend** (React/Vite) on `http://localhost:5173`
- **Backend** (FastAPI) on `http://localhost:8000`
- **Neo4j** database on `http://localhost:7474` (browser)
- **Ollama** LLM on `http://localhost:11434`

### 3. Populate Sample Network Data
```bash
# Install Python dependencies
pip install -r requirements.txt

# Populate sample professional network
python scripts/populate_sample_network.py
```

This creates **15 people** with **39 relationships** including:
- Executive team (CEO, CFO, CTO, HR Director, Sales Director)
- Department staff (Engineers, Designers, Analysts, etc.)
- Various relationship types (REPORTS_TO, COLLABORATES_WITH, MENTORS, FRIENDS_WITH)

### 4. Open the Chat Interface
Visit: **http://localhost:5173**

---

## 💬 Example Queries

### Shortest Path Queries
```
"How can Dan Developer reach John CEO?"
→ Shows: Dan Developer → Mike CTO → John CEO

"What's the quickest route from Grace Support to Sarah CFO?"
→ Analyzes multiple paths and returns the shortest
```

### Mutual Connections
```
"Who do Tom Engineer and Lisa Marketing both know?"
→ Lists: Mike CTO, Anna Designer, Carol PM

"What are the mutual connections between employees?"
→ Finds common colleagues
```

### Network Analysis
```
"Show me Mike CTO's network"
→ Returns:
  REPORTS_TO: 1 connection (John CEO)
  MENTORS: 1 connection (Tom Engineer)
  COLLABORATES_WITH: 5 connections...

"Analyze Sarah CFO's connections"
→ Shows complete network breakdown
```

### Introduction Requests
```
"Who can introduce Dan Developer to John CEO?"
→ Suggests: Mike CTO (best introducer)
  - Dan knows Mike (REPORTS_TO)
  - Mike knows John (REPORTS_TO)
  - Mike has network size: 12 connections

"Who should I ask to meet the Sales Director?"
→ Finds optimal introduction path
```

---

## 🎯 API Endpoints

### POST `/api/chat`
**Non-streaming chat endpoint**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How can John reach Sarah?", "use_streaming": false}'
```

Response:
```json
{
  "answer": "John CEO can reach Sarah CFO directly...",
  "context": "═══ NETWORK ANALYSIS ═══\n📍 Shortest Path Found...",
  "followups": ["Who else reports to John?", "What's Sarah's network?"]
}
```

### GET `/api/stream-chat`
**SSE streaming endpoint**

```javascript
const eventSource = new EventSource(
  `http://localhost:8000/api/stream-chat?message=How can John reach Sarah?`
);

eventSource.onmessage = (event) => {
  console.log(event.data); // Streaming tokens
};
```

---

## 🐳 Docker Services

### Backend Configuration
```yaml
environment:
  - LOCAL_LLM=ollama
  - LOCAL_LLM_MODEL=scb10x/typhoon2.1-gemma3-4b
  - OLLAMA_URL=http://ollama:11434
  - NEO4J_URI=bolt://neo4j:7687
  - NEO4J_USERNAME=neo4j
  - NEO4J_PASSWORD=letmein123
```

### View Logs
```bash
# Backend logs
docker logs -f neo4j-network-agent-backend-1

# Ollama logs
docker logs -f neo4j-network-agent-ollama-1

# Neo4j logs
docker logs -f neo4j-network-agent-neo4j-1
```

### Restart Services
```bash
# Restart specific service
docker compose -f docker-compose.dev.yml restart backend

# Rebuild and restart
docker compose -f docker-compose.dev.yml up -d --build backend
```

---

## 🔧 Development

### Local Development (without Docker)

1. **Start Neo4j**
```bash
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/letmein123 \
  neo4j:5.6
```

2. **Start Ollama**
```bash
ollama serve
ollama pull scb10x/typhoon2.1-gemma3-4b
```

3. **Start Backend**
```bash
pip install -r requirements.txt
cd backend
uvicorn app:app --reload --port 8000
```

4. **Start Frontend**
```bash
cd frontend
npm install
npm run dev
```

### Environment Variables

Create `.env` file:
```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=letmein123
NEO4J_DATABASE=neo4j

# Ollama
LOCAL_LLM=ollama
LOCAL_LLM_MODEL=scb10x/typhoon2.1-gemma3-4b
OLLAMA_URL=http://localhost:11434

# Fallback LLM (optional)
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

---

## 📊 Neo4j Browser

Access Neo4j Browser at: **http://localhost:7474**

Credentials:
- **Username**: `neo4j`
- **Password**: `letmein123`

### Useful Cypher Queries

```cypher
// View all people
MATCH (p:Person) RETURN p LIMIT 25

// Find shortest path
MATCH (a:Person {name: "Dan Developer"}), (b:Person {name: "John CEO"})
MATCH path = shortestPath((a)-[*]-(b))
RETURN path

// Find mutual connections
MATCH (a:Person {name: "Tom Engineer"})--(mutual)--(b:Person {name: "Lisa Marketing"})
RETURN DISTINCT mutual.name

// Network statistics
MATCH (p:Person)
RETURN p.name, size([(p)-[]-() | 1]) as connections
ORDER BY connections DESC
```

---

## 🎨 Customizing the UI

### Change Theme Colors

Edit `frontend/src/styles.css`:

```css
/* Primary gradient */
background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);

/* Accent colors */
background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
```

### Add Custom Components

The React app uses a simple structure:
- `App.jsx` - Main chat component
- `styles.css` - All styling
- No external UI libraries (pure CSS)

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🆘 Troubleshooting

### "Cannot connect to Neo4j"
```bash
# Check Neo4j is running
docker ps | grep neo4j

# View Neo4j logs
docker logs neo4j-network-agent-neo4j-1

# Restart Neo4j
docker compose -f docker-compose.dev.yml restart neo4j
```

### "Ollama model not found"
```bash
# Pull the model manually
docker exec neo4j-network-agent-ollama-1 ollama pull scb10x/typhoon2.1-gemma3-4b

# Verify model is available
docker exec neo4j-network-agent-ollama-1 ollama list
```

### "Frontend not loading"
```bash
# Check frontend logs
docker logs neo4j-network-agent-frontend-1

# Rebuild frontend
docker compose -f docker-compose.dev.yml up -d --build frontend
```

### "No network data found"
```bash
# Populate sample data
python scripts/populate_sample_network.py

# Verify data in Neo4j browser
# Visit http://localhost:7474
# Run: MATCH (p:Person) RETURN count(p)
```

---

## 📚 Learn More

- [Neo4j Graph Database](https://neo4j.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Hooks](https://react.dev/reference/react)
- [Ollama](https://ollama.ai/)
- [Docker Compose](https://docs.docker.com/compose/)

---

## 🎉 Acknowledgments

- **SCB 10X** for the Typhoon Thai LLM model
- **Neo4j** for the graph database platform
- **FastAPI** for the modern Python API framework
- **React** team for the awesome frontend library

---

**Made with ❤️ for professional network intelligence**

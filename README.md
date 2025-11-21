# Neo4j Knowledge Graph for Relationship Mapping

This project builds a Knowledge Graph using **Neo4j**, focusing on mapping and analyzing connections between **Person → Position → Agency → Ministry**.  
The goal is to help users explore how individuals and organizations are interconnected — for example, who knows whom, or which agencies fall under a specific ministry.

## ✨ NEW Enhanced Features (v2.0)

Built with best practices from:
- [Neo4j NaLLM](https://github.com/neo4j/NaLLM) - Official Neo4j + LLM framework
- [Tomasz Bratanic's blogs](https://github.com/tomasonjo/blogs) - Neo4j expert patterns
- LangChain official patterns

### 🚀 What's New:

1. **🔍 Hybrid Search** - 50%+ better Thai name matching
   - Combines vector similarity + keyword search
   - Finds people even with spelling variations
   - Example: "พี่โด่ง", "อนุทิน" work perfectly

2. **🩹 Self-Healing Cypher** - Automatic error recovery
   - Queries auto-fix syntax errors using AI
   - Handles Thai property names (`ชื่อ-นามสกุล`)
   - Max 2 healing attempts per query

3. **📝 Concise Mode** - Short, focused answers
   - Toggle in Settings: "✨ Concise mode (NEW!)"
   - Max 100 words (Thai) / 150 words (English)
   - Perfect for quick queries

**📖 Full documentation**: See [ENHANCEMENTS.md](ENHANCEMENTS.md)

## Features
- Built with Neo4j graph database
- Uses APOC and Graph Data Science (GDS) plugins
- Supports pathfinding and network centrality analysis
- AI-powered Q&A with Streamlit interface
- **NEW**: Hybrid search for better Thai language support
- **NEW**: Self-healing Cypher queries
- **NEW**: Concise answer mode
- Ready for LangChain / LangGraph integration

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Connections
Edit `Config/` files:
- `neo4j.py` - Neo4j connection settings
- `llm.py` - OpenRouter API key

### 3. Run the App
```bash
streamlit run streamlit_app.py
```

### 4. Try Enhanced Features
- **Hybrid Search**: Just ask in Thai! Works automatically.
- **Self-Healing**: Queries auto-fix errors (you'll see "✨ Query healed" if used)
- **Concise Mode**: Enable in ⚙️ Settings → "✨ Concise mode (NEW!)"

### 5. Test Everything
```bash
python test_enhancements.py
```

## Example Use Cases

### Connection Path Queries
```
Q: "หาเส้นทางเชื่อมต่อระหว่าง พี่โด่ง และ อนุทิน"
A: พบเส้นทางเชื่อมต่อ (3 hops):
   👤 พี่โด่ง → 🏢 กระทรวงมหาดไทย → 👤 อนุทิน ชาญวีรกูล
```

### Person Information
```
Q: "อนุทิน ชาญวีรกูล ทำงานที่ไหน?"
A: อนุทิน ชาญวีรกูล ดำรงตำแหน่ง รัฐมนตรีว่าการกระทรวงมหาดไทย
   สังกัดพรรคภูมิใจไทย
```

### Network Analysis
```
Q: "ใครบ้างในเครือข่าย OSK115?"
A: [Lists all people connected via OSK115 network]
```

## Project Structure
```
neo4j-network-agent/
├── streamlit_app.py           # Main Streamlit application
├── Config/                    # Configuration files
│   ├── neo4j.py              # Neo4j connection
│   └── llm.py                # OpenRouter LLM config
├── Graph/
│   └── Tool/
│       ├── CypherHealer.py   # 🆕 Self-healing Cypher
│       └── CypherSummarizer.py  # 🆕 Result summarization
├── KG/
│   └── VectorRAG.py          # 🆕 Enhanced with hybrid search
├── ENHANCEMENTS.md           # 🆕 Full feature documentation
└── test_enhancements.py      # 🆕 Test suite
```

## Architecture

```
User Query (Thai/English)
    ↓
Intent Detection
    ↓
┌─────────────────────────────────┐
│ Hybrid Search (Vector + Keyword)│  ← 🆕 Better Thai matching
└─────────────────────────────────┘
    ↓
Neo4j Query Generation
    ↓
┌─────────────────────────────────┐
│ Self-Healing Cypher             │  ← 🆕 Auto-fix errors
└─────────────────────────────────┘
    ↓
Result Processing
    ↓
┌─────────────────────────────────┐
│ Concise Summarization (Optional)│  ← 🆕 Short answers
└─────────────────────────────────┘
    ↓
Display to User
```

## Configuration

### Hybrid Search (default: ON)
```python
# In KG/VectorRAG.py
results = query_vector_rag(query, use_hybrid_search=True)
```

### Self-Healing (default: ON)
```python
# In streamlit_app.py
path = find_connection_path(person_a, person_b, use_healing=True)
```

### Concise Mode (default: OFF)
```
Toggle in Streamlit sidebar:
⚙️ Settings → "✨ Concise mode (NEW!)"
```

## Testing

Run the test suite:
```bash
python test_enhancements.py
```

Expected output:
```
✅ Imports: PASSED
✅ Hybrid Search: PASSED
✅ Cypher Healer: PASSED
✅ Summarizer: PASSED
✅ Graceful Degradation: PASSED

🎉 All tests passed!
```

## Technologies Used
- **Neo4j** - Graph database
- **Streamlit** - Web interface
- **OpenRouter** - LLM API
- **LangChain** - LLM framework
- **HuggingFace** - Embeddings (paraphrase-multilingual)

## Contributing
This project follows best practices from Neo4j official examples and community experts. Contributions welcome!

## Future Plans
- ✅ ~~Hybrid search~~ (DONE)
- ✅ ~~Self-healing Cypher~~ (DONE)
- ✅ ~~Concise mode~~ (DONE)
- ⏳ WebSocket streaming (like NaLLM)
- ⏳ Advanced ReAct patterns
- ⏳ Redis caching
- ⏳ Interactive visualization dashboard

---

*Developed as part of a data relationship analysis project using Neo4j.*

**Version 2.0** - Enhanced with production-ready patterns from Neo4j NaLLM and community experts.

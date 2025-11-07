# 📚 Knowledge Transfer Documentation - Neo4j Network Agent

## 🎯 Project Overview

**STelligence Network Agent** is a Thai-language Q&A chatbot that queries a Neo4j knowledge graph containing information about government officials, positions, ministries, and their relationships.

### Key Features
- 🇹🇭 Bilingual support (Thai/English)
- 🔍 Vector search + Graph traversal (hybrid approach)
- 💬 ChatGPT-like UI with conversation threads
- ✏️ Edit and regenerate responses
- 🌐 Stelligence network relationship queries

---

## 📂 Files to Share with Colleagues

### **Essential Files (Must Share)**
1. **`streamlit_app.py`** - Main application file (1,350+ lines)
2. **`requirements.txt`** - Python dependencies
3. **`.env.example`** or environment variables documentation
4. **`TEST_CASES.md`** - 34 test cases for quality validation
5. **`README.md`** - Project setup instructions

### **Configuration Files**
6. **`.streamlit/config.toml`** - Streamlit configuration
7. **`Config/` folder** - Neo4j and LLM configurations
   - `neo4j.py`
   - `llm.py`
   - `aura_neo4j.py`

### **Core Logic Files**
8. **`KG/VectorSearchDirect.py`** - Hybrid vector + graph search
9. **`Graph/` folder** - Graph processing logic
   - `Tool/Tools.py`
   - `Tool/ToolExecutor.py`
   - `Prompt/prompts.py`

### **Documentation Files**
10. **`ARCHITECTURE.md`** - System architecture
11. **`EMBEDDINGS_SETUP.md`** - Vector index setup guide
12. **`SETUP_VECTOR_SEARCH.md`** - Search configuration
13. **This file: `KNOWLEDGE_TRANSFER.md`**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│  (streamlit_app.py)                                     │
│  - Chat interface with threads                          │
│  - Edit/Regenerate buttons                              │
│  - Session state management                             │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              Query Processing Layer                      │
│  - User input → Vector search query                     │
│  - Stelligence network detection                        │
│  - Context building                                      │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  Vector Search   │    │  Graph Database  │
│  (HuggingFace)   │    │    (Neo4j)       │
│  - Embeddings    │    │  - Nodes         │
│  - Similarity    │    │  - Relationships │
└──────────────────┘    └──────────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
┌─────────────────────────────────────────────────────────┐
│            Context Aggregation                           │
│  - Merge vector search + graph results                  │
│  - Format for LLM prompt                                │
└───────────────────┬─────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────┐
│                LLM (OpenRouter API)                      │
│  - Model: deepseek/deepseek-chat                        │
│  - Processes context + question                         │
│  - Returns Thai/English answer                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Technology Stack

### **Frontend**
- **Streamlit** - Web UI framework
- **Custom CSS** - ChatGPT-like styling
- **Anuphan Font** - Thai-optimized Google Font

### **Backend**
- **Python 3.8+**
- **Neo4j Graph Database** (bolt://localhost:7687 or Aura)
- **HuggingFace Embeddings** - `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)

### **APIs**
- **OpenRouter API** - LLM inference (DeepSeek model)
- Alternative: OpenAI API compatible

### **Key Libraries**
```txt
streamlit>=1.28.0
neo4j>=5.0.0
langchain-huggingface
sentence-transformers
requests
python-dotenv
```

---

## 🚀 Setup Process (Step-by-Step)

### **1. Environment Setup**

```bash
# Clone repository
git clone https://github.com/pheechan/neo4j-network-agent.git
cd neo4j-network-agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Neo4j Setup**

**Option A: Local Neo4j**
```bash
# Download Neo4j Desktop from https://neo4j.com/download/
# Create database
# Set password
# Start database on bolt://localhost:7687
```

**Option B: Neo4j Aura (Cloud)**
```bash
# Sign up at https://neo4j.com/cloud/aura/
# Create free instance
# Copy connection URI and credentials
```

### **3. Environment Variables**

Create `.env` file:
```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# OpenRouter API (for LLM)
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=deepseek/deepseek-chat

# Vector Search Settings
VECTOR_INDEX_NAME=person_vector_index
VECTOR_NODE_LABEL=Person
VECTOR_SOURCE_PROPERTY=embedding_text
VECTOR_EMBEDDING_PROPERTY=embedding
VECTOR_TOP_K=5
```

### **4. Create Vector Index in Neo4j**

Run in Neo4j Browser or Cypher Shell:
```cypher
CREATE VECTOR INDEX person_vector_index IF NOT EXISTS
FOR (n:Person) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

-- Create indexes for other node types
CREATE VECTOR INDEX position_vector_index IF NOT EXISTS
FOR (n:Position) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX ministry_vector_index IF NOT EXISTS
FOR (n:Ministry) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};
```

### **5. Generate Embeddings**

```bash
python create_vector_index.py
```

This will:
- Load HuggingFace embedding model
- Generate embeddings for all nodes
- Store embeddings in Neo4j

### **6. Run Application**

```bash
streamlit run streamlit_app.py
```

Access at: `http://localhost:8501`

---

## 🎨 UI Components Explained

### **Sidebar (Left Panel)**
- **Background**: `#181818` (dark gray)
- **STelligence title**
- **+ New Chat button** - Creates new conversation thread
- **Chat History** - List of conversation threads
- **Settings expander** - Model and database info

### **Main Chat Area**
- **Background**: `#212121` (medium gray)
- **Welcome card** - Gradient purple card when no messages
- **Messages**:
  - **User messages** - Right-aligned, `#2f2f2f` background
  - **Assistant messages** - Left-aligned, `#1f2937` background
- **Input box** - Bottom, `#303030` background

### **Action Buttons**
- **✏️ Edit** - Edit user message and resend
- **🔄 Regenerate** - Regenerate assistant response

---

## 🔍 How the Search Works

### **1. User Input Processing**
```python
user_input = "Santisook มีความสัมพันธ์กับใครบ้าง"
```

### **2. Vector Search**
- Generate embedding for user query
- Search across multiple indexes:
  - `person_vector_index` (top 30 results)
  - `position_vector_index` (top 30 results)
  - `ministry_vector_index` (top 30 results)
  - `agency_vector_index` (top 30 results)

### **3. Graph Traversal**
For each vector search result, fetch relationships:
```cypher
MATCH (n)-[r]->(connected)
RETURN n, r, connected
```

### **4. Stelligence Network Detection**
If query mentions "Santisook", "Por", or "Knot":
```cypher
MATCH (n:Person)
WHERE n.Stelligence = 'Santisook' OR n.`Connect by` CONTAINS 'Santisook'
RETURN n
```

### **5. Context Building**
Format results as structured text:
```
📊 ข้อมูลจาก Neo4j Knowledge Graph:

👤 Person: อนุทิน ชาญวีรกูล
- ตำแหน่ง: รัฐมนตรีว่าการกระทรวงมหาดไทย
- 👥 ดำรงตำแหน่งโดย: [Position details]
...
```

### **6. LLM Processing**
Send to OpenRouter API with system prompt:
```python
system_prompt = """
You are a Thai government official knowledge assistant.
Answer questions in Thai or English based on the context provided.
Rules:
- No preamble
- Full ministry names
- Bullet points on separate lines
- No hallucination
"""
```

---

## 📝 Prompt Engineering

### **System Prompt Structure**

1. **Role Definition**
   - Thai government knowledge assistant
   - Bilingual capability

2. **Critical Rules**
   - No preamble ("ตามข้อมูล...", "จาก Context...")
   - Full position names ("รัฐมนตรีว่าการกระทรวงมหาดไทย" not "รัฐมนตรีว่าการ")
   - Bullet points on separate lines
   - No hallucination

3. **Data Analysis Skills**
   - Aggregate data for summary questions
   - Group by person/ministry/agency as appropriate
   - Show relationships and context

4. **Response Format**
   - Main answer first
   - Supporting details with bullets
   - Suggested follow-up questions at end

### **Example Prompts**

**Good Response:**
```
อนุทิน ชาญวีรกูล ดำรงตำแหน่ง:
• รัฐมนตรีว่าการกระทรวงมหาดไทย
• รองนายกรัฐมนตรี

คุณอาจสนใจ:
• ใครเป็นรัฐมนตรีช่วยว่าการกระทรวงมหาดไทย?
```

**Bad Response (Avoid):**
```
❌ ตามข้อมูลที่ได้รับจาก Knowledge Graph พบว่า อนุทิน ชาญวีรกูล เป็นรัฐมนตรีว่าการ
```

---

## 🧪 Testing Guide

Use `TEST_CASES.md` for systematic testing:

### **Priority Tests**
1. **Test 1.1**: `อนุทิน ชาญวีรกูล ตำแหน่งอะไร`
   - Must show full ministry name
   - No preamble

2. **Test 3.1**: `Santisook มีความสัมพันธ์กับใครบ้าง`
   - Should return 30-50+ people
   - Network summary at top

3. **Test 7.1**: `รัฐมนตรีว่าการกระทรวงอวกาศคือใคร`
   - Must return "ไม่พบข้อมูล"
   - No hallucination

### **Scoring System**
- ✅ 5/5 - Perfect
- ⚠️ 3-4/5 - Good (minor issues)
- ❌ 1-2/5 - Poor (major issues)
- 💥 0/5 - Failed

**Target**: Average ≥ 4.0/5.0

---

## 🐛 Common Issues & Solutions

### **Issue 1: Vector Search Returns No Results**
**Problem**: `query_with_relationships()` returns empty results

**Solution**:
```bash
# Check if embeddings exist
MATCH (n:Person) WHERE n.embedding IS NOT NULL RETURN count(n)

# If 0, run:
python create_vector_index.py
```

### **Issue 2: Sidebar Not Showing**
**Problem**: Hamburger menu hidden by CSS

**Solution**: Remove CSS that hides `#MainMenu` and `header`

### **Issue 3: Font Issues with Icons**
**Problem**: Material icons showing as text

**Solution**: Use font fallback chain or exclude buttons from custom font

### **Issue 4: Slow Response Time**
**Problem**: Queries take >10 seconds

**Solutions**:
- Reduce `top_k_per_index` from 30 to 10
- Add database indexes on frequently queried properties
- Use Neo4j Aura for better performance

### **Issue 5: Thai Language Breaks**
**Problem**: Thai characters garbled or incorrect

**Solutions**:
- Ensure UTF-8 encoding in all files
- Use Thai-optimized embedding model
- Verify Neo4j database encoding

---

## 🔒 Security Considerations

1. **API Keys**: Never commit `.env` file to git
2. **Neo4j Credentials**: Use environment variables
3. **Rate Limiting**: Implement for OpenRouter API
4. **Input Validation**: Sanitize user queries
5. **CORS**: Configured in `.streamlit/config.toml`

---

## 📊 Performance Metrics

### **Current Performance**
- **Query Time**: 3-8 seconds (including LLM)
- **Vector Search**: 1-2 seconds
- **Graph Traversal**: 0.5-1 second
- **LLM Response**: 2-5 seconds

### **Optimization Tips**
1. Cache frequent queries
2. Pre-compute common relationship paths
3. Use Neo4j APOC procedures for complex traversals
4. Implement query result pagination

---

## 🚀 Deployment

### **Streamlit Cloud**
1. Push code to GitHub
2. Connect at streamlit.io/cloud
3. Configure secrets in Streamlit Cloud dashboard
4. Auto-deploys on push to main branch
5. Deployment time: 3-5 minutes

### **Environment Variables in Streamlit Cloud**
Add in Settings → Secrets:
```toml
NEO4J_URI = "bolt://..."
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "..."
OPENROUTER_API_KEY = "..."
```

---

## 📞 Support & Maintenance

### **Key Maintenance Tasks**
1. **Weekly**: Review test case scores
2. **Monthly**: Update embeddings for new data
3. **Quarterly**: Review and update system prompts
4. **As Needed**: Add new test cases for edge cases

### **Monitoring**
- Check Streamlit Cloud logs
- Monitor OpenRouter API usage
- Track Neo4j database size
- Review user feedback

---

## 🎓 Learning Resources

### **Neo4j**
- [Neo4j Graph Academy](https://graphacademy.neo4j.com/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/)

### **Vector Search**
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers)
- [Neo4j Vector Search](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)

### **Streamlit**
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Chat Elements Guide](https://docs.streamlit.io/library/api-reference/chat)

---

## 📄 License & Credits

- **Project**: STelligence Network Agent
- **Repository**: github.com/pheechan/neo4j-network-agent
- **Developed**: 2025
- **Tech Stack**: Python, Streamlit, Neo4j, HuggingFace, OpenRouter

---

## 📋 Quick Reference Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Generate embeddings
python create_vector_index.py

# Run app locally
streamlit run streamlit_app.py

# Git workflow
git add -A
git commit -m "Your message"
git push

# Check Neo4j connection
python test_neo4j_conn.py
```

---

**Last Updated**: November 7, 2025
**Version**: 1.0
**Maintainer**: Pheemaphat Chan

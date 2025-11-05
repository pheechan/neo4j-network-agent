# 🚀 Setup Guide: Vector Search with Neo4j

## You have 13 labels in your database:
1. Person
2. Position
3. Level
4. Connect by
5. Agency
6. Santisook
7. Por
8. Remark
9. Knot
10. Ministry
11. Associate
12. NIckname
13. Document

---

## ✅ Step 1: Create Vector Indexes (Neo4j Browser)

Go to **Neo4j Browser** → https://console.neo4j.io

Run this Cypher query to create ALL vector indexes at once:

```cypher
CREATE VECTOR INDEX person_vector_index IF NOT EXISTS
FOR (n:Person) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX position_vector_index IF NOT EXISTS
FOR (n:Position) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX level_vector_index IF NOT EXISTS
FOR (n:Level) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX connect_by_vector_index IF NOT EXISTS
FOR (n:`Connect by`) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX agency_vector_index IF NOT EXISTS
FOR (n:Agency) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX santisook_vector_index IF NOT EXISTS
FOR (n:Santisook) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX por_vector_index IF NOT EXISTS
FOR (n:Por) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX remark_vector_index IF NOT EXISTS
FOR (n:Remark) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX knot_vector_index IF NOT EXISTS
FOR (n:Knot) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX ministry_vector_index IF NOT EXISTS
FOR (n:Ministry) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX associate_vector_index IF NOT EXISTS
FOR (n:Associate) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX nickname_vector_index IF NOT EXISTS
FOR (n:NIckname) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX document_vector_index IF NOT EXISTS
FOR (n:Document) ON n.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
```

**Check if it worked:**
```cypher
SHOW INDEXES;
```

You should see 13 new indexes with names ending in `_vector_index`.

---

## ✅ Step 2: Generate Embeddings (Streamlit Cloud)

### Option A: Use the Main App (Easiest) ⭐

1. Go to your Streamlit app: https://neo4j-network-agent.streamlit.app/
2. **Open the sidebar** (click the arrow on the left)
3. Scroll down to **"Admin Tools"**
4. Click **"⚡ Generate Embeddings"** button
5. Wait 1-2 minutes (it will process 100 nodes at a time)
6. Click again to process more nodes (repeat until all nodes have embeddings)

### Option B: Use the Admin Page

1. Deploy `admin_page.py` as a separate app on Streamlit Cloud
2. Use the visual interface to:
   - See database statistics
   - Search for "Santisook"
   - Create indexes (if you didn't do Step 1)
   - Generate embeddings with progress bars

---

## ✅ Step 3: Verify It Works

### In Neo4j Browser:
```cypher
// Count nodes with embeddings
MATCH (n)
WHERE n.embedding IS NOT NULL
RETURN count(n) as nodes_with_embeddings;

// Search for Santisook
MATCH (n)
WHERE any(prop in keys(n) WHERE toLower(toString(n[prop])) CONTAINS 'santisook')
RETURN labels(n), properties(n)
LIMIT 10;
```

### In Your Streamlit App:
Test these Thai queries:
- "Santisook รู้จักใครบ้าง" (Who does Santisook know?)
- "ใครทำงานในหน่วยงานอะไร" (Who works in which agency?)
- "บุคคลที่มีตำแหน่งสำคัญ" (Important positions)

---

## 🎯 What Happens After Setup

Once vector indexes and embeddings are created:

1. **Semantic Search Works** - The app can find similar content even without exact matches
2. **Thai Language Support** - Works with Thai queries naturally
3. **Relationship Queries** - Can answer "who knows whom" questions
4. **Fast Retrieval** - Vector search is much faster than property scanning

---

## 🔧 Troubleshooting

### "No embeddings generated"
- Make sure you ran the Cypher commands in Step 1 first
- Check that nodes have text properties (name, description, etc.)

### "HuggingFace embeddings not installed"
- On Streamlit Cloud: Add `langchain-huggingface` and `sentence-transformers` to `requirements.txt`
- Already done! Just redeploy your app

### "Connection failed" (local only)
- This is a Python 3.13 + Windows SSL issue
- **Solution**: Use Streamlit Cloud instead (it works there)

---

## 📊 Monitoring Progress

### Check how many nodes need embeddings:
```cypher
MATCH (n)
WHERE n.embedding IS NULL
RETURN labels(n)[0] as label, count(n) as count
ORDER BY count DESC;
```

### See which nodes have embeddings:
```cypher
MATCH (n)
WHERE n.embedding IS NOT NULL
RETURN labels(n)[0] as label, count(n) as count
ORDER BY count DESC;
```

---

## 🎉 Done!

After completing Steps 1-3, your app will:
- ✅ Find "Santisook" nodes
- ✅ Answer relationship queries
- ✅ Support Thai language
- ✅ Use semantic search for better results

**Need help?** Check the debug info in the app's sidebar or search logs in Neo4j Browser.

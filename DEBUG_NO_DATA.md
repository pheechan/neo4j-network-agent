# Debugging "No Data Found" Issue

## Problem
Queries like "Santisook รู้จักใครบ้าง" return "no data found" even though the data exists.

## New Debug Features Added

I've added comprehensive debugging tools to help identify the issue:

### 1. Database Status Checker (Sidebar)
Click **"📊 Check Database Status"** in the sidebar to see:
- ✅ Vector indexes (with dimensions)
- ✅ Nodes with embeddings (count per label)
- ✅ Test search for "Santisook"

### 2. Live Search Diagnostics (Chat)
When you send a query, you'll now see:
- 🔍 Which search method is being used (vector vs Cypher)
- ✅ How many results were found
- ⚠️ Error messages if searches fail
- 📊 Automatic fallback from vector → Cypher if needed

## Troubleshooting Steps

### Step 1: Check Database Status
1. Go to your Streamlit Cloud app
2. Click **"📊 Check Database Status"** in sidebar
3. Take note of:
   - Do vector indexes exist?
   - What dimensions are they? (should be 384)
   - Do nodes have embeddings?
   - Can it find "Santisook" in test search?

### Step 2: Try a Query and Watch Debug Output
1. Send query: "Santisook รู้จักใครบ้าง"
2. Look at the debug messages that appear
3. Note what it says:
   - Is it using vector search or Cypher?
   - How many results found?
   - Any error messages?

### Step 3: Fix Based on Results

**If NO VECTOR INDEXES exist:**
```cypher
-- Run in Neo4j Browser
CREATE VECTOR INDEX person_vector_index IF NOT EXISTS
FOR (n:Person) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

-- Repeat for other labels: Santisook, Position, etc.
```

**If indexes have WRONG DIMENSIONS (1536 instead of 384):**
```cypher
-- Run in Neo4j Browser
DROP INDEX person_vector_index IF EXISTS;
-- Then create with 384 dimensions as shown above
```

**If NO EMBEDDINGS exist:**
Click **"⚡ Generate Embeddings"** button in sidebar

**If Cypher search ALSO fails:**
Check that nodes actually exist:
```cypher
-- Run in Neo4j Browser
MATCH (n)
WHERE any(prop IN keys(n) WHERE 
  toLower(toString(n[prop])) CONTAINS 'santisook'
)
RETURN n
LIMIT 5
```

## Most Likely Issues

### Issue 1: Vector Index Doesn't Exist
**Symptom:** Debug shows "Vector search error: Index not found"
**Fix:** Create vector indexes (see Step 3 above)

### Issue 2: Wrong Index Name/Label
**Symptom:** Debug shows "Found 0 results from Vector Search"
**Current config:**
- Index name: `person_vector_index`
- Node label: `Person`

If your "Santisook" data is under label `Santisook` (not `Person`), either:
- Change config in sidebar settings, OR
- Create index for Santisook label:
```cypher
CREATE VECTOR INDEX santisook_vector_index IF NOT EXISTS
FOR (n:Santisook) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};
```

### Issue 3: No Embeddings Generated
**Symptom:** Database status shows "No nodes have embeddings"
**Fix:** Click "⚡ Generate Embeddings" button

### Issue 4: Dimension Mismatch
**Symptom:** Debug shows "Embedding dimension 384 does not match index dimension 1536"
**Fix:** Drop old indexes and create new ones with 384 dimensions

## Quick Test Commands

Run these in Neo4j Browser to understand your data:

```cypher
-- 1. Check what labels exist
CALL db.labels()

-- 2. Check which label has Santisook
MATCH (n)
WHERE any(prop IN keys(n) WHERE 
  toLower(toString(n[prop])) CONTAINS 'santisook'
)
RETURN DISTINCT labels(n) as labels, count(*) as count

-- 3. Check if those nodes have embeddings
MATCH (n)
WHERE any(prop IN keys(n) WHERE 
  toLower(toString(n[prop])) CONTAINS 'santisook'
)
RETURN labels(n), n.embedding IS NOT NULL as has_embedding, 
       size(n.embedding) as embedding_dimension
LIMIT 5

-- 4. Show current indexes
SHOW INDEXES
```

## What to Report Back

After running the Database Status check, tell me:
1. What indexes exist? (names and dimensions)
2. How many nodes have embeddings? (per label)
3. Did the test search find Santisook?
4. What error messages appear when you try a query?

This will help me pinpoint the exact issue!

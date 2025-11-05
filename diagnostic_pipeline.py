"""
Pipeline Diagnostic: Shows how the query flows through the system
"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

print("=" * 80)
print("PIPELINE DIAGNOSTIC: How 'Santisook รู้จักใครบ้าง' is processed")
print("=" * 80)

query = "Santisook รู้จักใครบ้าง"
search_term = "Santisook"

print(f"\n📥 INPUT: {query}")
print(f"🔍 Search term: {search_term}\n")

# Step 1: Check if HuggingFace embeddings work
print("=" * 80)
print("STEP 1: HuggingFace Embedding Generation")
print("=" * 80)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    query_embedding = embeddings.embed_query(query)
    print(f"✅ Embedding created successfully")
    print(f"   Dimensions: {len(query_embedding)}")
    print(f"   First 5 values: {query_embedding[:5]}")
    print(f"   Model: paraphrase-multilingual-MiniLM-L12-v2 (supports Thai)")
except Exception as e:
    print(f"❌ Failed: {e}")
    query_embedding = None

# Step 2: Connect to Neo4j
print("\n" + "=" * 80)
print("STEP 2: Connect to Neo4j")
print("=" * 80)
try:
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    print(f"✅ Connected to: {NEO4J_URI}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

# Step 3: Search for Santisook in properties
print("\n" + "=" * 80)
print("STEP 3: Cypher Property Search (Fallback Method)")
print("=" * 80)
print(f"Query: Search ALL properties for '{search_term}' (case-insensitive)\n")

cypher_query = """
MATCH (n)
WHERE any(prop in keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($search))
RETURN labels(n) as labels, properties(n) as props, id(n) as nodeId
LIMIT 10
"""

with driver.session(database=NEO4J_DATABASE) as session:
    result = session.run(cypher_query, search=search_term)
    nodes = [dict(record) for record in result]
    
    if nodes:
        print(f"✅ Found {len(nodes)} nodes containing '{search_term}':\n")
        for i, node in enumerate(nodes, 1):
            print(f"Node {i}:")
            print(f"  Labels: {node['labels']}")
            print(f"  Properties:")
            for key, value in node['props'].items():
                if search_term.lower() in str(value).lower():
                    print(f"    → {key}: {value} ⭐ (contains '{search_term}')")
                else:
                    print(f"    - {key}: {value}")
            print()
    else:
        print(f"❌ No nodes found containing '{search_term}'")
        print("\nPossible reasons:")
        print("1. The name is spelled differently in the database")
        print("2. It's stored in a relationship, not a node property")
        print("3. The data hasn't been imported yet")
        
        # Show sample data
        print("\n📊 Sample nodes from database:")
        result = session.run("MATCH (n) RETURN labels(n) as labels, properties(n) as props LIMIT 3")
        samples = [dict(record) for record in result]
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"  Labels: {sample['labels']}")
            print(f"  Properties: {list(sample['props'].keys())}")

# Step 4: Check embeddings
print("\n" + "=" * 80)
print("STEP 4: Check Existing Embeddings")
print("=" * 80)

with driver.session(database=NEO4J_DATABASE) as session:
    # Count nodes with embeddings
    result = session.run("""
    MATCH (n)
    WHERE n.embedding IS NOT NULL
    RETURN count(n) as count
    """)
    count = dict(result.single())['count']
    
    print(f"Nodes with embeddings: {count}")
    
    if count == 0:
        print("\n⚠️ NO EMBEDDINGS FOUND!")
        print("You need to generate embeddings using:")
        print("1. Streamlit app sidebar → '⚡ Generate Embeddings' button")
        print("2. Or wait for the admin page to complete")
    
    # Check if Santisook nodes have embeddings
    result = session.run("""
    MATCH (n)
    WHERE any(prop in keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($search))
    RETURN labels(n) as labels, 
           n.embedding IS NOT NULL as has_embedding,
           n.embedding_text as embedding_text
    LIMIT 5
    """, search=search_term)
    
    santisook_nodes = [dict(record) for record in result]
    if santisook_nodes:
        print(f"\nSantisook nodes embedding status:")
        for node in santisook_nodes:
            status = "✅ Has embedding" if node['has_embedding'] else "❌ No embedding"
            print(f"  {node['labels']}: {status}")
            if node['embedding_text']:
                print(f"    Text: {node['embedding_text'][:100]}...")

# Step 5: Check vector indexes
print("\n" + "=" * 80)
print("STEP 5: Check Vector Indexes")
print("=" * 80)

with driver.session(database=NEO4J_DATABASE) as session:
    result = session.run("SHOW INDEXES")
    indexes = [dict(record) for record in result]
    
    vector_indexes = [idx for idx in indexes if 'VECTOR' in idx.get('type', '')]
    
    if vector_indexes:
        print(f"✅ Found {len(vector_indexes)} vector indexes:\n")
        for idx in vector_indexes:
            print(f"  - {idx.get('name')}")
            print(f"    Label: {idx.get('labelsOrTypes', 'N/A')}")
            print(f"    State: {idx.get('state', 'N/A')}")
    else:
        print("❌ No vector indexes found!")
        print("Run the Cypher script to create them.")

# Step 6: Context Building
print("\n" + "=" * 80)
print("STEP 6: Build Context for LLM")
print("=" * 80)

if nodes:
    context_parts = []
    for node in nodes:
        props = node['props']
        labels = node['labels']
        
        # Extract text
        text_parts = []
        for key in ['name', 'title', 'text', 'description']:
            if key in props:
                text_parts.append(f"{key}: {props[key]}")
        
        if not text_parts:
            text_parts = [f"{k}: {v}" for k, v in props.items() if isinstance(v, str)]
        
        text = " | ".join(text_parts)
        context_parts.append(f"{labels}: {text}")
    
    context = "\n\n".join(context_parts)
    print("✅ Context built successfully:")
    print("\n" + context[:500] + "..." if len(context) > 500 else context)
else:
    context = ""
    print("❌ No context - no nodes found")

# Step 7: Show LLM Prompt
print("\n" + "=" * 80)
print("STEP 7: Prompt Sent to DeepSeek (via OpenRouter)")
print("=" * 80)

prompt = f"""คุณเป็นผู้ช่วยที่ชาญฉลาดและตอบคำถามอย่างเป็นกันเอง

Context from Neo4j:
{context if context else "ไม่พบข้อมูลที่เกี่ยวข้อง (No relevant information found)"}

คำถามของผู้ใช้:
{query}

คำตอบ:"""

print("\n📝 Full Prompt:")
print("-" * 80)
print(prompt)
print("-" * 80)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✅ HuggingFace embeddings: {'Working' if query_embedding else 'Failed'}")
print(f"✅ Neo4j connection: Working")
print(f"{'✅' if nodes else '❌'} Found Santisook nodes: {len(nodes) if nodes else 0}")
print(f"{'✅' if count > 0 else '❌'} Embeddings in database: {count}")
print(f"{'✅' if vector_indexes else '❌'} Vector indexes: {len(vector_indexes) if vector_indexes else 0}")
print(f"{'✅' if context else '❌'} Context for LLM: {'Available' if context else 'Empty'}")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

if not nodes:
    print("🔴 CRITICAL: No Santisook data found!")
    print("   1. Check if data was imported correctly")
    print("   2. Verify the spelling in your database")
    print("   3. Check if it's in relationships instead of nodes")
elif count == 0:
    print("🟡 WARNING: Data exists but no embeddings!")
    print("   1. Go to Streamlit app sidebar")
    print("   2. Click '⚡ Generate Embeddings' button")
    print("   3. Wait for completion (may take a few minutes)")
elif not vector_indexes:
    print("🟡 WARNING: Embeddings exist but no vector indexes!")
    print("   1. Run the Cypher script to create indexes")
else:
    print("✅ ALL SYSTEMS READY!")
    print("   Vector search should work now.")
    print("   If still not working, check:")
    print("   1. Index state (should be ONLINE)")
    print("   2. Try restarting Streamlit app")

driver.close()
print("\n✓ Diagnostic complete")

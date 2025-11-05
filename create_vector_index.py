"""
Script to create vector index and generate embeddings for existing Neo4j nodes
"""
import os
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection using LangChain (same as streamlit_app.py)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Initialize embeddings model
print("Loading HuggingFace embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print("✓ Model loaded successfully")

def create_vector_index(graph):
    """Create vector index for nodes"""
    # First, check existing indexes
    result = graph.query("SHOW INDEXES")
    existing_indexes = [record.get("name", "") for record in result]
    print(f"\nExisting indexes: {existing_indexes}")
    
    # Check what node labels exist
    result = graph.query("CALL db.labels()")
    labels = [record.get("label", "") for record in result]
    print(f"\nNode labels in database: {labels}")
    
    # Create vector index for each label
    for label in labels:
        if not label:
            continue
        index_name = f"{label.lower()}_vector_index"
        
        if index_name in existing_indexes:
            print(f"✓ Vector index '{index_name}' already exists")
        else:
            try:
                # Create vector index
                # Dimension 384 is for paraphrase-multilingual-MiniLM-L12-v2
                query = f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{label})
                ON n.embedding
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
                graph.query(query)
                print(f"✓ Created vector index '{index_name}' for label '{label}'")
            except Exception as e:
                print(f"✗ Error creating index for {label}: {e}")

def generate_embeddings(graph):
    """Generate embeddings for nodes that don't have them"""
    # Get all node labels
    result = graph.query("CALL db.labels()")
    labels = [record.get("label", "") for record in result]
    
    for label in labels:
        if not label:
            continue
        print(f"\n--- Processing {label} nodes ---")
        
        # Find nodes without embeddings
        query = f"""
        MATCH (n:{label})
        WHERE n.embedding IS NULL
        RETURN id(n) as nodeId, properties(n) as props
        LIMIT 100
        """
        result = graph.query(query)
        nodes = list(result)
        
        print(f"Found {len(nodes)} nodes without embeddings")
        
        if not nodes:
            continue
        
        # Generate embeddings for each node
        for i, record in enumerate(nodes):
            node_id = record.get("nodeId")
            props = record.get("props", {})
            
            # Create text from node properties
            text_parts = []
            for key, value in props.items():
                if key != "embedding" and value:
                    text_parts.append(f"{key}: {value}")
            
            if not text_parts:
                print(f"  Skipping node {node_id} - no text properties")
                continue
            
            text = " | ".join(text_parts)
            print(f"  [{i+1}/{len(nodes)}] Generating embedding for: {text[:100]}...")
            
            try:
                # Generate embedding
                embedding = embeddings.embed_query(text)
                
                # Store embedding
                update_query = f"""
                MATCH (n:{label})
                WHERE id(n) = $nodeId
                SET n.embedding = $embedding
                SET n.embedding_text = $text
                """
                graph.query(update_query, params={"nodeId": node_id, "embedding": embedding, "text": text})
                print(f"    ✓ Saved embedding")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")

def debug_search(graph):
    """Debug why Santisook isn't being found"""
    print("\n=== DEBUGGING SEARCH FOR 'Santisook' ===\n")
    
    # 1. Direct property search (case-insensitive)
    print("1. Searching for nodes with 'Santisook' in any property:")
    query = """
    MATCH (n)
    WHERE any(prop in keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($search))
    RETURN labels(n) as labels, properties(n) as props, id(n) as nodeId
    LIMIT 10
    """
    result = graph.query(query, params={"search": "Santisook"})
    nodes = list(result)
    
    if nodes:
        print(f"✓ Found {len(nodes)} nodes:")
        for record in nodes:
            print(f"  - Labels: {record.get('labels', [])}")
            print(f"    Properties: {record.get('props', {})}")
            print(f"    Node ID: {record.get('nodeId')}")
    else:
        print("✗ No nodes found with 'Santisook'")
    
    # 2. Check all unique property values
    print("\n2. Checking all nodes with 'name' property:")
    query = """
    MATCH (n)
    WHERE n.name IS NOT NULL
    RETURN DISTINCT n.name as name, labels(n) as labels
    LIMIT 20
    """
    result = graph.query(query)
    names = list(result)
    
    if names:
        print(f"Found {len(names)} nodes with 'name' property:")
        for record in names:
            print(f"  - {record.get('name')} ({record.get('labels', [])})")
    else:
        print("No nodes with 'name' property found")
    
    # 3. Check for any other text properties
    print("\n3. Checking all property keys in the database:")
    query = """
    MATCH (n)
    UNWIND keys(n) as key
    RETURN DISTINCT key
    LIMIT 50
    """
    result = graph.query(query)
    keys = [record.get('key') for record in result]
    print(f"Found property keys: {keys}")
    
    # 4. Show sample nodes
    print("\n4. Sample nodes from database:")
    query = """
    MATCH (n)
    RETURN labels(n) as labels, properties(n) as props
    LIMIT 5
    """
    result = graph.query(query)
    samples = list(result)
    
    if samples:
        print(f"Sample of {len(samples)} nodes:")
        for i, record in enumerate(samples, 1):
            print(f"\n  Node {i}:")
            print(f"    Labels: {record.get('labels', [])}")
            print(f"    Properties: {record.get('props', {})}")
    
    # 5. Show total node count
    print("\n5. Database statistics:")
    result = graph.query("MATCH (n) RETURN count(n) as total")
    if result:
        total = result[0].get("total", 0)
        print(f"  Total nodes in database: {total}")
    
    result = graph.query("CALL db.labels()")
    labels = [record.get("label", "") for record in result]
    for label in labels:
        if label:
            result = graph.query(f"MATCH (n:{label}) RETURN count(n) as count")
            if result:
                count = result[0].get("count", 0)
                print(f"  - {label}: {count} nodes")

def main():
    print("=== Neo4j Vector Index Creation Tool ===\n")
    print(f"Connecting to: {NEO4J_URI}")
    
    try:
        # Create graph connection using LangChain (same method as streamlit_app.py)
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        print("✓ Connected to Neo4j\n")
        
        # Step 1: Debug search first
        debug_search(graph)
        
        # Step 2: Create vector indexes
        print("\n=== CREATING VECTOR INDEXES ===")
        create_vector_index(graph)
        
        # Step 3: Generate embeddings
        print("\n=== GENERATING EMBEDDINGS ===")
        generate_embeddings(graph)
        
        # Step 4: Debug search again after embeddings
        print("\n=== CHECKING AFTER EMBEDDINGS ===")
        # Check if embeddings were created
        result = graph.query("""
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN count(n) as count
        """)
        if result:
            count = result[0].get("count", 0)
            print(f"\nNodes with embeddings: {count}")
        
        print("\n=== COMPLETED ===")
        print("\nNext steps:")
        print("1. If nodes were found but vector search failed, wait a minute for indexes to populate")
        print("2. Test in Streamlit app with query: 'Santisook รู้จักใครบ้าง'")
        print("3. Check the app shows embeddings working in the debug info")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

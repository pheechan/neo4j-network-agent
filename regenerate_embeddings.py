"""
Script to regenerate embeddings for all Person nodes after database import
Uses Thai property names: ชื่อ-นามสกุล, ตำแหน่ง, หน่วยงาน
"""
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

print("="*80)
print("REGENERATING EMBEDDINGS FOR NEO4J DATABASE")
print("="*80)

# Initialize embeddings model
print("\n1. Loading HuggingFace embeddings model...")
print("   Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print("   ✓ Model loaded successfully (384 dimensions)")

# Connect to Neo4j using LangChain (handles SSL correctly)
print("\n2. Connecting to Neo4j...")
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)
print(f"   ✓ Connected to: {NEO4J_URI}")

def create_vector_index(graph):
    """Create vector index for Person nodes"""
    print("\n3. Creating vector index...")
    
    # Check if index exists
    result = graph.query("SHOW INDEXES")
    existing = [r.get("name", "") for r in result if r.get("type") == "VECTOR"]
    
    if "person_vector_index" in existing:
        print("   ✓ Vector index 'person_vector_index' already exists")
    else:
        try:
            graph.query("""
                CREATE VECTOR INDEX person_vector_index IF NOT EXISTS
                FOR (p:Person)
                ON p.embedding
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)
            print("   ✓ Created vector index 'person_vector_index'")
        except Exception as e:
            print(f"   ⚠ Error creating index: {e}")

def generate_person_embeddings(graph):
    """Generate embeddings for all Person nodes"""
    print("\n4. Counting Person nodes...")
    
    # Count total persons
    result = graph.query("MATCH (p:Person) RETURN count(p) as total")
    total = result[0]["total"] if result else 0
    print(f"   Total Person nodes: {total}")
    
    # Count persons without embeddings
    result = graph.query("""
        MATCH (p:Person)
        WHERE p.embedding IS NULL
        RETURN count(p) as without_embedding
    """)
    without_embedding = result[0]["without_embedding"] if result else 0
    print(f"   Persons WITHOUT embedding: {without_embedding}")
    
    if without_embedding == 0:
        print("   ✓ All Person nodes already have embeddings!")
        return
    
    print(f"\n5. Generating embeddings for {without_embedding} Person nodes...")
    print("   (This may take a few minutes...)")
    
    # Get all persons without embeddings
    result = graph.query("""
        MATCH (p:Person)
        WHERE p.embedding IS NULL
        RETURN 
            id(p) as node_id,
            p.`ชื่อ-นามสกุล` as full_name,
            p.`ตำแหน่ง` as position,
            p.`หน่วยงาน` as agency,
            p.name as name,
            properties(p) as all_props
    """)
    
    persons = list(result)
    success_count = 0
    error_count = 0
    
    # Process each person with progress display
    total_persons = len(persons)
    for i, person in enumerate(persons, 1):
        if i % 10 == 0:
            print(f"   Progress: {i}/{total_persons} ({i/total_persons*100:.1f}%)")
        
        node_id = person["node_id"]
        
        # Build text for embedding from available properties
        text_parts = []
        
        # Use Thai properties first (ชื่อ-นามสกุล, ตำแหน่ง, หน่วยงาน)
        if person.get("full_name"):
            text_parts.append(f"ชื่อ: {person['full_name']}")
        elif person.get("name"):
            text_parts.append(f"Name: {person['name']}")
        
        if person.get("position"):
            text_parts.append(f"ตำแหน่ง: {person['position']}")
        
        if person.get("agency"):
            text_parts.append(f"หน่วยงาน: {person['agency']}")
        
        # If no standard properties, use all available properties
        if not text_parts:
            all_props = person.get("all_props", {})
            for key, value in all_props.items():
                if key not in ["embedding", "embedding_text"] and value:
                    text_parts.append(f"{key}: {value}")
        
        if not text_parts:
            error_count += 1
            continue
        
        # Create embedding text
        embedding_text = " | ".join(text_parts)
        
        try:
            # Generate embedding
            embedding_vector = embeddings.embed_query(embedding_text)
            
            # Store in Neo4j
            graph.query("""
                MATCH (p:Person)
                WHERE id(p) = $node_id
                SET p.embedding = $embedding,
                    p.embedding_text = $text
            """, params={"node_id": node_id, "embedding": embedding_vector, "text": embedding_text})
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"   ✗ Error for node {node_id}: {e}")
    
    print(f"\n   ✓ Successfully generated {success_count} embeddings")
    if error_count > 0:
        print(f"   ⚠ Failed to generate {error_count} embeddings")

def verify_embeddings(graph):
    """Verify that embeddings were created"""
    print("\n6. Verifying embeddings...")
    
    result = graph.query("""
        MATCH (p:Person)
        RETURN 
            count(p) as total,
            count(p.embedding) as with_embedding,
            count(p.embedding_text) as with_text
    """)
    
    if result:
        stats = result[0]
        print(f"   Total Person nodes: {stats['total']}")
        print(f"   With embedding: {stats['with_embedding']}")
        print(f"   With embedding_text: {stats['with_text']}")
        
        coverage = (stats['with_embedding'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"   Coverage: {coverage:.1f}%")
    
    # Show sample
    print("\n7. Sample persons with embeddings:")
    result = graph.query("""
        MATCH (p:Person)
        WHERE p.embedding IS NOT NULL
        RETURN p.`ชื่อ-นามสกุล` as name, p.embedding_text as text
        LIMIT 5
    """)
    
    for i, record in enumerate(result, 1):
        print(f"   {i}. {record['name']}")
        print(f"      Text: {record['text'][:80]}...")

# Main execution
try:
    create_vector_index(graph)
    generate_person_embeddings(graph)
    verify_embeddings(graph)
    
    print("\n" + "="*80)
    print("✅ EMBEDDINGS REGENERATION COMPLETE!")
    print("="*80)
    print("\nYou can now:")
    print("  1. Run your Streamlit app")
    print("  2. Test vector search queries")
    print("  3. Verify with: python check_embedding.py")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

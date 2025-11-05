"""
Simple diagnostic script - works around SSL issues
Run with: python simple_setup.py
"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

print("=" * 60)
print("NEO4J DATABASE DIAGNOSTIC & SETUP")
print("=" * 60)
print(f"\nConnecting to: {NEO4J_URI}")

# Create driver
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    max_connection_lifetime=3600
)

def run_query(query, params=None):
    """Run a query and return results"""
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, params or {})
        return [dict(record) for record in result]

try:
    # Test connection
    result = run_query("RETURN 'Connected!' as message")
    print(f"✓ {result[0]['message']}\n")
    
    # 1. Show database statistics
    print("=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)
    
    result = run_query("MATCH (n) RETURN count(n) as count")
    print(f"Total Nodes: {result[0]['count']}")
    
    result = run_query("MATCH ()-[r]->() RETURN count(r) as count")
    print(f"Total Relationships: {result[0]['count']}")
    
    # 2. Show labels and counts
    print("\n" + "=" * 60)
    print("NODE LABELS")
    print("=" * 60)
    
    result = run_query("CALL db.labels()")
    labels = [record["label"] for record in result]
    
    for label in labels:
        result = run_query(f"MATCH (n:{label}) RETURN count(n) as count")
        count = result[0]['count']
        print(f"{label:20s} : {count:4d} nodes")
    
    # 3. Show sample data for each label
    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    
    for label in labels[:5]:  # Show first 5 labels
        result = run_query(f"MATCH (n:{label}) RETURN properties(n) as props LIMIT 1")
        if result:
            print(f"\n{label} example:")
            props = result[0]['props']
            for key, value in props.items():
                print(f"  - {key}: {value}")
    
    # 4. Search for Santisook
    print("\n" + "=" * 60)
    print("SEARCHING FOR 'SANTISOOK'")
    print("=" * 60)
    
    # Check if Santisook is a label
    if "Santisook" in labels:
        result = run_query("MATCH (n:Santisook) RETURN count(n) as count")
        count = result[0]['count']
        print(f"\n✓ Found {count} nodes with label 'Santisook'")
        
        # Show sample
        result = run_query("MATCH (n:Santisook) RETURN properties(n) as props LIMIT 3")
        for i, record in enumerate(result, 1):
            print(f"\nSantisook node {i}:")
            for key, value in record['props'].items():
                print(f"  - {key}: {value}")
    
    # Search for Santisook in properties
    result = run_query("""
    MATCH (n)
    WHERE any(prop in keys(n) WHERE toLower(toString(n[prop])) CONTAINS 'santisook')
    RETURN labels(n) as labels, properties(n) as props
    LIMIT 5
    """)
    
    if result:
        print(f"\n✓ Found {len(result)} nodes with 'santisook' in properties:")
        for i, record in enumerate(result, 1):
            print(f"\nMatch {i} - Labels: {record['labels']}")
            for key, value in record['props'].items():
                if 'santisook' in str(value).lower():
                    print(f"  → {key}: {value}")
    
    # 5. Show relationship types
    print("\n" + "=" * 60)
    print("RELATIONSHIP TYPES")
    print("=" * 60)
    
    result = run_query("CALL db.relationshipTypes()")
    rel_types = [record["relationshipType"] for record in result]
    
    for rel_type in rel_types:
        result = run_query(f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count")
        count = result[0]['count']
        print(f"{rel_type:30s} : {count:4d} relationships")
    
    # 6. Sample relationship pattern
    if rel_types:
        print("\n" + "=" * 60)
        print("SAMPLE RELATIONSHIP PATTERN")
        print("=" * 60)
        
        result = run_query("""
        MATCH (a)-[r]->(b)
        RETURN labels(a) as from_labels, type(r) as rel_type, labels(b) as to_labels
        LIMIT 5
        """)
        
        for record in result:
            print(f"{record['from_labels']} -[{record['rel_type']}]-> {record['to_labels']}")
    
    # 7. Create vector indexes
    print("\n" + "=" * 60)
    print("VECTOR INDEX SETUP")
    print("=" * 60)
    
    # Check existing indexes
    result = run_query("SHOW INDEXES")
    existing_indexes = [r.get("name", "") for r in result]
    print(f"\nExisting indexes: {len(existing_indexes)}")
    
    input("\nPress ENTER to create vector indexes for all labels (or Ctrl+C to skip)...")
    
    for label in labels:
        index_name = f"{label.replace(' ', '_').lower()}_vector_index"
        
        if index_name in existing_indexes:
            print(f"✓ {index_name} already exists")
        else:
            try:
                query = f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:`{label}`)
                ON n.embedding
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
                run_query(query)
                print(f"✓ Created {index_name}")
            except Exception as e:
                print(f"✗ Error creating {index_name}: {e}")
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run your Streamlit app to generate embeddings")
    print("2. Or use the admin_page.py for a visual interface")
    print("3. Test queries like: 'Santisook รู้จักใครบ้าง'")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    driver.close()
    print("\n✓ Connection closed")

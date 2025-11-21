"""Check if a specific person has embedding in Neo4j"""
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

with driver.session() as session:
    # Check อนุทิน ชาญวีรกูล
    print("=" * 80)
    print("Checking: อนุทิน ชาญวีรกูล")
    print("=" * 80)
    
    result = session.run('''
        MATCH (p:Person)
        WHERE p.`ชื่อ-นามสกุล` = "อนุทิน ชาญวีรกูล"
        RETURN p.name as name,
               p.`ชื่อ` as thai_name,
               p.`ชื่อ-นามสกุล` as full_name,
               p.embedding IS NOT NULL as has_embedding,
               p.embedding_text as embedding_text,
               CASE WHEN p.embedding IS NOT NULL THEN size(p.embedding) ELSE 0 END as embedding_size
    ''')
    
    record = result.single()
    if record:
        print(f"✅ FOUND in database!")
        print(f"  Name: {record['name']}")
        print(f"  Thai name: {record['thai_name']}")
        print(f"  Full name: {record['full_name']}")
        print(f"  Has embedding: {record['has_embedding']}")
        print(f"  Embedding text: {record['embedding_text']}")
        print(f"  Embedding size: {record['embedding_size']} dimensions")
        
        if not record['has_embedding']:
            print("\n❌ PROBLEM: Person exists but has NO EMBEDDING!")
            print("   This is why vector search cannot find this person.")
    else:
        print("❌ NOT FOUND in database")
    
    print("\n" + "=" * 80)
    print("Checking vector indexes...")
    print("=" * 80)
    
    # Check what vector indexes exist
    indexes_result = session.run('''
        SHOW INDEXES
        YIELD name, type, labelsOrTypes, properties
        WHERE type = "VECTOR"
        RETURN name, labelsOrTypes, properties
    ''')
    
    print("\nVector indexes in database:")
    for idx_record in indexes_result:
        print(f"  • {idx_record['name']}")
        print(f"    Labels: {idx_record['labelsOrTypes']}")
        print(f"    Property: {idx_record['properties']}")
    
    print("\n" + "=" * 80)
    print("Checking sample Person nodes with embeddings...")
    print("=" * 80)
    
    # Check how many Person nodes have embeddings
    stats_result = session.run('''
        MATCH (p:Person)
        RETURN count(p) as total_persons,
               count(p.embedding) as persons_with_embedding,
               count(p.embedding_text) as persons_with_embedding_text
    ''')
    
    stats = stats_result.single()
    print(f"\nTotal Person nodes: {stats['total_persons']}")
    print(f"Persons WITH embedding: {stats['persons_with_embedding']}")
    print(f"Persons WITH embedding_text: {stats['persons_with_embedding_text']}")
    
    # Show some examples
    examples_result = session.run('''
        MATCH (p:Person)
        WHERE p.embedding IS NOT NULL
        RETURN p.`ชื่อ-นามสกุล` as full_name, p.embedding_text as embedding_text
        LIMIT 5
    ''')
    
    print("\nSample Person nodes WITH embeddings:")
    for ex in examples_result:
        print(f"  • {ex['full_name']}")
        print(f"    Embedding text: {ex['embedding_text'][:100]}...")

driver.close()

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("""
If 'อนุทิน ชาญวีรกูล' has NO EMBEDDING:
  → Vector search CANNOT find this person
  → Solution: Run create_vector_index.py to regenerate embeddings
  
If 'อนุทิน ชาญวีรกูล' HAS embedding:
  → Check if embedding_text is meaningful
  → Check if vector index includes this node
  → May need to adjust search similarity threshold
""")

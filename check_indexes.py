"""
Quick diagnostic: Check if vector indexes exist and test searches
"""
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

print("=" * 80)
print("1. CHECKING VECTOR INDEXES")
print("=" * 80)

with driver.session() as session:
    # Show all indexes
    result = session.run("SHOW INDEXES")
    indexes = list(result)
    
    if indexes:
        print(f"\nFound {len(indexes)} indexes:")
        for idx in indexes:
            print(f"\n  Name: {idx.get('name')}")
            print(f"  Type: {idx.get('type')}")
            print(f"  Labels: {idx.get('labelsOrTypes')}")
            print(f"  Properties: {idx.get('properties')}")
            if idx.get('type') == 'VECTOR':
                print(f"  Options: {idx.get('options')}")
    else:
        print("\n⚠️ NO INDEXES FOUND!")

print("\n" + "=" * 80)
print("2. CHECKING FOR NODES WITH EMBEDDINGS")
print("=" * 80)

with driver.session() as session:
    # Count nodes with embeddings by label
    result = session.run("""
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN labels(n)[0] as label, count(n) as count
        ORDER BY count DESC
    """)
    
    has_embeddings = False
    for record in result:
        has_embeddings = True
        print(f"  {record['label']}: {record['count']} nodes with embeddings")
    
    if not has_embeddings:
        print("\n⚠️ NO NODES HAVE EMBEDDINGS!")

print("\n" + "=" * 80)
print("3. SEARCHING FOR 'SANTISOOK' IN ALL PROPERTIES")
print("=" * 80)

with driver.session() as session:
    result = session.run("""
        MATCH (n)
        WHERE any(prop IN keys(n) WHERE 
            toLower(toString(n[prop])) CONTAINS 'santisook'
        )
        RETURN labels(n) as labels, n
        LIMIT 5
    """)
    
    found = False
    for record in result:
        found = True
        labels = record['labels']
        node = dict(record['n'])
        print(f"\n  Found node with labels: {labels}")
        print(f"  Properties:")
        for key, val in node.items():
            if key not in ['embedding', 'embedding_text']:  # Skip embedding arrays
                print(f"    {key}: {val}")
    
    if not found:
        print("\n⚠️ NO NODES FOUND WITH 'SANTISOOK'!")

print("\n" + "=" * 80)
print("4. CHECKING STELLIGENCE PROPERTY")
print("=" * 80)

with driver.session() as session:
    result = session.run("""
        MATCH (n)
        WHERE n.Stelligence IS NOT NULL
        RETURN labels(n) as labels, n.Stelligence as stelligence, count(*) as total
        LIMIT 10
    """)
    
    found = False
    for record in result:
        found = True
        print(f"  Labels: {record['labels']}, Stelligence: {record['stelligence']}")
    
    if not found:
        print("\n⚠️ NO NODES HAVE 'STELLIGENCE' PROPERTY!")

driver.close()
print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)

"""
Fix Neo4j Browser Display Issue
================================
Problem: Nodes show vector embeddings [0.00...] instead of names
Solution: Remove embedding property or use Neo4j Browser settings to hide it

This script provides options to:
1. Check current node properties
2. Remove embedding property (keeps vector index intact)
3. Show proper Cypher queries for Neo4j Browser
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DB = os.getenv('NEO4J_DB', 'neo4j')

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def check_node_properties():
    """Check what properties exist on Person nodes"""
    driver = get_driver()
    session = driver.session(database=NEO4J_DB)
    
    print("Checking Person node properties...")
    result = session.run('''
        MATCH (p:Person)
        RETURN keys(p) as properties
        LIMIT 1
    ''')
    
    record = result.single()
    if record:
        print("Properties found on Person nodes:")
        for prop in record['properties']:
            print(f"  - {prop}")
    
    session.close()
    return record['properties'] if record else []

def count_embeddings():
    """Count how many nodes have embedding property"""
    driver = get_driver()
    session = driver.session(database=NEO4J_DB)
    
    result = session.run('''
        MATCH (p:Person)
        WHERE p.embedding IS NOT NULL
        RETURN count(p) as count
    ''')
    
    record = result.single()
    count = record['count'] if record else 0
    print(f"\nNodes with embedding property: {count}")
    
    session.close()
    return count

def remove_embeddings():
    """Remove embedding property from nodes (keeps vector index)"""
    driver = get_driver()
    session = driver.session(database=NEO4J_DB)
    
    print("\n⚠️  WARNING: This will remove the 'embedding' property from all Person nodes!")
    print("The vector index will still work, but you'll need to recreate embeddings if you want them back.")
    response = input("Continue? (yes/no): ")
    
    if response.lower() == 'yes':
        print("\nRemoving embeddings...")
        result = session.run('''
            MATCH (p:Person)
            WHERE p.embedding IS NOT NULL
            REMOVE p.embedding
            RETURN count(p) as removed
        ''')
        
        record = result.single()
        print(f"✅ Removed embedding from {record['removed']} nodes")
    else:
        print("Cancelled.")
    
    session.close()

def show_browser_tips():
    """Show tips for Neo4j Browser"""
    print("\n" + "="*70)
    print("Neo4j Browser Display Tips")
    print("="*70)
    
    print("\n1. **Use better Cypher queries to hide embeddings:**")
    print("   Instead of: MATCH (p:Person) RETURN p")
    print("   Use this:")
    print('''
   MATCH (p:Person)
   RETURN p.name as name, 
          p.`ชื่อ` as thai_name,
          labels(p) as labels
   LIMIT 25
   ''')
    
    print("\n2. **For visualization without embeddings:**")
    print('''
   MATCH (p:Person)-[r]-(connected)
   RETURN p.name as person, 
          type(r) as relationship,
          connected.name as connected_to
   LIMIT 100
   ''')
    
    print("\n3. **Neo4j Browser Settings:**")
    print("   - Click the gear icon (⚙️) in Neo4j Browser")
    print("   - Go to 'Graph Visualization'")
    print("   - Set 'Caption' to show 'name' or 'ชื่อ' property")
    print("   - This will display names instead of embeddings")
    
    print("\n4. **Quick fix: Remove embedding property**")
    print("   Run this in Neo4j Browser:")
    print('''
   MATCH (p:Person)
   WHERE p.embedding IS NOT NULL
   REMOVE p.embedding
   RETURN count(p) as removed
   ''')
    print("\n   ⚠️  You can recreate embeddings later with: python create_vector_index.py")

if __name__ == "__main__":
    print("Neo4j Browser Display Fixer")
    print("="*70)
    
    # Check current state
    properties = check_node_properties()
    
    if 'embedding' in properties:
        count = count_embeddings()
        
        print("\n" + "="*70)
        print("Options:")
        print("="*70)
        print("1. Show Neo4j Browser tips (recommended)")
        print("2. Remove embedding property from all nodes")
        print("3. Exit")
        
        choice = input("\nChoose option (1-3): ")
        
        if choice == "1":
            show_browser_tips()
        elif choice == "2":
            remove_embeddings()
        else:
            print("Exiting...")
    else:
        print("\n✅ No embedding property found. Display should be fine!")
        show_browser_tips()

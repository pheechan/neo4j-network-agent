"""
Debug script to explore Neo4j database structure and test searches.
Run this to see what nodes exist and why search might not be working.
"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")

def explore_database():
    # URI neo4j+s already includes encryption, so no need for extra config
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    
    with driver.session(database=NEO4J_DB) as session:
        print("=" * 80)
        print("DATABASE EXPLORATION")
        print("=" * 80)
        
        # 1. Count all nodes
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
        print(f"\n📊 Total nodes in database: {count}")
        
        # 2. Get all node labels
        result = session.run("CALL db.labels()")
        labels = [r["label"] for r in result]
        print(f"\n🏷️  Node labels: {labels}")
        
        # 3. Sample nodes for each label
        for label in labels[:5]:  # Limit to first 5 labels
            print(f"\n--- Sample nodes with label '{label}' ---")
            query = f"""
            MATCH (n:{label})
            RETURN n
            LIMIT 3
            """
            result = session.run(query)
            for i, record in enumerate(result, 1):
                node = record["n"]
                props = dict(node)
                print(f"  Node {i}: {props}")
        
        # 4. Get all property keys used in the database
        result = session.run("CALL db.propertyKeys()")
        prop_keys = [r["propertyKey"] for r in result]
        print(f"\n🔑 Property keys in database: {prop_keys}")
        
        # 5. Test search for "นายกรัฐมนตรี"
        search_term = "นายกรัฐมนตรี"
        print(f"\n🔍 Testing search for '{search_term}'...")
        
        # Try searching all text properties
        query = """
        MATCH (n)
        WHERE any(prop IN keys(n) WHERE 
            n[prop] IS NOT NULL AND 
            toString(n[prop]) CONTAINS $search
        )
        RETURN n, labels(n) as labels
        LIMIT 10
        """
        result = session.run(query, search=search_term)
        matches = list(result)
        
        if matches:
            print(f"✅ Found {len(matches)} matches:")
            for i, record in enumerate(matches, 1):
                node = record["n"]
                node_labels = record["labels"]
                props = dict(node)
                print(f"\n  Match {i} (Labels: {node_labels}):")
                print(f"    {props}")
        else:
            print(f"❌ No matches found for '{search_term}'")
            
            # Try case-insensitive search
            print(f"\n🔍 Trying case-insensitive search...")
            query = """
            MATCH (n)
            WHERE any(prop IN keys(n) WHERE 
                n[prop] IS NOT NULL AND 
                toLower(toString(n[prop])) CONTAINS toLower($search)
            )
            RETURN n, labels(n) as labels
            LIMIT 10
            """
            result = session.run(query, search=search_term)
            matches = list(result)
            
            if matches:
                print(f"✅ Found {len(matches)} matches with case-insensitive search:")
                for i, record in enumerate(matches, 1):
                    node = record["n"]
                    node_labels = record["labels"]
                    props = dict(node)
                    print(f"\n  Match {i} (Labels: {node_labels}):")
                    print(f"    {props}")
            else:
                print(f"❌ Still no matches")
        
        # 6. Show sample of nodes with Thai text
        print(f"\n🇹🇭 Sample nodes containing Thai characters...")
        query = """
        MATCH (n)
        WHERE any(prop IN keys(n) WHERE 
            n[prop] IS NOT NULL AND 
            toString(n[prop]) =~ '.*[ก-๙].*'
        )
        RETURN n, labels(n) as labels
        LIMIT 5
        """
        result = session.run(query)
        thai_nodes = list(result)
        
        if thai_nodes:
            print(f"✅ Found {len(thai_nodes)} nodes with Thai text:")
            for i, record in enumerate(thai_nodes, 1):
                node = record["n"]
                node_labels = record["labels"]
                props = dict(node)
                print(f"\n  Node {i} (Labels: {node_labels}):")
                print(f"    {props}")
        else:
            print("❌ No nodes with Thai characters found")
    
    driver.close()
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    try:
        explore_database()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

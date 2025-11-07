"""
Check Neo4j node properties and fix display issues
Run this to diagnose why nodes are showing as numbers instead of names
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
# For Aura, remove :7687 port if present
if ":7687" in NEO4J_URI and "neo4j+s://" in NEO4J_URI:
    NEO4J_URI = NEO4J_URI.replace(":7687", "")
    
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")

print(f"Connecting to: {NEO4J_URI}")

def check_node_properties():
    """Check what properties exist on Person nodes"""
    # For Neo4j Aura, don't specify database parameter
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    
    # Use default database for Aura
    with driver.session() as session:
        print("=" * 60)
        print("CHECKING NEO4J NODE PROPERTIES")
        print("=" * 60)
        
        # Check total count
        result = session.run("MATCH (n:Person) RETURN count(n) AS total")
        total = result.single()["total"]
        print(f"\n✓ Total Person nodes: {total}")
        
        # Check properties
        result = session.run("MATCH (n:Person) RETURN keys(n) AS props LIMIT 1")
        record = result.single()
        if record:
            props = record["props"]
            print(f"\n✓ Properties on Person nodes: {props}")
        
        # Check sample data
        print("\n" + "=" * 60)
        print("SAMPLE PERSON NODES:")
        print("=" * 60)
        
        result = session.run("""
            MATCH (n:Person) 
            RETURN n.`ชื่อ-นามสกุล` AS thai_name, 
                   n.name AS name,
                   n.Stelligence AS stelligence,
                   id(n) AS node_id
            LIMIT 10
        """)
        
        for i, record in enumerate(result, 1):
            print(f"\n{i}. Node ID: {record['node_id']}")
            print(f"   ชื่อ-นามสกุล: {record['thai_name']}")
            print(f"   name: {record['name']}")
            print(f"   Stelligence: {record['stelligence']}")
        
        # Check how many have names
        print("\n" + "=" * 60)
        print("NAME PROPERTY STATISTICS:")
        print("=" * 60)
        
        result = session.run("""
            MATCH (n:Person)
            RETURN 
                count(CASE WHEN n.`ชื่อ-นามสกุล` IS NOT NULL THEN 1 END) AS with_thai_name,
                count(CASE WHEN n.name IS NOT NULL THEN 1 END) AS with_name,
                count(CASE WHEN n.`ชื่อ-นามสกุล` IS NULL AND n.name IS NULL THEN 1 END) AS no_name,
                count(*) AS total
        """)
        
        stats = result.single()
        print(f"\nNodes with 'ชื่อ-นามสกุล': {stats['with_thai_name']}/{stats['total']}")
        print(f"Nodes with 'name': {stats['with_name']}/{stats['total']}")
        print(f"Nodes with NO name: {stats['no_name']}/{stats['total']}")
        
        if stats['no_name'] > 0:
            print("\n⚠️  WARNING: Some nodes don't have name properties!")
            print("   This is why they show as numbers in Neo4j Browser.")
        
        # Check other node types
        print("\n" + "=" * 60)
        print("OTHER NODE TYPES:")
        print("=" * 60)
        
        for label in ['Position', 'Ministry', 'Agency']:
            result = session.run(f"""
                MATCH (n:{label}) 
                RETURN count(n) AS total,
                       count(n.name) AS with_name
            """)
            record = result.single()
            if record and record['total'] > 0:
                print(f"\n{label}: {record['total']} nodes, {record['with_name']} have 'name' property")
                
                # Show sample
                result2 = session.run(f"MATCH (n:{label}) RETURN n.name AS name LIMIT 3")
                samples = [r['name'] for r in result2]
                print(f"  Samples: {samples}")
        
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS:")
        print("=" * 60)
        print("""
1. In Neo4j Browser, click on a Person node in the graph
2. On the right side, click the node type label "Person"
3. Find the "Caption" dropdown
4. Change it from "<id>" to "ชื่อ-นามสกุล"
5. The nodes should now show names!

Alternative: Run this in Neo4j Browser
MATCH (n:Person) 
WHERE n.`ชื่อ-นามสกุล` IS NOT NULL
RETURN n.`ชื่อ-นามสกุล` AS name, n.`กระทรวง` AS ministry
LIMIT 25
        """)
    
    driver.close()

if __name__ == "__main__":
    try:
        check_node_properties()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure your .env file has correct Neo4j credentials:")
        print(f"  NEO4J_URI: {NEO4J_URI}")
        print(f"  NEO4J_USERNAME: {NEO4J_USER}")
        print(f"  NEO4J_DATABASE: {NEO4J_DB}")

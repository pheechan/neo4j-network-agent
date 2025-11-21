"""
Simple script to verify connections by querying Neo4j directly
Uses same connection method as streamlit_app.py
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Get connection details
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")

print(f"Connecting to: {NEO4J_URI}")

# Simple connection test
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    
    # Test query - find all people in database
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Person)
            RETURN p.`ชื่อ-นามสกุล` as name
            LIMIT 20
        """)
        
        print("\n✅ Connection successful!")
        print("\nFirst 20 people in database:")
        for record in result:
            print(f"  - {record['name']}")
    
    # Now check specific connection: วรวุฒิ หลายพูนสวัสดิ์ -> พิพัฒน์ รัชกิจประการ
    print("\n" + "="*80)
    print("Checking specific connection (from your selected line):")
    print("วรวุฒิ หลายพูนสวัสดิ์ -> พิพัฒน์ รัชกิจประการ")
    print("="*80)
    
    with driver.session() as session:
        result = session.run("""
            MATCH path = shortestPath(
                (start:Person)-[*..6]-(target:Person)
            )
            WHERE start.`ชื่อ-นามสกุล` CONTAINS 'วรวุฒิ หลายพูนสวัสดิ์'
            AND target.`ชื่อ-นามสกุล` CONTAINS 'พิพัฒน์ รัชกิจประการ'
            RETURN 
                start.`ชื่อ-นามสกุล` as start_name,
                target.`ชื่อ-นามสกุล` as target_name,
                length(path) as distance,
                [n in nodes(path) | n.`ชื่อ-นามสกุล`] as path_names
            LIMIT 1
        """)
        
        record = result.single()
        if record:
            print(f"\n✅ CONNECTION FOUND!")
            print(f"\nStart: {record['start_name']}")
            print(f"Target: {record['target_name']}")
            print(f"Distance: {record['distance']} hops")
            print(f"\nPath:")
            for i, name in enumerate(record['path_names'], 1):
                print(f"  {i}. {name}")
        else:
            print(f"\n❌ NO CONNECTION FOUND between these two people")
            print("They may not be connected in the database.")
    
    driver.close()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"\nError type: {type(e).__name__}")
    sys.exit(1)

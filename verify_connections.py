"""
Script to verify actual connections in Neo4j database
and find real connected people for test cases
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Connect to Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")

# For Neo4j Aura, the neo4j+s:// or neo4j+ssc:// scheme handles encryption automatically
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))

def find_connected_people_to_target(target_name="พิพัฒน์ รัชกิจประการ", max_distance=4):
    """Find all people connected to target within max_distance hops"""
    with driver.session() as session:
        query = """
        MATCH path = (start:Person)-[*1..4]-(target:Person)
        WHERE target.`ชื่อ-นามสกุล` CONTAINS $target_name
        WITH start, target, 
             [n in nodes(path) | n.`ชื่อ-นามสกุล`] as path_names,
             length(path) as distance
        WHERE start <> target
        RETURN DISTINCT 
            start.`ชื่อ-นามสกุล` as start_name,
            target.`ชื่อ-นามสกุล` as target_name,
            distance,
            path_names
        ORDER BY distance ASC, start_name
        LIMIT 50
        """
        result = session.run(query, target_name=target_name)
        connections = []
        for record in result:
            connections.append({
                'start': record['start_name'],
                'target': record['target_name'],
                'distance': record['distance'],
                'path': ' -> '.join(record['path_names'])
            })
        return connections

def check_specific_connections(person_pairs):
    """Check if specific person pairs are actually connected"""
    with driver.session() as session:
        results = []
        for start_name, target_name in person_pairs:
            query = """
            MATCH path = shortestPath(
                (start:Person)-[*..6]-(target:Person)
            )
            WHERE start.`ชื่อ-นามสกุล` CONTAINS $start_name
            AND target.`ชื่อ-นามสกุล` CONTAINS $target_name
            RETURN 
                start.`ชื่อ-นามสกุล` as start_name,
                target.`ชื่อ-นามสกุล` as target_name,
                length(path) as distance,
                [n in nodes(path) | n.`ชื่อ-นามสกุล`] as path_names
            """
            result = session.run(query, start_name=start_name, target_name=target_name)
            record = result.single()
            if record:
                results.append({
                    'start': record['start_name'],
                    'target': record['target_name'],
                    'distance': record['distance'],
                    'path': ' -> '.join(record['path_names']),
                    'connected': True
                })
            else:
                results.append({
                    'start': start_name,
                    'target': target_name,
                    'connected': False
                })
        return results

if __name__ == "__main__":
    print("=" * 80)
    print("VERIFYING CONNECTIONS IN NEO4J DATABASE")
    print("=" * 80)
    
    # Check the people from Test Case 4
    print("\n1. Checking Test Case 4 connections to 'พิพัฒน์ รัชกิจประการ'...")
    print("-" * 80)
    
    test_case_4_people = [
        ("ฉัฐชัย มีชั้นช่วง", "พิพัฒน์ รัชกิจประการ"),
        ("วรวุฒิ หลายพูนสวัสดิ์", "พิพัฒน์ รัชกิจประการ"),
        ("ประมุข อุณหเลขกะ", "พิพัฒน์ รัชกิจประการ"),
        ("ประเสริฐ สินสุขประเสริฐ", "พิพัฒน์ รัชกิจประการ"),
        ("รุ่งโรจน์กิติยศ -", "พิพัฒน์ รัชกิจประการ"),
        ("ปิติ นฤขัตรพิชัย", "พิพัฒน์ รัชกิจประการ"),
        ("ชื่นสุมน นิวาทวงษ์", "พิพัฒน์ รัชกิจประการ"),
        ("สามารถ ถิระศักดิ์", "พิพัฒน์ รัชกิจประการ"),
        ("วรยุทธ อันเพียร", "พิพัฒน์ รัชกิจประการ"),
        ("พูลพัฒน์ ลีสมบัติไพบูลย์", "พิพัฒน์ รัชกิจประการ"),
    ]
    
    connections = check_specific_connections(test_case_4_people)
    
    connected_count = 0
    not_connected = []
    
    for conn in connections:
        if conn['connected']:
            connected_count += 1
            print(f"✅ {conn['start']}")
            print(f"   -> {conn['target']}")
            print(f"   Distance: {conn['distance']} hops")
            print(f"   Path: {conn['path'][:150]}...")
            print()
        else:
            not_connected.append(conn)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {connected_count}/{len(test_case_4_people)} connections verified")
    print(f"{'='*80}")
    
    if not_connected:
        print("\n❌ NOT CONNECTED:")
        for conn in not_connected:
            print(f"   • {conn['start']} -> {conn['target']}")
    
    # Find alternative connected people
    print("\n\n2. Finding alternative people actually connected to 'พิพัฒน์ รัชกิจประการ'...")
    print("-" * 80)
    
    real_connections = find_connected_people_to_target("พิพัฒน์ รัชกิจประการ")
    
    if real_connections:
        print(f"\nFound {len(real_connections)} people connected to target:")
        print("\nSuggested replacements for Test Case 4:\n")
        
        for i, conn in enumerate(real_connections[:10], 1):
            print(f"{i}. {conn['start']}")
            print(f"   Distance: {conn['distance']} hops")
            print(f"   Path preview: {conn['path'][:100]}...")
            print()
    else:
        print("❌ No connections found!")
    
    driver.close()

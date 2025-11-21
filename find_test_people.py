"""
Find all Person nodes in the database for testing connection paths
"""
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("Finding Person nodes with connections for testing")
print("=" * 80)

try:
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )
    
    with driver.session() as session:
        # Find all Person nodes with their connection counts
        print("\n1. Person nodes with most connections:")
        print("-" * 80)
        
        result = session.run('''
            MATCH (p:Person)
            WITH p, size([(p)-[]-() | 1]) as connections
            WHERE connections > 0
            RETURN p.`ชื่อ-นามสกุล` as full_name,
                   p.name as name,
                   p.`ชื่อ` as thai_name,
                   connections
            ORDER BY connections DESC
            LIMIT 20
        ''')
        
        people = []
        for record in result:
            name = record['full_name'] or record['thai_name'] or record['name'] or 'N/A'
            connections = record['connections']
            people.append((name, connections))
            print(f"  {name}: {connections} connections")
        
        print("\n" + "=" * 80)
        print("2. Finding connected pairs for testing:")
        print("-" * 80)
        
        # Find pairs that are actually connected
        result = session.run('''
            MATCH (a:Person)-[*1..3]-(b:Person)
            WHERE a <> b
            WITH a, b, shortestPath((a)-[*]-(b)) as path
            WHERE length(path) <= 3
            RETURN DISTINCT
                   coalesce(a.`ชื่อ-นามสกุล`, a.name, a.`ชื่อ`) as person_a,
                   coalesce(b.`ชื่อ-นามสกุล`, b.name, b.`ชื่อ`) as person_b,
                   length(path) as hops
            ORDER BY hops ASC
            LIMIT 10
        ''')
        
        print("\nConnected pairs to test:")
        for record in result:
            print(f"  • {record['person_a']} → {record['person_b']} ({record['hops']} hops)")
        
        print("\n" + "=" * 80)
        print("3. Checking specific people:")
        print("-" * 80)
        
        test_names = ['พี่โด่ง', 'พี่เต๊ะ', 'พี่จู๊ฟ', 'อนุทิน ชาญวีรกูล']
        for name in test_names:
            result = session.run('''
                MATCH (p:Person)
                WHERE p.`ชื่อ-นามสกุล` CONTAINS $name
                   OR p.name CONTAINS $name
                   OR p.`ชื่อ` CONTAINS $name
                RETURN coalesce(p.`ชื่อ-นามสกุล`, p.name, p.`ชื่อ`) as name,
                       size([(p)-[]-() | 1]) as connections
            ''', name=name)
            
            record = result.single()
            if record:
                print(f"  ✅ {record['name']}: {record['connections']} connections")
            else:
                print(f"  ❌ {name}: NOT FOUND")
    
    driver.close()
    
    print("\n" + "=" * 80)
    print("RECOMMENDED TEST QUERIES:")
    print("=" * 80)
    if people and len(people) >= 2:
        print(f'\n1. หาเส้นทางจาก "{people[0][0]}" ไป "{people[1][0]}"')
        print(f'2. หาเส้นทางจาก "พี่เต๊ะ" ไป "พี่โด่ง"')
        print(f'3. หาเส้นทางจาก "พี่จู๊ฟ" ไป "พี่เต๊ะ"')

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nNote: SSL errors are expected when running locally.")
    print("This script works fine in the Streamlit Cloud environment.")

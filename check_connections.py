"""
Check if อนุทิน ชาญวีรกูล exists and find connections to พี่โด่ง
"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DB = os.getenv('NEO4J_DB', 'neo4j')

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

driver = get_driver()
session = driver.session(database=NEO4J_DB)

print("="*70)
print("1. Checking if 'อนุทิน ชาญวีรกูล' exists...")
print("="*70)

# Check all possible property names
result = session.run('''
    MATCH (p:Person)
    WHERE p.name CONTAINS "อนุทิน" 
       OR p.`ชื่อ` CONTAINS "อนุทิน"
       OR p.`ชื่อ-นามสกุล` CONTAINS "อนุทิน"
    RETURN p.name as name, 
           p.`ชื่อ` as thai_name,
           p.`ชื่อ-นามสกุล` as full_name,
           keys(p) as properties,
           COUNT { (p)-[]-() } as connections
    LIMIT 10
''')

found_anutin = False
for r in result:
    found_anutin = True
    print(f"\n✅ Found person with 'อนุทิน':")
    print(f"   name: {r['name']}")
    print(f"   ชื่อ: {r['thai_name']}")
    print(f"   ชื่อ-นามสกุล: {r['full_name']}")
    print(f"   connections: {r['connections']}")
    print(f"   properties: {r['properties']}")

if not found_anutin:
    print("❌ No person found with 'อนุทิน' in any property!")

print("\n" + "="*70)
print("2. Checking if 'พี่โด่ง' exists...")
print("="*70)

result = session.run('''
    MATCH (p:Person)
    WHERE p.name CONTAINS "พี่โด่ง" 
       OR p.`ชื่อ` CONTAINS "พี่โด่ง"
       OR p.`ชื่อ-นามสกุล` CONTAINS "พี่โด่ง"
    RETURN p.name as name, 
           p.`ชื่อ` as thai_name,
           p.`ชื่อ-นามสกุล` as full_name,
           keys(p) as properties,
           COUNT { (p)-[]-() } as connections
''')

pidong_found = False
for r in result:
    pidong_found = True
    print(f"\n✅ Found พี่โด่ง:")
    print(f"   name: {r['name']}")
    print(f"   ชื่อ: {r['thai_name']}")
    print(f"   ชื่อ-นามสกุล: {r['full_name']}")
    print(f"   connections: {r['connections']}")
    print(f"   properties: {r['properties']}")

if not pidong_found:
    print("❌ พี่โด่ง not found!")

print("\n" + "="*70)
print("3. Checking if there's ANY path between them...")
print("="*70)

if found_anutin and pidong_found:
    result = session.run('''
        MATCH (a:Person), (b:Person)
        WHERE (a.name CONTAINS "อนุทิน" OR a.`ชื่อ` CONTAINS "อนุทิน" OR a.`ชื่อ-นามสกุล` CONTAINS "อนุทิน")
          AND (b.name CONTAINS "พี่โด่ง" OR b.`ชื่อ` CONTAINS "พี่โด่ง" OR b.`ชื่อ-นามสกุล` CONTAINS "พี่โด่ง")
        WITH a, b
        MATCH path = shortestPath((a)-[*..15]-(b))
        RETURN length(path) as hops,
               [node in nodes(path) | coalesce(node.`ชื่อ-นามสกุล`, node.name, node.`ชื่อ`, 'Unknown')] as names
        LIMIT 1
    ''')
    
    record = result.single()
    if record:
        print(f"\n✅ PATH FOUND! {record['hops']} hops")
        print(f"\nPath:")
        for i, name in enumerate(record['names'], 1):
            print(f"  {i}. {name}")
    else:
        print("\n❌ NO PATH EXISTS between อนุทิน and พี่โด่ง (even up to 15 hops)")

print("\n" + "="*70)
print("4. Finding people who ARE connected to พี่โด่ง...")
print("="*70)

result = session.run('''
    MATCH (pidong:Person)
    WHERE pidong.`ชื่อ-นามสกุล` CONTAINS "พี่โด่ง"
    WITH pidong
    MATCH (pidong)-[r]-(connected:Person)
    RETURN DISTINCT 
           coalesce(connected.`ชื่อ-นามสกุล`, connected.name, connected.`ชื่อ`) as name,
           type(r) as relationship,
           COUNT { (connected)-[]-() } as connections
    ORDER BY connections DESC
    LIMIT 10
''')

print("\nPeople directly connected to พี่โด่ง:")
for r in result:
    print(f"  • {r['name']} ({r['connections']} connections) - via {r['relationship']}")

print("\n" + "="*70)
print("5. Recommended test pairs (both people exist and are connected):")
print("="*70)

# Find pairs that are definitely connected
result = session.run('''
    MATCH (a:Person)-[r]-(b:Person)
    WHERE a.`ชื่อ-นามสกุล` IS NOT NULL 
      AND b.`ชื่อ-นามสกุล` IS NOT NULL
      AND COUNT { (a)-[]-() } > 2
      AND COUNT { (b)-[]-() } > 2
    RETURN DISTINCT
           a.`ชื่อ-นามสกุล` as person1,
           b.`ชื่อ-นามสกุล` as person2,
           COUNT { (a)-[]-() } as conn1,
           COUNT { (b)-[]-() } as conn2
    LIMIT 5
''')

print("\nSuggested test queries (guaranteed to work):")
for i, r in enumerate(result, 1):
    print(f"\n{i}. Test with:")
    print(f'   From: "{r["person1"]}" ({r["conn1"]} connections)')
    print(f'   To:   "{r["person2"]}" ({r["conn2"]} connections)')
    print(f'   Query: หาเส้นทางที่สั้นที่สุดจาก "{r["person1"]}" ไป "{r["person2"]}"')

session.close()
print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

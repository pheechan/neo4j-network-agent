import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Config.neo4j import get_driver, NEO4J_DB

driver = get_driver()
session = driver.session(database=NEO4J_DB)

# Get list of people in database
print("People in database (first 30):")
print("=" * 60)
result = session.run('''
    MATCH (p:Person) 
    WHERE p.name IS NOT NULL OR p.`ชื่อ` IS NOT NULL
    RETURN coalesce(p.name, p.`ชื่อ`) as name, 
           COUNT { (p)-[]-() } as connections
    ORDER BY connections DESC
    LIMIT 30
''')

for i, r in enumerate(result, 1):
    print(f"{i:2}. {r['name']:40} ({r['connections']} connections)")

# Check if specific people exist
print("\n" + "=" * 60)
print("Checking specific people:")
print("=" * 60)

test_names = ["อนุทิน", "พลเอก ประวิตร", "Boss", "พี่โด่ง", "เศรษฐา"]

for name in test_names:
    result = session.run('''
        MATCH (p:Person)
        WHERE p.name CONTAINS $name OR p.`ชื่อ` CONTAINS $name
        RETURN coalesce(p.name, p.`ชื่อ`) as full_name,
               COUNT { (p)-[]-() } as connections
        LIMIT 1
    ''', name=name)
    
    record = result.single()
    if record:
        print(f"✅ Found '{name}': {record['full_name']} ({record['connections']} connections)")
    else:
        print(f"❌ Not found: '{name}'")

session.close()

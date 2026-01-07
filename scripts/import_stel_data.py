"""
Import STelligence Thai Government Connections Database into Neo4j
Handles Thai names, positions, agencies, ministries, and networks
"""
import os
import sys
import csv
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "letmein123")

def import_stel_data(csv_path: str):
    """Import STelligence connections data from CSV"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    
    with driver.session() as session:
        # Clear existing data
        print("🗑️  Clearing existing data...")
        session.run("MATCH (n) DETACH DELETE n")
        
        # Create constraints for better performance
        print("📐 Creating constraints...")
        try:
            session.run("CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
            session.run("CREATE CONSTRAINT agency_name IF NOT EXISTS FOR (a:Agency) REQUIRE a.name IS UNIQUE")
            session.run("CREATE CONSTRAINT ministry_name IF NOT EXISTS FOR (m:Ministry) REQUIRE m.name IS UNIQUE")
            session.run("CREATE CONSTRAINT position_name IF NOT EXISTS FOR (pos:Position) REQUIRE pos.name IS UNIQUE")
        except Exception as e:
            print(f"Note: Constraints may already exist: {e}")
        
        # Read CSV
        print(f"📂 Reading CSV: {csv_path}")
        people = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                people.append({
                    'full_name': row['ชื่อ-นามสกุล'].strip(),
                    'first_name': row['ชื่อ'].strip(),
                    'last_name': row['นามสกุล'].strip(),
                    'nickname': row['ชื่อเล่น'].strip() if row['ชื่อเล่น'].strip() else None,
                    'prefix': row['คำนำหน้า'].strip(),
                    'position': row['ตำแหน่ง'].strip(),
                    'agency': row['หน่วยงาน'].strip(),
                    'ministry': row['กระทรวง'].strip() if row['กระทรวง'].strip() else None,
                    'level': row['Level'].strip() if row['Level'].strip() else None,
                    'remark': row['Remark'].strip() if row['Remark'].strip() else None,
                    'stelligence': row['Stelligence'].strip() if row['Stelligence'].strip() else None,
                    'connect_by': row['Connect by'].strip() if row['Connect by'].strip() else None,
                    'associate': row['Associate'].strip() if row['Associate'].strip() else None
                })
        
        print(f"✅ Found {len(people)} people in CSV")
        
        # Create Person nodes
        print("👤 Creating Person nodes...")
        for person in people:
            session.run("""
                MERGE (p:Person {name: $full_name})
                SET p.first_name = $first_name,
                    p.last_name = $last_name,
                    p.nickname = $nickname,
                    p.prefix = $prefix,
                    p.level = $level,
                    p.remark = $remark,
                    p.stelligence = $stelligence,
                    p.connect_by = $connect_by,
                    p.associate = $associate
            """, 
                full_name=person['full_name'],
                first_name=person['first_name'],
                last_name=person['last_name'],
                nickname=person['nickname'],
                prefix=person['prefix'],
                level=person['level'],
                remark=person['remark'],
                stelligence=person['stelligence'],
                connect_by=person['connect_by'],
                associate=person['associate']
            )
        
        # Create Position, Agency, Ministry nodes and relationships
        print("🏢 Creating positions, agencies, and ministries...")
        for person in people:
            # Create Position
            if person['position']:
                session.run("""
                    MERGE (pos:Position {name: $position})
                """, position=person['position'])
                
                # Link Person to Position
                session.run("""
                    MATCH (p:Person {name: $full_name})
                    MATCH (pos:Position {name: $position})
                    MERGE (p)-[:HOLDS_POSITION]->(pos)
                """, full_name=person['full_name'], position=person['position'])
            
            # Create Agency
            if person['agency']:
                session.run("""
                    MERGE (a:Agency {name: $agency})
                """, agency=person['agency'])
                
                # Link Person to Agency
                session.run("""
                    MATCH (p:Person {name: $full_name})
                    MATCH (a:Agency {name: $agency})
                    MERGE (p)-[:WORKS_AT]->(a)
                """, full_name=person['full_name'], agency=person['agency'])
            
            # Create Ministry
            if person['ministry']:
                session.run("""
                    MERGE (m:Ministry {name: $ministry})
                """, ministry=person['ministry'])
                
                # Link Person to Ministry
                session.run("""
                    MATCH (p:Person {name: $full_name})
                    MATCH (m:Ministry {name: $ministry})
                    MERGE (p)-[:WORKS_IN_MINISTRY]->(m)
                """, full_name=person['full_name'], ministry=person['ministry'])
                
                # Link Agency to Ministry
                if person['agency']:
                    session.run("""
                        MATCH (a:Agency {name: $agency})
                        MATCH (m:Ministry {name: $ministry})
                        MERGE (a)-[:BELONGS_TO_MINISTRY]->(m)
                    """, agency=person['agency'], ministry=person['ministry'])
        
        # Create network connections (Connect by)
        print("🌐 Creating network connections (Connect by)...")
        connect_by_map = {}
        for person in people:
            if person['connect_by']:
                networks = [n.strip() for n in person['connect_by'].split(',')]
                for network in networks:
                    if network:
                        if network not in connect_by_map:
                            connect_by_map[network] = []
                        connect_by_map[network].append(person['full_name'])
        
        # Create Network nodes and connections
        for network_name, members in connect_by_map.items():
            session.run("""
                MERGE (n:Network {name: $network})
            """, network=network_name)
            
            for member in members:
                session.run("""
                    MATCH (p:Person {name: $member})
                    MATCH (n:Network {name: $network})
                    MERGE (p)-[:CONNECTED_BY]->(n)
                """, member=member, network=network_name)
        
        # Create associate connections
        print("🤝 Creating associate relationships...")
        associate_map = {}
        for person in people:
            if person['associate']:
                associates = [a.strip() for a in person['associate'].split(',')]
                for associate in associates:
                    if associate:
                        if associate not in associate_map:
                            associate_map[associate] = []
                        associate_map[associate].append(person['full_name'])
        
        # Create mutual associate connections
        for assoc_name, members in associate_map.items():
            for i, member1 in enumerate(members):
                for member2 in members[i+1:]:
                    session.run("""
                        MATCH (p1:Person {name: $member1})
                        MATCH (p2:Person {name: $member2})
                        MERGE (p1)-[:ASSOCIATE_WITH {via: $assoc}]-(p2)
                    """, member1=member1, member2=member2, assoc=assoc_name)
        
        # Create Stelligence network connections
        print("⭐ Creating Stelligence network connections...")
        stel_members = [p['full_name'] for p in people if p['stelligence']]
        
        for i, member1 in enumerate(stel_members):
            for member2 in stel_members[i+1:]:
                session.run("""
                    MATCH (p1:Person {name: $member1})
                    MATCH (p2:Person {name: $member2})
                    MERGE (p1)-[:STELLIGENCE_NETWORK]-(p2)
                """, member1=member1, member2=member2)
        
        # Get statistics
        stats = {}
        stats['persons'] = session.run("MATCH (p:Person) RETURN count(p) as count").single()['count']
        stats['positions'] = session.run("MATCH (pos:Position) RETURN count(pos) as count").single()['count']
        stats['agencies'] = session.run("MATCH (a:Agency) RETURN count(a) as count").single()['count']
        stats['ministries'] = session.run("MATCH (m:Ministry) RETURN count(m) as count").single()['count']
        stats['networks'] = session.run("MATCH (n:Network) RETURN count(n) as count").single()['count']
        stats['relationships'] = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
        
        print("\n" + "="*60)
        print("✅ Import Complete!")
        print("="*60)
        print(f"📊 Statistics:")
        print(f"   Persons: {stats['persons']}")
        print(f"   Positions: {stats['positions']}")
        print(f"   Agencies: {stats['agencies']}")
        print(f"   Ministries: {stats['ministries']}")
        print(f"   Networks: {stats['networks']}")
        print(f"   Total Relationships: {stats['relationships']}")
        print()
        print("🎯 Example queries to try:")
        print('   - "เส้นทางที่สั้นที่สุดจาก Santisook ไป อนุทิน"')
        print('   - "Santisook รู้จักใครบ้าง"')
        print('   - "ใครบ้างที่ connect by OSK115"')
        print('   - "ถ้าอยากรู้จัก พี่โด่ง ต้อง connect ผ่านใคร"')
        print()
    
    driver.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default path
        csv_path = r"c:\Users\(Phee)PheemaphatChan\Downloads\STEL Connections\Network-Agent-Main\STel_connection_database.csv.utf8.csv"
    
    print("="*60)
    print("STelligence Database Import")
    print("="*60)
    print(f"CSV Path: {csv_path}")
    print(f"Neo4j URI: {NEO4J_URI}")
    print()
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at {csv_path}")
        print("Usage: python import_stel_data.py <path_to_csv>")
        sys.exit(1)
    
    try:
        import_stel_data(csv_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

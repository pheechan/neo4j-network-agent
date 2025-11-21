from Config.neo4j import get_driver, NEO4J_DB

driver = get_driver()
session = driver.session(database=NEO4J_DB)

# Check if people exist
print("Searching for อนุทิน...")
result = session.run('''
    MATCH (p:Person) 
    WHERE p.name CONTAINS "อนุทิน" OR p.`ชื่อ` CONTAINS "อนุทิน"
    RETURN p.name as name, p.`ชื่อ` as thai_name, labels(p) as labels
    LIMIT 5
''')
for r in result:
    print(f"  {r['name']} | {r['thai_name']} | {r['labels']}")

print("\nSearching for จุรินทร์...")
result = session.run('''
    MATCH (p:Person) 
    WHERE p.name CONTAINS "จุรินทร์" OR p.`ชื่อ` CONTAINS "จุรินทร์"
    RETURN p.name as name, p.`ชื่อ` as thai_name, labels(p) as labels
    LIMIT 5
''')
for r in result:
    print(f"  {r['name']} | {r['thai_name']} | {r['labels']}")

# Test connection path query with new syntax
print("\n\nTesting connection path query...")
result = session.run('''
    MATCH (a:Person), (b:Person)
    WHERE (a.name CONTAINS "อนุทิน" OR a.`ชื่อ` CONTAINS "อนุทิน")
      AND (b.name CONTAINS "จุรินทร์" OR b.`ชื่อ` CONTAINS "จุรินทร์")
    WITH a, b
    MATCH path = allShortestPaths((a)-[*..3]-(b))
    WITH path, length(path) as hops,
         nodes(path) as path_nodes,
         relationships(path) as path_rels
    WITH path, hops, path_nodes, path_rels,
         [node in path_nodes[1..-1] | COUNT { (node)-[]-() }] as intermediate_connections
    WITH path, hops, path_nodes, path_rels,
         reduce(total = 0, conn in intermediate_connections | total + conn) as total_connections
    RETURN hops,
           [node in path_nodes | {
               name: coalesce(node.name, node.`ชื่อ`, 'Unknown'), 
               labels: labels(node),
               connections: COUNT { (node)-[]-() }
           }] as path_nodes,
           [rel in path_rels | type(rel)] as path_rels,
           total_connections
    ORDER BY hops ASC, total_connections DESC
    LIMIT 1
''')

record = result.single()
if record:
    print(f"✅ Path found: {record['hops']} hops")
    print(f"Total intermediate connections: {record['total_connections']}")
    print("\nPath:")
    for i, node in enumerate(record['path_nodes']):
        print(f"  {i+1}. {node['name']} ({node['connections']} connections)")
    print("\nRelationships:", record['path_rels'])
else:
    print("❌ No path found")

session.close()

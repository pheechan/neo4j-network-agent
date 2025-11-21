"""Check what relationships connect Person nodes in the graph"""
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# Note: This will fail locally with SSL error, but shows the query structure
print("=" * 80)
print("Query to check Person-to-Person relationships")
print("=" * 80)
print("""
Run this in Neo4j Browser:

// 1. Check what relationship types exist between Person nodes
MATCH (p1:Person)-[r]-(p2:Person)
RETURN DISTINCT type(r) as relationship_type, count(*) as count
ORDER BY count DESC

// 2. Check if Person connects through intermediate nodes
MATCH (p1:Person)-[r1]->(intermediate)-[r2]->(p2:Person)
WHERE NOT 'Person' IN labels(intermediate)
RETURN labels(intermediate) as intermediate_type, 
       type(r1) as from_person,
       type(r2) as to_person,
       count(*) as count
ORDER BY count DESC
LIMIT 10

// 3. Check specific path for อนุทิน to พี่โด่ง
MATCH (a:Person), (b:Person)
WHERE a.`ชื่อ-นามสกุล` CONTAINS "อนุทิน"
  AND b.`ชื่อ-นามสกุล` CONTAINS "พี่โด่ง"
WITH a, b
MATCH path = shortestPath((a)-[*..10]-(b))
RETURN 
  [node in nodes(path) | {name: coalesce(node.`ชื่อ-นามสกุล`, node.name, 'N/A'), labels: labels(node)}] as nodes,
  [rel in relationships(path) | type(rel)] as relationships,
  length(path) as hops

// 4. If no direct Person-to-Person relationships, find common connections
MATCH (a:Person {`ชื่อ-นามสกุล`: "อนุทิน ชาญวีรกูล"})-[r1]->(common)<-[r2]-(b:Person)
WHERE b.`ชื่อ-นามสกุล` CONTAINS "พี่โด่ง"
RETURN 
  a.`ชื่อ-นามสกุล` as person_a,
  labels(common) as common_node_type,
  coalesce(common.name, common.`ชื่อ-นามสกุล`) as common_node,
  b.`ชื่อ-นามสกุล` as person_b,
  type(r1) as rel_a_to_common,
  type(r2) as rel_common_to_b
LIMIT 10
""")

print("\n" + "=" * 80)
print("EXPLANATION:")
print("=" * 80)
print("""
The issue is that Person nodes might not be DIRECTLY connected.
Instead, they connect through intermediate nodes like:
  - Connect_by (network names like "Santisook")
  - Position
  - Agency
  - Ministry

Current path shows:
  อนุทิน → Santisook → พี่โด่ง

This means:
  - อนุทิน has relationship to Santisook (Connect_by node)
  - พี่โด่ง has relationship to same Santisook node
  - They share a common connection point

SOLUTION OPTIONS:

1. EXPAND PATH to include intermediate Person nodes:
   อนุทิน → Santisook → [Other People] → พี่โด่ง
   
2. FIND PEOPLE in Santisook network:
   Show: "Both are in Santisook network, connected through: [list of people]"

3. CHANGE QUERY to find Person-to-Person path:
   Skip non-Person nodes, only show actual people in the path
""")

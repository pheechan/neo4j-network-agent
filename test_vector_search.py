"""Test vector search for specific person"""
import os
from dotenv import load_dotenv

load_dotenv()

# Import the vector search function
from KG.VectorSearchDirect import query_with_relationships

# Test search for อนุทิน
print("=" * 80)
print("Testing vector search for: อนุทิน ชาญวีรกูล")
print("=" * 80)

queries_to_test = [
    "อนุทิน ชาญวีรกูล",
    "อนุทิน",
    "ชาญวีรกูล",
    "นายอนุทิน ชาญวีรกูล",
]

for query in queries_to_test:
    print(f"\n🔍 Query: '{query}'")
    print("-" * 80)
    
    try:
        results = query_with_relationships(query, top_k_per_index=5)
        
        if results:
            print(f"✅ Found {len(results)} results:")
            for i, result in enumerate(results[:3], 1):
                node = result
                labels = node.get('__labels__', [])
                
                if 'Person' in labels:
                    name = node.get('ชื่อ-นามสกุล') or node.get('name') or node.get('ชื่อ')
                    print(f"  {i}. {name} (Person)")
                    if 'embedding_text' in node:
                        print(f"     Embedding text: {node['embedding_text'][:80]}...")
                else:
                    print(f"  {i}. {labels[0] if labels else 'Unknown'}: {node.get('name', 'N/A')}")
        else:
            print("❌ No results found")
            
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print("""
If NO results found for any variation of the name:
  → Person either doesn't have an embedding OR
  → The embedding doesn't match the query embedding well
  
Solutions:
1. Check if person has embedding in Neo4j directly
2. Regenerate embeddings with create_vector_index.py
3. Check if embedding_text field contains searchable content
""")

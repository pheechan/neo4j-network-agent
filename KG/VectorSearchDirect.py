"""
Custom vector search that actually works with Neo4j.
Instead of using LangChain's Neo4jVector (which can't read embedding_text),
we'll query Neo4j directly and get all the node properties.
"""
from typing import List, Tuple
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Try to import embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


def get_embeddings_model():
    """Get HuggingFace embeddings model."""
    if not EMBEDDINGS_AVAILABLE:
        return None
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def query_vector_search_direct(
    question: str,
    vector_index_name: str = "person_vector_index",
    top_k: int = 5
) -> List[Tuple[dict, float]]:
    """
    Query Neo4j vector index DIRECTLY without LangChain.
    Returns list of (node_properties_dict, score) tuples.
    """
    embeddings_model = get_embeddings_model()
    if not embeddings_model:
        raise ValueError("HuggingFace embeddings not available")
    
    # Generate embedding for the question
    question_embedding = embeddings_model.embed_query(question)
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    
    # Query vector index directly with Cypher
    query = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
    YIELD node, score
    RETURN node, labels(node) as labels, properties(node) as props, score
    ORDER BY score DESC
    """
    
    results = []
    with driver.session(database=os.getenv("NEO4J_DB", "neo4j")) as session:
        result = session.run(
            query,
            index_name=vector_index_name,
            top_k=top_k,
            embedding=question_embedding
        )
        
        for record in result:
            node_props = dict(record["props"])
            node_props["__labels__"] = record["labels"]
            score = record["score"]
            results.append((node_props, score))
    
    driver.close()
    return results


if __name__ == "__main__":
    # Test the function
    print("Testing direct vector search...")
    try:
        results = query_vector_search_direct("Santisook", top_k=3)
        print(f"\nFound {len(results)} results:")
        for i, (node, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.4f}")
            print(f"   Labels: {node.get('__labels__')}")
            print(f"   Properties: {[k for k in node.keys() if k not in ['embedding', 'embedding_text', '__labels__']]}")
            if 'embedding_text' in node:
                print(f"   embedding_text: {node['embedding_text'][:100]}...")
    except Exception as e:
        print(f"Error: {e}")

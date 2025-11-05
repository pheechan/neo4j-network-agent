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


def query_multiple_vector_indexes(
    question: str,
    vector_indexes: List[str] = None,
    top_k_per_index: int = 3
) -> List[Tuple[dict, float]]:
    """
    Query MULTIPLE Neo4j vector indexes and combine results.
    This allows searching across Person, Position, Agency, etc. simultaneously.
    
    Args:
        question: The query string
        vector_indexes: List of index names to search. If None, uses default set.
        top_k_per_index: How many results to get from each index
        
    Returns:
        Combined list of (node_properties_dict, score) tuples, sorted by score
    """
    # Default indexes to search - covers most important node types
    if vector_indexes is None:
        vector_indexes = [
            "person_vector_index",      # Person nodes
            "position_vector_index",    # Position nodes
            "agency_vector_index",      # Agency nodes
            "ministry_vector_index",    # Ministry nodes
            "remark_vector_index",      # Remark nodes
            "connect_by_vector_index",  # Connect by nodes
        ]
    
    embeddings_model = get_embeddings_model()
    if not embeddings_model:
        raise ValueError("HuggingFace embeddings not available")
    
    # Generate embedding once for the question
    question_embedding = embeddings_model.embed_query(question)
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    
    # Query each index and collect all results
    all_results = []
    with driver.session(database=os.getenv("NEO4J_DB", "neo4j")) as session:
        for index_name in vector_indexes:
            try:
                query = """
                CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                YIELD node, score
                RETURN node, labels(node) as labels, properties(node) as props, score
                """
                
                result = session.run(
                    query,
                    index_name=index_name,
                    top_k=top_k_per_index,
                    embedding=question_embedding
                )
                
                for record in result:
                    node_props = dict(record["props"])
                    node_props["__labels__"] = record["labels"]
                    score = record["score"]
                    all_results.append((node_props, score))
                    
            except Exception as e:
                # Skip indexes that don't exist or have errors
                print(f"Warning: Could not query index {index_name}: {e}")
                continue
    
    driver.close()
    
    # Sort all results by score (descending)
    all_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top results across all indexes
    return all_results[:top_k_per_index * 2]  # Return roughly 2x top_k_per_index results total


if __name__ == "__main__":
    # Test the function
    print("Testing direct vector search (single index)...")
    try:
        results = query_vector_search_direct("Santisook", top_k=3)
        print(f"\nFound {len(results)} results:")
        for node_props, score in results:
            labels = node_props.get("__labels__", [])
            print(f"\nScore: {score:.4f}")
            print(f"Labels: {labels}")
            print(f"Properties: {node_props}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*80)
    print("\nTesting multi-index vector search...")
    try:
        results = query_multiple_vector_indexes("อนุทินทำงานตำแหน่งอะไร", top_k_per_index=3)
        print(f"\nFound {len(results)} results across all indexes:")
        for node_props, score in results:
            labels = node_props.get("__labels__", [])
            print(f"\nScore: {score:.4f}")
            print(f"Labels: {labels}")
            print(f"Properties: {node_props}")
        for i, (node, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.4f}")
            print(f"   Labels: {node.get('__labels__')}")
            print(f"   Properties: {[k for k in node.keys() if k not in ['embedding', 'embedding_text', '__labels__']]}")
            if 'embedding_text' in node:
                print(f"   embedding_text: {node['embedding_text'][:100]}...")
    except Exception as e:
        print(f"Error: {e}")

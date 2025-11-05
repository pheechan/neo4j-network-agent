"""
Vector RAG using HuggingFace embeddings (free, no API key required).

To use this instead of OpenAI embeddings:
1. Install: pip install sentence-transformers
2. In streamlit_app.py, change:
   from KG.VectorRAG import query_vector_rag
   to:
   from KG.VectorRAG_HuggingFace import query_vector_rag
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from dotenv import load_dotenv
import os

load_dotenv()

def query_vector_rag(
    question: str,
    vector_index_name: str,
    vector_node_label: str,
    vector_source_property: str,
    vector_embedding_property: str,
    top_k: int = 3
):
    """
    Query vector store using HuggingFace embeddings (no API key needed).
    Uses the 'all-MiniLM-L6-v2' model by default (small, fast, decent quality).
    """
    # Use a free, local embedding model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = Neo4jVector.from_existing_graph(
        embedding=embedding,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        index_name=vector_index_name,
        node_label=vector_node_label,
        text_node_properties=[vector_source_property],
        embedding_node_property=vector_embedding_property,
    )
    
    docs_and_scores = vector_store.similarity_search_with_score(
        question,
        k=top_k
    )

    return docs_and_scores

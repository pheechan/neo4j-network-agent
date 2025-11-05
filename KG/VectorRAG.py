
from langchain_community.vectorstores import Neo4jVector
from dotenv import load_dotenv
import os

load_dotenv()

# Try to import embeddings providers in order of preference
EMBEDDINGS_PROVIDER = None
try:
    # Option 1: Try HuggingFace embeddings (free, no API key)
    from langchain_huggingface import HuggingFaceEmbeddings
    EMBEDDINGS_PROVIDER = "huggingface"
except ImportError:
    try:
        # Option 2: Fall back to OpenAI if available
        from langchain_openai import OpenAIEmbeddings
        EMBEDDINGS_PROVIDER = "openai"
    except ImportError:
        pass

def get_embeddings():
    """Get the appropriate embeddings instance based on what's available."""
    if EMBEDDINGS_PROVIDER == "huggingface":
        # Use a lightweight, multilingual model that supports Thai
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif EMBEDDINGS_PROVIDER == "openai":
        # Check if OpenAI key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI embeddings selected but OPENAI_API_KEY not set")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    else:
        raise ImportError(
            "No embeddings provider available. Install one of:\n"
            "  pip install langchain-huggingface sentence-transformers\n"
            "  pip install langchain-openai (requires OPENAI_API_KEY)"
        )

def query_vector_rag(
    question: str,
    vector_index_name: str,
    vector_node_label: str,
    vector_source_property: str,
    vector_embedding_property: str,
    top_k: int = 3
):
    """
    Query vector store using either HuggingFace (free) or OpenAI embeddings.
    Automatically selects the best available provider.
    """
    # Get embeddings instance
    embedding = get_embeddings()
    
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

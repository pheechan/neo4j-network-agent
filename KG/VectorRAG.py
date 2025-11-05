
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from dotenv import load_dotenv
import os

load_dotenv()

# OpenRouter keys (sk-or-...) are NOT compatible with OpenAI's servers.
# Two options:
# 1) Use an actual OpenAI API key in OPENAI_API_KEY
# 2) Configure OpenAIEmbeddings to use OpenRouter's endpoint (if supported)
#
# For now, if only OpenRouter key exists, set up the client to try OpenRouter:
if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
    # Set base URL to OpenRouter - note: OpenRouter may not support embeddings endpoint
    # If this fails, you'll need to either:
    # - Get a real OpenAI key, or
    # - Use a different embeddings provider (HuggingFace, sentence-transformers, etc.)
    base = os.getenv("OPENROUTER_API_BASE") or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    os.environ["OPENAI_API_BASE"] = base

def query_vector_rag(
    question: str,
    vector_index_name: str,
    vector_node_label: str,
    vector_source_property: str,
    vector_embedding_property: str,
    top_k: int = 3
):
    # initialize the Neo4j-backed vector store
    # initialize the embeddings object; this will raise a clear error if
    # no compatible API key is available to the underlying client.
    embedding = OpenAIEmbeddings()

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
    # directly run a similarity search with scores
    docs_and_scores = vector_store.similarity_search_with_score(
        question,
        k=top_k
    )

    return docs_and_scores

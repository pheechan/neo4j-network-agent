import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

load_dotenv()


def load_neo4j_graph() -> Neo4jGraph:
    """Create and return a Neo4jGraph object using environment variables.

    Environment variables used:
    - NEO4J_URI (e.g. neo4j+s://<host>:7687)
    - NEO4J_USERNAME
    - NEO4J_PASSWORD
    - NEO4J_DATABASE
    """
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '')
    NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')

    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    return graph

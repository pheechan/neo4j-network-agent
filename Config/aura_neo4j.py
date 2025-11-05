import os
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector

load_dotenv()

def vector_search(question, index_name, node_label, text_prop, emb_prop, k=3):
    vector_store = Neo4jVector.from_existing_graph(
        embedding=OpenAIEmbeddings(),
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        index_name=index_name,
        node_label=node_label,
        text_node_properties=[text_prop],
        embedding_node_property=emb_prop,
    )
    return vector_store.similarity_search_with_score(question, k=k)

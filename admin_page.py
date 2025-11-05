"""
Admin page for Neo4j database management - run with: streamlit run admin_page.py
"""
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Read from Streamlit secrets (cloud) or environment variables (local)
def get_config(key, default=""):
    """Get config from st.secrets (Streamlit Cloud) or os.getenv (local)"""
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except:
        return os.getenv(key, default)

NEO4J_URI = get_config("NEO4J_URI")
NEO4J_USERNAME = get_config("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = get_config("NEO4J_PASSWORD")
NEO4J_DATABASE = get_config("NEO4J_DATABASE", "neo4j")

st.set_page_config(page_title="Neo4j Admin", page_icon="🔧", layout="wide")

st.title("🔧 Neo4j Database Admin")
st.markdown("Manage your Neo4j database: create vector indexes and generate embeddings")

# Initialize connection
@st.cache_resource
def get_graph():
    # For Python 3.13+ with neo4j+s:// URIs, we need to handle SSL differently
    # Use the neo4j driver directly with proper configuration
    from neo4j import GraphDatabase
    import ssl
    
    try:
        # Try with standard LangChain connection first
        return Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
    except Exception as e:
        # If that fails, create driver manually and wrap it
        st.warning(f"Standard connection failed, trying alternative method...")
        
        # For neo4j+s:// URIs, the SSL is handled automatically
        # We just need to ensure the driver is created properly
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
            max_connection_lifetime=3600,
            keep_alive=True
        )
        
        # Create a simple wrapper that mimics Neo4jGraph.query()
        class SimpleGraphWrapper:
            def __init__(self, driver, database):
                self._driver = driver
                self._database = database
            
            def query(self, cypher, params=None):
                with self._driver.session(database=self._database) as session:
                    result = session.run(cypher, params or {})
                    return [dict(record) for record in result]
        
        return SimpleGraphWrapper(driver, NEO4J_DATABASE)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

try:
    graph = get_graph()
    embeddings = get_embeddings()
    st.success("✓ Connected to Neo4j")
except Exception as e:
    st.error(f"✗ Connection failed: {e}")
    st.stop()

# Tab layout
tab1, tab2, tab3 = st.tabs(["📊 Database Info", "🔍 Search Debug", "⚡ Vector Setup"])

with tab1:
    st.header("Database Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Node count
        result = graph.query("MATCH (n) RETURN count(n) as count")
        node_count = result[0].get("count", 0) if result else 0
        st.metric("Total Nodes", node_count)
        
        # Nodes with embeddings
        result = graph.query("MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) as count")
        embedding_count = result[0].get("count", 0) if result else 0
        st.metric("Nodes with Embeddings", embedding_count)
    
    with col2:
        # Relationship count
        result = graph.query("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = result[0].get("count", 0) if result else 0
        st.metric("Total Relationships", rel_count)
        
        # Indexes
        result = graph.query("SHOW INDEXES")
        index_count = len(result)
        st.metric("Indexes", index_count)
    
    st.subheader("Node Labels")
    result = graph.query("CALL db.labels()")
    labels = [record.get("label", "") for record in result]
    
    if labels:
        for label in labels:
            if label:
                result = graph.query(f"MATCH (n:{label}) RETURN count(n) as count")
                count = result[0].get("count", 0) if result else 0
                st.write(f"**{label}**: {count} nodes")
    
    st.subheader("Property Keys")
    result = graph.query("""
    MATCH (n)
    UNWIND keys(n) as key
    RETURN DISTINCT key
    LIMIT 50
    """)
    keys = [record.get('key') for record in result]
    st.write(", ".join(keys) if keys else "No properties found")
    
    st.subheader("Sample Nodes")
    result = graph.query("""
    MATCH (n)
    RETURN labels(n) as labels, properties(n) as props
    LIMIT 10
    """)
    
    for i, record in enumerate(result, 1):
        with st.expander(f"Node {i}: {record.get('labels', [])}"):
            st.json(record.get('props', {}))

with tab2:
    st.header("Search for 'Santisook'")
    
    search_term = st.text_input("Search term:", value="Santisook")
    
    if st.button("🔍 Search"):
        with st.spinner("Searching..."):
            # Search in all properties
            query = """
            MATCH (n)
            WHERE any(prop in keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($search))
            RETURN labels(n) as labels, properties(n) as props, id(n) as nodeId
            LIMIT 20
            """
            result = graph.query(query, params={"search": search_term})
            
            if result:
                st.success(f"✓ Found {len(result)} nodes")
                for i, record in enumerate(result, 1):
                    with st.expander(f"Match {i}: {record.get('labels', [])}"):
                        st.write("**Node ID:**", record.get('nodeId'))
                        st.json(record.get('props', {}))
            else:
                st.warning(f"✗ No nodes found containing '{search_term}'")
                
                # Show alternative: list all nodes with 'name' property
                st.info("Showing all nodes with 'name' property instead:")
                result = graph.query("""
                MATCH (n)
                WHERE n.name IS NOT NULL
                RETURN DISTINCT n.name as name, labels(n) as labels
                LIMIT 20
                """)
                
                if result:
                    for record in result:
                        st.write(f"- **{record.get('name')}** ({record.get('labels', [])})")

with tab3:
    st.header("Vector Index Setup")
    
    st.info("""
    **What this does:**
    1. Creates vector indexes for storing embeddings
    2. Generates embeddings for all nodes using HuggingFace (free, multilingual)
    3. Stores embeddings in Neo4j for semantic search
    
    **Why you need this:**
    - Enables AI-powered semantic search
    - Finds similar content even without exact keyword matches
    - Required for vector-based RAG (Retrieval Augmented Generation)
    """)
    
    # Get labels
    result = graph.query("CALL db.labels()")
    labels = [record.get("label", "") for record in result if record.get("label")]
    
    if not labels:
        st.warning("No node labels found in database")
        st.stop()
    
    st.subheader("1. Create Vector Indexes")
    
    for label in labels:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{label}** index")
        
        with col2:
            # Check if index exists
            result = graph.query("SHOW INDEXES")
            existing_indexes = [r.get("name", "") for r in result]
            index_name = f"{label.lower()}_vector_index"
            
            if index_name in existing_indexes:
                st.success("✓ Exists")
            else:
                st.warning("✗ Missing")
        
        with col3:
            if st.button(f"Create", key=f"create_{label}"):
                try:
                    query = f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR (n:{label})
                    ON n.embedding
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: 384,
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                    """
                    graph.query(query)
                    st.success(f"✓ Created {index_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    st.subheader("2. Generate Embeddings")
    
    for label in labels:
        # Count nodes without embeddings
        result = graph.query(f"""
        MATCH (n:{label})
        WHERE n.embedding IS NULL
        RETURN count(n) as count
        """)
        missing_count = result[0].get("count", 0) if result else 0
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{label}** nodes")
        
        with col2:
            if missing_count > 0:
                st.warning(f"{missing_count} without embeddings")
            else:
                st.success("✓ All have embeddings")
        
        with col3:
            if missing_count > 0:
                if st.button(f"Generate ({missing_count})", key=f"gen_{label}"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Get nodes
                    query = f"""
                    MATCH (n:{label})
                    WHERE n.embedding IS NULL
                    RETURN id(n) as nodeId, properties(n) as props
                    LIMIT 100
                    """
                    result = graph.query(query)
                    nodes = list(result)
                    
                    success_count = 0
                    for i, record in enumerate(nodes):
                        node_id = record.get("nodeId")
                        props = record.get("props", {})
                        
                        # Create text from properties
                        text_parts = []
                        for key, value in props.items():
                            if key != "embedding" and value:
                                text_parts.append(f"{key}: {value}")
                        
                        if text_parts:
                            text = " | ".join(text_parts)
                            status_text.text(f"Processing node {i+1}/{len(nodes)}: {text[:50]}...")
                            
                            try:
                                # Generate embedding
                                embedding = embeddings.embed_query(text)
                                
                                # Store embedding
                                update_query = f"""
                                MATCH (n:{label})
                                WHERE id(n) = $nodeId
                                SET n.embedding = $embedding
                                SET n.embedding_text = $text
                                """
                                graph.query(update_query, params={"nodeId": node_id, "embedding": embedding, "text": text})
                                success_count += 1
                            except Exception as e:
                                st.error(f"Error on node {node_id}: {e}")
                        
                        progress_bar.progress((i + 1) / len(nodes))
                    
                    status_text.text(f"✓ Generated {success_count} embeddings")
                    st.success("Done! Refresh page to see updated counts.")
                    st.balloons()

st.markdown("---")
st.caption("💡 Tip: After generating embeddings, wait 1-2 minutes for indexes to populate before testing searches")

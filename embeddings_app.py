"""
Streamlit App to Regenerate Embeddings for Neo4j Database
Run this after importing new data to regenerate all embeddings
"""

import streamlit as st
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(page_title="Regenerate Embeddings", page_icon="🔄", layout="wide")

st.title("🔄 Regenerate Embeddings")
st.markdown("---")

# Get Neo4j credentials (same as streamlit_app.py)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Initialize embeddings model (cached)
@st.cache_resource
def load_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

# Connect to Neo4j (cached) - using same method as streamlit_app.py
@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

st.info("📊 **Step 1:** Check current embedding status")

if st.button("🔍 Check Embedding Status", type="secondary"):
    try:
        driver = get_neo4j_driver()
        st.success(f"✅ Connected to Neo4j: {NEO4J_URI}")
        
        # Check current status
        with driver.session() as session:
            result = session.run("""
                MATCH (p:Person)
                RETURN 
                    count(p) as total,
                    count(p.embedding) as with_embedding,
                    count(p.embedding_text) as with_text
            """)
            stats = result.single()
        
        if stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Person Nodes", stats['total'])
            with col2:
                st.metric("With Embeddings", stats['with_embedding'])
            with col3:
                missing = stats['total'] - stats['with_embedding']
                st.metric("Missing Embeddings", missing)
            
            coverage = (stats['with_embedding'] / stats['total'] * 100) if stats['total'] > 0 else 0
            st.progress(coverage / 100)
            st.caption(f"Coverage: {coverage:.1f}%")
            
            if missing == 0:
                st.success("🎉 All Person nodes already have embeddings!")
            else:
                st.warning(f"⚠️ {missing} Person nodes need embeddings")
                
                st.markdown("---")
                st.info("📊 **Step 2:** Generate missing embeddings")
                
                if st.button("🚀 Start Embedding Generation", type="primary"):
                    embeddings = load_embeddings_model()
                    st.success("✅ Embeddings model loaded")
                    
                    # Get persons without embeddings
                    with driver.session() as session:
                        result = session.run("""
                            MATCH (p:Person)
                            WHERE p.embedding IS NULL
                            RETURN 
                                id(p) as node_id,
                                p.`ชื่อ-นามสกุล` as full_name,
                                p.`ตำแหน่ง` as position,
                                p.`หน่วยงาน` as agency,
                                p.name as name,
                                properties(p) as all_props
                        """)
                        persons = list(result)
                    total_persons = len(persons)
                    
                    st.write(f"Processing {total_persons} persons...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    success_count = 0
                    error_count = 0
                    
                    for i, person in enumerate(persons, 1):
                        node_id = person["node_id"]
                        
                        # Build text for embedding
                        text_parts = []
                        
                        if person.get("full_name"):
                            text_parts.append(f"ชื่อ: {person['full_name']}")
                        elif person.get("name"):
                            text_parts.append(f"Name: {person['name']}")
                        
                        if person.get("position"):
                            text_parts.append(f"ตำแหน่ง: {person['position']}")
                        
                        if person.get("agency"):
                            text_parts.append(f"หน่วยงาน: {person['agency']}")
                        
                        if not text_parts:
                            all_props = person.get("all_props", {})
                            for key, value in all_props.items():
                                if key not in ["embedding", "embedding_text"] and value:
                                    text_parts.append(f"{key}: {value}")
                        
                        if not text_parts:
                            error_count += 1
                            continue
                        
                        embedding_text = " | ".join(text_parts)
                        
                        try:
                            # Generate embedding
                            embedding_vector = embeddings.embed_query(embedding_text)
                            
                            # Store in Neo4j
                            with driver.session() as session:
                                session.run("""
                                    MATCH (p:Person)
                                    WHERE id(p) = $node_id
                                    SET p.embedding = $embedding,
                                        p.embedding_text = $text
                                """, node_id=node_id, embedding=embedding_vector, text=embedding_text)
                            
                            success_count += 1
                        except Exception as e:
                            error_count += 1
                            st.error(f"Error for node {node_id}: {e}")
                        
                        # Update progress
                        progress_bar.progress(i / total_persons)
                        status_text.text(f"Processing {i}/{total_persons} ({i/total_persons*100:.1f}%)")
                    
                    st.success(f"✅ Successfully generated {success_count} embeddings")
                    if error_count > 0:
                        st.warning(f"⚠️ Failed to generate {error_count} embeddings")
                    
                    # Verify
                    st.markdown("---")
                    st.info("📊 **Step 3:** Verify results")
                    
                    with driver.session() as session:
                        result = session.run("""
                            MATCH (p:Person)
                            RETURN 
                                count(p) as total,
                                count(p.embedding) as with_embedding
                        """)
                        verify_stats = result.single()
                        
                        if verify_stats:
                            new_coverage = (verify_stats['with_embedding'] / verify_stats['total'] * 100) if verify_stats['total'] > 0 else 0
                            st.metric("New Coverage", f"{new_coverage:.1f}%")
                            
                            # Show sample
                            st.subheader("Sample persons with embeddings:")
                            sample_result = session.run("""
                                MATCH (p:Person)
                                WHERE p.embedding IS NOT NULL
                                RETURN p.`ชื่อ-นามสกุล` as name, p.embedding_text as text
                                LIMIT 5
                            """)
                            
                            for record in sample_result:
                                with st.expander(record['name']):
                                    st.text(record['text'])
                            
                            st.balloons()
    
    except Exception as e:
        st.error(f"❌ Error connecting to Neo4j: {e}")
        st.info("Make sure your .env file has correct Neo4j credentials")

st.markdown("---")
st.markdown("""
### ℹ️ About This Tool

This tool regenerates embeddings for all Person nodes in your Neo4j database that are missing them.

**When to use:**
- After importing new data into Neo4j
- After accidentally deleting embeddings
- To fix nodes that show up as numbers instead of names

**What it does:**
1. Finds all Person nodes without embeddings
2. Creates text from properties: ชื่อ-นามสกุล, ตำแหน่ง, หน่วยงาน
3. Generates 384-dimensional vectors using HuggingFace model
4. Stores embeddings back in Neo4j

**Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (supports Thai)
""")

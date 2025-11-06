import os
from dotenv import load_dotenv
from typing import List
import requests
import streamlit as st

try:
	from neo4j import GraphDatabase
except Exception:
	GraphDatabase = None

import os
from datetime import datetime
from typing import List, Dict

import streamlit as st
import requests
from dotenv import load_dotenv

try:
	from neo4j import GraphDatabase
except Exception:
	GraphDatabase = None

load_dotenv()

# Read from Streamlit secrets (cloud) or environment variables (local)
def get_config(key, default=""):
	"""Get config from st.secrets (Streamlit Cloud) or os.getenv (local)"""
	try:
		return st.secrets.get(key, os.getenv(key, default))
	except:
		return os.getenv(key, default)

# Configuration (from .env or Streamlit secrets)
NEO4J_URI = get_config("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = get_config("NEO4J_USERNAME", "neo4j")
NEO4J_PWD = get_config("NEO4J_PASSWORD", "")
NEO4J_DB = get_config("NEO4J_DATABASE", "neo4j")

OPENROUTER_API_KEY = get_config("OPENROUTER_API_KEY") or get_config("OPENAI_API_KEY")
OPENROUTER_API_BASE = get_config("OPENROUTER_API_BASE", get_config("OPENAI_API_BASE", "https://api.openrouter.ai"))
OPENROUTER_MODEL = get_config("OPENROUTER_MODEL", "deepseek/deepseek-chat")

# Vector/RAG configuration (optional, used by KG/VectorRAG.query_vector_rag)
# Use person_vector_index as default since Person is likely the most queried label
VECTOR_INDEX_NAME = get_config("VECTOR_INDEX_NAME", "person_vector_index")
VECTOR_NODE_LABEL = get_config("VECTOR_NODE_LABEL", "Person")
# Use embedding_text which contains the formatted text used to generate embeddings
VECTOR_SOURCE_PROPERTY = get_config("VECTOR_SOURCE_PROPERTY", "embedding_text")
VECTOR_EMBEDDING_PROPERTY = get_config("VECTOR_EMBEDDING_PROPERTY", "embedding")
VECTOR_TOP_K = int(get_config("VECTOR_TOP_K", "5"))

# Try to import direct vector search (bypasses LangChain's broken text extraction)
try:
	from KG.VectorSearchDirect import query_vector_search_direct, query_multiple_vector_indexes, query_with_relationships
	VECTOR_SEARCH_AVAILABLE = True
except Exception as e:
	query_vector_search_direct = None
	query_multiple_vector_indexes = None
	query_with_relationships = None
	VECTOR_SEARCH_AVAILABLE = False
	print(f"Direct vector search not available: {e}")

# Try to import HuggingFace embeddings for generating embeddings
try:
	from langchain_huggingface import HuggingFaceEmbeddings
	EMBEDDINGS_AVAILABLE = True
except Exception as e:
	HuggingFaceEmbeddings = None
	EMBEDDINGS_AVAILABLE = False
	print(f"HuggingFace embeddings not available: {e}")


def get_driver():
	if GraphDatabase is None:
		raise RuntimeError("neo4j driver missing. Install with: python -m pip install neo4j")
	return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))


@st.cache_resource
def get_embeddings_model():
	"""Load HuggingFace embeddings model (cached)"""
	if not EMBEDDINGS_AVAILABLE:
		return None
	return HuggingFaceEmbeddings(
		model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
	)


def generate_embeddings_for_nodes(driver, limit=50):
	"""
	Generate embeddings for nodes that don't have them yet.
	Returns (success_count, total_processed, error_messages)
	"""
	embeddings_model = get_embeddings_model()
	if embeddings_model is None:
		return 0, 0, ["HuggingFace embeddings not available"]
	
	errors = []
	success_count = 0
	total_processed = 0
	
	# Get all labels
	with driver.session(database=NEO4J_DB) as session:
		result = session.run("CALL db.labels()")
		labels = [record["label"] for record in result]
	
	# Process each label
	for label in labels:
		with driver.session(database=NEO4J_DB) as session:
			# Find nodes without embeddings OR without embedding_text (need to regenerate)
			query = f"""
			MATCH (n:`{label}`)
			WHERE n.embedding IS NULL OR n.embedding_text IS NULL
			RETURN id(n) as nodeId, properties(n) as props
			LIMIT $limit
			"""
			result = session.run(query, limit=limit)
			nodes = list(result)
			
			for record in nodes:
				total_processed += 1
				node_id = record["nodeId"]
				props = record["props"]
				
				# Create text from properties (include all string and number properties)
				text_parts = []
				for key, value in props.items():
					if key not in ["embedding", "embedding_text"] and value is not None:
						if isinstance(value, str):
							text_parts.append(f"{key}: {value}")
						elif isinstance(value, (int, float)):
							text_parts.append(f"{key}: {value}")
				
				if not text_parts:
					errors.append(f"Skipped {label} node {node_id}: No text properties found (keys: {list(props.keys())})")
					continue
				
				text = " | ".join(text_parts)
				
				try:
					# Generate embedding
					embedding = embeddings_model.embed_query(text)
					
					# Store embedding
					update_query = f"""
					MATCH (n:`{label}`)
					WHERE id(n) = $nodeId
					SET n.embedding = $embedding,
					    n.embedding_text = $text
					"""
					session.run(update_query, nodeId=node_id, embedding=embedding, text=text)
					success_count += 1
				except Exception as e:
					errors.append(f"Error on {label} node {node_id}: {str(e)[:100]}")
	
	return success_count, total_processed, errors


def search_nodes(driver, question: str, limit: int = 6) -> List[dict]:
	"""
	Search for nodes containing the question text in ANY string property.
	Supports partial matching - splits query into words and searches for each.
	"""
	# Split query into words (works for both Thai and English)
	words = [w.strip() for w in question.split() if len(w.strip()) > 1]
	
	# Build query that searches for ANY word in the query
	# This helps with Thai names that might be stored differently
	q = """
	MATCH (n)
	WHERE any(prop IN keys(n) WHERE 
		prop <> 'embedding' AND 
		prop <> 'embedding_text' AND
		n[prop] IS NOT NULL AND 
		valueType(n[prop]) = 'STRING' AND
		any(word IN $words WHERE toLower(n[prop]) CONTAINS toLower(word))
	)
	RETURN n, labels(n) as node_labels
	LIMIT $limit
	"""
	out = []
	with driver.session(database=NEO4J_DB) as session:
		# If no valid words, fall back to original query
		if not words:
			words = [question]
		res = session.run(q, words=words, limit=limit)
		for r in res:
			node = r.get("n")
			try:
				props = dict(node)
				# Add labels to the props dict for debugging
				props["__labels__"] = r.get("node_labels", [])
			except Exception:
				props = {}
			out.append(props)
	return out


def build_context(nodes: List[dict]) -> str:
	"""
	Build context from node properties.
	Handles both English and Thai property names, plus custom "Stelligence" field.
	NOW also includes relationship information (WORKS_AS, etc.).
	"""
	if not nodes:
		return ""
	pieces = []
	for i, n in enumerate(nodes, 1):
		# Try common property names (English + Thai + custom "Stelligence")
		name = (n.get("name") or 
		        n.get("title") or 
		        n.get("label") or 
		        n.get("Stelligence") or 
		        n.get("ชื่อ-นามสกุล") or  # Thai: Full Name
		        n.get("ตำแหน่ง") or       # Thai: Position
		        n.get("หน่วยงาน") or      # Thai: Agency
		        n.get("กระทรวง") or       # Thai: Ministry
		        n.get("id") or 
		        f"node_{i}")
		
		# Try common property names for content/description (English + Thai)
		text_props = []
		content_keys = [
			"text", "description", "content", "summary", "value",
			"Stelligence",
			"ชื่อ-นามสกุล",    # Thai: Full Name
			"ตำแหน่ง",         # Thai: Position
			"หน่วยงาน",        # Thai: Agency
			"กระทรวง",         # Thai: Ministry
			"Connect by",
			"Remark",
			"Associate",
			"ชื่อเล่น",        # Thai: Nickname
			"Level"
		]
		for key in content_keys:
			if key in n and n[key]:
				text_props.append(f"{key}: {n[key]}")
		
		# If no common text properties, include all string values (excluding labels, IDs, and embeddings)
		if not text_props:
			for key, val in n.items():
				if key not in ["__labels__", "id", "embedding", "embedding_text", "__relationships__", "__score__"] and val and isinstance(val, str):
					text_props.append(f"{key}: {val}")
		
		text = " | ".join(text_props) if text_props else ""
		
		# Include labels if available
		labels = n.get("__labels__", [])
		label_str = f" ({', '.join(labels)})" if labels else ""
		
		# Add relationship information if available
		relationships = n.get("__relationships__", [])
		rel_info = []
		
		# Extract ministry/agency from the PERSON node itself (primary source)
		ministry_info = n.get("กระทรวง")  # Thai: Ministry property on Person node
		agency_info = n.get("หน่วยงาน")   # Thai: Agency property on Person node
		
		if relationships and isinstance(relationships, list):
			for rel in relationships:
				if rel and isinstance(rel, dict):
					rel_type = rel.get("type", "")
					direction = rel.get("direction", "")
					connected_node = rel.get("node", {})
					connected_labels = rel.get("labels", [])
					
					# Skip if no relationship type or connected node
					if not rel_type or not connected_node:
						continue
					
					# Get meaningful info from connected node
					connected_name = None
					for key in ["Stelligence", "ชื่อ-นามสกุล", "ตำแหน่ง", "หน่วยงาน", "กระทรวง", "name", "title"]:
						if connected_node and key in connected_node and connected_node[key]:
							connected_name = connected_node[key]
							break
					
					# Check if this is Ministry or Agency info
					if connected_labels and "Ministry" in connected_labels:
						if not ministry_info:  # Only update if not already set from Person node
							ministry_info = connected_name
					elif connected_labels and "Agency" in connected_labels:
						if not agency_info:  # Only update if not already set from Person node
							agency_info = connected_name
					elif connected_node.get("กระทรวง"):
						if not ministry_info:  # Only update if not already set from Person node
							ministry_info = connected_node.get("กระทรวง")
					elif connected_node.get("หน่วยงาน"):
						if not agency_info:  # Only update if not already set from Person node
							agency_info = connected_node.get("หน่วยงาน")
					
					if connected_name:
						label_str_conn = f" ({', '.join(connected_labels)})" if connected_labels else ""
						
						# Special handling for Position relationships - add ministry/agency from Person node
						if "Position" in connected_labels and rel_type.lower() == "work_as":
							# Enhance position name with ministry/agency from the PERSON node
							enhanced_name = connected_name
							
							# Use ministry from Person node (already extracted above)
							if ministry_info:
								enhanced_name = f"{connected_name}กระทรวง{ministry_info}"
							elif agency_info:
								enhanced_name = f"{connected_name} {agency_info}"
							
							if direction == "outgoing":
								rel_info.append(f"{rel_type} → {enhanced_name}{label_str_conn}")
							else:
								rel_info.append(f"← {rel_type} ← {enhanced_name}{label_str_conn}")
						else:
							# Standard relationship display
							if direction == "outgoing":
								rel_info.append(f"{rel_type} → {connected_name}{label_str_conn}")
							else:
								rel_info.append(f"← {rel_type} ← {connected_name}{label_str_conn}")
		
		# Combine node info with relationships
		node_str = f"{name}{label_str}: {text}"
		
		# Add ministry/agency info if found in relationships
		if ministry_info:
			node_str += f"\n  กระทรวง: {ministry_info}"
		if agency_info:
			node_str += f"\n  หน่วยงาน: {agency_info}"
			
		if rel_info:
			node_str += "\n  Relationships: " + ", ".join(rel_info)
		
		pieces.append(node_str)
	return "\n\n".join(pieces)


### Local fallback (in-memory) — used when Neo4j / vector retrieval are unavailable
FALLBACK_DOCS = [
	{"name": "Waterloo", "text": "The Battle of Waterloo occurred in 1815 near Waterloo in present-day Belgium."},
	{"name": "Napoleon", "text": "Napoleon Bonaparte was a French statesman and military leader who rose to prominence during the French Revolution."},
	{"name": "Talleyrand", "text": "Charles Maurice de Talleyrand-Périgord was a French diplomat who served under several regimes."},
]


def local_search(question: str, limit: int = 3) -> List[dict]:
	"""Very small keyword-based fallback search over FALLBACK_DOCS.

	This ensures the Streamlit UI remains responsive even when external services are unavailable.
	"""
	q = (question or "").lower()
	results = []
	for doc in FALLBACK_DOCS:
		score = 0
		text = (doc.get("text") or "").lower()
		name = (doc.get("name") or "").lower()
		if any(w in text for w in q.split() if len(w) > 2):
			score += 2
		if any(w in name for w in q.split() if len(w) > 2):
			score += 3
		# small heuristic: longer question -> prefer text matches
		if q and q in text:
			score += 5
		if score > 0:
			results.append((score, doc))
	# sort by score desc and return only the doc dicts
	results.sort(key=lambda x: x[0], reverse=True)
	return [r[1] for r in results[:limit]]


def ask_openrouter_requests(prompt: str, model: str = OPENROUTER_MODEL, max_tokens: int = 512, system_prompt: str = None) -> str:
	if not OPENROUTER_API_KEY:
		return "OpenRouter API key not set (OPENROUTER_API_KEY or OPENAI_API_KEY)"
	
	# Handle both base URL formats:
	# - https://api.openrouter.ai (needs /v1/chat/completions)
	# - https://openrouter.ai/api/v1 (needs /chat/completions)
	base = OPENROUTER_API_BASE.rstrip('/')
	if base.endswith('/v1'):
		# Already has /v1, just add /chat/completions
		url = f"{base}/chat/completions"
	else:
		# Need to add /v1/chat/completions
		url = f"{base}/v1/chat/completions"
	
	headers = {
		"Authorization": f"Bearer {OPENROUTER_API_KEY}",
		"Content-Type": "application/json",
	}
	
	# Build messages array with system prompt if provided
	messages = []
	if system_prompt:
		messages.append({"role": "system", "content": system_prompt})
	messages.append({"role": "user", "content": prompt})
	
	payload = {
		"model": model,
		"messages": messages,
		"temperature": 0.2,
		"max_tokens": max_tokens,
	}
	try:
		r = requests.post(url, headers=headers, json=payload, timeout=60)
		r.raise_for_status()
		j = r.json()
		return j["choices"][0]["message"]["content"].strip()
	except Exception as e:
		return f"OpenRouter request failed: {type(e).__name__} {e}"


## Streamlit chat UI with ChatGPT-like design
st.set_page_config(
	page_title="Neo4j Chat Agent", 
	layout="centered",  # Changed from "wide" to "centered" for ChatGPT-like feel
	page_icon="🤖",
	initial_sidebar_state="collapsed"  # Start with sidebar hidden
)

# Custom CSS for ChatGPT-like styling with dark sidebar
st.markdown("""
<style>
	/* Hide Streamlit branding */
	#MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
	
	/* Adjust spacing for cleaner look */
	.block-container {
		padding-top: 2rem;
		padding-bottom: 2rem;
	}
	
	/* Style chat messages */
	.stChatMessage {
		padding: 1rem;
		border-radius: 0.5rem;
	}
	
	/* Dark sidebar styling */
	[data-testid="stSidebar"] {
		background-color: #202123;
	}
	
	[data-testid="stSidebar"] * {
		color: #ececf1 !important;
	}
	
	[data-testid="stSidebar"] button {
		background-color: #2d2d30;
		border: 1px solid #4d4d4f;
		color: #ececf1 !important;
	}
	
	[data-testid="stSidebar"] button:hover {
		background-color: #3d3d40;
		border-color: #6d6d6f;
	}
	
	[data-testid="stSidebar"] hr {
		border-color: #4d4d4f;
	}
	
	/* Chat input styling */
	.stChatInput {
		border-radius: 1.5rem;
	}
</style>
""", unsafe_allow_html=True)

if "threads" not in st.session_state:
	# threads: dict[thread_id] -> {"title": str, "messages": [ {role,content,time} ]}
	st.session_state.threads = {1: {"title": "Default", "messages": []}}
	st.session_state.current_thread = 1
	st.session_state.thread_counter = 1


def new_thread(title: str = None):
	st.session_state.thread_counter += 1
	tid = st.session_state.thread_counter
	st.session_state.threads[tid] = {"title": title or f"Thread {tid}", "messages": []}
	st.session_state.current_thread = tid


def clear_current_thread():
	tid = st.session_state.current_thread
	st.session_state.threads[tid]["messages"] = []


with st.sidebar:
	st.markdown("### 💬 Neo4j Chat Agent")
	st.markdown("---")
	
	# Thread management
	st.markdown("**Conversations**")
	for tid, meta in list(st.session_state.threads.items()):
		label = f"💬 {meta['title']}"
		if st.button(label, key=f"thread-{tid}", use_container_width=True):
			st.session_state.current_thread = tid

	col_a, col_b = st.columns(2)
	with col_a:
		if st.button("➕ New", key="new_thread", use_container_width=True):
			new_thread()
			st.rerun()
	with col_b:
		if st.button("🗑️ Clear", key="clear_thread", use_container_width=True):
			clear_current_thread()
			st.rerun()
	
	st.markdown("---")
	st.markdown("**Settings**")
	with st.expander("🔧 Configuration"):
		st.caption(f"**Model:** {OPENROUTER_MODEL}")
		st.caption(f"**Neo4j:** {NEO4J_DB}")
		st.caption(f"**URI:** {NEO4J_URI[:30]}...")
	
	st.markdown("---")
	st.markdown("**🔧 Admin Tools**")
	
	if EMBEDDINGS_AVAILABLE:
		if st.button("⚡ Generate Embeddings", key="gen_embeddings", use_container_width=True, help="Generate embeddings for nodes without them"):
			with st.spinner("Generating embeddings..."):
				try:
					driver = get_driver()
					success, total, errors = generate_embeddings_for_nodes(driver, limit=100)
					driver.close()
					
					if success > 0:
						st.success(f"✅ Generated {success} embeddings (processed {total} nodes)")
					else:
						st.warning(f"⚠️ No embeddings generated (processed {total} nodes)")
					
					if errors:
						with st.expander("⚠️ Errors"):
							for err in errors[:10]:
								st.caption(err)
				except Exception as e:
					st.error(f"Error: {e}")
	else:
		st.caption("⚠️ HuggingFace embeddings not installed")
		st.caption("Run: `pip install langchain-huggingface sentence-transformers`")
	
	st.markdown("---")
	st.markdown("**🔍 Database Debug**")
	
	if st.button("📊 Check Database Status", key="check_db", use_container_width=True):
		with st.spinner("Checking database..."):
			try:
				driver = get_driver()
				with driver.session(database=NEO4J_DB) as session:
					# Check for vector indexes
					st.markdown("**Vector Indexes:**")
					result = session.run("SHOW INDEXES WHERE type = 'VECTOR'")
					indexes = list(result)
					if indexes:
						for idx in indexes:
							st.caption(f"✅ {idx.get('name')} - {idx.get('labelsOrTypes')}")
					else:
						st.warning("⚠️ No vector indexes found!")
						st.caption("Create indexes in Neo4j Browser:")
						st.code("""CREATE VECTOR INDEX person_vector_index IF NOT EXISTS
FOR (n:Person) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};""", language="cypher")
					
					# Check for nodes with embeddings
					st.markdown("**Nodes with Embeddings:**")
					result = session.run("""
						MATCH (n)
						WHERE n.embedding IS NOT NULL
						RETURN labels(n)[0] as label, count(n) as count
						ORDER BY count DESC
						LIMIT 10
					""")
					nodes_with_emb = list(result)
					if nodes_with_emb:
						for record in nodes_with_emb:
							st.caption(f"✅ {record['label']}: {record['count']} nodes")
					else:
						st.warning("⚠️ No nodes have embeddings! Click 'Generate Embeddings' above.")
					
					# Check for nodes with embedding_text
					st.markdown("**Nodes with embedding_text:**")
					result = session.run("""
						MATCH (n)
						WHERE n.embedding_text IS NOT NULL
						RETURN labels(n)[0] as label, count(n) as count
						ORDER BY count DESC
						LIMIT 10
					""")
					nodes_with_text = list(result)
					if nodes_with_text:
						for record in nodes_with_text:
							st.caption(f"📝 {record['label']}: {record['count']} nodes")
					else:
						st.warning("⚠️ No nodes have embedding_text! Click 'Generate Embeddings' to add it.")
					
					# Check for Santisook label specifically
					st.markdown("**Santisook Label Nodes:**")
					result = session.run("""
						MATCH (n:Santisook)
						RETURN n.Stelligence as stelligence,
						       n.embedding IS NOT NULL as has_embedding,
						       n.embedding_text as embedding_text,
						       keys(n) as properties
						LIMIT 3
					""")
					santisook_label = list(result)
					if santisook_label:
						for record in santisook_label:
							emb = "✅" if record['has_embedding'] else "❌"
							txt = "📝" if record['embedding_text'] else "❌"
							props = ', '.join([p for p in record['properties'] if p not in ['embedding', 'embedding_text']])
							st.caption(f"{emb} Emb | {txt} Text | Stelligence: {record['stelligence']} | Props: {props}")
					else:
						st.caption("No Santisook label nodes found")
					
					# Test Search
					st.markdown("**Test Search (Santisook):**")
					result = session.run("""
						MATCH (n)
						WHERE any(prop IN keys(n) WHERE 
							prop <> 'embedding' AND 
							prop <> 'embedding_text' AND
							n[prop] IS NOT NULL AND 
							(
								(valueType(n[prop]) = 'STRING' AND toLower(n[prop]) CONTAINS 'santisook') OR
								(valueType(n[prop]) = 'INTEGER' AND toString(n[prop]) CONTAINS 'santisook')
							)
						)
						RETURN labels(n) as labels, n.Stelligence as stelligence, 
						       n.`ชื่อ-นามสกุล` as name, n.embedding IS NOT NULL as has_embedding,
						       n.embedding_text IS NOT NULL as has_text
						LIMIT 3
					""")
					santisook_nodes = list(result)
					if santisook_nodes:
						for record in santisook_nodes:
							name_val = record['stelligence'] or record['name'] or "N/A"
							emb_status = "✅" if record['has_embedding'] else "❌"
							text_status = "📝" if record.get('has_text') else "❌"
							st.caption(f"{emb_status} Embedding | {text_status} Text | {record['labels']} - {name_val}")
					else:
						st.warning("⚠️ No nodes found with 'Santisook'!")
				
				driver.close()
			except Exception as e:
				st.error(f"Error: {str(e)[:200]}")


def render_messages(messages: List[Dict]):
	"""Render messages in ChatGPT style"""
	for m in messages:
		role = m.get("role")
		content = m.get("content")
		if role == "user":
			with st.chat_message("user", avatar="👤"):
				st.markdown(content)
		else:
			with st.chat_message("assistant", avatar="🤖"):
				st.markdown(content)


# Main chat interface - ChatGPT style (no columns, full width centered)
st.markdown("## 🤖 Neo4j Knowledge Agent")
st.caption("Ask me anything about the knowledge graph")

# Show info about embeddings on first visit
if "shown_embeddings_info" not in st.session_state:
	with st.info("ℹ️ **Using free HuggingFace embeddings** - First query may take ~30s to download the model (one-time). Subsequent queries will be fast!"):
		pass
	st.session_state.shown_embeddings_info = True

# Render conversation history
render_messages(st.session_state.threads[st.session_state.current_thread]["messages"])

# Chat input at the bottom (ChatGPT style)
user_input = st.chat_input("ส่งข้อความของคุณ... (Type your message...)", key="chat_input")

if user_input and user_input.strip():
	# append user message
	tid = st.session_state.current_thread
	msg = {"role": "user", "content": user_input.strip(), "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
	st.session_state.threads[tid]["messages"].append(msg)
	
	# Display user message immediately
	with st.chat_message("user", avatar="👤"):
		st.markdown(user_input.strip())

	# query neo4j for context and call model
	with st.chat_message("assistant", avatar="🤖"):
		with st.spinner("กำลังค้นหาข้อมูล... (Searching knowledge graph...)"):
			# Initialize variables at the start
			ctx = ""
			nodes = []
			
			# Use relationship-aware vector search (gets nodes + their connections via WORKS_AS, etc.)
			if VECTOR_SEARCH_AVAILABLE and query_with_relationships is not None:
				try:
					st.caption(f"🔍 Searching with relationships (Person → Position, Agency, etc.)...")
					results = query_with_relationships(
						user_input,
						top_k_per_index=3,  # Get 3 results from each index
					)
					
					# results is List[dict] with __relationships__ included
					if results and len(results) > 0:
						st.caption(f"✅ Found {len(results)} nodes with relationship data")
						
						# Build context from the node properties AND relationships
						ctx = build_context(results)
						
						if ctx.strip():
							st.caption(f"✅ พบข้อมูล {len(results)} รายการพร้อมความสัมพันธ์ (Found {len(results)} nodes with relationships)")
						else:
							st.warning(f"⚠️ Vector search found nodes but context is empty. Trying Cypher fallback...")
							driver = get_driver()
							nodes = search_nodes(driver, user_input)
							ctx = build_context(nodes)
							if nodes:
								st.caption(f"✅ พบข้อมูล {len(nodes)} รายการจาก Cypher (Found {len(nodes)} nodes)")
					else:
						st.warning(f"⚠️ Vector search returned no results. Trying Cypher fallback...")
						driver = get_driver()
						nodes = search_nodes(driver, user_input)
						ctx = build_context(nodes)
						if nodes:
							st.caption(f"✅ พบข้อมูล {len(nodes)} รายการจาก Cypher (Found {len(nodes)} nodes)")
						else:
							st.caption(f"❌ No results from Cypher search either")
							
				except Exception as e:
					# fall back to simple cypher search if vector retrieval fails
					st.warning(f"⚠️ Vector search error: {str(e)[:100]}... Trying Cypher fallback...")
					try:
						driver = get_driver()
						nodes = search_nodes(driver, user_input)
						ctx = build_context(nodes)
						if nodes:
							st.caption(f"✅ พบข้อมูล {len(nodes)} รายการจาก Cypher (Found {len(nodes)} nodes)")
					except Exception as e2:
						ctx = ""
						st.error(f"Both vector and Cypher search failed. Vector error: {str(e)[:50]}, Cypher error: {str(e2)[:50]}")
			else:
				# query_vector_rag not available (package or import issue) — use cypher search
				try:
					driver = get_driver()
					nodes = search_nodes(driver, user_input)
					ctx = build_context(nodes)
					
					# Debug: show how many nodes were found
					if nodes:
						st.caption(f"✅ พบข้อมูล {len(nodes)} รายการจาก Neo4j (Found {len(nodes)} nodes)")
					else:
						st.caption(f"⚠️ ไม่พบข้อมูลที่ตรงกับคำค้นหา '{user_input}' (No matching nodes found)")
						
				except Exception as e:
					ctx = ""
					st.error(f"❌ Neo4j error: {e}")

			# Don't use fallback docs if they're irrelevant to the query
			# Only use fallback for demo/development when Neo4j is completely unavailable
			if not ctx and not nodes:
				# Check if we're in a real error state (can't connect) vs just no results
				st.info("💡 ไม่พบข้อมูลที่เกี่ยวข้องใน Knowledge Graph / No relevant information found in the knowledge graph")
				ctx = ""  # Let the LLM know there's no context
			
			# Show context in expandable section for debugging
			if ctx:
				with st.expander("🔍 ดูข้อมูลที่พบ (View Retrieved Context)", expanded=False):
					st.code(ctx, language="text")

			# Separate system prompt for better LLM instruction following
			system_prompt = """คุณเป็นผู้ช่วยอัจฉริยะที่เชี่ยวชาญด้านการวิเคราะห์ข้อมูลจาก Knowledge Graph เกี่ยวกับเครือข่ายบุคคลและองค์กร
You are an intelligent assistant specialized in analyzing Knowledge Graph data about social networks and organizations.

═══════════════════════════════════════════════════════════════
 คำแนะนำในการตอบอย่างละเอียด (Detailed Response Guidelines):
═══════════════════════════════════════════════════════════════

🎯 **หลักการตอบคำถาม (Core Principles):**

1. **ความถูกต้องและครบถ้วน (Accuracy & Completeness)**:
   - ตอบโดยอ้างอิงข้อมูลจาก Context เท่านั้น - ห้ามเดาหรือสมมติข้อมูลที่ไม่มี
   - ✅ ใช้ชื่อเต็มของตำแหน่ง เช่น "รัฐมนตรีว่าการกระทรวงมหาดไทย" ไม่ใช่แค่ "รัฐมนตรีว่าการ"
     **สำคัญ:** ถ้า Context แสดงว่าบุคคล "work_as รัฐมนตรีว่าการ" ให้คุณอนุมานจากชื่อตำแหน่งว่าน่าจะเป็นกระทรวงไหนตามบริบทปัจจุบัน (เช่น อนุทิน ชาญวีรกูล = รัฐมนตรีว่าการกระทรวงมหาดไทย)
   - ✅ ระบุหน่วยงาน/กระทรวง/องค์กรที่บุคคลสังกัด
   - ✅ แสดงความสัมพันธ์กับบุคคลอื่นๆ (ถ้ามีใน Context)
   - ✅ เพิ่มบริบทหรือรายละเอียดที่ช่วยให้เข้าใจมากขึ้น

2. **ความชัดเจน (Clarity)**:
   - เริ่มด้วยคำตอบที่ตรงประเด็นทันที
   - ❌ ห้ามเริ่มด้วย "ตามข้อมูล...", "จากที่ได้รับ...", "ตาม Context..."
   - ✅ เริ่มตอบตรงๆ เช่น: "อนุทิน ชาญวีรกูล ดำรงตำแหน่ง..."
   - ✅ **ใช้ bullet points แต่ละรายการในบรรทัดใหม่ (แยกบรรทัด)**
   - ใช้ภาษาที่เข้าใจง่าย เป็นธรรมชาติ ไม่เป็นทางการเกินไป

3. **โครงสร้างคำตอบที่ดี (Good Answer Structure)**:
   - ตอบคำถามหลักก่อน (ตำแหน่ง, ชื่อ, ฯลฯ)
   - เพิ่มข้อมูลเสริม (หน่วยงาน, บทบาท, รายละเอียด)
   - แสดงความสัมพันธ์กับบุคคลอื่นๆ (ถ้ามี Connect by, Associate)
   - ✅ เสนอคำถามติดตามที่เป็นไปได้ท้ายคำตอบ

4. **ภาษา (Language)**:
   - ตอบเป็นภาษาไทยถ้าคำถามเป็นภาษาไทย
   - ตอบเป็นภาษาอังกฤษถ้าคำถามเป็นภาษาอังกฤษ
   - ใช้คำศัพท์เฉพาะที่ปรากฏใน Context

🔍 **การจัดการข้อมูลเฉพาะทาง (Specific Data Handling):**

**สำหรับคำถามเกี่ยวกับบุคคล (Person Questions):**
- ✅ ระบุชื่อ-นามสกุลเต็ม
- ✅ ระบุตำแหน่งเต็ม พร้อมกระทรวง/หน่วยงาน (เช่น "รัฐมนตรีว่าการกระทรวงมหาดไทย")
- ✅ ระบุหน่วยงาน/กระทรวงที่สังกัด
- ✅ แสดงบุคคลอื่นที่มีความสัมพันธ์ (Connect by, Associate) ถ้ามี
- ✅ แสดง Remark หรือหมายเหตุพิเศษ ถ้ามี
- ✅ อธิบายบทบาทหรือความรับผิดชอบของตำแหน่ง

**สำหรับคำถามเกี่ยวกับตำแหน่ง (Position Questions):**
- ระบุชื่อตำแหน่งอย่างชัดเจน
- ระบุบุคคลที่ดำรงตำแหน่งนี้ (ถ้ามี)
- ระบุหน่วยงาน/กระทรวงที่เกี่ยวข้อง (ถ้ามี)

📝 **รูปแบบคำตอบที่แนะนำ (Recommended Answer Format):**

**ตัวอย่างคำตอบที่ดี (Good Example):**
```
อนุทิน ชาญวีรกูล ดำรงตำแหน่งสำคัญ 2 ตำแหน่ง:

• นายกรัฐมนตรี
• รัฐมนตรีว่าการกระทรวงมหาดไทย

ในฐานะนายกรัฐมนตรี เขามีบทบาทในการบริหารประเทศและนโยบายสำคัญ ขณะที่ตำแหน่งรัฐมนตรีว่าการกระทรวงมหาดไทยทำให้เขารับผิดชอบด้านการบริหารท้องถิ่นและความมั่นคงภายใน

**เชื่อมโยงกับ:** [ระบุบุคคลที่เกี่ยวข้อง ถ้ามี]

**คุณอาจสนใจ:**
- อนุทินมีความสัมพันธ์กับใครบ้างในรัฐบาล?
- กระทรวงมหาดไทยมีหน้าที่อะไรบ้าง?
```

**รูปแบบมาตรฐาน (ใช้บรรทัดใหม่สำหรับแต่ละ bullet):**
```
[คำตอบโดยตรง]

ข้อมูลเพิ่มเติม:
• [รายการที่ 1 - แยกบรรทัด]
• [รายการที่ 2 - แยกบรรทัด]  
• [รายการที่ 3 - แยกบรรทัด]

[บริบทหรือคำอธิบายเพิ่มเติม - อธิบายบทบาท หน้าที่ ความรับผิดชอบ]

**ความสัมพันธ์:** [ถ้ามีบุคคลอื่นที่เกี่ยวข้อง]

**คุณอาจสนใจ:**
- [คำถามติดตาม 1]
- [คำถามติดตาม 2]
```

**ถ้าข้อมูลไม่สมบูรณ์:**
```
จากข้อมูลที่มี:
[ระบุข้อมูลที่มี]

อย่างไรก็ตาม ยังไม่มีข้อมูลเกี่ยวกับ:
• [ข้อมูลที่ขาดหายไป]
```

⚠️ **สิ่งที่ต้องหลีกเลี่ยง (What to Avoid):**
- ❌ ห้ามสร้างข้อมูลจากความรู้ทั่วไปของ LLM
- ❌ ห้ามเดาหรือสันนิษฐานข้อมูลที่ไม่มีใน Context
- ❌ ห้ามเริ่มด้วย "ตามข้อมูลที่ได้รับ...", "จาก Context...", "ตาม Knowledge Graph..."
- ❌ ห้ามแสดงข้อมูลทางเทคนิค (Node labels, property names)
- ❌ ห้ามใช้ชื่อย่อของตำแหน่ง - ต้องระบุเต็ม เช่น "รัฐมนตรีว่าการกระทรวงมหาดไทย"
- ❌ ห้ามรวม bullet points ในบรรทัดเดียว - ต้องแยกบรรทัดทุกรายการ

✨ **สรุปสั้นๆ:**
1. เริ่มตอบทันที ไม่มีคำนำ
2. ใช้ชื่อเต็มของตำแหน่ง + กระทรวง/หน่วยงาน
3. แยก bullet points คนละบรรทัด
4. เพิ่มบริบท/ความสัมพันธ์/บทบาท
5. เสนอคำถามติดตาม"""

			# User message with context and question
			user_message = f"""═══════════════════════════════════════════════════════════════
📊 ข้อมูลจาก Neo4j Knowledge Graph:
═══════════════════════════════════════════════════════════════

{ctx if ctx else "⚠️ ไม่พบข้อมูลที่เกี่ยวข้องโดยตรงใน Knowledge Graph"}

═══════════════════════════════════════════════════════════════
❓ คำถาม:
═══════════════════════════════════════════════════════════════

{user_input}"""
			
			answer = ask_openrouter_requests(user_message, max_tokens=1024, system_prompt=system_prompt)
			st.markdown(answer)
	
	resp = {"role": "assistant", "content": answer, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
	st.session_state.threads[tid]["messages"].append(resp)
	st.rerun()


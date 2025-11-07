"""
STelligence Network Agent - Neo4j Knowledge Graph Q&A System
Version: 1.1.0 - Bullet formatting fix applied
Last Updated: 2025-11-07
"""

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

# Vector search configuration
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
		# Use a list to track if person is connected to MULTIPLE ministries
		ministry_info = n.get("กระทรวง") if n.get("กระทรวง") else None  # From Person property
		agency_info = n.get("หน่วยงาน") if n.get("หน่วยงาน") else None    # From Person property
		ministry_from_relationship = None  # Track ministry from graph relationship
		person_ministry_list = []  # For Position nodes: track which Person works in which Ministry
		
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
					
					# Handle special 2-hop relationship for Position nodes
					if rel_type == "person_ministry":
						person_name = connected_node.get("ชื่อ-นามสกุล") or connected_node.get("Stelligence")
						person_ministry = connected_node.get("ministry")
						if person_name and person_ministry:
							person_ministry_list.append(f"{person_name} ({person_ministry})")
						continue  # Don't process this as a normal relationship
					
					# Get meaningful info from connected node
					connected_name = None
					for key in ["Stelligence", "ชื่อ-นามสกุล", "ตำแหน่ง", "หน่วยงาน", "กระทรวง", "name", "title"]:
						if connected_node and key in connected_node and connected_node[key]:
							connected_name = connected_node[key]
							break
					
					# Check if this is Ministry or Agency info from relationship
					if connected_labels and "Ministry" in connected_labels:
						# Store ministry from relationship - will override property if needed
						ministry_from_relationship = connected_name
						if not ministry_info:  # Use relationship if no property exists
							ministry_info = connected_name
					elif connected_labels and "Agency" in connected_labels:
						if not agency_info:
							agency_info = connected_name
					elif connected_node.get("กระทรวง"):
						if not ministry_from_relationship:  # Prefer actual Ministry node
							ministry_from_relationship = connected_node.get("กระทรวง")
						if not ministry_info:
							ministry_info = connected_node.get("กระทรวง")
					elif connected_node.get("หน่วยงาน"):
						if not agency_info:
							agency_info = connected_node.get("หน่วยงาน")
					
					if connected_name:
						label_str_conn = f" ({', '.join(connected_labels)})" if connected_labels else ""
						
						# Special handling for Position relationships - add ministry/agency from Person node
						if "Position" in connected_labels and rel_type.lower() == "work_as":
							# Enhance position name with ministry/agency from the PERSON node
							enhanced_name = connected_name
							
							# Use ministry from Person node or relationship (already extracted above)
							if ministry_from_relationship:
								enhanced_name = f"{connected_name}กระทรวง{ministry_from_relationship}"
							elif ministry_info:
								enhanced_name = f"{connected_name}กระทรวง{ministry_info}"
							elif agency_info:
								enhanced_name = f"{connected_name} {agency_info}"
							
							if direction == "outgoing":
								rel_info.append(f"{rel_type} → {enhanced_name}{label_str_conn}")
							else:
								rel_info.append(f"← {rel_type} ← {enhanced_name}{label_str_conn}")
						# Always show Ministry relationships explicitly
						elif "Ministry" in connected_labels:
							if direction == "outgoing":
								rel_info.append(f"🏛️ {rel_type} → {connected_name}{label_str_conn}")
							else:
								rel_info.append(f"🏛️ ← {rel_type} ← {connected_name}{label_str_conn}")
						else:
							# Standard relationship display
							if direction == "outgoing":
								rel_info.append(f"{rel_type} → {connected_name}{label_str_conn}")
							else:
								rel_info.append(f"← {rel_type} ← {connected_name}{label_str_conn}")
		
		# Combine node info with relationships
		node_str = f"{name}{label_str}: {text}"
		
		# Add ministry/agency info - prefer relationship over property
		display_ministry = ministry_from_relationship if ministry_from_relationship else ministry_info
		if display_ministry:
			node_str += f"\n  กระทรวง: {display_ministry}"
		if agency_info:
			node_str += f"\n  หน่วยงาน: {agency_info}"
		
		# For Position nodes: show which Person holds this position and their ministry
		if person_ministry_list:
			node_str += f"\n  👥 ดำรงตำแหน่งโดย: " + ", ".join(person_ministry_list)
			
		if rel_info:
			node_str += "\n  Relationships: " + ", ".join(rel_info)
		
		pieces.append(node_str)
	
	# Post-process: Add Stelligence network summary at the top if present
	stelligence_networks = {}
	for n in nodes:
		stelligence = n.get("Stelligence")
		if stelligence and stelligence in ["Santisook", "Por", "Knot"]:
			person_name = n.get("ชื่อ-นามสกุล") or n.get("name") or "Unknown"
			if stelligence not in stelligence_networks:
				stelligence_networks[stelligence] = []
			stelligence_networks[stelligence].append(person_name)
	
	# Add summary if networks found
	if stelligence_networks:
		summary_parts = ["🌐 เครือข่าย Stelligence Networks:"]
		for network_name, members in stelligence_networks.items():
			summary_parts.append(f"\n  📍 {network_name} Network: {len(members)} คน")
			summary_parts.append(f"     → {', '.join(members[:10])}")  # Show first 10
			if len(members) > 10:
				summary_parts.append(f"     → และอีก {len(members) - 10} คน...")
		summary = "\n".join(summary_parts)
		return summary + "\n\n" + "\n\n".join(pieces)
	
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


def fix_bullet_formatting(text: str) -> str:
	"""
	Fix bullet point formatting to ensure each bullet is on a new line.
	Converts inline bullets like '• item1 • item2 • item3' to separate lines.
	"""
	import re
	
	# Split by lines to process each line
	lines = text.split('\n')
	fixed_lines = []
	
	for line in lines:
		# Count bullets in this line
		bullet_count = line.count('•')
		
		if bullet_count > 1:
			# Multiple bullets on same line - split them
			# Replace ' •' with '\n•' but not the first bullet
			parts = line.split('•')
			if len(parts) > 1:
				# First part before first bullet
				result = parts[0]
				# Add each bullet on new line
				for i, part in enumerate(parts[1:], 1):
					if part.strip():  # Only add if not empty
						result += '•' + part.rstrip()
						if i < len(parts) - 1:  # Not the last one
							result += '\n'
				fixed_lines.append(result)
			else:
				fixed_lines.append(line)
		else:
			# Single bullet or no bullet - keep as is
			fixed_lines.append(line)
	
	return '\n'.join(fixed_lines)


## Streamlit chat UI with modern ChatGPT-like design
st.set_page_config(
	page_title="STelligence Network Agent", 
	layout="centered",
	page_icon="💬",
	initial_sidebar_state="expanded"
)

# Force show sidebar on mobile/small screens
st.markdown("""
<style>
	[data-testid="stSidebar"][aria-expanded="false"] {
		display: block !important;
		margin-left: 0px !important;
	}
	[data-testid="stSidebar"] {
		display: block !important;
	}
	/* Force sidebar to always show */
	section[data-testid="stSidebar"] {
		display: block !important;
		width: 21rem !important;
	}
</style>
<script>
	// Force expand sidebar on load
	window.addEventListener('load', function() {
		const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
		if (sidebar) {
			sidebar.setAttribute('aria-expanded', 'true');
		}
	});
</script>
""", unsafe_allow_html=True)

# Custom CSS for ChatGPT-like styling
def apply_custom_css():
	# Updated colors
	sidebar_bg = "#181818"  # Darker sidebar
	main_bg = "#212121"     # Chat interface background
	text_color = "#ececec"
	border_color = "#3d4451"
	input_bg = "#2f2f2f"
	user_msg_bg = "#2f2f2f"
	assistant_msg_bg = "#1f2937"
	hover_bg = "#374151"
	button_color = "#565869"
	
	st.markdown(f"""
	<style>
		/* Main container and all backgrounds */
		.stApp {{
			background-color: {main_bg} !important;
		}}
		
		/* Header area */
		header {{
			background-color: {main_bg} !important;
		}}
		
		/* Main content area */
		.main {{
			background-color: {main_bg} !important;
		}}
		
		/* Block container */
		.block-container {{
			background-color: {main_bg} !important;
		}}
		
		/* Sidebar styling */
		[data-testid="stSidebar"] {{
			background-color: {sidebar_bg} !important;
			border-right: 1px solid {border_color};
		}}
		
		[data-testid="stSidebar"] .stButton button {{
			width: 100%;
			background-color: transparent;
			color: {text_color};
			border: 1px solid {border_color};
			border-radius: 0.5rem;
			padding: 0.5rem 1rem;
			margin-bottom: 0.5rem;
		}}
		
		[data-testid="stSidebar"] .stButton button:hover {{
			background-color: {hover_bg};
		}}
		
		/* Text color */
		.stMarkdown, p, span, div {{
			color: {text_color};
		}}
		
		/* Left-right chat layout */
		/* Assistant messages on the left */
		[data-testid="stChatMessage"][data-testid*="assistant"] {{
			justify-content: flex-start;
			max-width: 70%;
			margin-right: auto;
			margin-left: 0;
		}}
		
		/* User messages on the right */
		[data-testid="stChatMessage"][data-testid*="user"] {{
			justify-content: flex-end;
			max-width: 70%;
			margin-left: auto;
			margin-right: 0;
		}}
		
		/* Chat message styling */
		[data-testid="stChatMessageContent"] {{
			border-radius: 1rem;
			padding: 1rem;
		}}
		
		/* Assistant message background */
		.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {{
			background-color: {assistant_msg_bg};
			border: 1px solid {border_color};
		}}
		
		/* User message background */
		.stChatMessage:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {{
			background-color: {user_msg_bg};
			border: 1px solid {border_color};
		}}
		
		/* Action buttons styling */
		.action-buttons {{
			display: flex;
			gap: 0.5rem;
			margin-top: 0.5rem;
			opacity: 0.7;
		}}
		
		.action-buttons:hover {{
			opacity: 1;
		}}
		
		.action-btn {{
			background-color: transparent;
			border: 1px solid {button_color};
			color: {text_color};
			padding: 0.25rem 0.75rem;
			border-radius: 0.5rem;
			font-size: 0.8rem;
			cursor: pointer;
			transition: all 0.2s;
		}}
		
		.action-btn:hover {{
			background-color: {button_color};
		}}
		
		/* Input area */
		.stChatInput {{
			border: 1px solid {border_color};
			border-radius: 1rem;
			background-color: #303030 !important;
		}}
		
		/* Input text color */
		.stChatInput input {{
			background-color: #303030 !important;
			color: {text_color};
		}}
		
		/* Bottom area (stBottom) */
		[data-testid="stBottom"] {{
			background-color: {main_bg} !important;
		}}
		
		/* Headers */
		h1, h2, h3 {{
			color: {text_color};
		}}
		
		/* Welcome card styling */
		.welcome-container {{
			background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
			padding: 3rem;
			border-radius: 1rem;
			text-align: center;
			margin: 2rem auto;
			max-width: 800px;
		}}
		
		.welcome-title {{
			font-size: 2.5rem;
			font-weight: 700;
			color: white;
			margin-bottom: 1rem;
		}}
		
		.welcome-subtitle {{
			font-size: 1.25rem;
			color: rgba(255, 255, 255, 0.9);
		}}
		
		.feature-card {{
			background-color: {input_bg};
			padding: 1.5rem;
			border-radius: 0.75rem;
			border: 1px solid {border_color};
			margin: 0.5rem;
			text-align: center;
			transition: all 0.3s;
		}}
		
		.feature-card:hover {{
			background-color: {hover_bg};
			transform: translateY(-2px);
		}}
		
		/* Button styling */
		.stButton button {{
			border-radius: 0.5rem;
			border: 1px solid {border_color};
			background-color: {input_bg};
			color: {text_color};
		}}
		
		.stButton button:hover {{
			background-color: {hover_bg};
		}}
		
		/* Hide Streamlit branding but keep hamburger menu */
		footer {{visibility: hidden;}}
		
		/* Divider */
		hr {{
			border-color: {border_color};
		}}
	</style>
	""", unsafe_allow_html=True)

apply_custom_css()

if "threads" not in st.session_state:
	# threads: dict[thread_id] -> {"title": str, "messages": [ {role,content,time} ], "created_at": str}
	st.session_state.threads = {
		1: {
			"title": "New Chat",
			"messages": [],
			"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		}
	}
	st.session_state.current_thread = 1
	st.session_state.thread_counter = 1

# Initialize edit mode state
if "edit_mode" not in st.session_state:
	st.session_state.edit_mode = None  # Will store (thread_id, message_index) when editing
if "edit_text" not in st.session_state:
	st.session_state.edit_text = ""


def new_thread(title: str = None):
	"""Create a new conversation thread"""
	st.session_state.thread_counter += 1
	tid = st.session_state.thread_counter
	st.session_state.threads[tid] = {
		"title": title or "New Chat",
		"messages": [],
		"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	}
	st.session_state.current_thread = tid


def delete_thread(tid: int):
	"""Delete a conversation thread"""
	if tid in st.session_state.threads and len(st.session_state.threads) > 1:
		del st.session_state.threads[tid]
		# Switch to another thread
		st.session_state.current_thread = list(st.session_state.threads.keys())[0]


def update_thread_title(tid: int, first_message: str):
	"""Update thread title based on first message"""
	# Use first 30 characters of first message as title
	if first_message:
		title = first_message[:50] + ("..." if len(first_message) > 50 else "")
		st.session_state.threads[tid]["title"] = title


def clear_current_thread():
	"""Clear messages in current thread"""
	tid = st.session_state.current_thread
	st.session_state.threads[tid]["messages"] = []
	st.session_state.threads[tid]["title"] = "New Chat"


with st.sidebar:
	st.title("STelligence")
	st.write("Network Agent")
	
	# New Chat button (prominent)
	if st.button("+ New Chat", key="new_chat", use_container_width=True):
		new_thread()
		st.rerun()
	
	st.divider()
	
	# Conversation History
	st.subheader("Chat History")
	
	# Sort threads by created_at (most recent first)
	sorted_threads = sorted(
		st.session_state.threads.items(),
		key=lambda x: x[1].get("created_at", ""),
		reverse=True
	)
	
	for tid, meta in sorted_threads:
		col1, col2 = st.columns([4, 1])
		
		with col1:
			# Highlight active thread
			thread_label = meta['title']
			
			if st.button(
				thread_label,
				key=f"thread-{tid}",
				use_container_width=True,
				disabled=tid == st.session_state.current_thread
			):
				st.session_state.current_thread = tid
				st.rerun()
		
		with col2:
			# Delete button (only show if not current thread and more than 1 thread exists)
			if len(st.session_state.threads) > 1:
				if st.button("X", key=f"del-{tid}", help="Delete conversation"):
					delete_thread(tid)
					st.rerun()
	
	st.divider()
	
	# Settings at bottom
	with st.expander("Settings"):
		st.caption(f"Model: {OPENROUTER_MODEL}")
		st.caption(f"Database: {NEO4J_DB}")


def render_messages_with_actions(messages: List[Dict], thread_id: int):
	"""Render messages with edit and regenerate buttons for assistant messages"""
	for idx, m in enumerate(messages):
		role = m.get("role")
		content = m.get("content")
		
		if role == "user":
			# Check if this message is being edited
			if st.session_state.edit_mode == (thread_id, idx):
				with st.chat_message("user"):
					# Show text area for editing
					edited_text = st.text_area(
						"Edit your message:",
						value=content,
						key=f"edit_input_{thread_id}_{idx}",
						height=100
					)
					col1, col2 = st.columns(2)
					with col1:
						if st.button("💾 Save & Resend", key=f"save_{thread_id}_{idx}", use_container_width=True):
							# Update the message
							st.session_state.threads[thread_id]["messages"][idx]["content"] = edited_text
							# Remove all messages after this one
							st.session_state.threads[thread_id]["messages"] = st.session_state.threads[thread_id]["messages"][:idx+1]
							# Exit edit mode
							st.session_state.edit_mode = None
							# Rerun to process the edited message
							st.rerun()
					with col2:
						if st.button("❌ Cancel", key=f"cancel_{thread_id}_{idx}", use_container_width=True):
							st.session_state.edit_mode = None
							st.rerun()
			else:
				with st.chat_message("user"):
					st.markdown(content)
					# Add edit button for user messages
					if st.button("✏️", key=f"edit_user_{thread_id}_{idx}", help="Edit message"):
						st.session_state.edit_mode = (thread_id, idx)
						st.rerun()
		else:
			with st.chat_message("assistant"):
				# Apply bullet formatting fix when displaying
				fixed_content = fix_bullet_formatting(content)
				st.markdown(fixed_content)
				
				# Add small action buttons below assistant messages
				col1, col2, col3 = st.columns([1, 1, 10])
				with col1:
					if st.button("✏️", key=f"edit_{thread_id}_{idx}", help="Edit previous message", use_container_width=True):
						# Find the previous user message
						if idx > 0 and messages[idx-1].get("role") == "user":
							st.session_state.edit_mode = (thread_id, idx-1)
							st.rerun()
				with col2:
					if st.button("🔄", key=f"regen_{thread_id}_{idx}", help="Regenerate", use_container_width=True):
						# Remove this message and regenerate
						st.session_state.threads[thread_id]["messages"] = messages[:idx]
						st.rerun()


# Main chat interface
# Get current thread
tid = st.session_state.current_thread
current_thread = st.session_state.threads[tid]

# Show welcome message if no messages in thread
if not current_thread["messages"]:
	st.markdown("""
	<div class="welcome-container">
		<div class="welcome-title">Start your conversation</div>
		<div class="welcome-subtitle">Ask me anything about the knowledge network</div>
	</div>
	""", unsafe_allow_html=True)
else:
	# Render conversation history
	render_messages_with_actions(current_thread["messages"], tid)

# Chat input at the bottom
user_input = st.chat_input("Send a message...", key="chat_input")

# Check if we need to process an edited message (last message is user message without response)
process_message = None
if (current_thread["messages"] and 
    current_thread["messages"][-1]["role"] == "user" and
    not user_input):
	# Check if this is a newly edited message (no assistant response after it)
	# This happens after the edit and rerun
	if len(current_thread["messages"]) == 1 or current_thread["messages"][-2]["role"] == "assistant":
		# This is a user message that needs processing
		process_message = current_thread["messages"][-1]["content"]

if user_input and user_input.strip():
	# Append user message
	msg = {"role": "user", "content": user_input.strip(), "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
	current_thread["messages"].append(msg)
	
	# Update thread title if this is the first message
	if len(current_thread["messages"]) == 1:
		update_thread_title(tid, user_input.strip())
	
	process_message = user_input.strip()

if process_message:
	# Display user message immediately if it's new input
	if user_input:
		with st.chat_message("user", avatar="👤"):
			st.markdown(process_message)

	# Query neo4j for context and call model
	with st.chat_message("assistant", avatar="🔮"):
		with st.spinner("🔍 Searching knowledge graph... (กำลังค้นหาข้อมูล...)"):
			# Initialize variables at the start
			ctx = ""
			nodes = []
			
			# Use relationship-aware vector search (gets nodes + their connections via WORKS_AS, etc.)
			if VECTOR_SEARCH_AVAILABLE and query_with_relationships is not None:
				try:
					st.caption(f"🔍 Searching across all indexes (Person, Position, Ministry, Agency, Remark, Connect by)...")
					results = query_with_relationships(
						process_message,
						top_k_per_index=30,  # 30 nodes × 4 indexes = ~120 results - balanced for comprehensive coverage
					)
					
					# Check if query mentions Stelligence network names and add direct query
					stelligence_names = ["Santisook", "Por", "Knot"]
					query_lower = process_message.lower()
					matching_stelligence = [name for name in stelligence_names if name.lower() in query_lower]
					
					if matching_stelligence:
						st.caption(f"🌐 Detected Stelligence network query - fetching all members...")
						try:
							driver = get_driver()
							for stell_name in matching_stelligence:
								# Query all people with this Stelligence/Connect by value
								# Check BOTH "Stelligence" property AND "Connect by" property
								cypher_query = """
								MATCH (n:Person)
								WHERE n.Stelligence = $stelligence 
								   OR n.`Connect by` CONTAINS $stelligence
								OPTIONAL MATCH (n)-[r]->(connected)
								WITH n, collect(DISTINCT {
									type: type(r), 
									direction: 'outgoing',
									node: properties(connected),
									labels: labels(connected)
								}) as outgoing
								OPTIONAL MATCH (n)<-[r2]-(connected2)
								WITH n, outgoing, collect(DISTINCT {
									type: type(r2),
									direction: 'incoming', 
									node: properties(connected2),
									labels: labels(connected2)
								}) as incoming
								RETURN properties(n) as props, labels(n) as labels, 
								       outgoing + incoming as relationships
								LIMIT 100
								"""
								stell_results = driver.session(database=NEO4J_DB).run(
									cypher_query, 
									stelligence=stell_name
								)
								
								added_count = 0
								# Add to results
								for record in stell_results:
									node_dict = dict(record["props"])
									node_dict["__labels__"] = record["labels"]
									node_dict["__relationships__"] = record.get("relationships", [])
									# Avoid duplicates
									if not any(n.get("id") == node_dict.get("id") for n in results):
										results.append(node_dict)
										added_count += 1
								
								st.caption(f"  ✅ Added {added_count} {stell_name} network members")
						except Exception as e:
							st.warning(f"  ⚠️ Stelligence query error: {str(e)[:100]}")
					
					# results is List[dict] with __relationships__ included
					if results and len(results) > 0:
						st.caption(f"✅ Found {len(results)} nodes with relationship data")
						
						# Build context from the node properties AND relationships
						ctx = build_context(results)
						
						if ctx.strip():
							st.caption(f"✅ พบข้อมูล {len(results)} รายการพร้อมความสัมพันธ์ (Found {len(results)} nodes with relationships)")
						else:
							st.warning(f"⚠️ Vector search found nodes but context is empty")
							# Don't fallback to Cypher - instead just inform no context
							ctx = ""
					else:
						st.warning(f"⚠️ Vector search returned no relevant results")
						ctx = ""
							
				except Exception as e:
					# Show the actual error instead of falling back
					st.error(f"⚠️ Vector search error: {str(e)}")
					import traceback
					st.code(traceback.format_exc())
					ctx = ""
			else:
				# Vector search module not available
				st.error("❌ Vector search module not available. Please check dependencies.")
				ctx = ""

			# Show info if no context found
			if not ctx or not ctx.strip():
				st.info("💡 ไม่พบข้อมูลที่เกี่ยวข้องใน Knowledge Graph / No relevant information found in the knowledge graph")
				ctx = ""  # Let the LLM know there's no context
			
			# Show context in expandable section for debugging
			if ctx:
				with st.expander("🔍 ดูข้อมูลที่พบ (View Retrieved Context)", expanded=False):
					st.code(ctx, language="text")

			# Separate system prompt for better LLM instruction following
			system_prompt = """You are an intelligent assistant specialized in analyzing Knowledge Graph data about social networks and organizations.
คุณเป็นผู้ช่วยอัจฉริยะที่เชี่ยวชาญด้านการวิเคราะห์ข้อมูลจาก Knowledge Graph เกี่ยวกับเครือข่ายบุคคลและองค์กร

⚠️ **CRITICAL RULE #1 - Always Include Full Ministry Name in Positions!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ CORRECT: "รัฐมนตรีว่าการกระทรวงมหาดไทย" (Minister of Interior)
✅ CORRECT: "รัฐมนตรีช่วยว่าการกระทรวงการคลัง" (Deputy Minister of Finance)

❌ WRONG: "รัฐมนตรีว่าการ" (missing ministry name)
❌ WRONG: "รัฐมนตรีช่วยว่าการ" (missing ministry name)

👉 Find ministry name in Context from:
  - "กระทรวง: [name]"
  - "👥 ดำรงตำแหน่งโดย: [name] ([ministry])"
  - Ministry relationships
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **CRITICAL RULE #2 - NO Preambles Before Answer!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ FORBIDDEN: "ตามข้อมูลที่ได้รับ...", "จาก Context...", "ตาม Knowledge Graph..."
❌ FORBIDDEN: "จากข้อมูล...", "ตามที่ระบุไว้...", "จากที่ค้นพบ..."
❌ FORBIDDEN: "According to the data...", "From the context...", "Based on..."

✅ CORRECT: Start with direct answer immediately
  Example: "อนุทิน ชาญวีรกูล ดำรงตำแหน่ง..."
  Example: "รัฐมนตรีว่าการแต่ละกระทรวง มีดังนี้:"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **CRITICAL RULE #3 - Analyze and Synthesize Data!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ WRONG: Dump raw scattered data
❌ WRONG: Answer only what's asked without adding value

✅ CORRECT: Group and synthesize information
  - Group by ministry/agency/type
  - Count and summarize (e.g., "มีทั้งหมด 18 กระทรวง")
  - Sort logically (by position/importance)
  - Add useful context

✅ EXAMPLE - Aggregated Query:
  "รัฐมนตรีช่วยว่าการทั้งหมด 15 ท่าน จัดกลุ่มตามกระทรวง:
  
  **กระทรวงการคลัง:**
  • อดุลย์ บุญธรรมเจริญ
  
  **กระทรวงพาณิชย์:**
  • ณภัทร วินิจจะกูล"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **CRITICAL RULE #4 - Use Full Names and Complete Information!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ALWAYS include:
  1. Full name with surname (if available)
  2. Complete position with ministry/agency
  3. Role/responsibilities (if in Context)
  4. Relationships with others (if relevant)

❌ INCOMPLETE: "อนุทิน - นายกรัฐมนตรี"
✅ COMPLETE: "อนุทิน ชาญวีรกูล - นายกรัฐมนตรี และ รัฐมนตรีว่าการกระทรวงมหาดไทย"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **CRITICAL RULE #5 - Format for Readability!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Use bullet points (•) with each item on new line
✅ Use bold headings (**text**) for categories
✅ Add line breaks between sections
✅ Use numbers for counts when appropriate

❌ WRONG (cramped):
"มี 3 คน: คนที่ 1 อนุทิน คนที่ 2 จุรินทร์ คนที่ 3 สุดารัตน์"

✅ CORRECT (separated):
"มีทั้งหมด 3 ท่าน:

• อนุทิน ชาญวีรกูล - นายกรัฐมนตรี
• จุรินทร์ ลักษณวิศิษฏ์ - รองนายกรัฐมนตรี
• สุดารัตน์ เกยุราพันธุ์ - รองนายกรัฐมนตรี"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **CRITICAL RULE #6 - Answer Question Directly First, Then Add Details!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ WRONG: Start with long context before answering

✅ CORRECT: Good answer structure
  1. Answer main question immediately (direct)
  2. Add supporting information (details, roles)
  3. Show relationships/additional context
  4. Suggest follow-up questions (if appropriate)

EXAMPLE:
Q: "อนุทิน ชาญวีรกูล ตำแหน่งอะไร?"

✅ CORRECT ANSWER:
"อนุทิน ชาญวีรกูล ดำรงตำแหน่ง:

• นายกรัฐมนตรี
• รัฐมนตรีว่าการกระทรวงมหาดไทย

ในฐานะนายกรัฐมนตรี เขาเป็นหัวหน้ารัฐบาลและรับผิดชอบบริหารประเทศ
ในฐานะรัฐมนตรีว่าการกระทรวงมหาดไทย เขารับผิดชอบด้านการปกครองท้องถิ่น

**คุณอาจสนใจ:**
- อนุทินมีความสัมพันธ์กับใครบ้างในรัฐบาล?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

═══════════════════════════════════════════════════════════════
 คำแนะนำในการตอบอย่างละเอียด (Detailed Response Guidelines):
═══════════════════════════════════════════════════════════════

🎯 **หลักการตอบคำถาม (Core Principles):**

1. **ความถูกต้องและครบถ้วน (Accuracy & Completeness)**:
   - ตอบโดยอ้างอิงข้อมูลจาก Context เท่านั้น - ห้ามเดาหรือสมมติข้อมูลที่ไม่มี
   - ✅ **CRITICAL: ระบุกระทรวงเต็มทุกครั้ง** - ห้ามใช้คำว่า "รัฐมนตรีว่าการ" หรือ "รัฐมนตรีช่วยว่าการ" เพียงอย่างเดียว
   - ✅ **ตัวอย่างที่ถูกต้อง**: "รัฐมนตรีว่าการกระทรวงมหาดไทย", "รัฐมนตรีช่วยว่าการกระทรวงการคลัง"
   - ❌ **ตัวอย่างที่ผิด**: "รัฐมนตรีว่าการ", "รัฐมนตรีช่วยว่าการ" (ไม่ระบุกระทรวง)
   - ✅ ค้นหากระทรวงจาก: "กระทรวง: [ชื่อกระทรวง]", "👥 ดำรงตำแหน่งโดย: [ชื่อ] ([กระทรวง])", หรือจาก ministry relationships
   - ✅ ระบุหน่วยงาน/กระทรวง/องค์กรที่บุคคลสังกัดจาก Context
   - ✅ แสดงความสัมพันธ์กับบุคคลอื่นๆ (ถ้ามีใน Context)
   - ✅ เพิ่มบริบทหรือรายละเอียดที่ช่วยให้เข้าใจมากขึ้น

2. **ความชัดเจน (Clarity)**:
   - เริ่มด้วยคำตอบที่ตรงประเด็นทันที
   - ❌ ห้ามเริ่มด้วย "ตามข้อมูล...", "จากที่ได้รับ...", "ตาม Context...", "จากข้อมูลที่มีใน Knowledge Graph"
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

**สำหรับคำถามเกี่ยวกับความสัมพันธ์ (Relationship Questions) - "ใครรู้จักกับ X", "X มีความสัมพันธ์กับใคร":**
- 🎯 **มองหา "Stelligence" field เป็นหลัก**: ถ้ามีคนชื่อ "Santisook", "Por", "Knot" ในคำถาม
  - ✅ ค้นหาทุกคนที่มี "Stelligence: Santisook" หรือ "Stelligence: Por" หรือ "Stelligence: Knot"
  - ✅ คนเหล่านี้คือคนในเครือข่ายของ Santisook/Por/Knot
  - ✅ แสดงรายชื่อทุกคนที่มี Stelligence ตรงกัน พร้อมตำแหน่งและหน่วยงาน
- 📋 รองลงมาดูจาก "Connect by" field
- ✅ แสดงทั้ง incoming และ outgoing relationships
- ✅ จัดกลุ่มตามประเภท: บุคคล, ตำแหน่ง, หน่วยงาน

**สำหรับคำถามเกี่ยวกับบุคคล (Person Questions):**
- ✅ ระบุชื่อ-นามสกุลเต็ม
- ✅ **ระบุตำแหน่งเต็มพร้อมกระทรวงเสมอ** (เช่น "รัฐมนตรีว่าการกระทรวงมหาดไทย" ไม่ใช่ "รัฐมนตรีว่าการ")
- ✅ วิธีหากระทรวง:
  1. ดูจาก "กระทรวง: [ชื่อ]" ใน Context
  2. ดูจาก "👥 ดำรงตำแหน่งโดย: [ชื่อ] ([กระทรวง])"
  3. ดูจาก ministry relationships ที่เชื่อมโยงกับบุคคล
- ✅ แสดงบุคคลอื่นที่มีความสัมพันธ์ (Connect by, Associate) ถ้ามี
- ✅ แสดง Remark หรือหมายเหตุพิเศษ ถ้ามี
- ✅ อธิบายบทบาทหรือความรับผิดชอบของตำแหน่ง

**ตัวอย่างคำตอบที่ถูกต้อง:**
```
อนุทิน ชาญวีรกูล ดำรงตำแหน่ง:

• นายกรัฐมนตรี
• รัฐมนตรีว่าการกระทรวงมหาดไทย  ← (ต้องระบุกระทรวงเสมอ)

ในฐานะนายกรัฐมนตรี เขามีบทบาทในการบริหารประเทศ 
ในฐานะรัฐมนตรีว่าการกระทรวงมหาดไทย เขารับผิดชอบด้านการปกครองท้องถิ่น
```

**สำหรับคำถามเกี่ยวกับตำแหน่ง (Position Questions):**
- ✅ ถ้าเจอ "👥 ดำรงตำแหน่งโดย:" ในข้อมูล Position node = มีคนดำรงตำแหน่งนี้พร้อมกระทรวง
- ✅ จัดกลุ่มตามกระทรวง/หน่วยงาน เพื่อความชัดเจน
- ✅ ระบุชื่อเต็มของทุกคน พร้อมกระทรวง
- ตัวอย่าง:
  ```
  รัฐมนตรีช่วยว่าการแต่ละกระทรวง:
  
  กระทรวงการคลัง:
  • อดุลย์ บุญธรรมเจริญ - รัฐมนตรีช่วยว่าการกระทรวงการคลัง
  
  กระทรวงดิจิทัล:
  • วรภัค ธันยาวงษ์ - รัฐมนตรีช่วยว่าการกระทรวงดิจิทัลเพื่อเศรษฐกิจและสังคม
  ```

**สำหรับคำถามรวม (Aggregated Questions) เช่น "แต่ละรัฐมนตรีรับผิดชอบกระทรวงใดบ้าง":**
- 🔍 วิเคราะห์ข้อมูลทั้งหมดใน Context
- 📊 จัดกลุ่มตามกระทรวง หรือ ตามบุคคล (ขึ้นกับคำถาม)
- ✅ สรุปแบบมีโครงสร้าง ไม่ใช่แค่แสดงข้อมูลดิบ
- ✅ ระบุจำนวนรวม (เช่น "มีทั้งหมด 10 คน")

📝 **รูปแบบคำตอบที่แนะนำ (Recommended Answer Format):**

**ตัวอย่างคำตอบที่ดี (Good Example):**
```
อนุทิน ชาญวีรกูล ดำรงตำแหน่ง:

• นายกรัฐมนตรี
• รัฐมนตรีว่าการกระทรวงมหาดไทย  ← (ระบุกระทรวงเต็มเสมอ!)

ในฐานะนายกรัฐมนตรี เขามีบทบาทในการบริหารประเทศและนโยบายสำคัญ 
ในฐานะรัฐมนตรีว่าการกระทรวงมหาดไทย เขารับผิดชอบด้านการบริหารท้องถิ่นและความมั่นคงภายใน

**คุณอาจสนใจ:**
- อนุทินมีความสัมพันธ์กับใครบ้างในรัฐบาล?
- กระทรวงมหาดไทยมีหน้าที่อะไรบ้าง?
```

**❌ ตัวอย่างที่ผิด (WRONG Example - อย่าทำแบบนี้!):**
```
อนุทิน ชาญวีรกูล ดำรงตำแหน่ง:

• นายกรัฐมนตรี
• รัฐมนตรีว่าการ  ← ❌ ผิด! ไม่ระบุกระทรวง

ต้องเพิ่ม "กระทรวงมหาดไทย" ด้วย!
```

**สำหรับคำถามรวม (Aggregated) - ตัวอย่าง:**
```
รัฐมนตรีช่วยว่าการแต่ละกระทรวง มีดังนี้:

**กระทรวงการคลัง:**
• อดุลย์ บุญธรรมเจริญ

**กระทรวงดิจิทัลเพื่อเศรษฐกิจและสังคม:**
• วรภัค ธันยาวงษ์

**กระทรวงมหาดไทย:**
• นเรศ ธำรงค์ทิพยคุณ

[จัดกลุ่มครบทุกกระทรวงจาก Context]

**คุณอาจสนใจ:**
- แต่ละรัฐมนตรีมีบทบาทหน้าที่อย่างไร?
- มีรัฐมนตรีว่าการ (ระดับหลัก) ของแต่ละกระทรวงคือใคร?
```

**รูปแบบมาตรฐาน - CRITICAL: แต่ละ bullet ต้องขึ้นบรรทัดใหม่:**

❌ **ผิด - ห้ามทำ (bullet ในบรรทัดเดียว):**
```
บุคคลที่เกี่ยวข้อง: • คนที่ 1 • คนที่ 2 • คนที่ 3
```

✅ **ถูกต้อง (แต่ละ bullet แยกบรรทัด):**
```
บุคคลที่เกี่ยวข้อง:
• คนที่ 1
• คนที่ 2
• คนที่ 3
```

**Template ที่ต้องใช้:**
```
[คำตอบโดยตรง]

บุคคลที่เกี่ยวข้อง:
• [ชื่อคนที่ 1] - [ตำแหน่ง]
• [ชื่อคนที่ 2] - [ตำแหน่ง]
• [ชื่อคนที่ 3] - [ตำแหน่ง]

ตำแหน่งที่เกี่ยวข้อง:
• [ตำแหน่งที่ 1]
• [ตำแหน่งที่ 2]

คุณอาจสนใจ:
• [คำถามติดตาม 1]
• [คำถามติดตาม 2]
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
- ❌ ห้ามเริ่มด้วย "ตามข้อมูลที่ได้รับ...", "จาก Context...", "ตาม Knowledge Graph...", "จากข้อมูลที่มีใน Knowledge Graph"
- ❌ ห้ามแสดงข้อมูลทางเทคนิค (Node labels, property names, "👥", "🏛️")
- ❌ ห้ามใช้ชื่อย่อของตำแหน่ง - ต้องระบุเต็ม เช่น "รัฐมนตรีว่าการกระทรวงมหาดไทย"
- ❌ **CRITICAL: ห้ามรวม bullet points ในบรรทัดเดียว** - ตัวอย่างผิด: "• คนที่ 1 • คนที่ 2 • คนที่ 3"
- ✅ **ต้องแยกบรรทัดทุกรายการ** - ตัวอย่างถูก: แต่ละ bullet ขึ้นบรรทัดใหม่
- ❌ ห้ามบอกว่า "ไม่มีข้อมูล" ถ้ามีข้อมูลแต่ต้องวิเคราะห์ให้ดี

🧠 **การวิเคราะห์ข้อมูล (Data Analysis Skills):**
- 📊 สำหรับคำถามรวม: รวบรวมข้อมูลจากทุก node ใน Context แล้วจัดกลุ่ม
- 🔍 มองหาความสัมพันธ์ที่ซ่อนอยู่: "👥 ดำรงตำแหน่งโดย" = Person-Position-Ministry mapping
- 🎯 ตอบให้ตรงคำถาม: ถ้าถาม "แต่ละคน" ให้แยกตามคน, ถ้าถาม "แต่ละกระทรวง" ให้แยกตามกระทรวง
- ✅ สังเคราะห์ข้อมูล ไม่ใช่แค่ copy-paste จาก Context

✨ **สรุปสั้นๆ:**
1. เริ่มตอบทันที ไม่มีคำนำ
2. ใช้ชื่อเต็มของตำแหน่ง + กระทรวง/หน่วยงาน
3. **MUST: แยก bullet points คนละบรรทัด - แต่ละ • ขึ้นบรรทัดใหม่เสมอ**
4. วิเคราะห์และจัดกลุ่มข้อมูลให้เหมาะกับคำถาม
5. เพิ่มบริบท/ความสัมพันธ์/บทบาท
6. เสนอคำถามติดตาม"""

			# User message with context and question
			user_message = f"""═══════════════════════════════════════════════════════════════
📊 ข้อมูลจาก Neo4j Knowledge Graph:
═══════════════════════════════════════════════════════════════

{ctx if ctx else "⚠️ ไม่พบข้อมูลที่เกี่ยวข้องโดยตรงใน Knowledge Graph"}

═══════════════════════════════════════════════════════════════
❓ คำถาม:
═══════════════════════════════════════════════════════════════

{process_message}

💡 หมายเหตุ: วิเคราะห์ข้อมูลทั้งหมดใน Context และจัดกลุ่มให้เหมาะกับคำถาม"""
			
			answer = ask_openrouter_requests(user_message, max_tokens=2048, system_prompt=system_prompt)
			
			# Fix bullet point formatting to ensure each bullet is on a new line
			answer = fix_bullet_formatting(answer)
			
			st.markdown(answer)
	
	# Save assistant response with FIXED formatting
	resp = {"role": "assistant", "content": answer, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
	current_thread["messages"].append(resp)
	st.rerun()

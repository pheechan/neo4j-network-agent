"""
STelligence Network Agent - Neo4j Knowledge Graph Q&A System
Version: 2.0.0 - Major improvements
Last Updated: 2025-11-07

Changelog v2.0.0:
- ✅ Added retry logic with exponential backoff (handles 429 rate limits)
- ✅ Added caching for vector search and LLM responses (1-hour TTL)
- ✅ Added query intent detection (person/org/relationship/timeline)
- ✅ Added multi-hop path finding for relationship queries
- ✅ Added follow-up question generation
- ✅ Added streaming response support (toggle in settings)
- ✅ Added query analytics tracking (success rate, response time)
- ✅ Improved error handling and user feedback
"""

import os
from dotenv import load_dotenv
from typing import List
import requests
import streamlit as st
import time
import json
from functools import wraps

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
OPENROUTER_API_BASE = get_config("OPENROUTER_BASE_URL", get_config("OPENROUTER_API_BASE", get_config("OPENAI_API_BASE", "https://openrouter.ai/api/v1")))
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


@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_vector_search(query: str, top_k_per_index: int = 30, _cache_bypass: bool = False):
	"""
	Cached vector search to avoid repeated API calls for same queries.
	TTL=3600 means cache expires after 1 hour.
	_cache_bypass: Use different value to force cache miss (e.g., timestamp)
	"""
	if VECTOR_SEARCH_AVAILABLE and query_with_relationships is not None:
		try:
			return query_with_relationships(query, top_k_per_index=top_k_per_index)
		except Exception as e:
			st.error(f"Vector search error: {e}")
			return []
	return []


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def cached_llm_response(prompt: str, context: str, model: str, max_tokens: int = 512, system_prompt: str = None, _cache_bypass: bool = False):
	"""
	Cached LLM responses for identical query+context combinations.
	Saves API costs and reduces latency for repeat queries.
	_cache_bypass: Use different value to force cache miss (e.g., timestamp)
	"""
	full_prompt = f"{context}\n\n{prompt}" if context else prompt
	return ask_openrouter_requests(
		prompt=full_prompt,
		model=model,
		max_tokens=max_tokens,
		system_prompt=system_prompt
	)


def detect_query_intent(query: str) -> dict:
	"""
	Detect the intent and focus of the user's query.
	Returns dict with: intent_type, focus_entities, search_strategy
	"""
	query_lower = query.lower()
	
	intent_info = {
		'intent_type': 'general',
		'focus_entities': [],
		'search_strategy': 'broad',
		'is_relationship_query': False,
		'is_comparison_query': False
	}
	
	# Person-focused queries
	person_keywords = ['ใคร', 'who', 'คน', 'บุคคล', 'ชื่อ', 'name', 'นาม']
	if any(word in query_lower for word in person_keywords):
		intent_info['intent_type'] = 'person'
		intent_info['search_strategy'] = 'person_focused'
	
	# Ministry/Organization queries
	org_keywords = ['กระทรวง', 'ministry', 'องค์กร', 'organization', 'หน่วยงาน', 'agency', 'department']
	if any(word in query_lower for word in org_keywords):
		intent_info['intent_type'] = 'organization'
		intent_info['search_strategy'] = 'org_focused'
	
	# Relationship/Connection queries
	relationship_keywords = ['รู้จัก', 'connect', 'ผ่าน', 'through', 'เชื่อมโยง', 'relation', 'เส้นสาย', 'network']
	if any(word in query_lower for word in relationship_keywords):
		intent_info['is_relationship_query'] = True
		intent_info['search_strategy'] = 'relationship_focused'
	
	# Position/Role queries
	position_keywords = ['ตำแหน่ง', 'position', 'role', 'หน้าที่', 'duty', 'job']
	if any(word in query_lower for word in position_keywords):
		intent_info['intent_type'] = 'position'
	
	# Comparison queries
	comparison_keywords = ['เปรียบเทียบ', 'compare', 'แตกต่าง', 'difference', 'เหมือน', 'similar']
	if any(word in query_lower for word in comparison_keywords):
		intent_info['is_comparison_query'] = True
	
	# Timeline queries
	timeline_keywords = ['เมื่อไหร่', 'when', 'วันที่', 'date', 'ปี', 'year', 'ช่วงเวลา', 'period']
	if any(word in query_lower for word in timeline_keywords):
		intent_info['intent_type'] = 'timeline'
	
	return intent_info


def find_connection_path(person_a: str, person_b: str, max_hops: int = 3) -> dict:
	"""
	Find the shortest path between two people with the most connections.
	Strategy: Among all shortest paths, pick the one where intermediate nodes have the most total connections.
	Returns dict with: path_found, hops, path_nodes, path_relationships, total_connections
	"""
	try:
		driver = get_driver()
		with driver.session(database=NEO4J_DB) as session:
			# Find ALL shortest paths, then pick the one with most intermediate connections
			query = f"""
			MATCH (a:Person), (b:Person)
			WHERE a.name CONTAINS $person_a OR a.`ชื่อ` CONTAINS $person_a
			  AND b.name CONTAINS $person_b OR b.`ชื่อ` CONTAINS $person_b
			WITH a, b
			MATCH path = allShortestPaths((a)-[*..{max_hops}]-(b))
			WITH path, 
			     length(path) as hops,
			     nodes(path) as path_nodes,
			     relationships(path) as path_rels
			// Calculate connection count for each node
			UNWIND path_nodes as node
			WITH path, hops, path_nodes, path_rels, node,
			     size([(node)-[]-() | 1]) as node_connections
			WITH path, hops, path_nodes, path_rels,
			     sum(node_connections) as total_connections
			// Return path with node details
			RETURN path, hops,
			       [node in path_nodes | {{
			           name: coalesce(node.name, node.`ชื่อ`, 'Unknown'), 
			           labels: labels(node),
			           connections: size([(node)-[]-() | 1])
			       }}] as path_nodes,
			       [rel in path_rels | type(rel)] as path_rels,
			       total_connections
			ORDER BY hops ASC, total_connections DESC
			LIMIT 1
			"""
			
			result = session.run(query, person_a=person_a, person_b=person_b)
			record = result.single()
			
			if record:
				return {
					'path_found': True,
					'hops': record['hops'],
					'path_nodes': record['path_nodes'],
					'path_relationships': record['path_rels'],
					'total_connections': record['total_connections']
				}
			else:
				return {
					'path_found': False,
					'hops': None,
					'path_nodes': [],
					'path_relationships': [],
					'total_connections': 0
				}
	except Exception as e:
		st.error(f"Error finding connection path: {e}")
		return {'path_found': False, 'error': str(e)}


def generate_followup_questions(context: str, original_query: str, max_questions: int = 3) -> List[str]:
	"""
	Generate relevant follow-up questions based on context and original query.
	Returns list of suggested questions in Thai.
	"""
	if not context or len(context) < 50:
		return []
	
	# Limit context to avoid token overflow
	context_snippet = context[:800] if len(context) > 800 else context
	
	prompt = f"""Based on this information:
{context_snippet}

And the user's question: "{original_query}"

Generate {max_questions} relevant follow-up questions in Thai that the user might want to ask next.
Each question should:
- Be specific and related to the information provided
- Help explore deeper connections or details
- Be in natural Thai language

Format: Return ONLY the questions, one per line, each starting with "•"
Do not include explanations or numbering."""

	try:
		response = ask_openrouter_requests(
			prompt=prompt,
			model=OPENROUTER_MODEL,
			max_tokens=200,
			system_prompt="You are a helpful assistant generating follow-up questions in Thai."
		)
		
		# Extract questions starting with •
		questions = [q.strip() for q in response.split('\n') if q.strip().startswith('•')]
		return questions[:max_questions]
	except Exception as e:
		# Silently fail - follow-up questions are nice-to-have
		return []


def log_query_analytics(query: str, success: bool, error_type: str = None, response_time: float = None):
	"""
	Log query analytics to track performance and errors.
	Helps identify which queries work well and which need improvement.
	"""
	try:
		log_entry = {
			'timestamp': datetime.now().isoformat(),
			'query': query,
			'success': success,
			'error_type': error_type,
			'response_time': response_time,
			'model': OPENROUTER_MODEL
		}
		
		# Append to analytics log file
		with open('query_analytics.jsonl', 'a', encoding='utf-8') as f:
			f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
	except Exception:
		# Don't let logging errors break the app
		pass


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
		
		# Add "Connect by" information (important for networking)
		connect_by = n.get("Connect by")
		if connect_by:
			node_str += f"\n  🤝 Connect by: {connect_by}"
		
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


def retry_with_backoff(max_retries=3, base_delay=2):
	"""
	Decorator to retry API calls with exponential backoff.
	Handles 429 (rate limit) and 5xx (server) errors automatically.
	"""
	def decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			for attempt in range(max_retries):
				try:
					return func(*args, **kwargs)
				except requests.HTTPError as e:
					status_code = e.response.status_code if hasattr(e.response, 'status_code') else 0
					
					# Retry on rate limits (429) or server errors (5xx)
					if status_code == 429 or (500 <= status_code < 600):
						if attempt < max_retries - 1:
							delay = base_delay * (2 ** attempt)  # Exponential: 2s, 4s, 8s
							st.warning(f"⏳ Rate limited or server error. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
							time.sleep(delay)
							continue
					# For other errors, raise immediately
					raise
				except Exception as e:
					# For non-HTTP errors, raise immediately
					raise
			# If all retries exhausted, make final attempt
			return func(*args, **kwargs)
		return wrapper
	return decorator


@retry_with_backoff(max_retries=3, base_delay=2)
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


def ask_openrouter_streaming(prompt: str, model: str = OPENROUTER_MODEL, max_tokens: int = 512, system_prompt: str = None):
	"""
	Stream responses token by token for better UX (like ChatGPT).
	Yields text chunks as they arrive.
	"""
	if not OPENROUTER_API_KEY:
		yield "OpenRouter API key not set"
		return
	
	# Handle both base URL formats
	base = OPENROUTER_API_BASE.rstrip('/')
	if base.endswith('/v1'):
		url = f"{base}/chat/completions"
	else:
		url = f"{base}/v1/chat/completions"
	
	headers = {
		"Authorization": f"Bearer {OPENROUTER_API_KEY}",
		"Content-Type": "application/json",
	}
	
	# Build messages array
	messages = []
	if system_prompt:
		messages.append({"role": "system", "content": system_prompt})
	messages.append({"role": "user", "content": prompt})
	
	payload = {
		"model": model,
		"messages": messages,
		"temperature": 0.2,
		"max_tokens": max_tokens,
		"stream": True  # Enable streaming
	}
	
	try:
		response = requests.post(url, headers=headers, json=payload, timeout=60, stream=True)
		response.raise_for_status()
		
		# Parse Server-Sent Events (SSE)
		for line in response.iter_lines():
			if line:
				line_text = line.decode('utf-8')
				if line_text.startswith('data: '):
					data_str = line_text[6:]  # Remove 'data: ' prefix
					if data_str.strip() == '[DONE]':
						break
					try:
						data = json.loads(data_str)
						if 'choices' in data and len(data['choices']) > 0:
							delta = data['choices'][0].get('delta', {})
							content = delta.get('content', '')
							if content:
								yield content
					except json.JSONDecodeError:
						continue
	except Exception as e:
		yield f"\n\n[Error: {type(e).__name__} {e}]"


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
			parts = line.split('•')
			if len(parts) > 1:
				# First part before first bullet (usually empty or heading)
				if parts[0].strip():
					fixed_lines.append(parts[0].strip())
				
				# Add each bullet on its own line
				for part in parts[1:]:
					if part.strip():  # Only add if not empty
						fixed_lines.append('• ' + part.strip())
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
	
	# ⚙️ Settings Section
	with st.expander("⚙️ Settings", expanded=False):
		st.markdown("**Response Options:**")
		use_streaming = st.checkbox(
			"🌊 Streaming responses",
			value=st.session_state.get('use_streaming', False),
			help="Stream responses token-by-token (like ChatGPT)"
		)
		st.session_state['use_streaming'] = use_streaming
		
		use_cache = st.checkbox(
			"💾 Enable caching",
			value=st.session_state.get('use_cache', True),
			help="Cache responses for faster repeat queries. Disable for always fresh answers."
		)
		st.session_state['use_cache'] = use_cache
		
		if st.button("🗑️ Clear all caches", use_container_width=True):
			st.cache_data.clear()
			st.success("✅ All caches cleared!")
			time.sleep(0.5)
			st.rerun()
		
		st.markdown("**Model Settings:**")
		st.caption(f"Current model: `{OPENROUTER_MODEL}`")
		
		# Analytics summary
		try:
			if os.path.exists('query_analytics.jsonl'):
				with open('query_analytics.jsonl', 'r', encoding='utf-8') as f:
					logs = [json.loads(line) for line in f]
				total_queries = len(logs)
				success_count = sum(1 for log in logs if log.get('success'))
				avg_time = sum(log.get('response_time', 0) for log in logs) / total_queries if total_queries > 0 else 0
				
				st.markdown("**📊 Analytics:**")
				st.caption(f"Total queries: {total_queries}")
				st.caption(f"Success rate: {success_count}/{total_queries} ({100*success_count/total_queries:.1f}%)")
				st.caption(f"Avg response time: {avg_time:.2f}s")
		except Exception:
			pass
	
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
						# Remove this message and mark for regeneration (bypass cache)
						st.session_state.threads[thread_id]["messages"] = messages[:idx]
						st.session_state['force_regenerate'] = True  # Flag to bypass cache
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
		# Track response time for analytics
		start_time = time.time()
		
		with st.spinner("🔍 Searching knowledge graph... (กำลังค้นหาข้อมูล...)"):
			# Initialize variables at the start
			ctx = ""
			nodes = []
			
			# Detect query intent
			intent = detect_query_intent(process_message)
			if intent['intent_type'] != 'general':
				st.caption(f"🎯 Detected query type: {intent['intent_type']}")
			
			# Check for multi-hop path queries
			if intent['is_relationship_query']:
				# Try to extract person names for path finding
				import re
				# Simple regex to find Thai names (this could be improved)
				potential_names = re.findall(r'[ก-๙]+(?:\s+[ก-๙]+)?', process_message)
				if len(potential_names) >= 2:
					st.caption(f"🔗 Checking connection path between people...")
					path_result = find_connection_path(potential_names[0], potential_names[1])
					if path_result.get('path_found'):
						st.success(f"✅ Found connection in {path_result['hops']} hops!")
			
			# Use cached vector search for better performance
			if VECTOR_SEARCH_AVAILABLE and query_with_relationships is not None:
				try:
					st.caption(f"🔍 Searching across all indexes (Person, Position, Ministry, Agency, Remark, Connect by)...")
					
					# Check if we should bypass cache (regenerate or cache disabled)
					use_cache = st.session_state.get('use_cache', True)
					force_regenerate = st.session_state.get('force_regenerate', False)
					bypass_cache = force_regenerate or not use_cache
					
					if bypass_cache:
						st.caption("🔄 Bypassing cache for fresh results...")
					
					# Use cached version to avoid repeated API calls (unless bypassing)
					results = cached_vector_search(
						process_message,
						top_k_per_index=30,  # 30 nodes × 4 indexes = 120 results - balanced for free tier
						_cache_bypass=time.time() if bypass_cache else False  # Different value = cache miss
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

⚠️ **CRITICAL RULE #0 - NEVER HALLUCINATE! SEARCH THOROUGHLY FIRST!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ **ABSOLUTELY FORBIDDEN:**
1. ❌ DO NOT use your general knowledge about Thailand, government, or politics
2. ❌ DO NOT make assumptions about positions, roles, or responsibilities
3. ❌ DO NOT add information that is NOT EXPLICITLY in the Context
4. ❌ DO NOT guess connections, relationships, or associations
5. ❌ DO NOT say "ไม่มีข้อมูล" without THOROUGHLY searching all Context first
6. ❌ DO NOT explain roles unless explicitly mentioned in Context

✅ **MANDATORY SEARCH PROCESS BEFORE ANSWERING:**
1. ✅ FIRST: Search the ENTIRE Context for ministry/position information
2. ✅ Look in multiple places:
   - Direct property: "กระทรวง: [name]"
   - Position relationships: "WORKS_AS → [Position] → [Ministry]"
   - Ministry relationships: "→ Ministry: [name]"
   - Remark field: May contain additional context
3. ✅ ONLY if truly NOT FOUND after thorough search → say "ไม่มีข้อมูลในระบบ"
4. ✅ Copy information EXACTLY as written in Context

**Example of CORRECT thorough search:**
Context has:
```
Person: อนุทิน ชาญวีรกูล
- ตำแหน่ง: นายกรัฐมนตรี
- Relationships:
  → WORKS_AS → Position: รัฐมนตรีว่าการ
  → Ministry: กระทรวงมหาดไทย
```
✅ Correct: "อนุทิน ชาญวีรกูล ดำรงตำแหน่ง นายกรัฐมนตรี และ รัฐมนตรีว่าการกระทรวงมหาดไทย"
❌ Wrong: "อนุทิน ชาญวีรกูล ดำรงตำแหน่ง นายกรัฐมนตรี และ รัฐมนตรีว่าการ (ไม่มีข้อมูลกระทรวงในระบบ)" ← DIDN'T SEARCH RELATIONSHIPS!

**Example when TRULY missing:**
Context has ONLY:
```
Person: สมชาย
- ตำแหน่ง: รองรัฐมนตรี
(No ministry in properties, no relationships, no remarks)
```
✅ Correct: "สมชาย ดำรงตำแหน่ง รองรัฐมนตรี แต่ไม่พบข้อมูลกระทรวงในระบบ"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **CRITICAL RULE #1 - Connection Direction Matters!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**For questions like "ถ้าอยากรู้จัก X ต้องรู้จักผ่านอะไร" or "How to connect to X":**

✅ **CORRECT Logic - Find WHO connects TO the target:**
- Look for people who KNOW the target person
- Look for people in SAME "Connect by" network
- Look for people in SAME ministry/agency
- Look for people with SHARED relationships TO the target

❌ **WRONG Logic - Target's outgoing connections:**
- Don't focus on who the target knows
- Focus on who KNOWS the target

**Example:**
Q: "ถ้าอยากรู้จักพี่โด่งต้องรู้จักผ่านอะไร?"

✅ CORRECT approach:
1. Find people/networks that connect TO พี่โด่ง
2. Find "Connect by" networks พี่โด่ง belongs to (e.g., OSK115)
3. Find people who KNOW พี่โด่ง (incoming relationships)

Answer format:
"ถ้าต้องการรู้จักพี่โด่ง สามารถเชื่อมต่อได้ผ่าน:

1. 🤝 Connect by: OSK115 (โรงเรียนสวนกุหลาบรุ่น 115)
   - บุคคลที่จบ OSK115 สามารถเชื่อมต่อกับพี่โด่งได้

2. 🌐 เครือข่าย Santisook
   - Santisook รู้จักพี่โด่ง (shown in Context as 'known' relationship)
   - สมาชิกใน Santisook network สามารถเชื่อมต่อผ่าน Santisook

3. 📋 กระทรวงพลังงาน
   - บุคคลที่ทำงานในกระทรวงพลังงานสามารถเชื่อมต่อผ่านการทำงาน"

❌ WRONG approach (focusing on who พี่โด่ง knows):
"พี่โด่งรู้จัก..." ← This is backwards!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **CRITICAL RULE #1.1 - Optimal Connection Path Strategy!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**When finding connection paths between two people:**

🎯 **Strategy:** Find the SHORTEST path, but if multiple paths have same length, choose the one with MOST CONNECTED intermediate people.

**Why?** 
- Well-connected people = More influential = Better networking opportunity
- Path through highly connected people = More reliable connections

**Example:**
Q: "หาเส้นทางจาก Boss ไป พี่โด่ง"

Path A (3 hops): Boss → Person1 (5 connections) → Person2 (3 connections) → พี่โด่ง
Total intermediate connections: 8

Path B (3 hops): Boss → Person3 (10 connections) → Person4 (12 connections) → พี่โด่ง
Total intermediate connections: 22

✅ **CHOOSE Path B** because:
- Same length (3 hops)
- Person3 and Person4 are more well-connected (22 total vs 8 total)
- Higher chance of successful introduction

**When displaying path - USE THIS EXACT FORMAT:**

**🎯 เส้นทางที่แนะนำ:**

**ระยะทาง:** 3 ขั้น (shortest path)
**Connections รวมของคนกลาง:** 22 connections

**เส้นทาง:**
1. **Boss** (ต้นทาง)
   
2. **Person3** (คนกลาง)
   - Connections: 10 🌟
   - ตำแหน่ง: [position if available]
   
3. **Person4** (คนกลาง) 
   - Connections: 12 🌟🌟 ← Most connected!
   - ตำแหน่ง: [position if available]
   
4. **พี่โด่ง** (เป้าหมาย)

**สรุป:** เส้นทางนี้ผ่านคนที่มี connections สูง ทำให้มีโอกาสติดต่อสำเร็จสูง

❌ DON'T show messy format like:
"อนุทิน ชาญวีรกูล
พี่เต๊ะ (มี 2 connections: อธิบดี, Santisook)
พี่โด่ง"

✅ DO use clear numbered list with proper sections and spacing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ **CRITICAL RULE #2 - Always Include Full Ministry Name in Positions!**
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
   - ✅ **IMPORTANT: Always show "Connect by" information** - มองหา "🤝 Connect by:" ใน Context และระบุทุกครั้ง
   - ✅ **Connect by is the KEY to networking** - อธิบายว่าบุคคลนี้เชื่อมต่อกับเครือข่ายผ่านอะไร (เช่น OSK115, โรงเรียน, มหาวิทยาลัย)
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
- ✅ **CRITICAL: วิธีหากระทรวง - ค้นหาอย่างละเอียดก่อนบอกว่า "ไม่มีข้อมูล":**
  1. ✅ ดูจาก "กระทรวง: [ชื่อ]" ใน direct properties
  2. ✅ ดูจาก "👥 ดำรงตำแหน่งโดย: [ชื่อ] ([กระทรวง])" ใน Position nodes
  3. ✅ ดูจาก "→ Ministry: [ชื่อ]" ใน relationships section
  4. ✅ ดูจาก "WORKS_AS → Position → Ministry" ใน relationship chains
  5. ✅ ดูจาก Remark field ที่อาจมีข้อมูลเพิ่มเติม
  6. ❌ อย่าบอกว่า "ไม่มีข้อมูล" ถ้ายังไม่ได้ค้นหาทุกแหล่งข้างต้น!
- ✅ แสดงบุคคลอื่นที่มีความสัมพันธ์ (Connect by, Associate) ถ้ามี
- ✅ แสดง Remark หรือหมายเหตุพิเศษ ถ้ามี
- ✅ อธิบายบทบาทหรือความรับผิดชอบของตำแหน่ง

**ตัวอย่างคำตอบที่ถูกต้อง:**
```
อนุทิน ชาญวีรกูล ดำรงตำแหน่ง:

• นายกรัฐมนตรี
• รัฐมนตรีว่าการกระทรวงมหาดไทย  ← (ต้องระบุกระทรวงเสมอ)

🤝 Connect by: OSK115 (โรงเรียนสวนกุหลาบวิทยาลัย รุ่น 115)

ในฐานะนายกรัฐมนตรี เขามีบทบาทในการบริหารประเทศ 
ในฐานะรัฐมนตรีว่าการกระทรวงมหาดไทย เขารับผิดชอบด้านการปกครองท้องถิ่น

**การเชื่อมต่อกับเครือข่าย:**
• รู้จักผ่าน Santisook (Stelligence Network)
• เชื่อมโยงผ่าน OSK115 (โรงเรียนเดียวกัน)
```

**สำหรับคำถาม "รู้จักผ่านอะไร" (How to Know/Connect Questions):**
```
พี่โด่งมีช่องทางการเชื่อมต่อดังนี้:

1. 🤝 Connect by: OSK115 (โรงเรียนสวนกุหลาบวิทยาลัย รุ่น 115)
   - บุคคลที่อยู่ในเครือข่ายเดียวกันสามารถเชื่อมต่อกันได้ผ่านรุ่นเดียวกัน

2. 🌐 รู้จักผ่าน Santisook (Stelligence Network)
   - เป็นส่วนหนึ่งของเครือข่าย Santisook

3. 📋 ตำแหน่ง: รัฐมนตรีว่าการกระทรวงพลังงาน
   - สามารถเชื่อมต่อผ่านการทำงานในกระทรวง

**คุณอาจสนใจ:**
• ใครบ้างที่จบ OSK115 เหมือนกัน?
• Santisook network มีใครบ้าง?
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

═══════════════════════════════════════════════════════════════
⚠️ CRITICAL REMINDERS:
═══════════════════════════════════════════════════════════════
1. ✅ USE ONLY information from Context above - DO NOT use general knowledge
2. ✅ If asking "how to connect to X" → find WHO connects TO X (incoming connections)
3. ✅ For connection paths: Choose shortest path with MOST CONNECTED intermediate people
4. ✅ Always show "Connect by" networks (e.g., OSK115) - this is KEY for networking
5. ✅ If ministry not in Context → say "ไม่มีข้อมูลกระทรวงในระบบ"
6. ❌ DO NOT add explanations or responsibilities not in Context
7. ❌ DO NOT guess or assume any information"""
			
			# Use streaming for better UX (optional - can be toggled)
			use_streaming = st.session_state.get('use_streaming', False)
			use_cache = st.session_state.get('use_cache', True)
			force_regenerate = st.session_state.get('force_regenerate', False)
			bypass_cache = force_regenerate or not use_cache
			
			# Clear regenerate flag after using it
			if force_regenerate:
				st.session_state['force_regenerate'] = False
			
			if use_streaming:
				# Streaming response (like ChatGPT) - always bypasses cache
				answer_placeholder = st.empty()
				answer = ""
				for chunk in ask_openrouter_streaming(user_message, max_tokens=2048, system_prompt=system_prompt):
					answer += chunk
					answer_placeholder.markdown(answer + "▌")  # Show cursor
				answer_placeholder.markdown(answer)  # Remove cursor
			else:
				# Regular response (with optional caching and retry)
				if bypass_cache:
					# Direct call without cache
					st.caption("🔄 Generating fresh response...")
					answer = ask_openrouter_requests(user_message, max_tokens=2048, system_prompt=system_prompt)
				else:
					# Try to use cached response first
					try:
						answer = cached_llm_response(
							prompt=process_message,
							context=ctx,
							model=OPENROUTER_MODEL,
							max_tokens=2048,
							system_prompt=system_prompt,
							_cache_bypass=time.time() if bypass_cache else False
						)
					except Exception as e:
						# If cached fails, try direct call (has retry logic)
						st.warning(f"Cache miss, calling API directly...")
						answer = ask_openrouter_requests(user_message, max_tokens=2048, system_prompt=system_prompt)
			
			# Fix bullet point formatting to ensure each bullet is on a new line
			answer = fix_bullet_formatting(answer)
			
			# Track analytics
			response_time = time.time() - start_time
			success = not answer.startswith("OpenRouter request failed")
			log_query_analytics(
				query=process_message,
				success=success,
				error_type=None if success else "API_ERROR",
				response_time=response_time
			)
			
			if not use_streaming:
				st.markdown(answer)
			
			# Generate and display follow-up questions
			if success and ctx:
				followup_questions = generate_followup_questions(ctx, process_message)
				if followup_questions:
					st.markdown("---")
					st.markdown("**💡 คำถามที่คุณอาจสนใจ:**")
					for q in followup_questions:
						st.markdown(q)
	
	# Save assistant response with FIXED formatting
	resp = {"role": "assistant", "content": answer, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
	current_thread["messages"].append(resp)
	st.rerun()

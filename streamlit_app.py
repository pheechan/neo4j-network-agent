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

# Configuration (from .env)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.openrouter.ai"))
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free")

# Vector/RAG configuration (optional, used by KG/VectorRAG.query_vector_rag)
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "default_index")
VECTOR_NODE_LABEL = os.getenv("VECTOR_NODE_LABEL", "Document")
VECTOR_SOURCE_PROPERTY = os.getenv("VECTOR_SOURCE_PROPERTY", "text")
VECTOR_EMBEDDING_PROPERTY = os.getenv("VECTOR_EMBEDDING_PROPERTY", "embedding")
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "3"))

# Try to import the project's vector RAG helper (LangChain + Neo4jVector)
try:
	from KG.VectorRAG import query_vector_rag
except Exception:
	query_vector_rag = None


def get_driver():
	if GraphDatabase is None:
		raise RuntimeError("neo4j driver missing. Install with: python -m pip install neo4j")
	return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))


def search_nodes(driver, question: str, limit: int = 6) -> List[dict]:
	q = """
	MATCH (n)
	WHERE (n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower($q))
	   OR (n.text IS NOT NULL AND toLower(n.text) CONTAINS toLower($q))
	RETURN n LIMIT $limit
	"""
	out = []
	with driver.session(database=NEO4J_DB) as session:
		res = session.run(q, q=question, limit=limit)
		for r in res:
			node = r.get("n")
			try:
				props = dict(node)
			except Exception:
				props = {}
			out.append(props)
	return out


def build_context(nodes: List[dict]) -> str:
	if not nodes:
		return ""
	pieces = []
	for i, n in enumerate(nodes, 1):
		name = n.get("name") or n.get("title") or f"node_{i}"
		text = n.get("text") or n.get("description") or ""
		pieces.append(f"{name}: {text}")
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


def ask_openrouter_requests(prompt: str, model: str = OPENROUTER_MODEL, max_tokens: int = 512) -> str:
	if not OPENROUTER_API_KEY:
		return "OpenRouter API key not set (OPENROUTER_API_KEY or OPENAI_API_KEY)"
	url = f"{OPENROUTER_API_BASE.rstrip('/')}" + "/v1/chat/completions"
	headers = {
		"Authorization": f"Bearer {OPENROUTER_API_KEY}",
		"Content-Type": "application/json",
	}
	payload = {
		"model": model,
		"messages": [{"role": "user", "content": prompt}],
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


## Streamlit chat UI with threads
st.set_page_config(page_title="Neo4j Chat — Streamlit", layout="wide")

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
	st.header("Threads")
	for tid, meta in list(st.session_state.threads.items()):
		label = f"{meta['title']} ({len(meta['messages'])})"
		if st.button(label, key=f"thread-{tid}"):
			st.session_state.current_thread = tid

	st.markdown("---")
	if st.button("New Thread"):
		new_thread()
	if st.button("Clear Thread"):
		clear_current_thread()
	st.markdown("---")
	st.write("Config")
	st.text_input("Model", value=OPENROUTER_MODEL, key="_model", disabled=True)
	st.write("Neo4j:")
	st.write(f"{NEO4J_URI} ({NEO4J_DB})")


def render_messages(messages: List[Dict]):
	for m in messages:
		role = m.get("role")
		content = m.get("content")
		ts = m.get("time")
		label = f"{role.capitalize()} - {ts}" if ts else role.capitalize()
		if role == "user":
			st.chat_message("user", avatar="🧑").write(content)
		else:
			st.chat_message("assistant", avatar="🤖").write(content)


st.title("Neo4j Chatbot")

col1, col2 = st.columns([3, 1])

with col1:
	st.subheader(st.session_state.threads[st.session_state.current_thread]["title"])
	render_messages(st.session_state.threads[st.session_state.current_thread]["messages"])

	with st.form(key="input_form", clear_on_submit=True):
		user_input = st.text_area("", placeholder="Type your message and press Send...", key="user_input", height=100)
		submitted = st.form_submit_button("Send")

	if submitted and user_input and user_input.strip():
		# append user message
		tid = st.session_state.current_thread
		msg = {"role": "user", "content": user_input.strip(), "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
		st.session_state.threads[tid]["messages"].append(msg)

		# query neo4j for context and call model
		with st.spinner("Querying Neo4j and generating answer..."):
			# prefer vector RAG retrieval (if available), otherwise fall back to simple node search
			ctx = ""
			if query_vector_rag is not None:
				try:
					docs_and_scores = query_vector_rag(
						user_input,
						vector_index_name=VECTOR_INDEX_NAME,
						vector_node_label=VECTOR_NODE_LABEL,
						vector_source_property=VECTOR_SOURCE_PROPERTY,
						vector_embedding_property=VECTOR_EMBEDDING_PROPERTY,
						top_k=VECTOR_TOP_K,
					)
					# build a short context from returned docs (each item may be (doc, score))
					snippets = []
					for item in docs_and_scores:
						try:
							doc = item[0]
							content = getattr(doc, "page_content", None) or getattr(doc, "content", None) or str(doc)
						except Exception:
							# item might itself be a Document
							doc = item
							content = getattr(doc, "page_content", None) or getattr(doc, "content", None) or str(doc)
						snippets.append(content)
					ctx = "\n\n".join(snippets)
				except Exception as e:
					# fall back to simple cypher search if vector retrieval fails
					try:
						driver = get_driver()
						nodes = search_nodes(driver, user_input)
						ctx = build_context(nodes)
					except Exception as e2:
						ctx = ""
						st.error(f"Vector RAG error: {e}; fallback error: {e2}")
			else:
				# query_vector_rag not available (package or import issue) — use cypher search
				try:
					driver = get_driver()
					nodes = search_nodes(driver, user_input)
					ctx = build_context(nodes)
				except Exception as e:
					ctx = ""
					st.error(f"Neo4j error: {e}")

			# final fallback: if we still don't have context, try local in-memory docs
			if not ctx:
				try:
					local_nodes = local_search(user_input, limit=VECTOR_TOP_K)
					if local_nodes:
						ctx = build_context(local_nodes)
						st.info("Using local fallback documents for context")
				except Exception:
					# silently ignore — ctx remains empty
					pass

			prompt = f"""You are an assistant. Use the following context extracted from a Neo4j graph to answer the user's question.\n\nQuestion:\n{user_input}\n\nContext:\n{ctx}\n\nAnswer concisely."""
			answer = ask_openrouter_requests(prompt)
			resp = {"role": "assistant", "content": answer, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
			st.session_state.threads[tid]["messages"].append(resp)
			# Streamlit will usually rerun after form submit; try to trigger a rerun if available
			try:
				st.experimental_rerun()
			except Exception:
				# older/newer streamlit versions may not expose experimental_rerun
				pass

with col2:
	st.subheader("Context Preview")
	# show most recent context for the current thread (if any)
	msgs = st.session_state.threads[st.session_state.current_thread]["messages"]
	last_user = None
	for m in reversed(msgs):
		if m["role"] == "user":
			last_user = m["content"]
			break
	if last_user:
		try:
			# prefer vector retrieval preview when available
			if query_vector_rag is not None:
				docs_and_scores = query_vector_rag(
					last_user,
					vector_index_name=VECTOR_INDEX_NAME,
					vector_node_label=VECTOR_NODE_LABEL,
					vector_source_property=VECTOR_SOURCE_PROPERTY,
					vector_embedding_property=VECTOR_EMBEDDING_PROPERTY,
					top_k=VECTOR_TOP_K,
				)
				if docs_and_scores:
					snippets = []
					for item in docs_and_scores:
						try:
							doc = item[0]
							content = getattr(doc, "page_content", None) or getattr(doc, "content", None) or str(doc)
						except Exception:
							doc = item
							content = getattr(doc, "page_content", None) or getattr(doc, "content", None) or str(doc)
						snippets.append(content)
					ctx = "\n\n".join(snippets)
					st.write(ctx)
				else:
					st.write("(no matching nodes)")
			else:
				driver = get_driver()
				nodes = search_nodes(driver, last_user)
				ctx = build_context(nodes)
				if ctx:
					st.write(ctx)
				else:
					st.write("(no matching nodes)")
		except Exception as e:
			st.write(f"Neo4j error: {e}")
	else:
		st.write("No user message yet in this thread")


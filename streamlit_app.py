import os
from dotenv import load_dotenv
from typing import List
import requests
import streamlit as st

try:
	from neo4j import GraphDatabase
except Exception:
	GraphDatabase = None

load_dotenv()

# Neo4j settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")

# OpenRouter / OpenAI-compatible settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.openrouter.ai"))
OR_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free")


def get_driver():
	if GraphDatabase is None:
		raise RuntimeError("neo4j driver not installed in this environment. Install with: python -m pip install neo4j")
	return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))


def search_nodes(driver, question: str, limit: int = 10) -> List[dict]:
	q = """
	MATCH (n)
	WHERE (exists(n.name) AND toLower(n.name) CONTAINS toLower($q))
	   OR (exists(n.text) AND toLower(n.text) CONTAINS toLower($q))
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


def ask_openrouter_requests(prompt: str, model: str = OR_MODEL, max_tokens: int = 512) -> str:
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
		# OpenRouter uses OpenAI-compatible response shape
		return j["choices"][0]["message"]["content"].strip()
	except Exception as e:
		return f"OpenRouter request failed: {type(e).__name__} {e}"


## Streamlit UI
st.set_page_config(page_title="Neo4j + Streamlit Chatbot", page_icon=":robot:")
st.title("Neo4j Chatbot (Streamlit) — OpenRouter")

ui_token = os.getenv("CHAT_UI_TOKEN", "devtoken")
token = st.text_input("UI token (for demo)", type="password")
if token != ui_token:
	st.warning("Enter valid UI token to use the chat (set CHAT_UI_TOKEN in .env).")
	st.stop()

question = st.text_input("Ask a question about the graph", key="q")
if st.button("Ask") and question.strip():
	with st.spinner("Querying Neo4j..."):
		try:
			driver = get_driver()
		except Exception as e:
			st.error(f"Neo4j driver error: {e}")
			st.stop()
		nodes = search_nodes(driver, question)
		ctx = build_context(nodes)
		if not ctx:
			st.info("No direct node context found; answer may be limited.")

	st.subheader("Context from Neo4j")
	st.write(ctx or "(no context)")

	st.subheader("Answer")
	prompt = f"""You are an assistant. Use the following context extracted from a Neo4j graph to answer the user's question.

Question:
{question}

Context:
{ctx}

Answer concisely. If context doesn't cover the answer, say you couldn't find enough info.
"""
	answer = ask_openrouter_requests(prompt)
	st.write(answer)


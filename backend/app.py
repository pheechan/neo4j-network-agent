import os
import time
import json
import asyncio
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment
load_dotenv()

try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None

# Import vector search helpers from KG (these modules are present in repo)
try:
    from KG.VectorSearchDirect import query_with_relationships, search_all_nodes_direct
except Exception:
    query_with_relationships = None
    search_all_nodes_direct = None

# Import CypherHealer / summarizer if available
try:
    from Graph.Tool.CypherHealer import CypherHealer
    from Graph.Tool.CypherSummarizer import CypherResultSummarizer, summarize_path_result
except Exception:
    CypherHealer = None
    CypherResultSummarizer = None
    summarize_path_result = None

import requests
import httpx

app = FastAPI(title="STelligence Network Agent API")

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    use_streaming: bool = False
    use_cache: bool = True


class ChatResponse(BaseModel):
    answer: str
    context: str = ""
    followups: List[str] = []


def get_config(key: str, default: str = "") -> str:
    return os.getenv(key, default)


NEO4J_URI = get_config("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = get_config("NEO4J_USERNAME", "neo4j")
NEO4J_PWD = get_config("NEO4J_PASSWORD", "")
NEO4J_DB = get_config("NEO4J_DATABASE", "neo4j")

LLM_PROVIDER = get_config("LLM_PROVIDER", "gemini")
GOOGLE_API_KEY = get_config("GOOGLE_API_KEY")
OPENROUTER_API_KEY = get_config("OPENROUTER_API_KEY") or get_config("OPENAI_API_KEY")
OPENROUTER_API_BASE = get_config("OPENROUTER_BASE_URL", get_config("OPENROUTER_API_BASE", get_config("OPENAI_API_BASE", "https://openrouter.ai/api/v1")))
OPENROUTER_MODEL = get_config("OPENROUTER_MODEL", "deepseek/deepseek-chat")
GEMINI_MODEL = get_config("GEMINI_MODEL", "gemini-2.5-flash")
# Local LLM options (if you want to run a model locally)
LOCAL_LLM = get_config("LOCAL_LLM", "")  # e.g., "transformers"
LOCAL_LLM_MODEL = get_config("LOCAL_LLM_MODEL", "")  # e.g., path or model id
OLLAMA_URL = get_config("OLLAMA_URL", "http://localhost:11434")


def get_driver():
    if GraphDatabase is None:
        raise RuntimeError("neo4j driver missing. Install neo4j package")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))


def detect_query_intent(query: str) -> dict:
    q = (query or "").lower()
    info = {
        'intent_type': 'general',
        'focus_entities': [],
        'search_strategy': 'broad',
        'is_relationship_query': False,
        'is_comparison_query': False
    }
    person_keywords = ['ใคร', 'who', 'คน', 'บุคคล', 'ชื่อ', 'name']
    if any(w in q for w in person_keywords):
        info['intent_type'] = 'person'
        info['search_strategy'] = 'person_focused'

    relationship_keywords = ['รู้จัก', 'connect', 'ผ่าน', 'เชื่อมโยง', 'relation', 'เส้นสาย', 'network', 'relationship', 'path', 'link', 'สัมพันธ์', 'connection']
    if any(w in q for w in relationship_keywords):
        info['is_relationship_query'] = True
        info['search_strategy'] = 'relationship_focused'

    comparison_keywords = ['เปรียบเทียบ', 'compare', 'vs', 'versus']
    if any(w in q for w in comparison_keywords):
        info['is_comparison_query'] = True

    return info


def build_context(nodes: List[dict]) -> str:
    # Simplified build_context adapted from streamlit app
    if not nodes:
        return ""
    pieces = []
    for n in nodes:
        name = (n.get('name') or n.get('title') or n.get('ชื่อ-นามสกุล') or n.get('id') or 'Unknown')
        text_parts = []
        for key, val in n.items():
            if key in ['embedding', 'embedding_text', '__labels__', '__relationships__', '__score__']:
                continue
            if isinstance(val, str) and val:
                text_parts.append(f"{key}: {val}")
        text = ' | '.join(text_parts)
        rels = n.get('__relationships__', [])
        rel_str = ''
        if rels:
            items = []
            for r in rels:
                t = r.get('type') if isinstance(r, dict) else str(r)
                node = r.get('node') if isinstance(r, dict) else {}
                connected_name = (node.get('ชื่อ-นามสกุล') or node.get('name') or '') if node else ''
                items.append(f"{t} -> {connected_name}")
            rel_str = ' Relationships: ' + ', '.join(items)
        pieces.append(f"{name}: {text}{rel_str}")
    return "\n\n".join(pieces)


SYSTEM_PROMPT = """You are an intelligent assistant specialized in analyzing Knowledge Graph data about social networks and organizations.
You must use only the provided context when answering and avoid hallucinations.
"""


def ask_openrouter_requests(prompt: str, model: str = OPENROUTER_MODEL, max_tokens: int = 512, system_prompt: str = None) -> str:
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not set"
    base = OPENROUTER_API_BASE.rstrip('/')
    if base.endswith('/v1'):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"OpenRouter request failed: {type(e).__name__} {e}"


def ask_gemini(prompt: str, model: str = GEMINI_MODEL, max_tokens: int = 512, system_prompt: str = None) -> str:
    # Try to use google.generativeai if available, otherwise return an informative message
    try:
        import google.generativeai as genai
        if not GOOGLE_API_KEY:
            return "Google API key not set"
        genai.configure(api_key=GOOGLE_API_KEY)
        # Basic generate - keep simple
        system_instruction = system_prompt or "You are a professional network analysis assistant."
        full_prompt = (system_instruction + "\n\n" + prompt) if system_instruction else prompt
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(full_prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens))
        return response.text.strip() if getattr(response, 'text', None) else str(response)
    except Exception as e:
        return f"Gemini request failed: {type(e).__name__} {e}"


def ask_llm(prompt: str, max_tokens: int = 512, system_prompt: str = None) -> str:
    # If a local LLM is configured, prefer it (e.g., transformers or ollama)
    if LOCAL_LLM:
        if LOCAL_LLM.lower() == 'transformers' and LOCAL_LLM_MODEL:
            try:
                return ask_local_transformers(prompt, model_name=LOCAL_LLM_MODEL, max_tokens=max_tokens)
            except Exception as e:
                # fall through to configured provider
                print(f"Local transformers LLM failed: {e}")
        if LOCAL_LLM.lower() == 'ollama' and LOCAL_LLM_MODEL:
            try:
                return ask_local_ollama(prompt, model_name=LOCAL_LLM_MODEL, max_tokens=max_tokens)
            except Exception as e:
                print(f"Local Ollama LLM failed: {e}")

    if LLM_PROVIDER == 'gemini':
        return ask_gemini(prompt, max_tokens=max_tokens, system_prompt=system_prompt)
    else:
        return ask_openrouter_requests(prompt, max_tokens=max_tokens, system_prompt=system_prompt)


def ask_local_transformers(prompt: str, model_name: str, max_tokens: int = 512) -> str:
    """Attempt to run a local HuggingFace Transformers text-generation model.

    This will try to import transformers and run a simple generate. For large models
    you should ensure the model is downloaded and you have adequate hardware.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as e:
        raise RuntimeError(f"Transformers or Torch not installed: {e}")

    # Load tokenizer and model (cached by transformers)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Auto device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device=='cuda' else None)
    model.to(device)

    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    # Generate
    gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=False)
    outputs = model.generate(input_ids, **gen_kwargs)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the generated text if model echoes it
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()


def ask_local_ollama(prompt: str, model_name: str, max_tokens: int = 512) -> str:
    """Call a local Ollama instance (HTTP API).

    Expects Ollama daemon to run (default http://localhost:11434).
    Model should be pulled locally (e.g. `ollama pull scb10x/typhoon2.1-gemma3-4b`).
    """
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    # Ollama expects 'model' and 'prompt' keys; set stream=false for full response
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,  # Important: disable streaming for synchronous call
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.2
        }
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        # Parse JSON result - Ollama returns {"model", "response", "done", ...}
        try:
            j = r.json()
            if isinstance(j, dict):
                # Ollama's /api/generate returns 'response' field
                if 'response' in j:
                    return j['response']
                if 'text' in j:
                    return j['text']
                if 'output' in j:
                    return j['output'] if isinstance(j['output'], str) else json.dumps(j['output'])
            # Fallback to raw text
            return r.text
        except ValueError:
            # Not JSON — return text
            return r.text
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")


async def ask_local_ollama_stream(prompt: str, model_name: str, max_tokens: int = 512):
    """Async stream reader for local Ollama HTTP API.

    Uses the Ollama /api/generate endpoint with streaming enabled (stream: true)
    and yields partial text tokens as they arrive. The exact streaming JSON
    format can vary across Ollama versions; this reader attempts to parse
    common keys ('token', 'delta', 'text', 'content') and falls back to
    yielding raw lines.
    """
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        # Use stream context to iterate over incoming lines
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                # Try to parse JSON chunks
                try:
                    j = json.loads(line)
                    text = None
                    if isinstance(j, dict):
                        # Common fields: token, delta, text, content
                        if 'token' in j:
                            text = j['token']
                        elif 'delta' in j:
                            text = j['delta']
                        elif 'text' in j:
                            text = j['text']
                        elif 'content' in j:
                            text = j['content']
                        elif 'output' in j and isinstance(j['output'], str):
                            text = j['output']
                    if text:
                        yield text
                        continue
                except Exception:
                    # Not JSON — pass through the raw text
                    pass

                # Fallback: yield the raw line
                yield line


def generate_followup_questions(context: str, original_query: str, max_questions: int = 3) -> List[str]:
    if not context or len(context) < 50:
        return []
    prompt = f"Based on this information:\n{context[:800]}\nAnd the user's question: \"{original_query}\"\nGenerate {max_questions} follow-up questions in Thai, one per line."
    try:
        resp = ask_llm(prompt, max_tokens=200)
        lines = [l.strip() for l in resp.split('\n') if l.strip()]
        return lines[:max_questions]
    except Exception:
        return []


@app.post('/api/chat', response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail='Empty message')

    # Detect intent
    intent = detect_query_intent(message)

    # Try vector search for context
    ctx = ""
    nodes = []
    try:
        if query_with_relationships is not None:
            nodes = query_with_relationships(message, top_k_per_index=6)
        # If few results, try search_all_nodes_direct fallback
        if (not nodes or len(nodes) < 6) and search_all_nodes_direct is not None:
            extra = search_all_nodes_direct(message, top_k=20)
            if extra:
                # append unique
                existing_names = {n.get('name') or n.get('id') for n in nodes}
                for n in extra:
                    name = n.get('name') or n.get('id')
                    if name not in existing_names:
                        nodes.append(n)
    except Exception as e:
        nodes = []

    if nodes:
        ctx = build_context(nodes)

    # Special handling for relationship queries: attempt path search
    if intent.get('is_relationship_query'):
        # Simple heuristic to extract two names in quotes
        import re
        quoted = re.findall(r'\"([^\"]+)\"', message)
        if len(quoted) >= 2:
            # try to run Cypher path search
            try:
                driver = get_driver()
                # use function from streamlit logic simplified: find shortest path
                query = f"""
MATCH (a:Person), (b:Person)
WHERE (a.name CONTAINS $person_a OR a.`ชื่อ` CONTAINS $person_a OR a.`ชื่อ-นามสกุล` CONTAINS $person_a)
  AND (b.name CONTAINS $person_b OR b.`ชื่อ` CONTAINS $person_b OR b.`ชื่อ-นามสกุ` CONTAINS $person_b)
WITH a, b
MATCH path = allShortestPaths((a)-[*..10]-(b))
WITH path, length(path) as hops, nodes(path) as all_nodes, relationships(path) as path_rels
UNWIND all_nodes as node
WITH path, hops, all_nodes, path_rels, node, size([(node)-[]-() | 1]) as node_connections
WITH path, hops, all_nodes, path_rels, sum(node_connections) as total_connections
RETURN path, hops, [node in all_nodes | {name: coalesce(node.`ชื่อ-นามสกุล`, node.name, node.`ชื่อ`, 'Unknown'), labels: labels(node), connections: size([(node)-[]-() | 1])}] as path_nodes, [rel in path_rels | type(rel)] as path_rels, total_connections
ORDER BY hops ASC, total_connections DESC
LIMIT 1
"""
                with driver.session(database=NEO4J_DB) as session:
                    rec = session.run(query, person_a=quoted[0], person_b=quoted[1]).single()
                    if rec:
                        path_nodes = rec['path_nodes']
                        hops = rec['hops']
                        summary = f"Found path ({hops} hops): {' → '.join([n.get('name') for n in path_nodes])}"
                        # Add to context
                        ctx = (ctx + "\n\n" + summary) if ctx else summary
            except Exception:
                pass

    # Build user_message and call LLM
    user_message = f"Context:\n{ctx}\n\nQuestion:\n{message}\n"
    answer = ask_llm(user_message, max_tokens=1024, system_prompt=SYSTEM_PROMPT)

    followups = generate_followup_questions(ctx, message)

    return ChatResponse(answer=answer, context=ctx, followups=followups)


@app.get('/api/stream-chat')
async def stream_chat(request: Request, message: str):
    """Simple SSE streaming endpoint. For providers that support streaming, integrate provider streams here.
    Currently it synthesizes a stream by splitting the full answer into chunks to simulate tokens.
    """
    if not message:
        raise HTTPException(status_code=400, detail='Empty message')

    # Build context quickly (non-blocking)
    intent = detect_query_intent(message)
    ctx = ""
    try:
        if query_with_relationships is not None:
            nodes = query_with_relationships(message, top_k_per_index=6)
            if nodes:
                ctx = build_context(nodes)
    except Exception:
        ctx = ""

    user_message = f"Context:\n{ctx}\n\nQuestion:\n{message}\n"

    # If a local Ollama model is configured, prefer true streaming from Ollama
    if LOCAL_LLM and LOCAL_LLM.lower() == 'ollama' and LOCAL_LLM_MODEL:
        async def ollama_event_generator():
            try:
                async for token in ask_local_ollama_stream(user_message, LOCAL_LLM_MODEL, max_tokens=1024):
                    if await await_client_disconnect(request):
                        return
                    # Send each partial token as an SSE data event
                    yield f"data: {token}\n\n"
                    # small sleep to give clients a chance to render
                    await asyncio.sleep(0)
            except Exception as e:
                # Forward an error message as a final SSE event
                yield f"event: error\ndata: {str(e)}\n\n"

        return StreamingResponse(ollama_event_generator(), media_type='text/event-stream')

    # Otherwise fallback: call LLM (non-streaming) then yield chunks to simulate streaming
    full_answer = ask_llm(user_message, max_tokens=1024, system_prompt=SYSTEM_PROMPT)

    async def event_generator(text: str, chunk_size: int = 80):
        # Break into chunks and yield as SSE `data:` lines
        i = 0
        while i < len(text):
            if await await_client_disconnect(request):
                return
            chunk = text[i:i+chunk_size]
            i += chunk_size
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.03)

    # StreamingResponse with text/event-stream
    return StreamingResponse(event_generator(full_answer), media_type='text/event-stream')


async def await_client_disconnect(request: Request) -> bool:
    # Helper to stop streaming if client disconnects
    if await request.is_disconnected():
        return True
    return False

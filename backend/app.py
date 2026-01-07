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

# Import smart network agent
try:
    from backend.network_agent import NetworkAgent, get_network_agent
except Exception:
    try:
        from network_agent import NetworkAgent, get_network_agent
    except Exception:
        NetworkAgent = None
        get_network_agent = None

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


@app.get('/')
def root():
    """Root endpoint with API info"""
    return {
        "name": "STelligence Network Agent API",
        "version": "1.0.0",
        "description": "Thai Government Network Knowledge Graph Chat API",
        "endpoints": {
            "chat": "/api/chat (POST)",
            "health": "/api/health (GET)",
            "search": "/api/search (POST)",
            "docs": "/docs"
        },
        "status": "running"
    }


@app.get('/api/health')
def health_check():
    """Comprehensive health check endpoint - checks Neo4j, Ollama, and agent status"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {}
    }
    
    # Check Neo4j connection
    neo4j_status = {"status": "unknown", "message": ""}
    if GraphDatabase is not None:
        try:
            uri = os.getenv('NEO4J_URI')
            user = os.getenv('NEO4J_USERNAME', 'neo4j')
            pwd = os.getenv('NEO4J_PASSWORD')
            
            if uri and pwd:
                driver = GraphDatabase.driver(uri, auth=(user, pwd))
                driver.verify_connectivity()
                # Quick test query
                with driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    result.single()
                driver.close()
                neo4j_status = {"status": "healthy", "message": "Connected to Neo4j Aura"}
            else:
                neo4j_status = {"status": "unhealthy", "message": "Missing NEO4J_URI or NEO4J_PASSWORD"}
        except Exception as e:
            neo4j_status = {"status": "unhealthy", "message": str(e)}
            health_status["status"] = "degraded"
    else:
        neo4j_status = {"status": "unavailable", "message": "Neo4j driver not installed"}
    health_status["services"]["neo4j"] = neo4j_status
    
    # Check Ollama connection
    ollama_status = {"status": "unknown", "message": ""}
    try:
        ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            ollama_status = {"status": "healthy", "message": f"Models: {', '.join(model_names[:3])}"}
        else:
            ollama_status = {"status": "unhealthy", "message": f"HTTP {response.status_code}"}
            health_status["status"] = "degraded"
    except Exception as e:
        ollama_status = {"status": "unhealthy", "message": str(e)}
        health_status["status"] = "degraded"
    health_status["services"]["ollama"] = ollama_status
    
    # Check Network Agent
    agent_status = {"status": "unknown", "message": ""}
    if get_network_agent is not None:
        try:
            agent = get_network_agent()
            agent.driver.verify_connectivity()
            agent_status = {"status": "healthy", "message": "Network agent ready"}
        except Exception as e:
            agent_status = {"status": "unhealthy", "message": str(e)}
            health_status["status"] = "degraded"
    else:
        agent_status = {"status": "unavailable", "message": "Network agent not loaded"}
    health_status["services"]["network_agent"] = agent_status
    
    return health_status


@app.get('/api/debug/networks')
def debug_list_networks():
    """Debug endpoint: List all Connect by networks and their member counts"""
    if GraphDatabase is None:
        return {"error": "Neo4j driver not available"}
    
    try:
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USERNAME', 'neo4j')
        pwd = os.getenv('NEO4J_PASSWORD')
        
        if not uri or not pwd:
            return {"error": "NEO4J_URI or NEO4J_PASSWORD not configured"}
        
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        with driver.session() as session:
            # List all Connect by networks with counts
            result = session.run("""
                MATCH (cb:`Connect by`)
                OPTIONAL MATCH (p:Person)-[:connect_by]->(cb)
                RETURN cb.`Connect by` as network, 
                       count(DISTINCT p) as member_count
                ORDER BY member_count DESC
                LIMIT 50
            """)
            networks = [{"network": r["network"], "member_count": r["member_count"]} for r in result]
            
            # Also check Stelligence networks
            stell_result = session.run("""
                MATCH (s:Santisook)-[:santisook_known]->(p:Person)
                RETURN 'Santisook' as network_type, count(DISTINCT p) as count
                UNION ALL
                MATCH (s:Por)-[:por_known]->(p:Person)
                RETURN 'Por' as network_type, count(DISTINCT p) as count
                UNION ALL
                MATCH (s:Knot)-[:knot_known]->(p:Person)
                RETURN 'Knot' as network_type, count(DISTINCT p) as count
            """)
            stelligence = [{"network_type": r["network_type"], "count": r["count"]} for r in stell_result]
            
        driver.close()
        return {
            "connect_by_networks": networks,
            "stelligence_networks": stelligence
        }
    except Exception as e:
        return {"error": str(e)}


class ChatRequest(BaseModel):
    message: str
    use_streaming: bool = False
    use_cache: bool = True


class ChatResponse(BaseModel):
    answer: str
    context: str = ""
    debug: Dict[str, Any] = {}  # Debug metadata for UI


# Lightweight Aura proxy request for minimal frontend mask
class AuraQueryRequest(BaseModel):
    # We intentionally do NOT accept credentials from the client anymore.
    # The frontend is a single URL app and the backend will use configured
    # environment variables for the Aura connection (NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD).
    query_type: str  # 'shortest_path' | 'person_network' | 'network_members'
    params: Dict[str, Any] = {}


@app.post('/api/aura/query')
def aura_query(req: AuraQueryRequest):
    """Run a safe, parameterized query against a user-provided Neo4j Aura URI.

    This endpoint is intentionally limited to a small set of allowed query types to
    avoid arbitrary code execution. The frontend (Next.js) will call this endpoint
    with URI / credentials entered by the user for ephemeral access.
    """
    if GraphDatabase is None:
        raise HTTPException(status_code=500, detail='Neo4j driver not available in backend')

    allowed = {'shortest_path', 'person_network', 'network_members'}
    if req.query_type not in allowed:
        raise HTTPException(status_code=400, detail=f'query_type must be one of {allowed}')

    # Prepare cypher and params
    cypher = ''
    cypher_params = {}
    if req.query_type == 'person_network':
        # params: { name }
        cypher = """
        MATCH (p:Person)
        WHERE p.name CONTAINS $name OR p.nickname = $name
        OPTIONAL MATCH (p)-[r]-(x)
        RETURN properties(p) AS person, collect({rel: type(r), node: properties(x)}) AS relationships
        LIMIT 200
        """
        cypher_params = {'name': req.params.get('name', '')}

    elif req.query_type == 'shortest_path':
        # params: { from_name, to_name }
        cypher = """
        MATCH (a:Person), (b:Person)
        WHERE (a.name CONTAINS $from_name OR a.nickname = $from_name)
          AND (b.name CONTAINS $to_name OR b.nickname = $to_name)
        MATCH p = shortestPath((a)-[*..10]-(b))
        RETURN [n IN nodes(p) | properties(n)] AS nodes, [r IN relationships(p) | {type: type(r), props: properties(r)}] AS rels
        LIMIT 1
        """
        cypher_params = {
            'from_name': req.params.get('from_name', ''),
            'to_name': req.params.get('to_name', ''),
        }

    elif req.query_type == 'network_members':
        # params: { network }
        cypher = """
        MATCH (n:Network {name: $network})<-[:CONNECTED_BY]-(p:Person)
        RETURN collect(properties(p)) AS members LIMIT 500
        """
        cypher_params = {'network': req.params.get('network', '')}

    # Execute against configured Aura (or NEO4J_URI provided in environment)
    driver = None
    try:
        uri = os.getenv('NEO4J_URI', NEO4J_URI)
        user = os.getenv('NEO4J_USERNAME', NEO4J_USER)
        pwd = os.getenv('NEO4J_PASSWORD', NEO4J_PWD)
        if not uri or not user or not pwd:
            raise HTTPException(status_code=500, detail='Backend Neo4j credentials are not configured (NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD)')
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        with driver.session() as session:
            result = session.run(cypher, cypher_params)
            records = [r.data() for r in result]
            return {'ok': True, 'records': records}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error querying Aura: {str(e)}')
    finally:
        if driver:
            try:
                driver.close()
            except Exception:
                pass
    followups: List[str] = []


def get_config(key: str, default: str = "") -> str:
    return os.getenv(key, default)


NEO4J_URI = get_config("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = get_config("NEO4J_USERNAME", "neo4j")
NEO4J_PWD = get_config("NEO4J_PASSWORD", "")
NEO4J_DB = get_config("NEO4J_DATABASE", "neo4j")

LLM_PROVIDER = get_config("LLM_PROVIDER", "ollama")
OPENROUTER_API_KEY = get_config("OPENROUTER_API_KEY") or get_config("OPENAI_API_KEY")
OPENROUTER_API_BASE = get_config("OPENROUTER_BASE_URL", get_config("OPENROUTER_API_BASE", get_config("OPENAI_API_BASE", "https://openrouter.ai/api/v1")))
OPENROUTER_MODEL = get_config("OPENROUTER_MODEL", "deepseek/deepseek-chat")

# Local LLM options (if you want to run a model locally)
LOCAL_LLM = get_config("LOCAL_LLM", "ollama")
LOCAL_LLM_MODEL = get_config("LOCAL_LLM_MODEL", "scb10x/typhoon2.1-gemma3-4b")
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


SYSTEM_PROMPT = """คุณคือ STelligence Agent ช่วยวิเคราะห์เครือข่ายบุคคลภาครัฐไทย ตอบภาษาไทย กระชับ ใช้ข้อมูลจาก Context ที่ให้มา"""


THAI_FOLLOWUP_SUGGESTIONS = {
    "network_members": [
        "ดูรายละเอียดเพิ่มเติมของสมาชิกคนใดคนหนึ่ง",
        "ค้นหาคนที่รู้จักกับสมาชิกในเครือข่ายนี้",
        "ดูเครือข่ายอื่นที่เกี่ยวข้อง"
    ],
    "stelligence_network": [
        "ดูเครือข่าย Por / Knot / Santisook อื่น",
        "หาคนที่อยู่หลายเครือข่ายพร้อมกัน",
        "ค้นหาตามกระทรวงหรือหน่วยงาน"
    ],
    "organization_search": [
        "ดูกระทรวงหรือหน่วยงานอื่น",
        "หาเครือข่ายของคนในหน่วยงานนี้",
        "ค้นหาตามตำแหน่งงาน"
    ],
    "cohort_search": [
        "ดูรุ่นอื่นๆ (NEXIS รุ่น 2, วปอ. รุ่น 69)",
        "ค้นหาคนที่อยู่หลายรุ่น",
        "ดูเครือข่ายอื่นของสมาชิก"
    ],
    "general": [
        "ใครทำงานกระทรวงพลังงาน",
        "NEXIS รุ่น 1 มีใครบ้าง",
        "ใครรู้จัก Santisook / Por / Knot"
    ]
}


def get_followup_suggestions(intent_type: str) -> List[str]:
    """Get follow-up question suggestions based on intent type"""
    return THAI_FOLLOWUP_SUGGESTIONS.get(intent_type, THAI_FOLLOWUP_SUGGESTIONS["general"])


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


def ask_llm(prompt: str, max_tokens: int = 512, system_prompt: str = None) -> str:
    """Use Ollama exclusively for all LLM requests."""
    # Use Ollama as the only LLM provider
    if LOCAL_LLM_MODEL:
        try:
            return ask_local_ollama(prompt, model_name=LOCAL_LLM_MODEL, max_tokens=max_tokens, system_prompt=system_prompt)
        except Exception as e:
            print(f"Ollama LLM failed: {e}")
            return f"ขออภัย ไม่สามารถประมวลผลได้: Ollama error - {e}"
    else:
        return "ขออภัย ไม่ได้กำหนดโมเดล LLM (LOCAL_LLM_MODEL) กรุณาตั้งค่าใน environment variables"


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


def ask_local_ollama(prompt: str, model_name: str, max_tokens: int = 512, system_prompt: str = None) -> str:
    """Call a local Ollama instance (HTTP API).

    Expects Ollama daemon to run (default http://localhost:11434).
    Model should be pulled locally (e.g. `ollama pull scb10x/typhoon2.1-gemma3-4b`).
    
    Speed optimizations for CPU inference:
    - Lower temperature for more deterministic (faster) responses
    - Reduced num_ctx to minimize memory/compute
    - Lower num_predict for faster responses
    - num_thread optimized for CPU
    """
    import time
    start_time = time.time()
    
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    
    # Build prompt with system prompt if provided - keep it concise
    full_prompt = prompt
    if system_prompt:
        # Use a more concise format to reduce token processing
        full_prompt = f"{system_prompt}\n\n{prompt}"
    
    # Aggressive CPU optimization settings
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False,  # Important: disable streaming for synchronous call
        "keep_alive": -1, # Keep model loaded in memory indefinitely for faster subsequent requests
        "options": {
            "num_predict": min(max_tokens, 256),  # Limit output tokens for faster response
            "temperature": 0.0,  # Greedy decoding = fastest
            "num_ctx": 2048,  # Reduced context window for speed on CPU
            "repeat_penalty": 1.0,  # Disable for speed
            "top_p": 1.0,  # Disable nucleus sampling for speed
            "top_k": 1,  # Greedy - only pick top token
            "num_thread": 8,  # Utilize multiple CPU threads
        }
    }
    
    print(f"[DEBUG OLLAMA] Starting request to {url}")
    print(f"[DEBUG OLLAMA] Model: {model_name}, max_tokens: {max_tokens}")
    print(f"[DEBUG OLLAMA] Prompt length: {len(full_prompt)} chars")
    try:
        # Increase timeout to 600s (10 min) for slow models like Typhoon
        r = requests.post(url, json=payload, timeout=600)
        elapsed = time.time() - start_time
        print(f"[DEBUG OLLAMA] Request completed in {elapsed:.2f}s")
        r.raise_for_status()
        # Parse JSON result - Ollama returns {"model", "response", "done", ...}
        try:
            j = r.json()
            if isinstance(j, dict):
                # Log performance metrics from Ollama
                if 'total_duration' in j:
                    total_ns = j.get('total_duration', 0)
                    prompt_eval_ns = j.get('prompt_eval_duration', 0)
                    eval_ns = j.get('eval_duration', 0)
                    eval_count = j.get('eval_count', 0)
                    print(f"[DEBUG OLLAMA] Total: {total_ns/1e9:.2f}s, Prompt eval: {prompt_eval_ns/1e9:.2f}s, Generation: {eval_ns/1e9:.2f}s, Tokens: {eval_count}")
                    if eval_count > 0 and eval_ns > 0:
                        tokens_per_sec = eval_count / (eval_ns / 1e9)
                        print(f"[DEBUG OLLAMA] Speed: {tokens_per_sec:.2f} tokens/sec")
                
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
        elapsed = time.time() - start_time
        print(f"[DEBUG OLLAMA] Request FAILED after {elapsed:.2f}s: {e}")
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
        "keep_alive": -1, # Keep loaded
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


def generate_followup_questions(context: str, original_query: str, max_questions: int = 3, intent_type: str = None) -> List[str]:
    """Generate follow-up questions based on context and query intent"""
    
    # If we have a known intent type, use predefined suggestions
    if intent_type and intent_type in THAI_FOLLOWUP_SUGGESTIONS:
        suggestions = THAI_FOLLOWUP_SUGGESTIONS[intent_type][:max_questions]
        # Try to make suggestions more specific based on query
        query_lower = original_query.lower()
        
        # Extract network/org names to personalize suggestions
        specific_suggestions = []
        for suggestion in suggestions:
            # Keep as-is for now, could be personalized further
            specific_suggestions.append(suggestion)
        
        return specific_suggestions[:max_questions]
    
    # Fallback: use LLM to generate (slower but more context-aware)
    if not context or len(context) < 50:
        return THAI_FOLLOWUP_SUGGESTIONS.get("general", [])[:max_questions]
    
    try:
        prompt = f"""จากข้อมูลนี้:
{context[:800]}

และคำถามของผู้ใช้: "{original_query}"

สร้างคำถามต่อเนื่อง {max_questions} ข้อ เป็นภาษาไทย บรรทัดละข้อ 
ตัวอย่าง:
- ใครรู้จัก X บ้าง
- X ทำงานที่ไหน
- มีใครอีกที่อยู่ในเครือข่ายเดียวกัน

คำถามต่อเนื่อง:"""
        
        resp = ask_llm(prompt, max_tokens=200)
        lines = [l.strip().lstrip('-').lstrip('•').strip() for l in resp.split('\n') if l.strip() and len(l.strip()) > 5]
        return lines[:max_questions] if lines else THAI_FOLLOWUP_SUGGESTIONS.get("general", [])[:max_questions]
    except Exception:
        return THAI_FOLLOWUP_SUGGESTIONS.get("general", [])[:max_questions]


@app.post('/api/chat', response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    import time
    start_time = time.time()
    
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail='Empty message')

    # Debug metadata to return to UI
    debug_info = {
        "intent_type": "general",
        "nodes_found": 0,
        "nodes_per_index": 6,
        "network_result_found": False,
        "llm_tokens": 0,
        "processing_time_ms": 0
    }

    # 1) First, try the smart Network Agent for relationship queries
    network_result = None
    network_context = ""
    network_intent_type = "general"  # Track intent type for follow-up suggestions
    
    if get_network_agent is not None:
        try:
            agent = get_network_agent()
            smart_query_result = agent.execute_smart_query(message)
            
            print(f"[DEBUG] Network agent result: {smart_query_result}")
            
            # Track the intent type
            network_intent_type = smart_query_result["intent"]["type"]
            debug_info["intent_type"] = network_intent_type
            
            # If we got a network-specific result, format it nicely
            if network_intent_type != "general":
                network_result = smart_query_result["result"]
                network_context = format_network_result(smart_query_result)
                debug_info["network_result_found"] = True
                print(f"[DEBUG] Network context generated: {network_context[:200]}...")
        except Exception as e:
            print(f"[ERROR] Network agent error: {e}")
            import traceback
            traceback.print_exc()
            # Provide user-friendly error message
            error_msg = str(e)
            if "ServiceUnavailable" in error_msg or "routing" in error_msg.lower():
                network_context = "\n⚠️ ขณะนี้มีปัญหาในการเชื่อมต่อฐานข้อมูล กรุณาลองใหม่อีกครั้ง"
            elif "timeout" in error_msg.lower():
                network_context = "\n⚠️ การเชื่อมต่อใช้เวลานานเกินไป กรุณาลองใหม่อีกครั้ง"
            else:
                network_context = ""
    else:
        print("[WARNING] Network agent not available (get_network_agent is None)")

    # 2) Detect intent (legacy)
    intent = detect_query_intent(message)

    # 3) Try vector search for context
    ctx = ""
    nodes = []
    nodes_per_index = 6
    try:
        if query_with_relationships is not None:
            nodes = query_with_relationships(message, top_k_per_index=nodes_per_index)
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
    
    debug_info["nodes_found"] = len(nodes)
    debug_info["nodes_per_index"] = nodes_per_index

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

    # 4) Combine network context with vector search context
    combined_context = ""
    if network_context:
        combined_context = f"═══ NETWORK ANALYSIS ═══\n{network_context}\n\n"
    if ctx:
        combined_context += f"═══ ADDITIONAL CONTEXT FROM DATABASE ═══\n{ctx}\n\n"

    # 5) Build the final prompt for LLM
    if network_context or ctx:
        # Use compact system prompt for faster CPU inference
        system_prompt = """คุณคือ STelligence Agent ช่วยวิเคราะห์เครือข่ายบุคคลภาครัฐไทย ตอบภาษาไทย กระชับ ใช้ข้อมูลจาก Context"""
        
        user_message = f"""{combined_context}คำถาม: {message}
ตอบกระชับ:"""
    else:
        # Use default system prompt for general queries
        system_prompt = SYSTEM_PROMPT
        user_message = f"Context:\n{combined_context if combined_context else 'No specific context available.'}\n\nQuestion:\n{message}\n"

    # 6) For structured queries with pre-formatted data, SKIP LLM entirely - use direct answer
    # This reduces response time from ~150s to ~5s for path/network queries
    is_structured_query = network_intent_type in ["shortest_path", "network_members", "stelligence_network", 
                                                   "organization_search", "cohort_search", "person_network"]
    
    if is_structured_query and network_context:
        # Generate human-readable answer directly from structured data
        answer = generate_structured_answer(smart_query_result, message)
        debug_info["llm_tokens"] = 0
        debug_info["llm_skipped"] = True
    else:
        # Use LLM for non-structured queries
        llm_max_tokens = 256
        debug_info["llm_tokens"] = llm_max_tokens
        answer = ask_llm(user_message, max_tokens=llm_max_tokens, system_prompt=system_prompt)

    # 7) Generate follow-up questions with intent type for better suggestions
    followups = generate_followup_questions(combined_context, message, intent_type=network_intent_type)

    # Calculate processing time
    debug_info["processing_time_ms"] = int((time.time() - start_time) * 1000)

    return ChatResponse(answer=answer, context=combined_context, debug=debug_info)


def generate_structured_answer(smart_query_result: Dict, original_question: str) -> str:
    """
    Generate a human-readable answer directly from structured query results.
    This bypasses the LLM entirely for massive speed improvements on CPU.
    """
    if not smart_query_result:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    
    intent = smart_query_result.get("intent", {})
    intent_type = intent.get("type", "general")
    result = smart_query_result.get("result", {})
    
    if not result or result.get("found") == False:
        return result.get("message", "ไม่พบข้อมูลที่ตรงกับคำถาม")
    
    lines = []
    
    # Handle shortest_path queries
    if intent_type == "shortest_path":
        from_name = result.get("from", intent.get("from_person", ""))
        to_name = result.get("to", intent.get("to_person", ""))
        path = result.get("path", [])
        distance = result.get("distance", len(path) - 1 if path else 0)
        
        # Build path string
        if path:
            path_str = " → ".join(str(p) for p in path)
            if from_name and not path_str.startswith(str(from_name)):
                path_str = f"{from_name} → {path_str}"
            lines.append(f"เส้นทางจาก {from_name} ไป {to_name}:")
            lines.append(f"📍 {path_str}")
            lines.append(f"📏 ระยะทาง: {distance} ขั้นตอน")
        
        # Show person details - fetch from result with debug
        person_details = result.get("person_details", [])
        print(f"[DEBUG] generate_structured_answer - person_details: {person_details}")
        
        if person_details and len(person_details) > 0:
            lines.append(f"\n👥 รายละเอียดบุคคลในเส้นทาง:")
            for person in person_details:
                if isinstance(person, dict):
                    name = person.get("name", person.get("full_name", ""))
                    pos = person.get("position", "")
                    ministry = person.get("ministry", "")
                    agency = person.get("agency", "")
                    conns = person.get("connections", 0)
                    
                    detail_parts = []
                    if pos:
                        detail_parts.append(f"ตำแหน่ง: {pos}")
                    if ministry:
                        detail_parts.append(f"กระทรวง: {ministry}")
                    elif agency:
                        detail_parts.append(f"หน่วยงาน: {agency}")
                    if conns:
                        detail_parts.append(f"{conns} connections")
                    
                    detail_str = ", ".join(detail_parts) if detail_parts else ""
                    lines.append(f"  • {name}" + (f" ({detail_str})" if detail_str else ""))
        
        # Show best connector
        best = result.get("best_connector")
        if best and isinstance(best, dict):
            name = best.get("name", "")
            pos = best.get("position", "")
            conns = best.get("connections", 0)
            lines.append(f"\n⭐ Best Connector: {name}")
            if pos:
                lines.append(f"   ตำแหน่ง: {pos}")
            if conns:
                lines.append(f"   มี {conns} connections ในเครือข่าย")
    
    # Handle network members/stelligence queries
    elif intent_type in ["stelligence_network", "network_members"]:
        network = result.get("network", result.get("network_type", ""))
        members = result.get("members", [])
        count = result.get("member_count", len(members))
        
        lines.append(f"เครือข่าย {network}:")
        lines.append(f"👥 สมาชิก {count} คน")
        
        # Show top members with details
        if members:
            lines.append("\nรายชื่อสมาชิก:")
            for member in members[:15]:  # Show top 15
                if isinstance(member, dict):
                    name = member.get("name", "")
                    pos = member.get("position", "")
                    ministry = member.get("ministry", "")
                    
                    if pos and ministry:
                        lines.append(f"  • {name} - {pos}, {ministry}")
                    elif pos:
                        lines.append(f"  • {name} - {pos}")
                    else:
                        lines.append(f"  • {name}")
                else:
                    lines.append(f"  • {member}")
            
            if len(members) > 15:
                lines.append(f"  ...และอีก {len(members) - 15} คน")
    
    # Handle organization search
    elif intent_type == "organization_search":
        org = result.get("organization", result.get("org_name", ""))
        people = result.get("people", result.get("members", []))
        
        lines.append(f"บุคคลใน {org}:")
        for person in people[:20]:
            if isinstance(person, dict):
                name = person.get("name", "")
                pos = person.get("position", "")
                lines.append(f"  • {name}" + (f" - {pos}" if pos else ""))
            else:
                lines.append(f"  • {person}")
        
        if len(people) > 20:
            lines.append(f"  ...และอีก {len(people) - 20} คน")
    
    # Default: return formatted result
    else:
        return format_network_result(smart_query_result)
    
    return "\n".join(lines) if lines else format_network_result(smart_query_result)


def format_network_result(smart_query_result: Dict) -> str:
    """
    Generic formatter that converts Neo4j query results into readable context for LLM.
    Uses data-driven approach - formats all available data with details.
    """
    result = smart_query_result.get("result", {})
    
    # If no result or not found, return the message
    if not result:
        return "ไม่พบข้อมูล"
    if result.get("found") == False:
        return result.get("message", "ไม่พบข้อมูล")
    
    lines = []
    
    # Helper to format a person/entity with ALL their details
    def format_entity_full(data, indent=""):
        """Format entity with full details on multiple lines"""
        if isinstance(data, str):
            return f"{indent}• {data}"
        if not isinstance(data, dict):
            return f"{indent}• {str(data)}"
        
        name = data.get("name", data.get("full_name", "Unknown"))
        result_lines = [f"{indent}• {name}"]
        
        if data.get("position"):
            result_lines.append(f"{indent}  ตำแหน่ง: {data['position']}")
        if data.get("ministry"):
            result_lines.append(f"{indent}  กระทรวง: {data['ministry']}")
        if data.get("agency"):
            result_lines.append(f"{indent}  หน่วยงาน: {data['agency']}")
        if data.get("connections"):
            result_lines.append(f"{indent}  Connections: {data['connections']}")
        if data.get("role_info"):
            result_lines.append(f"{indent}  ข้อมูล: {data['role_info']}")
        if data.get("recommendation"):
            result_lines.append(f"{indent}  แนะนำ: {data['recommendation']}")
        
        return "\n".join(result_lines)
    
    def format_entity_inline(data):
        """Format entity in single line for lists"""
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            return str(data)
        
        name = data.get("name", data.get("full_name", "Unknown"))
        details = []
        if data.get("position"):
            details.append(data['position'])
        if data.get("ministry"):
            details.append(f"กระทรวง{data['ministry']}")
        elif data.get("agency"):
            details.append(data['agency'])
        if data.get("connections"):
            details.append(f"{data['connections']} connections")
        
        if details:
            return f"{name} ({', '.join(details)})"
        return name
    
    # Add from/to context first
    if result.get("from") and result.get("to"):
        lines.append(f"จาก: {result['from']} ไปยัง: {result['to']}")
    
    # Process all fields generically
    for key, value in result.items():
        if key in ["found", "query", "intent", "from", "to"]:
            continue
        if value is None or (isinstance(value, list) and len(value) == 0):
            continue
        
        if isinstance(value, list):
            if key == "path":
                from_name = result.get("from", "")
                path_str = " → ".join(str(p) for p in value)
                if from_name and not path_str.startswith(str(from_name)):
                    path_str = f"{from_name} → {path_str}"
                lines.append(f"\nเส้นทาง: {path_str}")
            elif key == "relationships":
                lines.append(f"ความสัมพันธ์: {' → '.join(str(r) for r in value)}")
            elif key in ["members", "mutuals", "person_details", "introducers", "connections"]:
                # Format people/entities with full details
                lines.append(f"\n{key} ({len(value)} คน):")
                for item in value[:30]:
                    lines.append(format_entity_full(item, "  "))
                if len(value) > 30:
                    lines.append(f"  ...และอีก {len(value) - 30} คน")
            else:
                lines.append(f"{key}: {', '.join(str(v) for v in value[:10])}")
        
        elif isinstance(value, dict):
            if key in ["best_connector", "best_introducer"]:
                lines.append(f"\n★ {key}:")
                lines.append(format_entity_full(value, "  "))
            elif key == "by_relationship":
                lines.append(f"\nความสัมพันธ์แยกตามประเภท:")
                for rel_type, rel_data in value.items():
                    count = rel_data.get('count', 0)
                    conns = rel_data.get('connections', [])
                    lines.append(f"  {rel_type}: {count} ({', '.join(conns[:5])})")
            else:
                lines.append(f"{key}: {format_entity_inline(value)}")
        
        elif isinstance(value, (int, float)):
            if key == "distance":
                lines.append(f"ระยะทาง: {value} ขั้นตอน")
            elif key in ["member_count", "mutual_count", "total_connections"]:
                lines.append(f"จำนวน: {value}")
            else:
                lines.append(f"{key}: {value}")
        
        elif isinstance(value, str):
            if key in ["network", "organization", "cohort", "person", "network_type"]:
                lines.append(f"{key}: {value}")
            elif key in ["message", "description"]:
                lines.append(value)
            else:
                lines.append(f"{key}: {value}")
    
    return "\n".join(lines) if lines else str(result)


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

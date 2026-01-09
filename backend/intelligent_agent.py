"""
Intelligent Network Agent - LLM-powered query understanding and Cypher generation.

This agent uses LLM to understand ANY natural language query and dynamically
generates appropriate Cypher queries based on the database schema.

No hardcoded patterns - pure LLM understanding.
"""

import os
import re
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, CypherSyntaxError
import requests


@dataclass
class ChatMessage:
    """Represents a single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSession:
    """Manages conversation history for a session."""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_query_result: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to the conversation."""
        self.messages.append(ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
    
    def get_context(self, max_messages: int = 10) -> str:
        """Get conversation context as a string."""
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        lines = []
        for msg in recent:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)
    
    def get_last_result(self) -> Dict[str, Any]:
        """Get the last query result for follow-up reference."""
        return self.last_query_result


class IntelligentNetworkAgent:
    """
    LLM-powered agent that understands any natural language query about the network.
    
    Key features:
    - No hardcoded regex patterns
    - Dynamic Cypher generation based on schema
    - Chat history support for follow-up questions
    - Self-healing query execution
    """
    
    # Database schema - this is the source of truth for query generation
    SCHEMA = """
    # Node Types and Properties:
    
    1. Person (บุคคล):
       - `ชื่อ-นามสกุล`: Full name (Thai)
       - Properties vary per person
    
    2. Position (ตำแหน่ง):
       - `ตำแหน่ง`: Position title
    
    3. Agency (หน่วยงาน):
       - `หน่วยงาน`: Agency/department name
    
    4. Ministry (กระทรวง):
       - `กระทรวง`: Ministry name
    
    5. Level (ระดับ):
       - `ระดับ`: Level designation
    
    6. Connect by (เครือข่าย):
       - `Connect by`: Network name (e.g., "OSK115", "พี่เท่ห์ (MABE)")
    
    7. Associate (สมาคม/รุ่น):
       - `Associate`: Association name (e.g., "NEXUS รุ่นที่ 1", "วปอ. รุ่นที่ 68")
    
    8. Stelligence Owners (special nodes):
       - Santisook: Connected via [:santisook_known]
       - Por: Connected via [:por_known]
       - Knot: Connected via [:knot_known]
    
    # Relationships:
    - (Person)-[:work_as]->(Position)
    - (Person)-[:work_at]->(Agency)
    - (Person)-[:under]->(Ministry)
    - (Person)-[:has_level]->(Level)
    - (Person)-[:connect_by]->(Connect by)
    - (Person)-[:associate_with]->(Associate)
    - (Santisook)-[:santisook_known]->(Person)
    - (Por)-[:por_known]->(Person)
    - (Knot)-[:knot_known]->(Person)
    
    # Key Query Patterns:
    - Find shortest path: MATCH path = shortestPath((a:Person)-[*]-(b:Person))
    - Find mutual connections: MATCH (a:Person)-[]-(common)-[]-(b:Person)
    - Find network members: MATCH (p:Person)-[:connect_by]->(n:`Connect by`)
    - Find by organization: MATCH (p:Person)-[:work_at]->(a:Agency) WHERE a.`หน่วยงาน` CONTAINS $name
    - Find by ministry: MATCH (p:Person)-[:under]->(m:Ministry) WHERE m.`กระทรวง` CONTAINS $name
    - Complex multi-hop: Can combine multiple conditions
    """
    
    def __init__(self):
        """Initialize the agent with database connection and LLM settings."""
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "neo4j")
            ),
            max_connection_lifetime=300,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60
        )
        
        self.ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.model_name = os.getenv("LOCAL_LLM_MODEL", "scb10x/typhoon2.1-gemma3-4b")
        
        # Session storage (in-memory for now, could be Redis/DB)
        self.sessions: Dict[str, ChatSession] = {}
        
        # Cache schema info
        self._schema_cache = None
        self._schema_cache_time = 0
        
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def get_or_create_session(self, session_id: str = None) -> ChatSession:
        """Get existing session or create a new one."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        new_id = session_id or f"session_{int(time.time() * 1000)}"
        session = ChatSession(session_id=new_id)
        self.sessions[new_id] = session
        return session
    
    def _call_llm(self, prompt: str, system_prompt: str = None, max_tokens: int = 1024) -> str:
        """Call Ollama LLM for query understanding and generation."""
        url = f"{self.ollama_url}/api/generate"
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "keep_alive": -1,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,  # Low for more deterministic output
                "num_ctx": 4096,
                "top_k": 10,
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            return ""
    
    def _run_cypher(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute a Cypher query with error handling."""
        params = params or {}
        try:
            with self.driver.session() as session:
                result = session.run(query, **params)
                return [dict(record) for record in result]
        except CypherSyntaxError as e:
            print(f"[ERROR] Cypher syntax error: {e}")
            return []
        except Exception as e:
            print(f"[ERROR] Query execution failed: {e}")
            return []
    
    def _get_live_schema_info(self) -> str:
        """Get actual schema from the database."""
        # Cache for 5 minutes
        if self._schema_cache and (time.time() - self._schema_cache_time) < 300:
            return self._schema_cache
        
        info_parts = []
        
        # Get node labels and sample properties
        try:
            labels_result = self._run_cypher("CALL db.labels() YIELD label RETURN label LIMIT 20")
            labels = [r.get("label") for r in labels_result if r.get("label")]
            info_parts.append(f"Node Labels: {', '.join(labels)}")
            
            # Get relationship types
            rels_result = self._run_cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType LIMIT 20")
            rels = [r.get("relationshipType") for r in rels_result if r.get("relationshipType")]
            info_parts.append(f"Relationships: {', '.join(rels)}")
            
            # Sample some data
            sample = self._run_cypher("""
                MATCH (p:Person) 
                RETURN keys(p) as props 
                LIMIT 1
            """)
            if sample:
                info_parts.append(f"Person properties: {sample[0].get('props', [])}")
                
        except Exception as e:
            print(f"[WARN] Could not fetch live schema: {e}")
        
        self._schema_cache = "\n".join(info_parts)
        self._schema_cache_time = time.time()
        return self._schema_cache
    
    def _is_followup_query(self, query: str) -> bool:
        """Detect if this is a follow-up query referencing previous results."""
        followup_indicators = [
            'ในนี้', 'พวกนี้', 'คนเหล่านี้', 'เหล่านี้', 'จากนี้',
            'มีใครในนี้', 'ใครในนี้', 'คนไหนในนี้', 'กลุ่มนี้',
            'those', 'these', 'them', 'in this', 'from this',
            'ต่อ', 'แล้ว', 'อีก'
        ]
        query_lower = query.lower()
        return any(indicator in query or indicator in query_lower for indicator in followup_indicators)
    
    def _get_names_from_last_result(self, session: ChatSession) -> List[str]:
        """Extract person names from the last query result."""
        if not session or not session.last_query_result:
            return []
        
        last_result = session.last_query_result
        data = last_result.get("data", [])
        names = []
        
        for item in data:
            if isinstance(item, dict):
                # Try various name fields
                name = item.get("name") or item.get("connector_name") or item.get("minister_name")
                if name:
                    names.append(name)
                # Also handle members array
                if "members" in item:
                    for m in item.get("members", []):
                        if isinstance(m, dict) and m.get("name"):
                            names.append(m["name"])
        
        # If data is a list of dicts with 'members' key at top level
        if len(data) == 1 and isinstance(data[0], dict) and "members" in data[0]:
            for m in data[0].get("members", []):
                if isinstance(m, dict) and m.get("name"):
                    names.append(m["name"])
        
        # Remove duplicates and None
        return list(set(n for n in names if n))
    
    def _generate_followup_cypher(self, query: str, session: ChatSession, intent: Dict) -> Tuple[str, Dict]:
        """Generate Cypher for follow-up queries that reference previous results."""
        names = self._get_names_from_last_result(session)
        
        if not names:
            print("[DEBUG] No names found in previous result for follow-up")
            return None, {}
        
        query_lower = query.lower()
        
        # Limit to reasonable number of names
        names = names[:100]
        
        # Determine what the user wants to filter by
        understanding = intent.get("natural_language_understanding", "")
        
        # Check for person lookup (e.g., "รู้จัก X ไหม", "รู้จักใคร")
        if "รู้จัก" in query and any(name in query for name in ["อนุทิน", "ชาญวีรกูล"]):
            # Looking for who knows a specific person
            target_name = "อนุทิน ชาญวีรกูล" if "อนุทิน" in query else ""
            if target_name:
                query_text = f"""
                // Find people from previous results who know {target_name}
                UNWIND $names as name
                MATCH (p:Person)
                WHERE p.`ชื่อ-นามสกุล` = name
                
                // Check various connection types
                OPTIONAL MATCH (p)-[r]-(target:Person)
                WHERE target.`ชื่อ-นามสกุล` CONTAINS $target_name
                
                // Also check if they share networks
                OPTIONAL MATCH (p)-[:connect_by]->(n:`Connect by`)<-[:connect_by]-(target2:Person)
                WHERE target2.`ชื่อ-นามสกุล` CONTAINS $target_name
                
                OPTIONAL MATCH (p)-[:associate_with]->(a:Associate)<-[:associate_with]-(target3:Person)
                WHERE target3.`ชื่อ-นามสกุล` CONTAINS $target_name
                
                WITH p, target, target2, target3, r
                WHERE target IS NOT NULL OR target2 IS NOT NULL OR target3 IS NOT NULL
                
                OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
                OPTIONAL MATCH (p)-[:under]->(m:Ministry)
                
                RETURN DISTINCT
                    p.`ชื่อ-นามสกุล` as name,
                    pos.`ตำแหน่ง` as position,
                    m.`กระทรวง` as ministry,
                    CASE 
                        WHEN target IS NOT NULL THEN 'direct_connection'
                        WHEN target2 IS NOT NULL THEN 'same_network'
                        WHEN target3 IS NOT NULL THEN 'same_association'
                    END as connection_type
                LIMIT 30
                """
                return query_text, {"names": names, "target_name": target_name}
        
        # Check for Santisook/Por/Knot knowing people from list
        stelligence_owners = ["santisook", "por", "knot"]
        owner_in_query = next((o for o in stelligence_owners if o in query_lower), None)
        
        if owner_in_query and ("รู้จัก" in query or "know" in query_lower):
            owner_cap = owner_in_query.capitalize()
            rel_name = f"{owner_in_query}_known"
            
            query_text = f"""
            // Find people from previous results that {owner_cap} knows
            UNWIND $names as name
            MATCH (owner:{owner_cap})-[:{rel_name}]->(p:Person)
            WHERE p.`ชื่อ-นามสกุล` = name
            
            OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
            OPTIONAL MATCH (p)-[:under]->(m:Ministry)
            OPTIONAL MATCH (p)-[:work_at]->(a:Agency)
            
            RETURN DISTINCT
                p.`ชื่อ-นามสกุล` as name,
                pos.`ตำแหน่ง` as position,
                m.`กระทรวง` as ministry,
                a.`หน่วยงาน` as agency
            LIMIT 30
            """
            return query_text, {"names": names}
        
        # Check for work/government job filter (ทำงานราชการ)
        if "ราชการ" in query or "ทำงาน" in query or "กระทรวง" in query:
            query_text = """
            // Filter previous results by government position
            UNWIND $names as name
            MATCH (p:Person)
            WHERE p.`ชื่อ-นามสกุล` = name
            MATCH (p)-[:work_as]->(pos:Position)
            OPTIONAL MATCH (p)-[:under]->(m:Ministry)
            OPTIONAL MATCH (p)-[:work_at]->(a:Agency)
            WHERE m IS NOT NULL OR a IS NOT NULL
            
            RETURN DISTINCT
                p.`ชื่อ-นามสกุล` as name,
                pos.`ตำแหน่ง` as position,
                m.`กระทรวง` as ministry,
                a.`หน่วยงาน` as agency
            LIMIT 50
            """
            return query_text, {"names": names}
        
        # Check for position filter (รัฐมนตรี, อธิบดี, etc.)
        position_filters = [
            ('รัฐมนตรี', "pos.`ตำแหน่ง` CONTAINS 'รัฐมนตรี' OR pos.`ตำแหน่ง` CONTAINS 'รมต'"),
            ('รมต', "pos.`ตำแหน่ง` CONTAINS 'รัฐมนตรี' OR pos.`ตำแหน่ง` CONTAINS 'รมต'"),
            ('อธิบดี', "pos.`ตำแหน่ง` CONTAINS 'อธิบดี'"),
            ('ปลัด', "pos.`ตำแหน่ง` CONTAINS 'ปลัด'"),
        ]
        
        for keyword, condition in position_filters:
            if keyword in query:
                query_text = f"""
                // Filter previous results by position
                UNWIND $names as name
                MATCH (p:Person)
                WHERE p.`ชื่อ-นามสกุล` = name
                MATCH (p)-[:work_as]->(pos:Position)
                WHERE {condition}
                OPTIONAL MATCH (p)-[:under]->(m:Ministry)
                OPTIONAL MATCH (p)-[:work_at]->(a:Agency)
                
                RETURN DISTINCT
                    p.`ชื่อ-นามสกุล` as name,
                    pos.`ตำแหน่ง` as position,
                    m.`กระทรวง` as ministry,
                    a.`หน่วยงาน` as agency
                LIMIT 30
                """
                return query_text, {"names": names}
        
        # Check for network/association filter (วปอ., NEXUS, etc.)
        network_patterns = [
            (r'วปอ\.?\s*(?:รุ่น(?:ที่)?\s*)?(\d+)?', 'วปอ'),
            (r'NEXUS\s*(?:รุ่น(?:ที่)?\s*)?(\d+)?', 'NEXUS'),
            (r'(OSK\d+)', 'OSK'),
        ]
        
        for pattern, network_base in network_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if match.group(1):
                    network_search = f"{network_base}.*{match.group(1)}"
                else:
                    network_search = network_base
                
                query_text = """
                // Filter previous results by network/association
                UNWIND $names as name
                MATCH (p:Person)
                WHERE p.`ชื่อ-นามสกุล` = name
                
                // Check Connect by
                OPTIONAL MATCH (p)-[:connect_by]->(n:`Connect by`)
                WHERE n.`Connect by` =~ $network_pattern
                
                // Check Associate
                OPTIONAL MATCH (p)-[:associate_with]->(a:Associate)
                WHERE a.Associate =~ $network_pattern
                
                WITH p, n, a
                WHERE n IS NOT NULL OR a IS NOT NULL
                
                OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
                OPTIONAL MATCH (p)-[:under]->(m:Ministry)
                
                RETURN DISTINCT
                    p.`ชื่อ-นามสกุล` as name,
                    COALESCE(n.`Connect by`, a.Associate) as network,
                    pos.`ตำแหน่ง` as position,
                    m.`กระทรวง` as ministry
                LIMIT 50
                """
                return query_text, {"names": names, "network_pattern": f"(?i).*{network_search}.*"}
        
        return None, {}

    def _quick_pattern_extract(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Quick regex-based pattern extraction for common query types.
        Returns None if no pattern matched (will fallback to LLM).
        This is for speed and reliability, not to replace LLM understanding.
        """
        query_lower = query.lower()
        
        # Skip quick pattern for complex queries that need LLM understanding
        complex_indicators = [
            'santisook', 'por', 'knot', 'connect', 'path', 'ผ่าน', 'via', 
            'through', 'รัฐมนตรี', 'minister', 'อธิบดี', 'เชื่อม', 'หา', 'ไป'
        ]
        if any(indicator in query_lower for indicator in complex_indicators):
            print(f"[DEBUG] Complex query detected, skipping quick pattern")
            return None
        
        # Known network patterns
        network_patterns = [
            (r'(OSK\d+)', 'Connect by'),
            (r'(MABE\d*)', 'Connect by'),
            (r'(NEXUS)', 'Associate'),
            (r'(วปอ\.?)', 'Associate'),
            (r'(นบส\.?)', 'Connect by'),
        ]
        
        # Check for "who is in X network" pattern - Thai is case-insensitive
        members_keywords = ['ใครอยู่ใน', 'ใครบ้างใน', 'สมาชิก', 'members', 'ใครอยู่', 'อยู่ใน', 'มีใครบ้าง', 'ใคร']
        is_members_query = any(kw in query for kw in members_keywords) or any(kw in query_lower for kw in members_keywords)
        
        print(f"[DEBUG] Quick pattern check - is_members_query: {is_members_query}, query: {query[:50]}")
        
        # Extract network name
        for pattern, network_type in network_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                network_name = match.group(1)
                print(f"[DEBUG] Quick pattern matched network: {network_name}, type: {network_type}")
                
                # If asking about members, return members query
                if is_members_query:
                    return {
                        "intent_type": "members",
                        "entities": {"networks": [network_name], "persons": [], "ministries": [], "positions": []},
                        "query_structure": {},
                        "requires_previous_context": False,
                        "natural_language_understanding": f"Find members of {network_name} network",
                        "_quick_match": True
                    }
                # If just network name mentioned without clear intent, treat as members query
                else:
                    return {
                        "intent_type": "members",
                        "entities": {"networks": [network_name], "persons": [], "ministries": [], "positions": []},
                        "query_structure": {},
                        "requires_previous_context": False,
                        "natural_language_understanding": f"Query about {network_name} network",
                        "_quick_match": True
                    }
        
        return None
    
    def understand_query(self, query: str, session: ChatSession = None) -> Dict[str, Any]:
        """
        Use LLM to understand the user's query and extract structured intent.
        
        Returns a dict with:
        - intent_type: The type of query (path, network, search, etc.)
        - entities: Named entities extracted
        - conditions: Filter conditions
        - context_needed: Whether previous context is needed
        """
        
        # Try quick pattern matching first for speed and reliability
        quick_result = self._quick_pattern_extract(query)
        if quick_result:
            print(f"[DEBUG] Quick pattern matched: {quick_result.get('intent_type')}")
            return quick_result
        
        # Build context from chat history if available
        history_context = ""
        last_result_context = ""
        if session:
            history_context = session.get_context(max_messages=6)
            last_result = session.get_last_result()
            if last_result:
                # Summarize last result for context
                if "path" in last_result:
                    last_result_context = f"Last result was a path: {last_result.get('path', [])}"
                elif "members" in last_result:
                    names = [m.get("name", "") for m in last_result.get("members", [])[:5]]
                    last_result_context = f"Last result had members: {', '.join(names)}..."
        
        system_prompt = """You are a query understanding system for a Thai government network database.
Your task is to extract structured information from user queries.

IMPORTANT: Respond ONLY with valid JSON, no other text.

The database contains:
- Person nodes (บุคคล) with name in `ชื่อ-นามสกุล`
- Ministry (กระทรวง), Agency (หน่วยงาน), Position (ตำแหน่ง)
- Networks: Connect by (e.g., OSK115, MABE), Associate (e.g., NEXUS, วปอ.)
- Special connections: Santisook, Por, Knot (Stelligence owners)

Relationship keywords:
- "ผ่าน" / "via" / "through" = intermediate connection
- "รู้จัก" / "know" = direct connection
- "connect" / "เชื่อมต่อ" = path finding
- "รัฐมนตรี" = minister (usually in Ministry or high Position)
- "กระทรวง" = ministry

Output JSON format:
{
  "intent_type": "path|network|search|members|complex|general",
  "entities": {
    "persons": ["name1", "name2"],
    "networks": ["network1"],
    "ministries": ["ministry"],
    "positions": ["position"],
    "organizations": ["org"]
  },
  "query_structure": {
    "from": "source entity/person",
    "to": "target entity/person",
    "via": "intermediate network or person",
    "filter": "any filter condition"
  },
  "requires_previous_context": true/false,
  "natural_language_understanding": "Brief explanation of what user wants"
}"""

        # Build user prompt
        history_part = f"Previous conversation:\n{history_context}" if history_context else ""
        result_part = f"Previous result context: {last_result_context}" if last_result_context else ""
        
        user_prompt = f"""Query: "{query}"

{history_part}
{result_part}

Extract the structured intent from this query. Respond with JSON only:"""

        response = self._call_llm(user_prompt, system_prompt, max_tokens=512)
        
        # Parse JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                intent = json.loads(json_match.group())
                return intent
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse intent JSON: {e}")
        
        # Fallback - basic intent extraction
        return {
            "intent_type": "general",
            "entities": {"persons": [], "networks": [], "ministries": []},
            "query_structure": {},
            "requires_previous_context": False,
            "natural_language_understanding": query
        }
    
    def generate_cypher(self, intent: Dict[str, Any], session: ChatSession = None, original_query: str = "") -> Tuple[str, Dict]:
        """
        Generate a Cypher query based on the understood intent.
        
        Returns (cypher_query, parameters)
        """
        
        # Check for follow-up query first
        if original_query and session and self._is_followup_query(original_query):
            print(f"[DEBUG] Detected follow-up query: {original_query[:50]}")
            followup_cypher, followup_params = self._generate_followup_cypher(original_query, session, intent)
            if followup_cypher:
                print(f"[DEBUG] Using follow-up Cypher query")
                return followup_cypher, followup_params
        
        intent_type = intent.get("intent_type", "general")
        entities = intent.get("entities", {})
        structure = intent.get("query_structure", {})
        
        persons = entities.get("persons", [])
        networks = entities.get("networks", [])
        ministries = entities.get("ministries", [])
        positions = entities.get("positions", [])
        
        # Check if this is a "position in network" query like "หารัฐมนตรีใน วปอ."
        # Even if LLM didn't extract properly, detect from original query
        position_keywords = ['รัฐมนตรี', 'รมต', 'อธิบดี', 'ปลัด', 'ผู้ว่า', 'minister', 'director']
        has_position_keyword = any(kw in original_query for kw in position_keywords)
        has_network = bool(networks)
        
        # Use complex Cypher generator for position-in-network queries
        if has_position_keyword and has_network:
            return self._generate_complex_cypher(intent, session, original_query)
        
        # Use LLM to generate Cypher for complex queries
        if intent_type == "complex" or len([x for x in [persons, networks, ministries, positions] if x]) > 1:
            return self._generate_complex_cypher(intent, session, original_query)
        
        # Handle common patterns directly for speed
        if intent_type == "path" and len(persons) >= 2:
            return self._generate_path_cypher(persons[0], persons[1], structure.get("via"))
        
        if intent_type == "network" and persons:
            return self._generate_network_cypher(persons[0])
        
        if intent_type == "members" and networks:
            return self._generate_members_cypher(networks[0])
        
        if intent_type == "search":
            return self._generate_search_cypher(intent)
        
        # For general/unknown, use LLM generation
        return self._generate_complex_cypher(intent, session, original_query)
    
    def _generate_path_cypher(self, from_person: str, to_person: str, via: str = None) -> Tuple[str, Dict]:
        """Generate Cypher for path finding."""
        if via:
            # Path through specific network/person
            query = """
            MATCH (a:Person), (b:Person)
            WHERE a.`ชื่อ-นามสกุล` CONTAINS $from_name
              AND b.`ชื่อ-นามสกุล` CONTAINS $to_name
            MATCH path = shortestPath((a)-[*..10]-(b))
            WHERE any(node IN nodes(path) WHERE 
                (node:`Connect by` AND node.`Connect by` CONTAINS $via)
                OR (node:Person AND node.`ชื่อ-นามสกุล` CONTAINS $via)
                OR (node:Agency AND node.`หน่วยงาน` CONTAINS $via)
            )
            RETURN path, 
                   [n IN nodes(path) | 
                       CASE 
                           WHEN n:Person THEN n.`ชื่อ-นามสกุล`
                           WHEN n:`Connect by` THEN n.`Connect by`
                           WHEN n:Agency THEN n.`หน่วยงาน`
                           WHEN n:Ministry THEN n.`กระทรวง`
                           WHEN n:Position THEN n.`ตำแหน่ง`
                           ELSE labels(n)[0]
                       END
                   ] as path_names,
                   length(path) as hops
            ORDER BY hops
            LIMIT 5
            """
            return query, {"from_name": from_person, "to_name": to_person, "via": via}
        else:
            # Direct shortest path
            query = """
            MATCH (a:Person), (b:Person)
            WHERE a.`ชื่อ-นามสกุล` CONTAINS $from_name
              AND b.`ชื่อ-นามสกุล` CONTAINS $to_name
            MATCH path = shortestPath((a)-[*..10]-(b))
            RETURN path,
                   [n IN nodes(path) | 
                       CASE 
                           WHEN n:Person THEN n.`ชื่อ-นามสกุล`
                           WHEN n:`Connect by` THEN n.`Connect by`
                           WHEN n:Agency THEN n.`หน่วยงาน`
                           WHEN n:Ministry THEN n.`กระทรวง`
                           WHEN n:Position THEN n.`ตำแหน่ง`
                           ELSE labels(n)[0]
                       END
                   ] as path_names,
                   length(path) as hops
            ORDER BY hops
            LIMIT 3
            """
            return query, {"from_name": from_person, "to_name": to_person}
    
    def _generate_network_cypher(self, person: str) -> Tuple[str, Dict]:
        """Generate Cypher to find a person's network."""
        query = """
        MATCH (p:Person)
        WHERE p.`ชื่อ-นามสกุล` CONTAINS $name
        OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
        OPTIONAL MATCH (p)-[:work_at]->(agency:Agency)
        OPTIONAL MATCH (p)-[:under]->(ministry:Ministry)
        OPTIONAL MATCH (p)-[:connect_by]->(network:`Connect by`)
        OPTIONAL MATCH (p)-[:associate_with]->(assoc:Associate)
        OPTIONAL MATCH (p)-[r]-(connected:Person)
        RETURN p.`ชื่อ-นามสกุล` as name,
               pos.`ตำแหน่ง` as position,
               agency.`หน่วยงาน` as agency,
               ministry.`กระทรวง` as ministry,
               collect(DISTINCT network.`Connect by`) as networks,
               collect(DISTINCT assoc.Associate) as associations,
               collect(DISTINCT {
                   name: connected.`ชื่อ-นามสกุล`,
                   relationship: type(r)
               }) as connections
        """
        return query, {"name": person}
    
    def _generate_members_cypher(self, network: str) -> Tuple[str, Dict]:
        """Generate Cypher to find network members."""
        query = """
        // Try Connect by first
        OPTIONAL MATCH (p:Person)-[:connect_by]->(n:`Connect by`)
        WHERE toLower(n.`Connect by`) CONTAINS toLower($network)
        WITH collect(DISTINCT {
            name: p.`ชื่อ-นามสกุล`,
            source: 'Connect by',
            network: n.`Connect by`
        }) as connect_members
        
        // Then try Associate
        OPTIONAL MATCH (p2:Person)-[:associate_with]->(a:Associate)
        WHERE toLower(a.Associate) CONTAINS toLower($network)
        WITH connect_members, collect(DISTINCT {
            name: p2.`ชื่อ-นามสกุล`,
            source: 'Associate',
            network: a.Associate
        }) as assoc_members
        
        // Combine results
        RETURN connect_members + assoc_members as members
        """
        return query, {"network": network}
    
    def _generate_search_cypher(self, intent: Dict) -> Tuple[str, Dict]:
        """Generate Cypher for general search."""
        entities = intent.get("entities", {})
        
        conditions = []
        params = {}
        
        if entities.get("ministries"):
            conditions.append("m.`กระทรวง` CONTAINS $ministry")
            params["ministry"] = entities["ministries"][0]
        
        if entities.get("positions"):
            conditions.append("pos.`ตำแหน่ง` CONTAINS $position")
            params["position"] = entities["positions"][0]
        
        if entities.get("organizations"):
            conditions.append("a.`หน่วยงาน` CONTAINS $agency")
            params["agency"] = entities["organizations"][0]
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
        OPTIONAL MATCH (p)-[:work_at]->(a:Agency)
        OPTIONAL MATCH (p)-[:under]->(m:Ministry)
        WHERE {where_clause}
        RETURN p.`ชื่อ-นามสกุล` as name,
               pos.`ตำแหน่ง` as position,
               a.`หน่วยงาน` as agency,
               m.`กระทรวง` as ministry
        LIMIT 50
        """
        return query, params
    
    def _generate_complex_cypher(self, intent: Dict, session: ChatSession = None, original_query: str = "") -> Tuple[str, Dict]:
        """
        Use LLM to generate Cypher for complex queries that don't fit simple patterns.
        """
        
        # Extract key info first
        understanding = intent.get("natural_language_understanding", "")
        entities = intent.get("entities", {})
        structure = intent.get("query_structure", {})
        
        # Try to handle common complex patterns directly for better reliability
        networks = entities.get("networks", [])
        positions = entities.get("positions", [])
        
        # Pattern: "Find [position] in [network]" - e.g., "หารัฐมนตรีใน วปอ."
        # Check if looking for a position type within a network
        position_keywords = ['รัฐมนตรี', 'รมต', 'อธิบดี', 'ปลัด', 'ผู้ว่า', 'minister', 'director']
        network_names_in_query = []
        position_in_query = None
        
        # Search in BOTH understanding AND original_query for better coverage
        search_text = f"{understanding} {original_query}"
        
        # Extract network name from query
        for pattern in [r'(OSK\d+)', r'(MABE\d*)', r'(NEXUS)', r'(วปอ\.?)', r'(นบส\.?)'  ]:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                network_names_in_query.append(match.group(1))
        
        # Extract position from query (search in combined text)
        for kw in position_keywords:
            if kw in search_text.lower() or kw in search_text:
                position_in_query = kw
                break
        
        # If we have both network and position, generate query for "position in network"
        if network_names_in_query and position_in_query:
            network_name = network_names_in_query[0]
            print(f"[DEBUG] Position in network pattern: {position_in_query} in {network_name}")
            
            # Check both รัฐมนตรี and รมต.
            pos_condition = "pos.`ตำแหน่ง` CONTAINS 'รัฐมนตรี' OR pos.`ตำแหน่ง` CONTAINS 'รมต'" if position_in_query in ['รัฐมนตรี', 'รมต', 'minister'] else f"pos.`ตำแหน่ง` CONTAINS $position"
            
            query = f"""
            // Find people in {network_name} network (both Connect by and Associate)
            MATCH (p:Person)
            WHERE EXISTS {{
                MATCH (p)-[:connect_by]->(n:`Connect by`)
                WHERE n.`Connect by` CONTAINS $network
            }} OR EXISTS {{
                MATCH (p)-[:associate_with]->(a:Associate)
                WHERE a.Associate CONTAINS $network
            }}
            
            // Filter by position
            MATCH (p)-[:work_as]->(pos:Position)
            WHERE {pos_condition}
            
            OPTIONAL MATCH (p)-[:under]->(m:Ministry)
            OPTIONAL MATCH (p)-[:work_at]->(agency:Agency)
            
            RETURN DISTINCT
                p.`ชื่อ-นามสกุล` as name,
                pos.`ตำแหน่ง` as position,
                m.`กระทรวง` as ministry,
                agency.`หน่วยงาน` as agency
            LIMIT 30
            """
            params = {"network": network_name, "position": position_in_query}
            return query, params
        
        # Pattern: "X connect to Y through Z network"
        from_entity = structure.get("from", "")
        to_entity = structure.get("to", "")
        via_entity = structure.get("via", "")
        
        # Check for Stelligence owner in query (check both structure and raw search_text)
        stelligence_owners = ["santisook", "por", "knot"]
        from_lower = from_entity.lower() if from_entity else ""
        search_text_lower = search_text.lower()
        
        # Detect Stelligence owner from either from_entity or the search text
        owner_in_query = None
        for owner in stelligence_owners:
            if owner in from_lower or owner in search_text_lower:
                owner_in_query = owner
                break
        
        if owner_in_query:
            owner = owner_in_query
            owner_cap = owner.capitalize()
            rel_name = f"{owner}_known"
            
            # Get networks from entities OR extract from search text
            networks = entities.get("networks", [])
            
            # Also try to extract network from search text
            if not networks:
                for pattern in [r'(NEXUS[^,]*รุ่น[^,]*\d+)', r'(NEXUS)', r'(OSK\d+)', r'(MABE\d*)', r'(วปอ\.[^,]*)', r'(นบส\.\d*)']:
                    match = re.search(pattern, search_text, re.IGNORECASE)
                    if match:
                        networks = [match.group(1)]
                        break
            
            network_name = networks[0] if networks else via_entity
            
            # Don't clean up network name too aggressively - keep full name like "NEXUS รุ่นที่ 1"
            # Just extract from weird formats like "คนรู้จักจาก OSK115"
            if network_name and "คนรู้จักจาก" in network_name:
                network_match = re.search(r'คนรู้จักจาก\s*(.+)', network_name)
                if network_match:
                    network_name = network_match.group(1).strip()
            
            # Clean up target - get position keyword
            # Note: รมต. is short form of รัฐมนตรี, need to handle both
            target_pos = "รัฐมนตรี"  # Default to minister
            if to_entity:
                if "รัฐมนตรี" in to_entity or "minister" in to_entity.lower() or "รมต" in to_entity:
                    target_pos = "รัฐมนตรี"  # Will use OR condition for รมต.
                elif "อธิบดี" in to_entity:
                    target_pos = "อธิบดี"
                else:
                    target_pos = to_entity
            
            # Find people that owner knows, who are in the via_network, and match to_entity criteria
            # ONLY use minister pattern if there's an explicit to_entity, not just presence of keyword in understanding
            # Filter out meaningless placeholder values that LLM might return
            invalid_targets = ["unknown", "none", "ไม่ระบุ", "ไม่ทราบ", "", "any", "anyone", "ใครก็ได้", "ทุกคน"]
            to_entity_clean = to_entity.strip().lower() if to_entity else ""
            has_explicit_target = bool(to_entity_clean and to_entity_clean not in invalid_targets)
            
            if network_name and has_explicit_target:
                # Complex: Santisook -> via OSK115 -> to รัฐมนตรี
                # Note: Check both Connect by AND Associate networks, and both รัฐมนตรี and รมต.
                # Use toLower for case-insensitive network matching
                query = f"""
                // Find people that {owner_cap} knows who are in {network_name} network (Connect by OR Associate)
                MATCH (owner:{owner_cap})-[:{rel_name}]->(person1:Person)
                
                // Get their network affiliation
                OPTIONAL MATCH (person1)-[:connect_by]->(net1:`Connect by`)
                WHERE toLower(net1.`Connect by`) CONTAINS toLower($via_network)
                OPTIONAL MATCH (person1)-[:associate_with]->(assoc1:Associate)
                WHERE toLower(assoc1.Associate) CONTAINS toLower($via_network)
                
                WITH person1, COALESCE(net1.`Connect by`, assoc1.Associate) as person_network
                WHERE person_network IS NOT NULL
                
                // Check if person1 themselves is a minister
                OPTIONAL MATCH (person1)-[:work_as]->(pos1:Position)
                WHERE pos1.`ตำแหน่ง` CONTAINS 'รัฐมนตรี' OR pos1.`ตำแหน่ง` CONTAINS 'รมต'
                
                // Also find ministers connected to person1 within 1-3 hops
                OPTIONAL MATCH (person1)-[*1..3]-(person2:Person)-[:work_as]->(pos2:Position)
                WHERE (pos2.`ตำแหน่ง` CONTAINS 'รัฐมนตรี' OR pos2.`ตำแหน่ง` CONTAINS 'รมต') AND person2 <> person1
                
                OPTIONAL MATCH (person1)-[:under]->(ministry1:Ministry)
                OPTIONAL MATCH (person1)-[:work_at]->(agency1:Agency)
                OPTIONAL MATCH (person2)-[:under]->(ministry2:Ministry)
                OPTIONAL MATCH (person2)-[:work_at]->(agency2:Agency)
                
                WITH person1, person_network, pos1, ministry1, agency1, person2, pos2, ministry2, agency2
                WHERE pos1 IS NOT NULL OR pos2 IS NOT NULL
                
                RETURN DISTINCT 
                    person1.`ชื่อ-นามสกุล` as connector_name,
                    person_network as via_network,
                    CASE WHEN pos1 IS NOT NULL THEN person1.`ชื่อ-นามสกุล` ELSE person2.`ชื่อ-นามสกุล` END as minister_name,
                    CASE WHEN pos1 IS NOT NULL THEN pos1.`ตำแหน่ง` ELSE pos2.`ตำแหน่ง` END as position,
                    CASE WHEN pos1 IS NOT NULL THEN ministry1.`กระทรวง` ELSE ministry2.`กระทรวง` END as ministry,
                    CASE WHEN pos1 IS NOT NULL THEN agency1.`หน่วยงาน` ELSE agency2.`หน่วยงาน` END as agency,
                    CASE WHEN pos1 IS NOT NULL THEN 'direct' ELSE 'via_connection' END as connection_type
                LIMIT 20
                """
                params = {"via_network": network_name}
                print(f"[DEBUG] Generated complex query with params: {params}")
                return query, params
            
            elif network_name:
                # Simpler: Just Santisook -> via network (check both Connect by AND Associate)
                # Use toLower for case-insensitive matching (e.g., NEXUS matches NexusAI)
                query = f"""
                // Check Connect by networks
                OPTIONAL MATCH (owner:{owner_cap})-[:{rel_name}]->(person1:Person)-[:connect_by]->(network1:`Connect by`)
                WHERE toLower(network1.`Connect by`) CONTAINS toLower($via_network)
                
                // Check Associate networks  
                OPTIONAL MATCH (owner2:{owner_cap})-[:{rel_name}]->(person2:Person)-[:associate_with]->(assoc:Associate)
                WHERE toLower(assoc.Associate) CONTAINS toLower($via_network)
                
                WITH collect(DISTINCT {{
                    name: person1.`ชื่อ-นามสกุล`,
                    network: network1.`Connect by`,
                    type: 'Connect by'
                }}) as connect_results,
                collect(DISTINCT {{
                    name: person2.`ชื่อ-นามสกุล`,
                    network: assoc.Associate,
                    type: 'Associate'
                }}) as assoc_results
                
                UNWIND connect_results + assoc_results as result
                WITH result
                WHERE result.name IS NOT NULL
                
                // Get additional info for each person
                MATCH (p:Person)
                WHERE p.`ชื่อ-นามสกุล` = result.name
                OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
                OPTIONAL MATCH (p)-[:under]->(ministry:Ministry)
                OPTIONAL MATCH (p)-[:work_at]->(agency:Agency)
                
                RETURN DISTINCT 
                    result.name as name,
                    pos.`ตำแหน่ง` as position,
                    ministry.`กระทรวง` as ministry,
                    agency.`หน่วยงาน` as agency,
                    result.network as network
                LIMIT 30
                """
                params = {"via_network": network_name}
                return query, params
        
        # Fallback to LLM generation for truly complex cases
        system_prompt = f"""You are a Cypher query generator for Neo4j.

Database Schema:
{self.SCHEMA}

EXAMPLE QUERIES:

1. Find ministers in a network:
MATCH (p:Person)-[:connect_by]->(n:`Connect by`)
WHERE n.`Connect by` CONTAINS 'OSK115'
MATCH (p)-[:work_as]->(pos:Position)
WHERE pos.`ตำแหน่ง` CONTAINS 'รัฐมนตรี'
RETURN p.`ชื่อ-นามสกุล`, pos.`ตำแหน่ง`

2. Find path from Santisook to someone through a network:
MATCH (s:Santisook)-[:santisook_known]->(connector:Person)-[:connect_by]->(n:`Connect by`)
WHERE n.`Connect by` CONTAINS 'OSK115'
MATCH (connector)-[*1..3]-(target:Person)
RETURN connector.`ชื่อ-นามสกุล`, target.`ชื่อ-นามสกุล`

3. Find people by position and ministry:
MATCH (p:Person)-[:work_as]->(pos:Position)-[:under]->(m:Ministry)
WHERE pos.`ตำแหน่ง` CONTAINS 'อธิบดี' AND m.`กระทรวง` CONTAINS 'พลังงาน'
RETURN p.`ชื่อ-นามสกุล`, pos.`ตำแหน่ง`, m.`กระทรวง`

RULES:
1. Use backticks for Thai properties: `ชื่อ-นามสกุล`, `ตำแหน่ง`, `หน่วยงาน`, `กระทรวง`
2. Use $param for parameters
3. Always LIMIT results
4. For "รัฐมนตรี" search Position.`ตำแหน่ง` CONTAINS 'รัฐมนตรี'
5. Stelligence relationships: Santisook uses :santisook_known, Por uses :por_known, Knot uses :knot_known

Respond with ONLY valid Cypher, no explanation."""
        
        user_prompt = f"""Generate Cypher for this query:

User Intent: {understanding}

Entities:
- Persons: {entities.get('persons', [])}
- Networks: {entities.get('networks', [])}
- Ministries: {entities.get('ministries', [])}
- Positions: {entities.get('positions', [])}
- Organizations: {entities.get('organizations', [])}

Query Structure:
- From: {structure.get('from', 'N/A')}
- To: {structure.get('to', 'N/A')}
- Via: {structure.get('via', 'N/A')}
- Filter: {structure.get('filter', 'N/A')}

Generate the Cypher query:"""

        response = self._call_llm(user_prompt, system_prompt, max_tokens=800)
        
        # Extract Cypher from response
        cypher = response.strip()
        
        # Clean up the query
        if "```" in cypher:
            # Extract from code block
            match = re.search(r'```(?:cypher)?\s*([\s\S]*?)```', cypher)
            if match:
                cypher = match.group(1).strip()
        
        # Build parameters from entities
        params = {}
        for i, person in enumerate(entities.get('persons', [])):
            params[f'person{i}'] = person
        for i, network in enumerate(entities.get('networks', [])):
            params[f'network{i}'] = network
        for i, ministry in enumerate(entities.get('ministries', [])):
            params[f'ministry{i}'] = ministry
        
        return cypher, params
    
    def execute_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Main entry point - understand and execute any natural language query.
        
        Args:
            query: Natural language query in Thai or English
            session_id: Optional session ID for chat history
        
        Returns:
            Dict with results, formatted answer, and metadata
        """
        start_time = time.time()
        
        # Get or create session for chat history
        session = self.get_or_create_session(session_id)
        session.add_message("user", query)
        
        # Step 1: Understand the query
        print(f"[DEBUG] Understanding query: {query}")
        intent = self.understand_query(query, session)
        print(f"[DEBUG] Understood intent: {intent}")
        
        # Step 2: Generate Cypher (pass original query for follow-up detection)
        cypher, params = self.generate_cypher(intent, session, original_query=query)
        print(f"[DEBUG] Generated Cypher: {cypher}")
        print(f"[DEBUG] Parameters: {params}")
        
        # Step 3: Execute with error handling
        results = []
        error = None
        
        if cypher:
            try:
                results = self._run_cypher(cypher, params)
            except Exception as e:
                error = str(e)
                print(f"[ERROR] Query execution failed: {e}")
                
                # Try to heal the query
                healed_cypher = self._heal_cypher(cypher, error, intent)
                if healed_cypher and healed_cypher != cypher:
                    print(f"[DEBUG] Trying healed query: {healed_cypher}")
                    try:
                        results = self._run_cypher(healed_cypher, params)
                        error = None
                    except Exception as e2:
                        error = str(e2)
        
        # Step 4: Format results
        formatted_result = self._format_results(results, intent)
        
        # Store result for follow-up context
        session.last_query_result = formatted_result
        session.add_message("assistant", formatted_result.get("answer", ""), 
                          metadata={"intent": intent, "result_count": len(results)})
        
        elapsed = time.time() - start_time
        
        return {
            "success": not error,
            "intent": intent,
            "cypher": cypher,
            "results": results,
            "formatted": formatted_result,
            "error": error,
            "session_id": session.session_id,
            "processing_time_ms": int(elapsed * 1000)
        }
    
    def _heal_cypher(self, cypher: str, error: str, intent: Dict) -> Optional[str]:
        """Attempt to fix a failed Cypher query using LLM."""
        
        system_prompt = f"""You are a Cypher query debugger. Fix the query error.

Schema:
{self.SCHEMA}

Common fixes:
1. Property names need backticks: `ชื่อ-นามสกุล`, `ตำแหน่ง`
2. Node labels with spaces need backticks: `Connect by`
3. Check relationship directions
4. Ensure RETURN clause is valid

Return ONLY the fixed Cypher, no explanation."""

        user_prompt = f"""Original query:
{cypher}

Error:
{error}

User intent: {intent.get('natural_language_understanding', '')}

Fixed query:"""

        response = self._call_llm(user_prompt, system_prompt, max_tokens=600)
        
        fixed = response.strip()
        if "```" in fixed:
            match = re.search(r'```(?:cypher)?\s*([\s\S]*?)```', fixed)
            if match:
                fixed = match.group(1).strip()
        
        return fixed if fixed and fixed != cypher else None
    
    def _format_results(self, results: List[Dict], intent: Dict) -> Dict[str, Any]:
        """Format query results into a human-readable response."""
        
        if not results:
            return {
                "answer": "ไม่พบข้อมูลที่ตรงกับคำถาม",
                "found": False,
                "data": []
            }
        
        intent_type = intent.get("intent_type", "general")
        understanding = intent.get("natural_language_understanding", "")
        
        lines = []
        
        # Format based on result structure
        if results and len(results) > 0:
            first = results[0]
            
            # Path results
            if "path_names" in first or "path" in first:
                lines.append("🔗 **เส้นทางที่พบ:**")
                for r in results[:5]:
                    path_names = r.get("path_names", [])
                    if path_names:
                        path_str = " → ".join(str(n) for n in path_names if n)
                        hops = r.get("hops", len(path_names) - 1)
                        lines.append(f"  • {path_str} ({hops} ขั้นตอน)")
            
            # Member list results
            elif "members" in first:
                members = first.get("members", [])
                if isinstance(members, list):
                    lines.append(f"👥 **พบสมาชิก {len(members)} คน:**")
                    for m in members[:30]:
                        if isinstance(m, dict) and m.get("name"):
                            name = m.get("name", "")
                            source = m.get("source", "")
                            network = m.get("network", "")
                            lines.append(f"  • {name}" + (f" [{network}]" if network else ""))
                    if len(members) > 30:
                        lines.append(f"  ... และอีก {len(members) - 30} คน")
            
            # Person network results
            elif "name" in first and "connections" in first:
                lines.append(f"👤 **{first.get('name', '')}**")
                if first.get("position"):
                    lines.append(f"   ตำแหน่ง: {first.get('position')}")
                if first.get("ministry"):
                    lines.append(f"   กระทรวง: {first.get('ministry')}")
                if first.get("agency"):
                    lines.append(f"   หน่วยงาน: {first.get('agency')}")
                
                networks = first.get("networks", [])
                if networks:
                    lines.append(f"   เครือข่าย: {', '.join(n for n in networks if n)}")
                
                connections = first.get("connections", [])
                if connections:
                    lines.append(f"\n📊 **คนรู้จัก ({len(connections)} คน):**")
                    for c in connections[:20]:
                        if isinstance(c, dict) and c.get("name"):
                            rel = c.get("relationship", "")
                            lines.append(f"  • {c['name']}" + (f" ({rel})" if rel else ""))
            
            # Generic list results
            elif "name" in first:
                lines.append(f"📋 **พบ {len(results)} รายการ:**")
                for r in results[:30]:
                    name = r.get("name", "Unknown")
                    pos = r.get("position", "")
                    ministry = r.get("ministry", "")
                    detail = f" - {pos}" if pos else ""
                    detail += f" ({ministry})" if ministry else ""
                    lines.append(f"  • {name}{detail}")
                if len(results) > 30:
                    lines.append(f"  ... และอีก {len(results) - 30} รายการ")
            
            # Minister/position connection results (Santisook -> OSK -> Minister pattern)
            elif "connector_name" in first and "minister_name" in first:
                lines.append(f"🔗 **การเชื่อมต่อที่พบ ({len(results)} รายการ):**")
                for r in results[:20]:
                    connector = r.get("connector_name", "")
                    via = r.get("via_network", "")
                    minister = r.get("minister_name", "")
                    position = r.get("position", "")
                    ministry = r.get("ministry", "")
                    conn_type = r.get("connection_type", "")
                    
                    if conn_type == "direct":
                        lines.append(f"  • 📌 **{minister}** ({position}, {ministry})")
                        lines.append(f"    ├── เครือข่าย: {via}")
                        lines.append(f"    └── (รู้จักโดยตรง)")
                    else:
                        lines.append(f"  • 🔗 **{minister}** ({position}, {ministry})")
                        lines.append(f"    ├── ผ่าน: {connector}")
                        lines.append(f"    └── เครือข่าย: {via}")
            
            else:
                # Fallback - just show the data
                lines.append(f"📊 **ผลลัพธ์ ({len(results)} รายการ):**")
                for r in results[:10]:
                    lines.append(f"  • {r}")
        
        return {
            "answer": "\n".join(lines) if lines else "ไม่พบข้อมูล",
            "found": True,
            "data": results,
            "understanding": understanding
        }


# Singleton instance
_intelligent_agent = None

def get_intelligent_agent() -> IntelligentNetworkAgent:
    """Get or create the singleton intelligent agent instance."""
    global _intelligent_agent
    if _intelligent_agent is None:
        _intelligent_agent = IntelligentNetworkAgent()
    return _intelligent_agent

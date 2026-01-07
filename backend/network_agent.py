"""
Intelligent Network Agent - Analyzes Neo4j connections and provides smart relationship insights
"""
import re
import time
from typing import Dict, List, Optional, Tuple
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
import os


class NetworkAgent:
    """Smart agent that understands network queries and finds optimal paths/connections"""
    
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "neo4j")
        
        # Configure driver with connection pool settings for stability
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(username, password),
            max_connection_lifetime=300,  # 5 minutes max connection lifetime
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,
            connection_timeout=30,
            keep_alive=True
        )
        self._max_retries = 3
        self._retry_delay = 1  # seconds
        
    def close(self):
        self.driver.close()
    
    def _execute_with_retry(self, query_func, *args, **kwargs):
        """Execute a query function with automatic retry on connection failures"""
        last_exception = None
        for attempt in range(self._max_retries):
            try:
                return query_func(*args, **kwargs)
            except (ServiceUnavailable, SessionExpired, TransientError) as e:
                last_exception = e
                print(f"[WARN] Neo4j connection error (attempt {attempt + 1}/{self._max_retries}): {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))  # Exponential backoff
                    # Try to verify connection
                    try:
                        self.driver.verify_connectivity()
                    except Exception:
                        pass
        raise last_exception
    
    def _run_query(self, query: str, **params) -> list:
        """Execute a Cypher query with automatic retry on connection failures.
        Returns list of records."""
        def _execute():
            with self.driver.session() as session:
                result = session.run(query, **params)
                return list(result)
        return self._execute_with_retry(_execute)
    
    def _run_query_single(self, query: str, **params):
        """Execute a Cypher query and return single record with retry."""
        def _execute():
            with self.driver.session() as session:
                result = session.run(query, **params)
                return result.single()
        return self._execute_with_retry(_execute)
    
    def detect_query_intent(self, query: str) -> Dict[str, any]:
        """
        Analyze the query to understand what the user wants:
        - shortest_path: Find shortest connection between two people
        - mutual_connections: Find common connections
        - person_network: Get someone's network
        - introduction: Who can introduce person A to person B
        - network_members: Who is in a specific network (e.g., OSK115)
        - company_network: Analyze company connections
        """
        print(f"[DEBUG detect_query_intent] Received query: {repr(query)}")
        print(f"[DEBUG detect_query_intent] Query bytes: {query.encode('utf-8')}")
        
        query_lower = query.lower()
        
        # Thai keywords to exclude from person extraction
        thai_exclude_words = {
            'เส้นทาง', 'จาก', 'ไป', 'ถึง', 'หา', 'ติดต่อ', 'เครือข่าย', 'ใคร', 'บ้าง', 'ที่', 'อยู่', 
            'ทำงาน', 'สังกัด', 'รู้จัก', 'แนะนำ', 'ผ่าน', 'กระทรวง', 'กรม', 'สำนักงาน',
            'มี', 'กี่', 'คน', 'แสดง', 'ดู', 'ช่วย', 'อยาก', 'ต้องการ', 'หาทาง'
        }
        
        # Extract person names (capitalized words, Thai names, or nicknames)
        # Exclude common question words
        person_pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b|([ก-๙]+(?:\s+[ก-๙]+)*)'
        exclude_words = {'How', 'Can', 'Who', 'What', 'Where', 'When', 'Why', 'The', 'A', 'An', 'And', 'Or', 'But', 'Show', 'Tell', 'Find', 'Get', 'Help', 'Please', 'Me', 'My', 'I', 'You', 'Your', 'Their', 'His', 'Her', 'Between', 'To', 'From', 'With', 'Path', 'Shortest', 'Route'}
        
        persons_found = re.findall(person_pattern, query)
        # Filter out Thai keywords and English exclude words
        persons = []
        for p in persons_found:
            name = p[0] or p[1]
            if name and name not in exclude_words and name not in thai_exclude_words:
                persons.append(name)
        
        intent = {
            "type": "general",
            "persons": persons,
            "query": query
        }
        
        # Check for cohort/batch queries (NEXIS, วปอ., etc.)
        cohort_patterns = [
            (r'nexis\s*(?:รุ่น(?:ที่)?\s*)?(\d+)?', 'NEXIS'),
            (r'วปอ\.?\s*(?:รุ่น(?:ที่)?\s*)?(\d+)?', 'วปอ.'),
            (r'(?:รุ่น|batch|cohort)\s*(?:ที่\s*)?(\d+)?', None),
        ]
        for pattern, cohort_type in cohort_patterns:
            match = re.search(pattern, query_lower)
            if match:
                intent["type"] = "cohort_search"
                intent["cohort_type"] = cohort_type
                intent["cohort_number"] = match.group(1) if match.lastindex else None
                # Try to extract full cohort name from query
                full_pattern = r'(nexis\s*รุ่น(?:ที่)?\s*\d+|วปอ\.?\s*รุ่น(?:ที่)?\s*\d+)'
                full_match = re.search(full_pattern, query_lower)
                if full_match:
                    intent["cohort_name"] = full_match.group(1).strip()
                return intent
        
        # Check for organization/ministry search (ใครทำงานกระทรวงพลังงาน)
        org_keywords = ["ทำงาน", "อยู่ที่", "สังกัด", "ประจำ", "work at", "works at", "from", "in ministry", "at agency"]
        ministry_keywords = ["กระทรวง", "ministry"]
        agency_keywords = ["กรม", "สำนักงาน", "agency", "department", "office"]
        
        if any(kw in query_lower for kw in org_keywords) or any(kw in query_lower for kw in ministry_keywords + agency_keywords):
            # Try to extract organization name
            # Ministry pattern: กระทรวง + name
            ministry_match = re.search(r'กระทรวง\s*([ก-๙a-zA-Z\s]+?)(?:\s*$|\s*(?:บ้าง|มี|ใคร|กี่))', query)
            if ministry_match:
                intent["type"] = "organization_search"
                intent["org_type"] = "ministry"
                intent["org_name"] = ministry_match.group(1).strip()
                return intent
            
            # Agency pattern: กรม/สำนักงาน + name  
            agency_match = re.search(r'(?:กรม|สำนักงาน)\s*([ก-๙a-zA-Z\s]+?)(?:\s*$|\s*(?:บ้าง|มี|ใคร|กี่))', query)
            if agency_match:
                intent["type"] = "organization_search"
                intent["org_type"] = "agency"
                intent["org_name"] = agency_match.group(0).strip()
                return intent
        
        # STELLIGENCE OWNER NAMES - used for both Stelligence network queries AND pathfinding
        stelligence_keywords = {
            "santisook": "Santisook",
            "สันติสุข": "Santisook",
            "por": "Por",
            "พอ": "Por",
            "knot": "Knot",
            "น็อต": "Knot"
        }
        
        # Check for shortest path / connection intent FIRST
        # This must be checked BEFORE Stelligence network queries to handle "path from Santisook to X"
        path_keywords = ["เส้นทาง", "จาก", "ไป", "ถึง", "shortest", "quickest", "fastest", "path", "route", "reach", "get to know", "connect to", "how can", "หา", "ติดต่อ"]
        has_path_intent = any(keyword in query_lower for keyword in path_keywords)
        
        # For path queries, try to extract names using specific patterns
        if has_path_intent:
            # Pattern 1: "เส้นทางจาก X ไป Y" or "จาก X ไป Y" or "path from X to Y"
            # Thai pattern: จาก <name> ไป/ถึง/หา <name>
            # Also handle: X ไปหา Y, X ไป Y
            thai_path_patterns = [
                # Pattern: "เส้นทางจาก Por ไป อนุทิน" or "เส้นทางจาก X ไป Y"
                r'เส้นทาง\s*(?:จาก)?\s*([ก-๙A-Za-z]+(?:\s+[ก-๙A-Za-z]+)?)\s*(?:ไป|ถึง|หา)\s*([ก-๙A-Za-z]+(?:\s+[ก-๙A-Za-z]+)?)',
                # Pattern: "Santisook ไปหา อนุทิน ชาญวีรกูล" (name ไปหา name)
                r'([ก-๙A-Za-z]+(?:\s+[ก-๙A-Za-z]+)?)\s+(?:ไปหา|ไป(?:\s*)หา)\s+([ก-๙A-Za-z]+(?:\s+[ก-๙A-Za-z]+)?)',
                # Pattern: "จาก X ไป/ถึง/หา Y"
                r'(?:จาก|from)\s*([ก-๙A-Za-z]+(?:\s+[ก-๙A-Za-z]+)?)\s*(?:ไป|ถึง|to|หา)\s*([ก-๙A-Za-z]+(?:\s+[ก-๙A-Za-z]+)?)',
                # Pattern: "X ไป Y"
                r'([ก-๙A-Za-z]+(?:\s+[ก-๙A-Za-z]+)?)\s+(?:ไป|ถึง)\s+([ก-๙A-Za-z]+(?:\s+[ก-๙A-Za-z]+)?)',
            ]
            
            print(f"[DEBUG path patterns] Testing path patterns for: {query}")
            
            for pattern in thai_path_patterns:
                print(f"[DEBUG path patterns] Trying pattern: {pattern}")
                path_match = re.search(pattern, query, re.IGNORECASE)
                if path_match:
                    from_name = path_match.group(1).strip()
                    to_name = path_match.group(2).strip()
                    print(f"[DEBUG path patterns] Match found: from='{from_name}', to='{to_name}'")
                    
                    # Skip if to_name is a Thai keyword
                    if to_name.lower() in thai_exclude_words or len(to_name) < 2:
                        continue
                    
                    # Normalize Stelligence names
                    from_normalized = stelligence_keywords.get(from_name.lower(), from_name)
                    to_normalized = stelligence_keywords.get(to_name.lower(), to_name)
                    
                    intent["type"] = "shortest_path"
                    intent["from_person"] = from_normalized
                    intent["to_person"] = to_normalized
                    print(f"[DEBUG] Path extracted: from={from_normalized}, to={to_normalized}")
                    return intent
        
        # Fallback: if we have 2+ persons and path keywords
        if has_path_intent and len(persons) >= 2:
            intent["type"] = "shortest_path"
            intent["from_person"] = persons[0]
            intent["to_person"] = persons[1]
            return intent
        
        # Check if one name is Stelligence owner and there's another person
        if has_path_intent:
            stelligence_name = None
            other_person = None
            
            for keyword, network_type in stelligence_keywords.items():
                if keyword in query_lower:
                    stelligence_name = network_type
                    break
            
            if stelligence_name:
                # Find the other person mentioned (not the Stelligence keyword)
                for person in persons:
                    if person.lower() not in stelligence_keywords and person.lower() != stelligence_name.lower():
                        other_person = person
                        break
                
                # Check for Thai name after ไป/to keywords
                if not other_person:
                    to_pattern = r'(?:ไป|ถึง|to|หา)\s*([ก-๙]+(?:\s*[ก-๙]+)*)'
                    to_match = re.search(to_pattern, query)
                    if to_match:
                        other_person = to_match.group(1).strip()
                
                if other_person:
                    intent["type"] = "shortest_path"
                    # Determine direction based on from/to keywords
                    if any(kw in query_lower for kw in ["จาก " + keyword for keyword in stelligence_keywords.keys()]):
                        intent["from_person"] = stelligence_name
                        intent["to_person"] = other_person
                    else:
                        # Default: from Stelligence owner to other person
                        intent["from_person"] = stelligence_name
                        intent["to_person"] = other_person
                    return intent
        
        # Check for Stelligence network queries (only when NOT a path query)
        for keyword, network_type in stelligence_keywords.items():
            if keyword in query_lower:
                intent["type"] = "stelligence_network"
                intent["network_type"] = network_type
                return intent
        
        # Check for network members query (ใครบ้างที่ connect by OSK115)
        if "connect by" in query_lower or "เครือข่าย" in query_lower or "connect_by" in query_lower:
            # Extract network name (OSK115, CSIL, etc.)
            network_pattern = r'(?:by|เครือข่าย)\s*([A-Z0-9]+)'
            network_match = re.search(network_pattern, query, re.IGNORECASE)
            if network_match:
                intent["type"] = "network_members"
                intent["network"] = network_match.group(1)
                return intent
        
        # Shortest path / connection intent
        # Thai: "เส้นทางจาก X ไป Y", "จาก X ถึง Y"
        # English: "from X to Y", "path from X to Y"
        if any(keyword in query_lower for keyword in ["เส้นทาง", "จาก", "ไป", "ถึง", "shortest", "quickest", "fastest", "path", "route", "reach", "get to know", "connect to", "how can"]):
            if len(persons) >= 2:
                intent["type"] = "shortest_path"
                intent["from_person"] = persons[0]
                intent["to_person"] = persons[1]
                return intent
        
        # Introduction/connector query
        # Thai: "ต้อง connect ผ่านใคร", "ใครสามารถแนะนำ"
        # English: "who can introduce", "connect through who"
        elif any(keyword in query_lower for keyword in ["ผ่านใคร", "แนะนำ", "introduce", "connect through", "who can help me meet"]):
            if len(persons) >= 1:
                intent["type"] = "introduction"
                intent["from_person"] = "Me"  # User asking
                intent["to_person"] = persons[0]
                return intent
        
        # Mutual connections intent
        elif any(keyword in query_lower for keyword in ["mutual", "common", "both know", "share", "in common", "รู้จักร่วมกัน"]):
            if len(persons) >= 2:
                intent["type"] = "mutual_connections"
                intent["person1"] = persons[0]
                intent["person2"] = persons[1]
                return intent
        
        # Person's network intent  
        # Thai: "X รู้จักใครบ้าง", "เครือข่ายของ X"
        # English: "who does X know", "X's network"
        elif any(keyword in query_lower for keyword in ["รู้จัก", "เครือข่าย", "network", "connections", "knows", "connected to", "friends", "colleagues", "contacts"]):
            if len(persons) >= 1:
                intent["type"] = "person_network"
                intent["person"] = persons[0]
                return intent
        
        return intent
    
    def find_shortest_path(self, from_person: str, to_person: str) -> Dict:
        """Find shortest path between two people/network owners in the network.
        
        Uses retry logic for connection stability.
        """
        return self._execute_with_retry(self._find_shortest_path_impl, from_person, to_person)
    
    def _find_shortest_path_impl(self, from_person: str, to_person: str) -> Dict:
        """Internal implementation of find_shortest_path with actual Neo4j queries.
        
        Supports:
        - Person to Person paths
        - Stelligence network owner (Santisook/Por/Knot) to Person paths
        - Person to Stelligence network owner paths
        
        Also calculates 'best connector' - person along the path with most connections.
        
        Schema Notes:
        - Person properties: `ชื่อ-นามสกุล`
        - Stelligence: (network:Santisook/Por/Knot)-[:santisook_known/por_known/knot_known]->(Person)
        """
        # Known person name mappings for common partial matches (Thai + English)
        # Store keys in lowercase and do case-insensitive lookup to avoid misses on different capitalizations
        known_names = {
            # Thai names (kept as-is for matching but keys are lowered)
            "อนุทิน": "อนุทิน ชาญวีรกูล",
            "อนทน": "อนุทิน ชาญวีรกูล",  # Handle encoding issues
            "ประเสริฐ": "ประเสริฐสิน",
            # English transliterations (lowercase keys)
            "anutin": "อนุทิน ชาญวีรกูล",
            "charnvirakul": "อนุทิน ชาญวีรกูล",
            "prasertsin": "ประเสริฐสิน",
        }

        # Case-insensitive expansion: lower the incoming names for lookup
        to_lc = to_person.lower() if isinstance(to_person, str) else to_person
        from_lc = from_person.lower() if isinstance(from_person, str) else from_person

        if isinstance(to_lc, str) and to_lc in known_names:
            to_person = known_names[to_lc]
            print(f"[DEBUG] Expanded to_person to: {to_person}")
        if isinstance(from_lc, str) and from_lc in known_names:
            from_person = known_names[from_lc]
            print(f"[DEBUG] Expanded from_person to: {from_person}")
            
        # Check if from_person is a Stelligence network owner
        stelligence_owners = {"santisook": "Santisook", "por": "Por", "knot": "Knot"}
        stelligence_rels = {"Santisook": "santisook_known", "Por": "por_known", "Knot": "knot_known"}
        
        from_is_stelligence = from_person.lower() in stelligence_owners or from_person in stelligence_owners.values()
        to_is_stelligence = to_person.lower() in stelligence_owners or to_person in stelligence_owners.values()
        
        # Normalize names
        if from_is_stelligence:
            from_person = stelligence_owners.get(from_person.lower(), from_person)
        if to_is_stelligence:
            to_person = stelligence_owners.get(to_person.lower(), to_person)
        
        with self.driver.session() as session:
            # Build dynamic query based on whether we're starting from Stelligence owner or Person
            if from_is_stelligence:
                # Path from Stelligence network owner to a Person
                # Find a person known by that network owner, then path to target
                rel_type = stelligence_rels.get(from_person, "santisook_known")
                result = session.run(f"""
                    // Find a member of the Stelligence network to start from
                    MATCH (network:{from_person})-[:{rel_type}]->(member:Person)
                    WITH member LIMIT 1
                    
                    // Find target person (using actual property name)
                    MATCH (b:Person)
                    WHERE b.`ชื่อ-นามสกุล` CONTAINS $to_name
                    WITH member, b LIMIT 1
                    
                    // Find shortest path
                    MATCH path = shortestPath((member)-[*..12]-(b))
                    
                    // Calculate connection counts and get position/ministry for each person
                    WITH path, member, b,
                         [node in nodes(path) WHERE node:Person | {{
                             name: node.`ชื่อ-นามสกุล`,
                             full_name: node.`ชื่อ-นามสกุล`,
                             connections: size([(node)--() | 1]),
                             position: [(node)--(pos:Position) | pos.`ตำแหน่ง`][0],
                             ministry: [(node)--(m:Ministry) | m.`กระทรวง`][0],
                             agency: [(node)--(a:Agency) | a.`หน่วยงาน`][0]
                         }}] as person_details
                    
                    RETURN path,
                           length(path) as distance,
                           [node in nodes(path) | CASE 
                               WHEN node:Person THEN node.`ชื่อ-นามสกุล`
                               WHEN node:Ministry THEN 'กระทรวง' + coalesce(node.`กระทรวง`, node.name, '')
                               WHEN node:Agency THEN coalesce(node.`หน่วยงาน`, node.name, '')
                               WHEN node:`Connect by` THEN 'เครือข่าย ' + coalesce(node.Stelligence, node.`Connect by`, '')
                               WHEN node:Position THEN 'ตำแหน่ง ' + coalesce(node.`ตำแหน่ง`, '')
                               ELSE labels(node)[0]
                           END] as path_names,
                           [rel in relationships(path) | type(rel)] as relationship_types,
                           $from_name as from_full_name,
                           b.`ชื่อ-นามสกุล` as to_full_name,
                           person_details
                    LIMIT 1
                """, from_name=from_person, to_name=to_person)
            else:
                # Standard Person to Person path
                result = session.run("""
                    // Find source person
                    MATCH (a:Person)
                    WHERE a.`ชื่อ-นามสกุล` CONTAINS $from_name
                    WITH a LIMIT 1
                    
                    // Find target person
                    MATCH (b:Person)
                    WHERE b.`ชื่อ-นามสกุล` CONTAINS $to_name
                    WITH a, b LIMIT 1
                    
                    // Find shortest path
                    MATCH path = shortestPath((a)-[*..12]-(b))
                    
                    // Calculate connection counts and get position/ministry for each person
                    WITH path, a, b,
                         [node in nodes(path) WHERE node:Person | {
                             name: node.`ชื่อ-นามสกุล`,
                             full_name: node.`ชื่อ-นามสกุล`,
                             connections: size([(node)--() | 1]),
                             position: [(node)--(pos:Position) | pos.`ตำแหน่ง`][0],
                             ministry: [(node)--(m:Ministry) | m.`กระทรวง`][0],
                             agency: [(node)--(ag:Agency) | ag.`หน่วยงาน`][0]
                         }] as person_details
                    
                    RETURN path,
                           length(path) as distance,
                           [node in nodes(path) | CASE 
                               WHEN node:Person THEN node.`ชื่อ-นามสกุล`
                               WHEN node:Ministry THEN 'กระทรวง' + coalesce(node.`กระทรวง`, node.name, '')
                               WHEN node:Agency THEN coalesce(node.`หน่วยงาน`, node.name, '')
                               WHEN node:`Connect by` THEN 'เครือข่าย ' + coalesce(node.Stelligence, node.`Connect by`, '')
                               WHEN node:Position THEN 'ตำแหน่ง ' + coalesce(node.`ตำแหน่ง`, '')
                               ELSE labels(node)[0]
                           END] as path_names,
                           [rel in relationships(path) | type(rel)] as relationship_types,
                           a.`ชื่อ-นามสกุล` as from_full_name,
                           b.`ชื่อ-นามสกุล` as to_full_name,
                           person_details
                    LIMIT 1
                """, from_name=from_person, to_name=to_person)
            
            record = result.single()
            if not record:
                return {
                    "found": False,
                    "message": f"ไม่พบเส้นทางเชื่อมต่อระหว่าง {from_person} และ {to_person}"
                }
            
            path_names = record["path_names"]
            distance = record["distance"]
            rel_types = record["relationship_types"]
            from_full = record["from_full_name"]
            to_full = record["to_full_name"]
            person_details = record.get("person_details", [])
            
            # Find the best connector (person with most connections along the path)
            best_connector = None
            if person_details:
                # Exclude start and end persons, find one with most connections
                intermediate_persons = [p for p in person_details 
                                        if p["full_name"] not in [from_full, to_full]]
                if intermediate_persons:
                    best_connector = max(intermediate_persons, key=lambda x: x["connections"])
                elif len(person_details) > 0:
                    best_connector = max(person_details, key=lambda x: x["connections"])
            
            # Build human-readable path description
            path_desc = self._build_path_description(path_names, rel_types)
            
            result_dict = {
                "found": True,
                "from": from_full,
                "to": to_full,
                "distance": distance,
                "path": path_names,
                "relationships": rel_types,
                "description": path_desc
            }
            
            # Add best connector info if found
            if best_connector:
                # Build position/ministry info string
                role_info = []
                if best_connector.get("position"):
                    role_info.append(f"ตำแหน่ง: {best_connector['position']}")
                if best_connector.get("ministry"):
                    role_info.append(f"กระทรวง{best_connector['ministry']}")
                elif best_connector.get("agency"):
                    role_info.append(f"หน่วยงาน: {best_connector['agency']}")
                
                role_str = ", ".join(role_info) if role_info else "ไม่ระบุ"
                
                result_dict["best_connector"] = {
                    "name": best_connector["name"],
                    "full_name": best_connector["full_name"],
                    "connections": best_connector["connections"],
                    "position": best_connector.get("position"),
                    "ministry": best_connector.get("ministry"),
                    "agency": best_connector.get("agency"),
                    "role_info": role_str,
                    "recommendation": f"{best_connector['name']} ({role_str}) มีการเชื่อมต่อมากที่สุด ({best_connector['connections']} connections)"
                }
            
            # Also include person_details with position/ministry for all persons in path
            result_dict["person_details"] = person_details
            
            return result_dict
    
    def find_mutual_connections(self, person1: str, person2: str) -> Dict:
        """Find mutual connections between two people. Uses retry logic."""
        return self._execute_with_retry(self._find_mutual_connections_impl, person1, person2)
    
    def _find_mutual_connections_impl(self, person1: str, person2: str) -> Dict:
        """Internal implementation of find_mutual_connections"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Person)--(mutual)--(b:Person)
                WHERE a.`ชื่อ-นามสกุล` CONTAINS $name1 AND b.`ชื่อ-นามสกุล` CONTAINS $name2
                RETURN DISTINCT mutual.`ชื่อ-นามสกุล` as mutual_name,
                       labels(mutual) as labels,
                       [(mutual)-[r]-(a) | type(r)][0] as rel_to_person1,
                       [(mutual)-[r]-(b) | type(r)][0] as rel_to_person2
                LIMIT 20
            """, name1=person1, name2=person2)
            
            mutuals = []
            for record in result:
                mutuals.append({
                    "name": record["mutual_name"],
                    "labels": record["labels"],
                    "relation_to_person1": record["rel_to_person1"],
                    "relation_to_person2": record["rel_to_person2"]
                })
            
            return {
                "person1": person1,
                "person2": person2,
                "mutual_count": len(mutuals),
                "mutuals": mutuals
            }
    
    def get_person_network(self, person: str, depth: int = 2) -> Dict:
        """Get a person's network up to certain depth. Uses retry logic."""
        return self._execute_with_retry(self._get_person_network_impl, person, depth)
    
    def _get_person_network_impl(self, person: str, depth: int = 2) -> Dict:
        """Internal implementation of get_person_network"""
        with self.driver.session() as session:
            result = session.run("""
                // Fuzzy match person using actual property name
                MATCH (p:Person)
                WHERE p.`ชื่อ-นามสกุล` CONTAINS $name
                WITH p LIMIT 1
                
                // Get all connections
                OPTIONAL MATCH (p)-[r]-(connected)
                WITH p, type(r) as rel_type, collect(DISTINCT CASE
                    WHEN connected:Person THEN connected.`ชื่อ-นามสกุล`
                    WHEN connected:Ministry THEN connected.`กระทรวง`
                    WHEN connected:Agency THEN connected.`หน่วยงาน`
                    WHEN connected:`Connect by` THEN connected.`Connect by`
                    ELSE coalesce(connected.name, labels(connected)[0])
                END) as connections
                RETURN p.`ชื่อ-นามสกุล` as person,
                       rel_type,
                       connections,
                       size(connections) as count
                ORDER BY count DESC
            """, name=person)
            
            network = {
                "person": person,
                "person_full_name": None,
                "total_connections": 0,
                "by_relationship": {}
            }
            
            for record in result:
                if not network["person_full_name"]:
                    network["person_full_name"] = record["person"]
                    
                if record["rel_type"]:
                    rel_type = record["rel_type"]
                    connections = record["connections"]
                    count = record["count"]
                    
                    network["by_relationship"][rel_type] = {
                        "count": count,
                        "connections": connections[:10]  # Limit to 10 per type
                    }
                    network["total_connections"] += count
            
            return network
    
    def get_network_members(self, network_name: str) -> Dict:
        """Get all members connected by a specific network. Uses retry logic."""
        return self._execute_with_retry(self._get_network_members_impl, network_name)
    
    def _get_network_members_impl(self, network_name: str) -> Dict:
        """Internal implementation of get_network_members.
        
        Schema: 
        - Node label is `Connect by` (with space)
        - Property is also `Connect by`
        - Relationship from Person is `connect_by` (lowercase with underscore)
        
        Returns detailed info: name, position, agency, ministry for each member
        """
        with self.driver.session() as session:
            # Query with full details: Person -[:connect_by]-> (`Connect by` node)
            # Also fetch position, agency, ministry via relationships
            result = session.run("""
                MATCH (p:Person)-[:connect_by]->(cb:`Connect by`)
                WHERE cb.`Connect by` CONTAINS $network_name
                OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
                OPTIONAL MATCH (p)-[:work_at]->(agency:Agency)
                OPTIONAL MATCH (p)-[:under]->(ministry:Ministry)
                OPTIONAL MATCH (p)-[:has_level]->(lvl:Level)
                WITH cb, p, pos, agency, ministry, lvl
                RETURN cb.`Connect by` as network_name,
                       collect(DISTINCT {
                           name: p.`ชื่อ-นามสกุล`,
                           position: pos.`ตำแหน่ง`,
                           agency: agency.`หน่วยงาน`,
                           ministry: ministry.`กระทรวง`,
                           level: lvl.Level
                       }) as members,
                       count(DISTINCT p) as member_count
            """, network_name=network_name)
            
            record = result.single()
            
            if record and record["member_count"] > 0:
                # Format members with details
                members_detailed = []
                for m in record["members"]:
                    if m.get("name"):
                        member_info = {
                            "name": m["name"],
                            "position": m.get("position"),
                            "agency": m.get("agency"),
                            "ministry": m.get("ministry"),
                            "level": m.get("level")
                        }
                        members_detailed.append(member_info)
                
                return {
                    "found": True,
                    "network": record["network_name"] or network_name,
                    "members": members_detailed,
                    "member_count": len(members_detailed)
                }
            
            return {
                "found": False,
                "message": f"ไม่พบเครือข่าย {network_name}"
            }
    
    def get_stelligence_network(self, network_type: str) -> Dict:
        """Get all members of a Stelligence network. Uses retry logic."""
        relationship_map = {
            "Santisook": "santisook_known",
            "Por": "por_known",
            "Knot": "knot_known"
        }
        
        rel_type = relationship_map.get(network_type)
        if not rel_type:
            return {"found": False, "message": f"ไม่รู้จักประเภทเครือข่าย {network_type}"}
        
        return self._execute_with_retry(self._get_stelligence_network_impl, network_type, rel_type)
    
    def _get_stelligence_network_impl(self, network_type: str, rel_type: str) -> Dict:
        """Internal implementation of get_stelligence_network.
        
        Schema:
        - Santisook node -[:santisook_known]-> Person
        - Por node -[:por_known]-> Person
        - Knot node -[:knot_known]-> Person
        """
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (network:{network_type})-[:{rel_type}]->(p:Person)
                OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
                OPTIONAL MATCH (p)-[:work_at]->(agency:Agency)
                OPTIONAL MATCH (p)-[:under]->(ministry:Ministry)
                RETURN network.Stelligence as network_name,
                       collect(DISTINCT {{
                           name: p.`ชื่อ-นามสกุล`,
                           position: pos.`ตำแหน่ง`,
                           agency: agency.`หน่วยงาน`,
                           ministry: ministry.`กระทรวง`
                       }}) as members,
                       count(DISTINCT p) as member_count
            """)
            
            record = result.single()
            
            if record and record["member_count"] > 0:
                members_detailed = []
                for m in record["members"]:
                    if m.get("name"):
                        members_detailed.append({
                            "name": m["name"],
                            "position": m.get("position"),
                            "agency": m.get("agency"),
                            "ministry": m.get("ministry")
                        })
                
                return {
                    "found": True,
                    "network": f"{network_type} ({record['network_name'] or 'Stelligence'})",
                    "network_type": network_type,
                    "members": members_detailed,
                    "member_count": len(members_detailed)
                }
            
            return {
                "found": False,
                "message": f"ไม่พบข้อมูลเครือข่าย {network_type}"
            }
    
    def search_by_organization(self, org_name: str, org_type: str = "ministry") -> Dict:
        """Search for people by organization. Uses retry logic."""
        return self._execute_with_retry(self._search_by_organization_impl, org_name, org_type)
    
    def _search_by_organization_impl(self, org_name: str, org_type: str = "ministry") -> Dict:
        """Internal implementation of search_by_organization.
        
        Args:
            org_name: Name of the organization (e.g., "พลังงาน", "สำนักงานปลัด")
            org_type: "ministry" or "agency"
        """
        with self.driver.session() as session:
            if org_type == "ministry":
                result = session.run("""
                    MATCH (p:Person)-[:under]->(m:Ministry)
                    WHERE m.`กระทรวง` CONTAINS $org_name
                    OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
                    OPTIONAL MATCH (p)-[:work_at]->(agency:Agency)
                    OPTIONAL MATCH (p)-[:has_level]->(lvl:Level)
                    RETURN m.`กระทรวง` as organization,
                           collect(DISTINCT {
                               name: p.`ชื่อ-นามสกุล`,
                               position: pos.`ตำแหน่ง`,
                               agency: agency.`หน่วยงาน`,
                               level: lvl.Level
                           }) as members,
                           count(DISTINCT p) as member_count
                """, org_name=org_name)
            else:  # agency
                result = session.run("""
                    MATCH (p:Person)-[:work_at]->(a:Agency)
                    WHERE a.`หน่วยงาน` CONTAINS $org_name
                    OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
                    OPTIONAL MATCH (p)-[:under]->(ministry:Ministry)
                    OPTIONAL MATCH (p)-[:has_level]->(lvl:Level)
                    RETURN a.`หน่วยงาน` as organization,
                           collect(DISTINCT {
                               name: p.`ชื่อ-นามสกุล`,
                               position: pos.`ตำแหน่ง`,
                               ministry: ministry.`กระทรวง`,
                               level: lvl.Level
                           }) as members,
                           count(DISTINCT p) as member_count
                """, org_name=org_name)
            
            record = result.single()
            
            if record and record["member_count"] > 0:
                members_detailed = []
                for m in record["members"]:
                    if m.get("name"):
                        members_detailed.append({
                            "name": m["name"],
                            "position": m.get("position"),
                            "agency": m.get("agency"),
                            "ministry": m.get("ministry"),
                            "level": m.get("level")
                        })
                
                return {
                    "found": True,
                    "organization": record["organization"] or org_name,
                    "org_type": org_type,
                    "members": members_detailed,
                    "member_count": len(members_detailed)
                }
            
            return {
                "found": False,
                "message": f"ไม่พบบุคคลที่ทำงานใน {org_name}"
            }
    
    def search_by_cohort(self, cohort_name: str = None, cohort_type: str = None, cohort_number: str = None) -> Dict:
        """Search for people by cohort/batch. Uses retry logic."""
        # Build search pattern
        if cohort_name:
            search_pattern = cohort_name.upper()
        elif cohort_type and cohort_number:
            search_pattern = f"{cohort_type} รุ่นที่ {cohort_number}"
        elif cohort_type:
            search_pattern = cohort_type
        else:
            search_pattern = "NEXIS"  # Default
        
        return self._execute_with_retry(self._search_by_cohort_impl, search_pattern)
    
    def _search_by_cohort_impl(self, search_pattern: str) -> Dict:
        """Internal implementation of search_by_cohort.
        
        Uses the Associate relationship to find cohort members.
        """
        with self.driver.session() as session:
            
            result = session.run("""
                MATCH (p:Person)-[:associate_with]->(a:Associate)
                WHERE toLower(a.Associate) CONTAINS toLower($pattern)
                OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
                OPTIONAL MATCH (p)-[:work_at]->(agency:Agency)
                OPTIONAL MATCH (p)-[:under]->(ministry:Ministry)
                RETURN a.Associate as cohort_name,
                       collect(DISTINCT {
                           name: p.`ชื่อ-นามสกุล`,
                           position: pos.`ตำแหน่ง`,
                           agency: agency.`หน่วยงาน`,
                           ministry: ministry.`กระทรวง`
                       }) as members,
                       count(DISTINCT p) as member_count
            """, pattern=search_pattern)
            
            record = result.single()
            
            if record and record["member_count"] > 0:
                members_detailed = []
                for m in record["members"]:
                    if m.get("name"):
                        members_detailed.append({
                            "name": m["name"],
                            "position": m.get("position"),
                            "agency": m.get("agency"),
                            "ministry": m.get("ministry")
                        })
                
                return {
                    "found": True,
                    "cohort": record["cohort_name"] or search_pattern,
                    "members": members_detailed,
                    "member_count": len(members_detailed)
                }
            
            return {
                "found": False,
                "message": f"ไม่พบสมาชิกใน {search_pattern}"
            }

    def find_best_introducer(self, from_person: str, to_person: str) -> Dict:
        """Find the best person who can introduce from_person to to_person. Uses retry logic."""
        return self._execute_with_retry(self._find_best_introducer_impl, from_person, to_person)
    
    def _find_best_introducer_impl(self, from_person: str, to_person: str) -> Dict:
        """Internal implementation of find_best_introducer."""
        with self.driver.session() as session:
            # Find all people who know both - using correct property name
            result = session.run("""
                MATCH (a:Person)-[r1]-(introducer:Person)-[r2]-(b:Person)
                WHERE a.`ชื่อ-นามสกุล` CONTAINS $from_name 
                  AND b.`ชื่อ-นามสกุล` CONTAINS $to_name
                  AND introducer.`ชื่อ-นามสกุล` <> a.`ชื่อ-นามสกุล` 
                  AND introducer.`ชื่อ-นามสกุล` <> b.`ชื่อ-นามสกุล`
                WITH introducer,
                     type(r1) as rel_to_from,
                     type(r2) as rel_to_target,
                     [(introducer)-[]-(other) | other] as introducer_network
                RETURN DISTINCT introducer.`ชื่อ-นามสกุล` as name,
                       rel_to_from,
                       rel_to_target,
                       size(introducer_network) as network_size
                ORDER BY network_size DESC
                LIMIT 5
            """, from_name=from_person, to_name=to_person)
            
            introducers = []
            for record in result:
                introducers.append({
                    "name": record["name"],
                    "relationship_to_you": record["rel_to_from"],
                    "relationship_to_target": record["rel_to_target"],
                    "network_size": record["network_size"]
                })
            
            if not introducers:
                return {
                    "found": False,
                    "message": f"No mutual connections found between {from_person} and {to_person}"
                }
            
            return {
                "found": True,
                "from": from_person,
                "to": to_person,
                "introducers": introducers,
                "best_introducer": introducers[0]
            }
    
    def _build_path_description(self, path_names: List[str], rel_types: List[str]) -> str:
        """Build human-readable path description"""
        if len(path_names) == 1:
            return f"{path_names[0]}"
        
        desc_parts = []
        for i in range(len(path_names) - 1):
            from_name = path_names[i]
            to_name = path_names[i + 1]
            rel = rel_types[i] if i < len(rel_types) else "connected to"
            
            desc_parts.append(f"{from_name} --[{rel}]--> {to_name}")
        
        return " → ".join(desc_parts)
    
    def execute_smart_query(self, user_query: str) -> Dict:
        """
        Main entry point: Analyze query, execute appropriate Neo4j query, return results
        """
        intent = self.detect_query_intent(user_query)
        
        if intent["type"] == "shortest_path":
            return {
                "intent": intent,
                "result": self.find_shortest_path(intent["from_person"], intent["to_person"])
            }
        
        elif intent["type"] == "mutual_connections":
            return {
                "intent": intent,
                "result": self.find_mutual_connections(intent["person1"], intent["person2"])
            }
        
        elif intent["type"] == "person_network":
            return {
                "intent": intent,
                "result": self.get_person_network(intent["person"])
            }
        
        elif intent["type"] == "introduction":
            return {
                "intent": intent,
                "result": self.find_best_introducer(intent["from_person"], intent["to_person"])
            }
        
        elif intent["type"] == "network_members":
            return {
                "intent": intent,
                "result": self.get_network_members(intent["network"])
            }
        
        elif intent["type"] == "stelligence_network":
            return {
                "intent": intent,
                "result": self.get_stelligence_network(intent["network_type"])
            }
        
        elif intent["type"] == "organization_search":
            return {
                "intent": intent,
                "result": self.search_by_organization(intent["org_name"], intent["org_type"])
            }
        
        elif intent["type"] == "cohort_search":
            return {
                "intent": intent,
                "result": self.search_by_cohort(
                    cohort_name=intent.get("cohort_name"),
                    cohort_type=intent.get("cohort_type"),
                    cohort_number=intent.get("cohort_number")
                )
            }
        
        else:
            return {
                "intent": intent,
                "result": {
                    "message": "ฉันเข้าใจคำถามทั่วไป แต่ทำงานได้ดีที่สุดกับคำถามเกี่ยวกับเครือข่าย เช่น:\n" +
                              "- 'เส้นทางจาก X ไป Y'\n" +
                              "- 'X รู้จักใครบ้าง'\n" +
                              "- 'ใครบ้างที่ connect by OSK115'\n" +
                              "- 'ใครรู้จัก Santisook/Por/Knot'\n" +
                              "- 'ใครทำงานกระทรวงพลังงาน'\n" +
                              "- 'NEXIS รุ่น 1 มีใครบ้าง'"
                }
            }


# Singleton instance
_agent_instance = None

def get_network_agent() -> NetworkAgent:
    """Get or create singleton network agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = NetworkAgent()
    return _agent_instance

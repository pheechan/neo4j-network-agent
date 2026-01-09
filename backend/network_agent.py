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
        Generic query intent detection that works with any question format.
        Supports Thai and English queries for:
        - shortest_path: Find connection between two people
        - person_network: Get someone's connections/network
        - mutual_connections: Find common connections between people
        - introduction: Who can introduce person A to person B
        - network_members: Who is in a specific network
        - complex_query: Multi-condition queries (e.g. people in X who know Y)
        - general: Free-form queries
        """
        print(f"[DEBUG detect_query_intent] Received query: {repr(query)}")
        
        query_lower = query.lower()
        original_query = query
        
        # ============================================
        # STEP 1: Define keywords and known names
        # ============================================
        
        # Stelligence owners (define early for complex query check)
        stelligence_keywords = {
            "santisook": "Santisook", "สันติสุข": "Santisook",
            "por": "Por", "พอ": "Por",
            "knot": "Knot", "น็อต": "Knot",
            "stelligence": "Stelligence"
        }
        
        # ============================================
        # STEP 1.5: Check for COMPLEX QUERIES first
        # ============================================
        # Complex queries have multiple conditions connected with "และ", "ที่", "กับ", etc.
        
        complex_patterns = [
            # Pattern: หาคนที่ทำงาน[ที่ไหน]ที่รู้จักกับ[ใคร]และรู้จักกับ[ใคร]
            (r'หาคน(?:ที่)?ทำงาน(?:ที่)?(.+?)(?:ที่)?รู้จัก(?:กับ)?(.+)', 'org_with_connections'),
            # Pattern: [cohort] มีใครบ้างที่รู้จักกับ [network/person]
            (r'(.+?)\s*มีใครบ้าง(?:ที่)?รู้จัก(?:กับ)?(?:คนใน)?\s*(.+)', 'cohort_with_network'),
            # Pattern: ใคร[ใน X]ที่รู้จัก[Y]
            (r'ใคร(?:ใน|ที่อยู่ใน|ทำงาน(?:ที่)?)?(.+?)(?:ที่)?รู้จัก(?:กับ)?(.+)', 'who_in_x_knows_y'),
            # Pattern: คน[ใน X]รู้จัก[Y]
            (r'คน(?:ใน|ที่อยู่ใน|ทำงาน(?:ที่)?)?(.+?)(?:ที่)?รู้จัก(?:กับ)?(.+)', 'people_in_x_knows_y'),
        ]
        
        for pattern, pattern_name in complex_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                group1 = match.group(1).strip()
                group2 = match.group(2).strip()
                print(f"[DEBUG] Complex pattern '{pattern_name}' matched: group1='{group1}', group2='{group2}'")
                
                # Parse the complex query
                intent = {
                    "type": "complex_query",
                    "query": original_query,
                    "conditions": []
                }
                
                # Parse group1 (usually organization/cohort)
                if 'กระทรวง' in group1:
                    ministry_name = group1.replace('กระทรวง', '').strip()
                    # Get just the ministry name before any trailing words
                    ministry_name = re.sub(r'ที่$', '', ministry_name).strip()
                    intent["conditions"].append({"type": "ministry", "value": ministry_name or group1})
                elif re.search(r'วปอ\.?\s*(?:รุ่น(?:ที่)?\s*)?(\d+)?', group1.lower()):
                    cohort_match = re.search(r'วปอ\.?\s*(?:รุ่น(?:ที่)?\s*)?(\d+)?', group1.lower())
                    intent["conditions"].append({
                        "type": "cohort", 
                        "cohort_type": "วปอ.", 
                        "cohort_number": cohort_match.group(1) if cohort_match.lastindex else None
                    })
                elif re.search(r'nexus\s*(?:ai\s*)?(?:รุ่น(?:ที่)?\s*)?(\d+)?', group1.lower()):
                    cohort_match = re.search(r'nexus\s*(?:ai\s*)?(?:รุ่น(?:ที่)?\s*)?(\d+)?', group1.lower())
                    intent["conditions"].append({
                        "type": "cohort",
                        "cohort_type": "NEXUS",
                        "cohort_number": cohort_match.group(1) if cohort_match.lastindex else None
                    })
                else:
                    intent["conditions"].append({"type": "organization", "value": group1})
                
                # Parse group2 (usually network/person to connect with)
                # Split by "และ" or "กับ" for multiple connection targets
                connection_targets = re.split(r'\s*(?:และ|กับ|or|and)\s*', group2)
                for target in connection_targets:
                    target = target.strip()
                    if not target:
                        continue
                    
                    # Check if it's a Stelligence network
                    target_lower = target.lower().replace('คนใน', '').replace('เครือข่าย', '').strip()
                    is_stelligence = any(sk in target_lower for sk in stelligence_keywords.keys())
                    
                    if is_stelligence:
                        # Find which Stelligence network
                        for sk, sv in stelligence_keywords.items():
                            if sk in target_lower:
                                intent["conditions"].append({"type": "connected_to_stelligence", "network": sv})
                                break
                    elif 'คณะรัฐมนตรี' in target or 'รัฐมนตรี' in target:
                        intent["conditions"].append({"type": "connected_to_cabinet"})
                    else:
                        intent["conditions"].append({"type": "connected_to_person", "person": target})
                
                print(f"[DEBUG] Complex query parsed: {intent}")
                return intent
        
        # Thai question/action words to strip from names
        thai_question_words = [
            'รู้จักใครบ้าง', 'รู้จักใคร', 'รู้จัก', 'ใครบ้าง', 'บ้าง', 
            'มีใครบ้าง', 'ทำงานที่ไหน', 'อยู่ที่ไหน', 'เป็นใคร',
            'คือใคร', 'ติดต่อได้อย่างไร', 'เครือข่ายของ', 'network',
            'connections', 'หน่อย', 'ครับ', 'ค่ะ', 'นะ', 'ด้วย',
            'ได้ไหม', 'ได้มั้ย', 'ช่วย', 'กรุณา', 'please',
            'ที่รู้จักกับ', 'ที่รู้จัก', 'รู้จักกับ', 'มีใครบ้างที่'
        ]
        
        # Path-related keywords
        path_keywords_th = ['เส้นทาง', 'หาเส้นทาง', 'จาก', 'ไป', 'ถึง', 'ไปหา', 'หา', 'ติดต่อ']
        path_keywords_en = ['path', 'route', 'from', 'to', 'connect', 'reach', 'find path', 'shortest']
        
        # Network query keywords  
        network_keywords = ['รู้จัก', 'เครือข่าย', 'network', 'connections', 'knows', 'connected', 'contacts', 'friends']
        
        # (stelligence_keywords already defined above for complex query check)
        
        # Known person names (partial -> full name mapping)
        known_names = {
            # Thai names
            "อนุทิน": "อนุทิน ชาญวีรกูล",
            "อนุทิน ชาญวีรกูล": "อนุทิน ชาญวีรกูล",
            "ชาญวีรกูล": "อนุทิน ชาญวีรกูล",
            "เนติ": "เนติ วงกุหลาบ",
            "เนติ วงกุหลาบ": "เนติ วงกุหลาบ",
            "วงกุหลาบ": "เนติ วงกุหลาบ",
            "อรอนุตตร์": "อรอนุตตร์ สุทธิ์เสงี่ยม",
            "สุทธิ์เสงี่ยม": "อรอนุตตร์ สุทธิ์เสงี่ยม",
            "ประเสริฐ": "ประเสริฐสิน",
            # English transliterations
            "anutin": "อนุทิน ชาญวีรกูล",
            "charnvirakul": "อนุทิน ชาญวีรกูล",
            "neti": "เนติ วงกุหลาบ",
            "wongkulab": "เนติ วงกุหลาบ",
        }
        
        # ============================================
        # STEP 2: Clean query and extract names
        # ============================================
        
        def clean_and_extract_name(text: str) -> str:
            """Remove question words from text to get clean name"""
            cleaned = text.strip()
            # Remove Thai question words from end of string
            for word in sorted(thai_question_words, key=len, reverse=True):
                if cleaned.endswith(word):
                    cleaned = cleaned[:-len(word)].strip()
                if cleaned.startswith(word):
                    cleaned = cleaned[len(word):].strip()
            return cleaned.strip()
        
        def normalize_name(name: str) -> str:
            """Normalize a name to its full form"""
            if not name:
                return name
            name = clean_and_extract_name(name)
            name_lower = name.lower()
            
            # Check Stelligence owners
            if name_lower in stelligence_keywords:
                return stelligence_keywords[name_lower]
            
            # Check known names (exact match)
            if name in known_names:
                return known_names[name]
            if name_lower in known_names:
                return known_names[name_lower]
            
            # Check partial matches - look for known names inside the text
            for partial, full in known_names.items():
                if partial in name or name in partial:
                    return full
            
            return name
        
        def find_known_names_in_text(text: str) -> list:
            """Find any known names that appear in the text"""
            found = []
            text_lower = text.lower()
            
            # Check for Stelligence owners
            for keyword, name in stelligence_keywords.items():
                if keyword in text_lower and name not in found:
                    found.append(name)
            
            # Check for known Thai/English names (longer names first to avoid partial matches)
            sorted_names = sorted(known_names.keys(), key=len, reverse=True)
            for partial_name in sorted_names:
                if partial_name in text and known_names[partial_name] not in found:
                    found.append(known_names[partial_name])
                elif partial_name.lower() in text_lower and known_names[partial_name] not in found:
                    found.append(known_names[partial_name])
            
            return found
        
        def extract_thai_names(text: str) -> list:
            """Extract Thai person names from text, filtering out question words"""
            # FIRST: Try to find known names directly in the text
            known_found = find_known_names_in_text(text)
            if known_found:
                print(f"[DEBUG] Found known names in text: {known_found}")
                return known_found
            
            # If no known names found, try to extract by cleaning
            cleaned = text
            for word in sorted(thai_question_words, key=len, reverse=True):
                cleaned = cleaned.replace(word, ' ')
            
            # Split by common separators
            separators = ['จาก', 'ไป', 'ถึง', 'ไปหา', 'หา', 'และ', 'กับ', 'to', 'from', 'and']
            parts = [cleaned]
            for sep in separators:
                new_parts = []
                for part in parts:
                    new_parts.extend(part.split(sep))
                parts = new_parts
            
            # Filter and normalize names
            names = []
            for part in parts:
                part = part.strip()
                if len(part) >= 2:
                    # Check if it's a known name or looks like a person name
                    normalized = normalize_name(part)
                    if normalized and len(normalized) >= 2:
                        # Avoid duplicates
                        if normalized not in names:
                            names.append(normalized)
            
            return names
        
        # Extract names from query
        extracted_names = extract_thai_names(query)
        print(f"[DEBUG] Extracted names: {extracted_names}")
        
        # Also check for English capitalized names
        english_name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        english_exclude = {'How', 'Can', 'Who', 'What', 'Where', 'When', 'Why', 'The', 'Find', 'Path', 'Show', 'Tell', 'Get', 'Please', 'From', 'To', 'And', 'Or', 'Between'}
        english_names = [m for m in re.findall(english_name_pattern, query) if m not in english_exclude]
        for name in english_names:
            normalized = normalize_name(name)
            if normalized not in extracted_names:
                extracted_names.append(normalized)
        
        print(f"[DEBUG] Final extracted names: {extracted_names}")
        
        # ============================================
        # STEP 3: Determine query intent
        # ============================================
        
        intent = {
            "type": "general",
            "persons": extracted_names,
            "query": original_query
        }
        
        # Check for PATH queries (between two people)
        has_path_keywords = any(kw in query_lower for kw in path_keywords_th + path_keywords_en)
        
        if has_path_keywords and len(extracted_names) >= 2:
            # Path query with two people
            intent["type"] = "shortest_path"
            intent["from_person"] = extracted_names[0]
            intent["to_person"] = extracted_names[1]
            print(f"[DEBUG] Detected PATH intent: {extracted_names[0]} -> {extracted_names[1]}")
            return intent
        
        # Try specific path patterns for Thai
        path_patterns = [
            (r'(?:หา)?เส้นทาง\s*(?:จาก)?\s*(.+?)\s+(?:ไปหา|ไป\s*หา|ไป|ถึง)\s+(.+?)(?:\s*$)', 'th_path'),
            (r'(.+?)\s+(?:ไปหา|ไป\s*หา)\s+(.+?)(?:\s*$)', 'th_goto'),
            (r'จาก\s+(.+?)\s+(?:ไป|ถึง|ไปหา)\s+(.+?)(?:\s*$)', 'th_from'),
            (r'(?:find\s+)?path\s+(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\s*$)', 'en_path'),
            (r'from\s+(.+?)\s+to\s+(.+?)(?:\s*$)', 'en_from'),
        ]
        
        for pattern, pname in path_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                from_raw = clean_and_extract_name(match.group(1))
                to_raw = clean_and_extract_name(match.group(2))
                if len(from_raw) >= 2 and len(to_raw) >= 2:
                    intent["type"] = "shortest_path"
                    intent["from_person"] = normalize_name(from_raw)
                    intent["to_person"] = normalize_name(to_raw)
                    print(f"[DEBUG] Pattern '{pname}' matched: {intent['from_person']} -> {intent['to_person']}")
                    return intent
        
        # Cohort/batch queries (Associate column - NEXUS AI, วปอ.) - CHECK BEFORE network keywords
        # Note: Pattern handles Thai diacritics that may be stripped in some encodings
        # รุ่น can appear as รุ่น, รุน, รน, รุ่นที่, etc.
        cohort_patterns = [
            (r'nexus\s*(?:ai\s*)?(?:ร[ุู]?[่้๊๋]?น(?:ท[ีิ]?[่้๊๋]?)?\s*)?(\d+)?', 'NEXUS'),
            (r'วปอ\.?\s*(?:ร[ุู]?[่้๊๋]?น(?:ท[ีิ]?[่้๊๋]?)?\s*)?(\d+)?', 'วปอ.'),
        ]
        for pattern, cohort_type in cohort_patterns:
            match = re.search(pattern, query_lower)
            if match:
                intent["type"] = "cohort_search"
                intent["cohort_type"] = cohort_type
                intent["cohort_number"] = match.group(1) if match.lastindex else None
                print(f"[DEBUG] Detected COHORT intent: {cohort_type} batch {intent['cohort_number']}")
                return intent
        
        # Check for NETWORK queries (single person)
        has_network_keywords = any(kw in query_lower for kw in network_keywords)
        
        if has_network_keywords and len(extracted_names) >= 1:
            intent["type"] = "person_network"
            intent["person"] = extracted_names[0]
            print(f"[DEBUG] Detected NETWORK intent for: {extracted_names[0]}")
            return intent
        
        # Check for Stelligence owner network
        for keyword, owner_name in stelligence_keywords.items():
            if keyword in query_lower:
                # Check if it's a path query or network query
                if has_path_keywords and len(extracted_names) >= 1:
                    # Find the other person
                    other_person = None
                    for name in extracted_names:
                        if name != owner_name:
                            other_person = name
                            break
                    if other_person:
                        intent["type"] = "shortest_path"
                        intent["from_person"] = owner_name
                        intent["to_person"] = other_person
                        return intent
                
                # Default: show network
                intent["type"] = "stelligence_network"
                intent["network_type"] = owner_name
                return intent
        
        # Organization search
        ministry_match = re.search(r'กระทรวง\s*([ก-๙a-zA-Z\s]+?)(?:\s*$|\s*(?:บ้าง|มี|ใคร|กี่))', query)
        if ministry_match:
            intent["type"] = "organization_search"
            intent["org_type"] = "ministry"
            intent["org_name"] = ministry_match.group(1).strip()
            return intent
        
        # If we have at least one person name, default to person_network
        if len(extracted_names) >= 1:
            intent["type"] = "person_network"
            intent["person"] = extracted_names[0]
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
            # Thai partial names to full names
            "อนุทิน": "อนุทิน ชาญวีรกูล",
            "อนทน": "อนุทิน ชาญวีรกูล",  # Handle encoding issues
            "ชาญวีรกูล": "อนุทิน ชาญวีรกูล",
            "ประเสริฐ": "ประเสริฐสิน",
            "เนติ": "เนติ วงกุหลาบ",
            "วงกุหลาบ": "เนติ วงกุหลาบ",
            "อรอนุตตร์": "อรอนุตตร์ สุทธิ์เสงี่ยม",
            "สุทธิ์เสงี่ยม": "อรอนุตตร์ สุทธิ์เสงี่ยม",
            # English transliterations (lowercase keys)
            "anutin": "อนุทิน ชาญวีรกูล",
            "charnvirakul": "อนุทิน ชาญวีรกูล",
            "prasertsin": "ประเสริฐสิน",
            "neti": "เนติ วงกุหลาบ",
            "wongkulab": "เนติ วงกุหลาบ",
        }

        # Case-insensitive expansion: lower the incoming names for lookup
        to_lc = to_person.lower() if isinstance(to_person, str) else to_person
        from_lc = from_person.lower() if isinstance(from_person, str) else from_person

        # Try exact match first, then partial match
        if isinstance(to_lc, str):
            if to_lc in known_names:
                to_person = known_names[to_lc]
                print(f"[DEBUG] Expanded to_person (exact): {to_person}")
            else:
                # Try partial matching
                for partial, full in known_names.items():
                    if partial in to_lc or to_lc in partial:
                        to_person = full
                        print(f"[DEBUG] Expanded to_person (partial): {to_person}")
                        break
                        
        if isinstance(from_lc, str):
            if from_lc in known_names:
                from_person = known_names[from_lc]
                print(f"[DEBUG] Expanded from_person (exact): {from_person}")
            else:
                # Try partial matching
                for partial, full in known_names.items():
                    if partial in from_lc or from_lc in partial:
                        from_person = full
                        print(f"[DEBUG] Expanded from_person (partial): {from_person}")
                        break
            
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
        """Search for people by cohort/batch (from Associate column). Uses retry logic."""
        # Build search pattern
        if cohort_name:
            search_pattern = cohort_name.upper()
        elif cohort_type and cohort_number:
            search_pattern = f"{cohort_type} รุ่นที่ {cohort_number}"
        elif cohort_type:
            search_pattern = cohort_type
        else:
            search_pattern = "NEXUS"  # Default

        return self._execute_with_retry(self._search_by_cohort_impl, search_pattern)
    
    def _search_by_cohort_impl(self, search_pattern: str) -> Dict:
        """Internal implementation of search_by_cohort.
        
        Uses the Associate relationship to find cohort members.
        Aggregates across ALL matching Associate nodes (handles duplicates with different whitespace).
        """
        with self.driver.session() as session:
            
            # First, get all matching persons across all Associate variations
            result = session.run("""
                MATCH (p:Person)-[:associate_with]->(a:Associate)
                WHERE toLower(trim(a.Associate)) CONTAINS toLower(trim($pattern))
                WITH DISTINCT p, trim(a.Associate) as cohort
                OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
                OPTIONAL MATCH (p)-[:work_at]->(agency:Agency)
                OPTIONAL MATCH (p)-[:under]->(ministry:Ministry)
                WITH collect({
                    name: p.`ชื่อ-นามสกุล`,
                    position: pos.`ตำแหน่ง`,
                    agency: agency.`หน่วยงาน`,
                    ministry: ministry.`กระทรวง`
                }) as all_members, 
                collect(DISTINCT cohort)[0] as cohort_name
                RETURN cohort_name, all_members, size(all_members) as member_count
            """, pattern=search_pattern)
            
            record = result.single()
            
            if record and record["member_count"] > 0:
                # Deduplicate members by name, filter out empty/whitespace-only names
                seen_names = set()
                members_detailed = []
                for m in record["all_members"]:
                    name = m.get("name")
                    if name and name.strip() and name.strip() not in seen_names:
                        seen_names.add(name.strip())
                        members_detailed.append({
                            "name": name.strip(),
                            "position": m.get("position"),
                            "agency": m.get("agency"),
                            "ministry": m.get("ministry")
                        })
                
                return {
                    "found": True,
                    "cohort": (record["cohort_name"] or search_pattern).strip(),
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
        
        elif intent["type"] == "complex_query":
            return {
                "intent": intent,
                "result": self.execute_complex_query(intent["conditions"])
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
                              "- 'NEXUS AI รุ่น 1 มีใครบ้าง'\n" +
                              "- 'วปอ. รุ่น 68 ใครรู้จัก Stelligence'"
                }
            }
    
    def execute_complex_query(self, conditions: list) -> Dict:
        """
        Execute complex multi-condition queries.
        Builds dynamic Cypher based on conditions.
        """
        return self._execute_with_retry(self._execute_complex_query_impl, conditions)
    
    def _execute_complex_query_impl(self, conditions: list) -> Dict:
        """Internal implementation for complex queries.
        
        Uses a step-by-step approach: find people matching primary condition,
        then filter by additional connection requirements.
        """
        
        print(f"[DEBUG] Executing complex query with conditions: {conditions}")
        
        stelligence_rels = {"Santisook": "santisook_known", "Por": "por_known", "Knot": "knot_known", "Stelligence": None}
        
        # Separate primary conditions (what org/cohort person belongs to) 
        # from connection conditions (who they should know)
        primary_conditions = []
        connection_conditions = []
        
        for cond in conditions:
            cond_type = cond.get("type", "")
            if cond_type in ["ministry", "cohort", "organization"]:
                primary_conditions.append(cond)
            else:
                connection_conditions.append(cond)
        
        # Build query based on primary condition
        if not primary_conditions:
            # No primary condition, start with all people who match connections
            base_query = "MATCH (p:Person)"
            where_parts = []
            params = {}
        else:
            # Use first primary condition as base
            primary = primary_conditions[0]
            ptype = primary.get("type")
            params = {}
            
            if ptype == "ministry":
                ministry_name = primary.get("value", "")
                base_query = """
                MATCH (p:Person)-[:under]->(m:Ministry)
                WHERE m.`กระทรวง` CONTAINS $ministry_name
                """
                params["ministry_name"] = ministry_name
                
            elif ptype == "cohort":
                cohort_type = primary.get("cohort_type", "")
                cohort_number = primary.get("cohort_number")
                # Check both Connect by and Associate nodes for cohort data
                if cohort_number:
                    base_query = """
                    MATCH (p:Person)
                    WHERE (p)-[:connect_by]->(:`Connect by` {`Connect by`: $cohort_full})
                       OR (p)-[:associate_with]->(:Associate {Associate: $cohort_full})
                       OR EXISTS {
                           MATCH (p)-[:connect_by]->(cb:`Connect by`)
                           WHERE cb.`Connect by` CONTAINS $cohort_type AND cb.`Connect by` CONTAINS $cohort_num
                       }
                       OR EXISTS {
                           MATCH (p)-[:associate_with]->(a:Associate)
                           WHERE a.Associate CONTAINS $cohort_type AND a.Associate CONTAINS $cohort_num
                       }
                    """
                    # Build full cohort name like "NEXUS รุ่นที่ 1" or "วปอ. รุ่นที่ 68"
                    cohort_full = f"{cohort_type} รุ่นที่ {cohort_number}"
                    params["cohort_type"] = cohort_type
                    params["cohort_num"] = str(cohort_number)
                    params["cohort_full"] = cohort_full
                else:
                    base_query = """
                    MATCH (p:Person)
                    WHERE EXISTS {
                           MATCH (p)-[:connect_by]->(cb:`Connect by`)
                           WHERE cb.`Connect by` CONTAINS $cohort_type
                       }
                       OR EXISTS {
                           MATCH (p)-[:associate_with]->(a:Associate)
                           WHERE a.Associate CONTAINS $cohort_type
                       }
                    """
                    params["cohort_type"] = cohort_type
                    
            elif ptype == "organization":
                org_name = primary.get("value", "")
                # Try both Agency and Connect by - some orgs like OSK115 are stored as Connect by
                base_query = """
                MATCH (p:Person)
                WHERE (p)-[:work_at]->(:Agency {`หน่วยงาน`: $org_name})
                   OR (p)-[:connect_by]->(:`Connect by` {`Connect by`: $org_name})
                   OR EXISTS {
                       MATCH (p)-[:work_at]->(org:Agency)
                       WHERE org.`หน่วยงาน` CONTAINS $org_name
                   }
                   OR EXISTS {
                       MATCH (p)-[:connect_by]->(cb:`Connect by`)
                       WHERE cb.`Connect by` CONTAINS $org_name
                   }
                """
                params["org_name"] = org_name
            else:
                base_query = "MATCH (p:Person)"
        
        # Add connection conditions as additional patterns
        with_clause = "WITH DISTINCT p"
        connection_matches = []
        
        for i, conn_cond in enumerate(connection_conditions):
            conn_type = conn_cond.get("type", "")
            
            if conn_type == "connected_to_stelligence":
                network = conn_cond.get("network", "Stelligence")
                if network == "Stelligence":
                    # Any Stelligence network - find people who are 1-3 hops from any Stelligence member
                    connection_matches.append(f"""
                    MATCH (p)-[*1..3]-(stel_member{i}:Person)
                    WHERE (stel_member{i})<-[:santisook_known]-(:Santisook) 
                       OR (stel_member{i})<-[:por_known]-(:Por) 
                       OR (stel_member{i})<-[:knot_known]-(:Knot)
                    WITH DISTINCT p
                    """)
                else:
                    rel_type = stelligence_rels.get(network, "santisook_known")
                    connection_matches.append(f"""
                    MATCH (p)-[*1..3]-(stel_member{i}:Person)<-[:{rel_type}]-(:{network})
                    WITH DISTINCT p
                    """)
                    
            elif conn_type == "connected_to_cabinet":
                connection_matches.append(f"""
                MATCH (p)-[*1..3]-(cabinet{i}:Person)-[:work_as]->(cab_pos{i}:Position)
                WHERE cab_pos{i}.`ตำแหน่ง` CONTAINS 'รัฐมนตรี'
                WITH DISTINCT p
                """)
                
            elif conn_type == "connected_to_person":
                person_name = conn_cond.get("person", "")
                param_name = f"target_person_{i}"
                params[param_name] = person_name
                connection_matches.append(f"""
                MATCH (p)-[*1..3]-(target{i}:Person)
                WHERE target{i}.`ชื่อ-นามสกุล` CONTAINS ${param_name}
                WITH DISTINCT p
                """)
        
        # Build final query
        query = base_query + "\n" + with_clause + "\n"
        for conn_match in connection_matches:
            query += conn_match + "\n"
        
        # Final return with person details
        query += """
        OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
        OPTIONAL MATCH (p)-[:work_at]->(agency:Agency)
        OPTIONAL MATCH (p)-[:under]->(ministry:Ministry)
        WITH DISTINCT p, 
             head(collect(DISTINCT pos.`ตำแหน่ง`)) as position,
             head(collect(DISTINCT agency.`หน่วยงาน`)) as agency,
             head(collect(DISTINCT ministry.`กระทรวง`)) as ministry
        RETURN p.`ชื่อ-นามสกุล` as name,
               position,
               agency,
               ministry
        LIMIT 30
        """
        
        print(f"[DEBUG] Complex query Cypher:\n{query}")
        print(f"[DEBUG] Query params: {params}")
        
        # Execute
        with self.driver.session() as session:
            try:
                result = session.run(query, **params)
                
                people = []
                for record in result:
                    if record["name"]:  # Only include if name exists
                        people.append({
                            "name": record["name"],
                            "position": record["position"],
                            "agency": record["agency"],
                            "ministry": record["ministry"]
                        })
                
                return {
                    "found": len(people) > 0,
                    "count": len(people),
                    "people": people,
                    "conditions": conditions,
                    "message": f"พบ {len(people)} คนที่ตรงตามเงื่อนไข" if people else "ไม่พบข้อมูลที่ตรงตามเงื่อนไขที่กำหนด"
                }
            except Exception as e:
                print(f"[ERROR] Complex query failed: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "found": False,
                    "count": 0,
                    "people": [],
                    "conditions": conditions,
                    "message": f"เกิดข้อผิดพลาดในการค้นหา: {str(e)}"
                }


# Singleton instance
_agent_instance = None

def get_network_agent() -> NetworkAgent:
    """Get or create singleton network agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = NetworkAgent()
    return _agent_instance

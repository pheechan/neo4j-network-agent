# Neo4j Aura Agent Configuration - Refined

## 🎯 Agent Setup

### **Title**
`Thai Government Network Intelligence Agent`

### **Description**
`AI agent for analyzing Thai government personnel networks, direct relationships, and organizational hierarchies. Specializes in finding meaningful person-to-person connections.`

### **Instructions**
```
You are a Thai Government Network Analysis Assistant specialized in finding DIRECT relationships and connection paths between government officials.

**Your Core Mission:**
- Find REAL person-to-person connection paths (not just organizational hierarchy)
- Analyze direct relationships between officials
- Identify influence networks (Stelligence, Santisook, Por, Knot)
- Provide organizational context (positions, agencies, ministries)

**Critical Understanding:**
- Person-to-person connections use: connect_by, stelligence_known, santisook_known, por_known, knot_known
- Organizational structure uses: work_as (Person→Position), work_at (Person→Agency), under (Agency→Ministry)
- NEVER confuse organizational hierarchy with personal connections
- A path through "Ministry→Agency" is NOT a person-to-person connection

**Database Schema:**
Node Types:
- Person: ชื่อ-นามสกุล (full name)
- Position: ตำแหน่ง (position title)
- Agency: หน่วยงาน (agency name)
- Ministry: กระทรวง (ministry name)

Relationship Types:
1. PERSONAL CONNECTIONS (what users want):
   - connect_by: General connection between people
   - stelligence_known: Stelligence network member
   - santisook_known: Santisook network member
   - por_known: Por network connection
   - knot_known: Knot network connection

2. ORGANIZATIONAL STRUCTURE (context only):
   - work_as: Person → Position (their role)
   - work_at: Person → Agency (their workplace)
   - under: Agency → Ministry (hierarchy)

**Response Guidelines:**
1. Always respond in Thai unless asked otherwise
2. Use visual formatting:
   - Path boxes: ┌─────────┐
   - Arrows: → (normal), ⇒ (special network)
   - Person cards: ╔═══════╗
   - Bullets: • or ✓
3. For connection queries:
   - If REAL path exists: Show person→person connections with relationship types
   - If NO direct path: Say "ไม่พบเส้นทางเชื่อมโยงโดยตรง" then show both persons' details
   - Include special network indicators (🌟 for Stelligence/Santisook)
4. For "no connection": Still provide useful context:
   - Both persons' positions and agencies
   - Suggest they may be in different networks
   - Show if they work in same ministry (organizational proximity, not connection)
5. Always distinguish between:
   - "เชื่อมโยงโดยตรง" (direct personal connection)
   - "อยู่ในองค์กรเดียวกัน" (same organization, not necessarily connected)

**When to Use Each Tool:**
1. Vector Similarity: ALWAYS use first for any person name mentioned
2. find_connection_path: For "หาเส้นทาง", "เชื่อมโยง", "รู้จัก" queries
3. get_person_details: For "ข้อมูล", "รายละเอียด", single person info
4. get_ministry_hierarchy: For "โครงสร้าง", ministry/agency structure
5. find_colleagues: For "เพื่อนร่วมงาน", people in same agency
6. Text2Cypher: For complex stats, aggregations, or exploratory queries

**Common Mistakes to Avoid:**
❌ DON'T say people are connected just because they're in the same ministry
❌ DON'T show organizational hierarchy (Ministry→Agency) as a connection path
❌ DON'T return paths with only "under" or "work_at" relationships as connections
✅ DO show only person→person relationships as real connections
✅ DO provide organizational context separately
✅ DO be honest when no direct connection exists
```

---

## 🛠️ Tool Configuration

### **1. Vector Similarity Tool**
**Name:** `find_person_by_name`

**Description:** 
```
Find Thai government personnel by their name (ชื่อ-นามสกุล). Use this FIRST whenever a person's name is mentioned. Handles partial names, nicknames, and fuzzy matching for Thai names.
```

**Configuration:**
- Embedding Provider: OpenAI (or Vertex AI)
- Embedding Model: `text-embedding-3-small`
- Vector Index Name: `person_vector_index`
- Top-K: `5`

---

### **2. Cypher Template: Find Connection Path** ⭐ FIXED
**Name:** `find_connection_path`

**Description:**
```
Find DIRECT person-to-person connection paths between two officials. Only returns paths with actual personal relationships (connect_by, stelligence_known, santisook_known, por_known, knot_known). Does NOT return organizational hierarchy paths.
```

**Cypher Query:**
```cypher
MATCH (start:Person {`ชื่อ-นามสกุล`: $person1})
MATCH (end:Person {`ชื่อ-นามสกุล`: $person2})

// Only follow person-to-person relationship types
MATCH path = shortestPath((start)-[rels:connect_by|stelligence_known|santisook_known|por_known|knot_known*1..5]-(end))

WHERE ALL(r in relationships(path) WHERE type(r) IN ['connect_by', 'stelligence_known', 'santisook_known', 'por_known', 'knot_known'])

WITH path, relationships(path) as rels,
  size([r in relationships(path) WHERE type(r) IN ['stelligence_known', 'santisook_known']]) as special_count

ORDER BY special_count DESC, length(path) ASC
LIMIT 3

RETURN 
  [n in nodes(path) | {
    name: n.`ชื่อ-นามสกุล`,
    type: 'Person'
  }] as path_nodes,
  [r in rels | {
    type: type(r),
    is_special: type(r) IN ['stelligence_known', 'santisook_known']
  }] as relationships,
  length(path) as path_length,
  special_count
```

**Parameters:**
- `person1` (String): First person's full name
- `person2` (String): Second person's full name

**Notes:**
- Only returns REAL person-to-person connections
- Prioritizes paths through special networks (Stelligence/Santisook)
- Maximum 5 hops to avoid irrelevant distant connections

---

### **3. Cypher Template: Get Person Details**
**Name:** `get_person_details`

**Description:**
```
Get comprehensive information about a specific person including their position, agency, ministry, and all their personal connections to other people.
```

**Cypher Query:**
```cypher
MATCH (p:Person {`ชื่อ-นามสกุล`: $person_name})

// Get organizational context
OPTIONAL MATCH (p)-[:work_as]->(pos:Position)
OPTIONAL MATCH (p)-[:work_at]->(agency:Agency)
OPTIONAL MATCH (agency)-[:under]->(ministry:Ministry)

// Get ONLY personal connections (not organizational structure)
OPTIONAL MATCH (p)-[conn:connect_by|stelligence_known|santisook_known|por_known|knot_known]-(other:Person)

WITH p, pos, agency, ministry, 
  collect(DISTINCT {
    connection_type: type(conn), 
    person: other.`ชื่อ-นามสกุล`,
    is_special: type(conn) IN ['stelligence_known', 'santisook_known']
  }) as connections

RETURN 
  p.`ชื่อ-นามสกุล` as name,
  pos.ตำแหน่ง as position,
  agency.หน่วยงาน as agency,
  ministry.กระทรวง as ministry,
  connections,
  size(connections) as total_connections
LIMIT 1
```

**Parameters:**
- `person_name` (String): Person's full name

---

### **4. Cypher Template: Get Ministry Hierarchy**
**Name:** `get_ministry_hierarchy`

**Description:**
```
Get the organizational structure of a ministry, showing agencies and key personnel. Use for organizational questions, NOT for finding connections between people.
```

**Cypher Query:**
```cypher
MATCH (ministry:Ministry {กระทรวง: $ministry_name})
OPTIONAL MATCH (ministry)<-[:under]-(agency:Agency)
OPTIONAL MATCH (agency)<-[:work_at]-(person:Person)-[:work_as]->(pos:Position)

WITH ministry, agency, 
  collect({
    person: person.`ชื่อ-นามสกุล`,
    position: pos.ตำแหน่ง,
    agency: agency.หน่วยงาน
  }) as personnel

RETURN 
  ministry.กระทรวง as ministry_name,
  collect(DISTINCT agency.หน่วยงาน) as agencies,
  personnel[..30] as key_personnel,
  size(personnel) as total_personnel
LIMIT 1
```

**Parameters:**
- `ministry_name` (String): Ministry name in Thai

---

### **5. Cypher Template: Find Colleagues** (NEW)
**Name:** `find_colleagues`

**Description:**
```
Find people who work in the same agency as the specified person. Use when user asks about "เพื่อนร่วมงาน" or "คนในหน่วยงานเดียวกัน".
```

**Cypher Query:**
```cypher
MATCH (p:Person {`ชื่อ-นามสกุล`: $person_name})-[:work_at]->(agency:Agency)
MATCH (colleague:Person)-[:work_at]->(agency)
WHERE colleague.`ชื่อ-นามสกุล` <> $person_name

OPTIONAL MATCH (colleague)-[:work_as]->(pos:Position)

RETURN 
  agency.หน่วยงาน as agency_name,
  collect({
    name: colleague.`ชื่อ-นามสกุล`,
    position: pos.ตำแหน่ง
  })[..20] as colleagues,
  count(colleague) as total_colleagues
LIMIT 1
```

**Parameters:**
- `person_name` (String): Person's full name

---

### **6. Cypher Template: Check Same Ministry** (NEW)
**Name:** `check_same_ministry`

**Description:**
```
Check if two people work in the same ministry or related organizations. Use when no direct connection path exists to provide organizational proximity context.
```

**Cypher Query:**
```cypher
MATCH (p1:Person {`ชื่อ-นามสกุล`: $person1})
MATCH (p2:Person {`ชื่อ-นามสกุล`: $person2})

OPTIONAL MATCH (p1)-[:work_at]->(a1:Agency)-[:under]->(m1:Ministry)
OPTIONAL MATCH (p2)-[:work_at]->(a2:Agency)-[:under]->(m2:Ministry)

RETURN 
  p1.`ชื่อ-นามสกุล` as person1,
  p2.`ชื่อ-นามสกุล` as person2,
  a1.หน่วยงาน as agency1,
  a2.หน่วยงาน as agency2,
  m1.กระทรวง as ministry1,
  m2.กระทรวง as ministry2,
  CASE 
    WHEN m1 = m2 THEN true 
    ELSE false 
  END as same_ministry,
  CASE 
    WHEN a1 = a2 THEN true 
    ELSE false 
  END as same_agency
```

**Parameters:**
- `person1` (String): First person's full name
- `person2` (String): Second person's full name

---

### **7. Text2Cypher Tool**
**Name:** `text2cypher_search`

**Description:**
```
Generate and execute custom Cypher queries for complex analysis, statistics, aggregations, or exploratory questions about the network that aren't covered by other tools.
```

**Instructions:**
```
Use Text2Cypher ONLY when:
✅ User asks for statistics (e.g., "มีกี่คนในกระทรวง", "นับจำนวน")
✅ Complex aggregations or GROUP BY queries
✅ Exploratory network analysis (e.g., "ใครมี connections เยอะที่สุด")
✅ Filtering with multiple conditions
✅ Queries not covered by existing Cypher Template tools

DO NOT use Text2Cypher for:
❌ Finding people by name → Use Vector Similarity instead
❌ Finding connection paths → Use find_connection_path
❌ Person details → Use get_person_details
❌ Ministry structure → Use get_ministry_hierarchy
❌ Same agency check → Use find_colleagues

**Critical Database Rules:**
1. Node Labels: Person, Position, Agency, Ministry
2. Thai property names MUST use backticks:
   - Person: `ชื่อ-นามสกุล`
   - Position: ตำแหน่ง
   - Agency: หน่วยงาน
   - Ministry: กระทรวง
3. Relationship types (case-sensitive):
   - Personal: connect_by, stelligence_known, santisook_known, por_known, knot_known
   - Organizational: work_as, work_at, under
4. Always LIMIT results to 10-50 rows maximum
5. Return only text/numbers, NO embeddings or full node objects
6. Use DISTINCT to avoid duplicates
7. When finding connections, ONLY use personal relationship types

**Example patterns:**
- Count people: `MATCH (p:Person) RETURN count(p)`
- Top connected: `MATCH (p:Person)-[r:connect_by|stelligence_known|santisook_known]-() RETURN p.\`ชื่อ-นามสกุล\`, count(r) as connections ORDER BY connections DESC LIMIT 10`
- Ministry stats: `MATCH (m:Ministry)<-[:under]-(a:Agency)<-[:work_at]-(p:Person) RETURN m.กระทรวง, count(DISTINCT p) as people_count ORDER BY people_count DESC`
```

---

## 📋 Tool Selection Priority

For user queries, follow this decision tree:

1. **Does query mention person names?**
   → Use `find_person_by_name` (Vector Similarity) FIRST

2. **Query type:**
   - "หาเส้นทาง", "เชื่อมโยง", "รู้จัก" → `find_connection_path`
   - "ข้อมูล", "รายละเอียด" + single person → `get_person_details`
   - "เพื่อนร่วมงาน", "คนในหน่วยงาน" → `find_colleagues`
   - "โครงสร้าง", ministry info → `get_ministry_hierarchy`
   - "มีกี่คน", "นับจำนวน", statistics → `text2cypher_search`

3. **If no direct connection found:**
   - Use `check_same_ministry` to show organizational proximity
   - Use `get_person_details` for both people to show context

---

## 🎯 Example Responses

### When Direct Connection EXISTS:
```
✅ พบเส้นทางเชื่อมโยงโดยตรง:

╔═══════════════════════════════════════╗
║  ประเสริฐ สินสุขประเสริฐ              ║
╠═══════════════════════════════════════╣
║  ⇓ connect_by (เชื่อมโยงทั่วไป)      ║
╠═══════════════════════════════════════╣
║  สมชาย ใจดี                          ║
╠═══════════════════════════════════════╣
║  ⇓ stelligence_known 🌟              ║
╠═══════════════════════════════════════╣
║  พิพัฒน์ รัชกิจประการ                ║
╚═══════════════════════════════════════╝

📊 สรุป:
• ระยะทาง: 2 ขั้น (ผ่าน 1 คนกลาง)
• เครือข่ายพิเศษ: Stelligence 🌟
• ความแข็งแกร่งของเชื่อมโยง: สูง
```

### When NO Direct Connection:
```
❌ ไม่พบเส้นทางเชื่อมโยงโดยตรงระหว่าง:

┌───────────────────────────────────────┐
│ ประเสริฐ สินสุขประเสริฐ                │
├───────────────────────────────────────┤
│ 💼 ตำแหน่ง: รัฐมนตรีว่าการ            │
│ 🏢 หน่วยงาน: สำนักงานรัฐมนตรี         │
│ 🏛️ กระทรวง: กระทรวงพลังงาน           │
└───────────────────────────────────────┘

┌───────────────────────────────────────┐
│ พิพัฒน์ รัชกิจประการ                  │
├───────────────────────────────────────┤
│ 💼 ตำแหน่ง: ผู้อำนวยการ               │
│ 🏢 หน่วยงาน: สำนักงานรัฐมนตรี         │
│ 🏛️ กระทรวง: กระทรวงพลังงาน           │
└───────────────────────────────────────┘

ℹ️ ข้อสังเกต:
• ทั้งสองท่านอยู่ในกระทรวงเดียวกัน (กระทรวงพลังงาน)
• แต่ไม่มีการเชื่อมโยงโดยตรงในระบบ
• อาจอยู่คนละเครือข่าย หรือยังไม่มีข้อมูลความสัมพันธ์
```

---

## 🧪 Testing Queries

Test with these queries after setup:

1. **Direct Connection (should work):**
   - `"หาเส้นทางจาก [person1] ไป [person2]"` (if they're actually connected)

2. **No Connection (should handle gracefully):**
   - `"ประเสริฐ สินสุขประเสริฐ รู้จัก พิพัฒน์ รัชกิจประการ ไหม"`

3. **Person Details:**
   - `"บอกข้อมูลของ ประเสริฐ สินสุขประเสริฐ"`

4. **Organizational:**
   - `"มีใครบ้างในกระทรวงพลังงาน"`

5. **Colleagues:**
   - `"ใครเป็นเพื่อนร่วมงานของ ประเสริฐ สินสุขประเสริฐ"`

---

## ⚠️ Key Improvements Over Previous Config

1. ✅ **Cypher query now ONLY matches person-to-person relationships** - won't return organizational hierarchy as "connections"
2. ✅ **Added explicit relationship type filtering** - prevents false positives
3. ✅ **New tools for handling "no connection" cases** - check_same_ministry, find_colleagues
4. ✅ **Clear instructions to distinguish** organizational vs personal relationships
5. ✅ **Better agent reasoning** - knows when to use which tool
6. ✅ **Improved response templates** - clear visual distinction between connection types

This configuration will give you ACCURATE results! 🎯

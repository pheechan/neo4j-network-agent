# LLM Logic Improvements

## Current Issues

### 1. Intent Detection is Too Rigid
Current logic uses keyword matching which misses many natural queries:
- "ช่วยหาคนที่รู้จักกับปลัดกระทรวงพลังงาน" - won't be detected
- "มีใครบ้างที่ทำงานกระทรวงมหาดไทย" - won't be detected

### 2. System Prompt is Generic
Current prompt doesn't guide LLM on how to format Thai responses naturally.

### 3. No Context Prioritization
All retrieved nodes are weighted equally - no scoring by relevance.

### 4. No Query Reformulation
If first query fails, system doesn't try alternative phrasings.

---

## Recommended Improvements

### 1. Enhanced Intent Detection

```python
def detect_query_intent_v2(query: str) -> dict:
    """
    Use both keyword matching AND LLM-based classification
    """
    # First: Quick keyword classification
    intent = quick_keyword_classify(query)
    
    # If unclear, use LLM to classify
    if intent["type"] == "general" or intent["confidence"] < 0.7:
        intent = llm_classify_intent(query)
    
    return intent

def llm_classify_intent(query: str) -> dict:
    """Use LLM to classify the query intent"""
    prompt = f'''
    Classify this Thai query about a network/organization database:
    Query: "{query}"
    
    Categories:
    1. person_search - Looking for specific person(s)
    2. network_members - Who is in a network/group
    3. shortest_path - How to connect two people
    4. mutual_connections - Common connections between people
    5. organization_search - Looking for people in org/ministry
    6. introduction - Who can introduce me to someone
    7. general - Other queries
    
    Return JSON: {{"type": "...", "entities": [...], "confidence": 0.0-1.0}}
    '''
    # Call LLM and parse response
```

### 2. Better System Prompt for Thai

```python
THAI_SYSTEM_PROMPT = """คุณเป็นผู้ช่วยอัจฉริยะที่เชี่ยวชาญในการวิเคราะห์ข้อมูลเครือข่ายบุคคลและองค์กร

กฎสำคัญ:
1. ตอบเป็นภาษาไทยเสมอ ยกเว้นชื่อเฉพาะภาษาอังกฤษ
2. ใช้ข้อมูลจาก Context ที่ให้มาเท่านั้น ห้ามแต่งเอง
3. ถ้าไม่มีข้อมูล ให้บอกตรงๆ ว่า "ไม่พบข้อมูลที่ต้องการ"
4. แสดงผลลัพธ์ในรูปแบบที่อ่านง่าย ใช้หัวข้อและรายการ

รูปแบบการตอบ:
- เริ่มด้วยคำตอบโดยตรง
- ให้รายละเอียดสนับสนุน (ตำแหน่ง, หน่วยงาน, กระทรวง)
- แนะนำคำถามต่อเนื่องที่เกี่ยวข้อง

ตัวอย่าง:
คำถาม: "ใครบ้างอยู่ในเครือข่าย Santisook"
คำตอบที่ดี: 
"เครือข่าย Santisook มีสมาชิก 9 คน ได้แก่:

1. **เนติ วงกุหลาบ** - กองบัญชาการตำรวจสอบสวนกลาง (CIB)
2. **พี่โด่ง** - รมต. กระทรวงพลังงาน
...

💡 คุณอาจสนใจ:
- ค้นหาคนที่รู้จักกับสมาชิกคนใดคนหนึ่ง
- ดูเครือข่าย Por หรือ Knot ที่เกี่ยวข้อง"
"""
```

### 3. Context Scoring and Ranking

```python
def score_and_rank_context(nodes: List[dict], query: str) -> List[dict]:
    """
    Score nodes by relevance to query
    """
    query_terms = set(query.lower().split())
    
    scored_nodes = []
    for node in nodes:
        score = 0
        
        # Name match (highest weight)
        name = node.get('ชื่อ-นามสกุล', '').lower()
        if any(term in name for term in query_terms):
            score += 10
        
        # Position match
        position = node.get('ตำแหน่ง', '').lower()
        if any(term in position for term in query_terms):
            score += 5
            
        # Agency/Ministry match
        agency = node.get('หน่วยงาน', '').lower()
        ministry = node.get('กระทรวง', '').lower()
        if any(term in agency + ministry for term in query_terms):
            score += 3
        
        scored_nodes.append({**node, '__score__': score})
    
    # Sort by score descending
    return sorted(scored_nodes, key=lambda x: x['__score__'], reverse=True)
```

### 4. Query Reformulation

```python
def reformulate_query(original_query: str, attempt: int) -> str:
    """
    Generate alternative query phrasings if initial search fails
    """
    reformulations = {
        1: lambda q: q.replace("รู้จัก", "เชื่อมโยงกับ"),
        2: lambda q: q.replace("เครือข่าย", "network"),
        3: lambda q: extract_names_only(q),  # Just search for person names
    }
    
    if attempt in reformulations:
        return reformulations[attempt](original_query)
    return original_query
```

### 5. Smarter Follow-up Questions

```python
def generate_followup_questions(intent: dict, result: dict) -> List[str]:
    """
    Generate contextual follow-up questions
    """
    followups = []
    
    if intent["type"] == "network_members":
        network = intent.get("network")
        followups.append(f"ดูรายละเอียดตำแหน่งของสมาชิก {network}")
        followups.append(f"หาคนที่เชื่อมต่อกับเครือข่าย {network} อื่น")
        
    elif intent["type"] == "person_search":
        person = result.get("person_name")
        if person:
            followups.append(f"{person} รู้จักใครบ้าง")
            followups.append(f"ใครสามารถแนะนำ {person} ได้")
    
    return followups[:3]  # Max 3 suggestions
```

---

## New Query Types to Support

### 1. Organization-based Search
```
"ใครบ้างที่ทำงานกระทรวงพลังงาน"
"แสดงคนในสำนักงานปลัด"
```

### 2. Position-based Search
```
"หาปลัดกระทรวงทั้งหมด"
"ใครเป็น CEO บ้าง"
```

### 3. Cross-network Connections
```
"ใครที่อยู่ทั้ง Por และ Knot"
"คนที่รู้จักทั้ง Santisook และ OSK115"
```

### 4. Cohort/Batch Queries
```
"NEXIS รุ่น 1 มีใครบ้าง"
"หาคนจาก วปอ. รุ่น 68"
```

---

## Implementation Priority

1. **High Priority (Quick Wins)**
   - Better Thai system prompt
   - Add organization/ministry search
   - Add cohort/associate search
   
2. **Medium Priority**
   - Context scoring and ranking
   - Follow-up question generation
   
3. **Lower Priority (Requires Model Changes)**
   - Person-to-Person path finding
   - Cross-network analysis
   - LLM-based intent classification

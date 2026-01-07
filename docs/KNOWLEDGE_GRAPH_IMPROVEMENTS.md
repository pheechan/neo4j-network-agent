# Knowledge Graph Improvements

## Current Issues & Recommended Changes

### 1. Enhanced Person Node Properties

**Current:**
```cypher
(:Person {`ชื่อ-นามสกุล`: "ประเสริฐ สินสุขประเสริฐ"})
```

**Recommended:**
```cypher
(:Person {
  `ชื่อ-นามสกุล`: "ประเสริฐ สินสุขประเสริฐ",
  `ชื่อ`: "ประเสริฐ",
  `นามสกุล`: "สินสุขประเสริฐ",
  `ชื่อเล่น`: "เปิ้ล",
  `คำนำหน้า`: "นาย",
  fulltext_search: "ประเสริฐ สินสุขประเสริฐ เปิ้ล ปลัดกระทรวง สำนักงานปลัด กระทรวงพลังงาน Por"
})
```

Benefits:
- Better search with nicknames and partial names
- Fuzzy matching on any name variant
- Full-text search index for fast queries

### 2. Add Direct Person-to-Person Relationships

**New Relationship Type: `KNOWS`**
```cypher
(person1:Person)-[:KNOWS {
  via: "OSK115",
  strength: 3,  // 1-5 scale
  type: "professional"
}]->(person2:Person)
```

This enables:
- Finding "who can introduce me to X"
- Shortest path between any two people
- Network analysis (degree centrality, etc.)

### 3. Fix Stelligence Relationship Direction

**Current (inverted):**
```cypher
(:Santisook)-[:santisook_known]->(:Person)
```

**Recommended:**
```cypher
(:Person)-[:MEMBER_OF {network: "Santisook"}]->(:Network)
// OR keep current but document clearly
```

### 4. Merge Duplicate Nodes

**Problem:** Same agency appears multiple times with slight variations:
- "สำนักงานปลัด" vs "สำนักงานปลัดกระทรวง"
- "กรมสรรพากร" vs "กรมสรรพากร " (trailing space)

**Solution:** Data cleansing before import

### 5. Add Cohort/Batch Node

**For NEXIS, วปอ. connections:**
```cypher
(:Cohort {
  name: "NEXIS รุ่นที่ 1",
  type: "training_program",
  year: 2024
})

(:Person)-[:ALUMNI_OF]->(:Cohort)
```

### 6. Create Full-Text Search Index

```cypher
CREATE FULLTEXT INDEX person_search 
FOR (p:Person) 
ON EACH [p.`ชื่อ-นามสกุล`, p.`ชื่อ`, p.`นามสกุล`, p.`ชื่อเล่น`];

// Usage:
CALL db.index.fulltext.queryNodes("person_search", "ประเสริฐ~") 
YIELD node, score
RETURN node
```

---

## CSV Data Quality Issues to Fix

### Missing Names (Rows 50-76, 129-140)
These rows have position but no name. Options:
1. Fill in the actual names if known
2. Use position + agency as temporary identifier
3. Remove incomplete rows

### Inconsistent "Connect by" Values
Values like "สุวัฒน์ อ้นใจกล้า" should be Person references, not network names.
Consider:
- Separate column for "Introduced by" (person reference)
- Keep "Connect by" only for network names like OSK115

### Normalize Network Names
- "พี่เจ้ห์ (MABE)" → "MABE"
- "OSK115" ✓
- "ก่องระกูล(OSK115)" → "OSK115" (child network?)

---

## Recommended New Import Model

```json
{
  "nodes": [
    {
      "label": "Person",
      "properties": {
        "id": "UUID or ลำดับ",
        "fullName": "ชื่อ-นามสกุล",
        "firstName": "ชื่อ",
        "lastName": "นามสกุล",
        "nickname": "ชื่อเล่น",
        "title": "คำนำหน้า",
        "searchText": "combined search field"
      }
    },
    {
      "label": "Network",
      "properties": {
        "name": "OSK115, MABE, Santisook, Por, Knot, NEXIS รุ่นที่ 1, etc.",
        "type": "connection | stelligence | cohort",
        "description": "optional"
      }
    },
    {
      "label": "Organization", 
      "properties": {
        "name": "หน่วยงาน",
        "type": "ministry | agency | company",
        "parent": "กระทรวง (if agency)"
      }
    },
    {
      "label": "Position",
      "properties": {
        "name": "ตำแหน่ง",
        "level": 1-3
      }
    }
  ],
  "relationships": [
    "Person -[:WORKS_AT]-> Organization",
    "Person -[:HAS_POSITION]-> Position", 
    "Person -[:MEMBER_OF]-> Network",
    "Person -[:KNOWS {via, introduced_by}]-> Person",
    "Organization -[:PART_OF]-> Organization (for ministry hierarchy)"
  ]
}
```

---

## Quick Wins (No Model Change)

1. **Add missing names to CSV** - fix rows 50-76
2. **Create search index** in Neo4j Aura
3. **Clean duplicate agencies/ministries**
4. **Standardize network names**

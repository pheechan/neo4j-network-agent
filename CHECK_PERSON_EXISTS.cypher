// =============================================================================
// CHECK IF "ประเสริฐ สินสุขประเสริฐ" EXISTS IN DATABASE
// =============================================================================

// 1. Search for this exact name in Person label
MATCH (n:Person)
WHERE n.`ชื่อ-นามสกุล` CONTAINS 'ประเสริฐ' OR n.`ชื่อ-นามสกุล` CONTAINS 'สินสุข'
RETURN n.`ชื่อ-นามสกุล` as name, 
       n.`ตำแหน่ง` as position,
       n.`หน่วยงาน` as agency,
       n.embedding IS NOT NULL as has_embedding,
       n.embedding_text as embedding_text
LIMIT 5;

// 2. Search across ALL labels for this name
MATCH (n)
WHERE any(prop IN keys(n) WHERE 
  toString(n[prop]) CONTAINS 'ประเสริฐ' OR toString(n[prop]) CONTAINS 'สินสุข'
)
RETURN labels(n) as labels, properties(n) as props
LIMIT 5;

// 3. Check what Person nodes exist and their properties
MATCH (n:Person)
RETURN n.`ชื่อ-นามสกุล` as name,
       n.embedding_text as embedding_text,
       keys(n) as all_properties
LIMIT 10;

// 4. Check if embedding_text property exists on any nodes
MATCH (n)
WHERE n.embedding_text IS NOT NULL
RETURN labels(n) as labels, 
       n.embedding_text as embedding_text,
       n.embedding IS NOT NULL as has_embedding
LIMIT 10;

// =============================================================================
// SHOW ACTUAL SANTISOOK NODE CONTENT
// =============================================================================

// 1. Show ALL Santisook nodes with ALL their properties
MATCH (n:Santisook)
RETURN n, properties(n) as props, 
       n.embedding IS NOT NULL as has_embedding,
       n.embedding_text as embedding_text
LIMIT 5;

// 2. Show what the Stelligence property actually contains
MATCH (n:Santisook)
RETURN n.Stelligence as stelligence_value,
       n.embedding_text as embedding_text,
       keys(n) as all_keys
LIMIT 5;

// 3. Find ANY nodes with Stelligence property
MATCH (n)
WHERE n.Stelligence IS NOT NULL
RETURN labels(n) as labels, 
       n.Stelligence as stelligence,
       n.embedding_text as embedding_text,
       n.embedding IS NOT NULL as has_embedding
LIMIT 10;

// 4. Show sample Person nodes to compare structure
MATCH (n:Person)
RETURN n.`ชื่อ-นามสกุล` as name,
       n.embedding_text as embedding_text,
       keys(n) as all_keys
LIMIT 3;

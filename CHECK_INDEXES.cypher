// =============================================================================
// CHECK CURRENT VECTOR INDEXES
// =============================================================================
// Run this first to see what you have

SHOW INDEXES WHERE type = 'VECTOR';

// Look for:
// 1. How many indexes exist? (should be 12)
// 2. What are the dimensions? (should be 384)
// 3. Which labels are covered?
//
// Expected labels:
// - Person, Position, Level, Connect by, Agency
// - Santisook, Por, Knot, Remark, Ministry, Associate, NIckname

// =============================================================================
// CHECK IF NODES HAVE EMBEDDINGS
// =============================================================================

MATCH (n)
WHERE n.embedding IS NOT NULL
RETURN labels(n)[0] as label, count(n) as nodes_with_embeddings
ORDER BY nodes_with_embeddings DESC;

// This shows which labels have embeddings already

// =============================================================================
// DECISION GUIDE
// =============================================================================
//
// IF all 12 indexes exist with 384 dimensions:
//   → Skip to generating embeddings (if needed)
//   → No need to drop/recreate
//
// IF some indexes are missing:
//   → Just run CREATE_ALL_VECTOR_INDEXES.cypher
//   → IF NOT EXISTS will skip existing ones
//
// IF any indexes have 1536 dimensions:
//   → Run DROP_ALL_VECTOR_INDEXES.cypher first
//   → Then run CREATE_ALL_VECTOR_INDEXES.cypher
//
// IF indexes exist but Santisook label has no index:
//   → Just create the missing ones (no need to drop)

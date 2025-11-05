// =============================================================================
// CREATE VECTOR INDEXES FOR ALL NODE LABELS
// Based on your Neo4j import model schema
// =============================================================================
// 
// IMPORTANT: Run this in Neo4j Browser (https://neo4j.com/cloud/)
// 
// These indexes enable semantic search using 384-dimensional embeddings
// from HuggingFace model: paraphrase-multilingual-MiniLM-L12-v2
// =============================================================================

// 1. Person (ชื่อ-นามสกุล)
CREATE VECTOR INDEX person_vector_index IF NOT EXISTS
FOR (n:Person) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// 2. Position (ตำแหน่ง)
CREATE VECTOR INDEX position_vector_index IF NOT EXISTS
FOR (n:Position) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// 3. Level
CREATE VECTOR INDEX level_vector_index IF NOT EXISTS
FOR (n:Level) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// 4. Connect by (note the space in label name)
CREATE VECTOR INDEX connect_by_vector_index IF NOT EXISTS
FOR (n:`Connect by`) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// 5. Agency (หน่วยงาน)
CREATE VECTOR INDEX agency_vector_index IF NOT EXISTS
FOR (n:Agency) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// 6. Santisook (Stelligence property)
CREATE VECTOR INDEX santisook_vector_index IF NOT EXISTS
FOR (n:Santisook) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// 7. Por (Stelligence property)
CREATE VECTOR INDEX por_vector_index IF NOT EXISTS
FOR (n:Por) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// 8. Remark
CREATE VECTOR INDEX remark_vector_index IF NOT EXISTS
FOR (n:Remark) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// 9. Knot (Stelligence property)
CREATE VECTOR INDEX knot_vector_index IF NOT EXISTS
FOR (n:Knot) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// 10. Ministry (กระทรวง)
CREATE VECTOR INDEX ministry_vector_index IF NOT EXISTS
FOR (n:Ministry) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// 11. Associate
CREATE VECTOR INDEX associate_vector_index IF NOT EXISTS
FOR (n:Associate) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// 12. NIckname (ชื่อเล่น) - Note: typo in original schema
CREATE VECTOR INDEX nickname_vector_index IF NOT EXISTS
FOR (n:NIckname) ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// =============================================================================
// VERIFY INDEXES WERE CREATED
// =============================================================================

SHOW INDEXES WHERE type = 'VECTOR';

// You should see 12 vector indexes listed
// Each should show:
//   - name: <label>_vector_index
//   - type: VECTOR
//   - labelsOrTypes: [<Label>]
//   - properties: ["embedding"]
//   - options: {indexConfig: {vector.dimensions: 384, ...}}

// Fix Neo4j Browser Node Display
// Run these queries in Neo4j Browser to diagnose and fix the display issue

// 1. Check what properties Person nodes have
MATCH (n:Person) 
RETURN keys(n) AS properties, count(*) AS node_count
LIMIT 1;

// 2. Check sample Person node data
MATCH (n:Person) 
RETURN n LIMIT 5;

// 3. Check if ชื่อ-นามสกุล property exists and has values
MATCH (n:Person) 
WHERE n.`ชื่อ-นามสกุล` IS NOT NULL
RETURN n.`ชื่อ-นามสกุล` AS name, n.Stelligence, n.`กระทรวง`
LIMIT 10;

// 4. Count nodes with and without name property
MATCH (n:Person)
RETURN 
  count(CASE WHEN n.`ชื่อ-นามสกุล` IS NOT NULL THEN 1 END) AS with_name,
  count(CASE WHEN n.`ชื่อ-นามสกุล` IS NULL THEN 1 END) AS without_name;

// 5. If names exist, try different property names
MATCH (n:Person) 
RETURN DISTINCT 
  n.`ชื่อ-นามสกุล` AS thai_name,
  n.name AS name,
  n.Stelligence AS stelligence,
  id(n) AS node_id
LIMIT 10;

// 6. Set up graph visualization style
// Copy and paste this into the :style command in Neo4j Browser
:style

node {
  diameter: 50px;
  color: #A5ABB6;
  border-color: #9AA1AC;
  border-width: 2px;
  text-color-internal: #FFFFFF;
  font-size: 10px;
}

node.Person {
  color: #4C8EDA;
  border-color: #3A7BC8;
  text-color-internal: #FFFFFF;
  caption: "{ชื่อ-นามสกุล}";
  diameter: 60px;
}

node.Position {
  color: #FFA500;
  border-color: #FF8C00;
  caption: "{name}";
}

node.Ministry {
  color: #DA7194;
  border-color: #C8607B;
  caption: "{name}";
}

node.Agency {
  color: #57C7E3;
  border-color: #46B5D1;
  caption: "{name}";
}

relationship {
  color: #A5ABB6;
  shaft-width: 2px;
  font-size: 8px;
  padding: 3px;
  text-color-external: #000000;
  text-color-internal: #FFFFFF;
}

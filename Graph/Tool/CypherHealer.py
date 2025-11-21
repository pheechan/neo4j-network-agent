"""
Cypher Query Self-Healing Module
Based on Neo4j NaLLM pattern for automatic error recovery
"""

import re
from typing import Dict, Any, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, ClientError


class CypherHealer:
    """Self-healing Cypher query executor with LLM-based error recovery"""
    
    def __init__(self, driver, llm_invoke_func, max_attempts: int = 2):
        """
        Args:
            driver: Neo4j driver instance
            llm_invoke_func: Function to call LLM for query fixing
            max_attempts: Maximum healing attempts
        """
        self.driver = driver
        self.llm_invoke = llm_invoke_func
        self.max_attempts = max_attempts
    
    def execute_with_healing(
        self, 
        cypher_query: str, 
        params: Dict[str, Any] = None,
        database: str = "neo4j"
    ) -> Dict[str, Any]:
        """
        Execute Cypher query with automatic error healing.
        
        Returns:
            Dict with 'success', 'data', 'error', 'healed' keys
        """
        params = params or {}
        
        for attempt in range(self.max_attempts):
            try:
                with self.driver.session(database=database) as session:
                    result = session.run(cypher_query, params)
                    data = [record.data() for record in result]
                    
                    return {
                        'success': True,
                        'data': data,
                        'error': None,
                        'healed': attempt > 0,
                        'attempts': attempt + 1,
                        'final_query': cypher_query
                    }
            
            except CypherSyntaxError as e:
                if attempt < self.max_attempts - 1:
                    # Try to heal the query
                    cypher_query = self._heal_syntax_error(cypher_query, str(e))
                else:
                    return {
                        'success': False,
                        'data': None,
                        'error': f"Syntax error after {self.max_attempts} attempts: {str(e)}",
                        'healed': False,
                        'attempts': attempt + 1
                    }
            
            except ClientError as e:
                # Handle specific Neo4j errors
                error_code = e.code if hasattr(e, 'code') else None
                
                if error_code == "Neo.ClientError.Statement.PropertyNotFound":
                    if attempt < self.max_attempts - 1:
                        cypher_query = self._heal_property_error(cypher_query, str(e))
                    else:
                        return {
                            'success': False,
                            'data': None,
                            'error': f"Property not found: {str(e)}",
                            'healed': False,
                            'attempts': attempt + 1
                        }
                else:
                    return {
                        'success': False,
                        'data': None,
                        'error': f"Neo4j error: {str(e)}",
                        'healed': False,
                        'attempts': attempt + 1
                    }
            
            except Exception as e:
                return {
                    'success': False,
                    'data': None,
                    'error': f"Unexpected error: {str(e)}",
                    'healed': False,
                    'attempts': attempt + 1
                }
        
        return {
            'success': False,
            'data': None,
            'error': 'Max attempts reached',
            'healed': False,
            'attempts': self.max_attempts
        }
    
    def _heal_syntax_error(self, query: str, error_msg: str) -> str:
        """Use LLM to fix Cypher syntax errors"""
        prompt = f"""Fix this Cypher query syntax error:

ERROR: {error_msg}

QUERY:
{query}

Return ONLY the fixed Cypher query wrapped in triple backticks. No explanation.
"""
        
        response = self.llm_invoke(prompt)
        
        # Extract query from response
        match = re.search(r'```(?:cypher)?\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no backticks, return the response as-is
        return response.strip()
    
    def _heal_property_error(self, query: str, error_msg: str) -> str:
        """Fix property name errors (e.g., using 'name' instead of 'ชื่อ-นามสกุล')"""
        
        # Common Thai property name fixes
        replacements = {
            r'\.name\b': '.`ชื่อ-นามสกุล`',
            r'\.ชื่อ\b': '.`ชื่อ-นามสกุล`',
            r'\{name:': '{`ชื่อ-นามสกุล`:',
            r'\{ชื่อ:': '{`ชื่อ-นามสกุล`:',
        }
        
        healed_query = query
        for pattern, replacement in replacements.items():
            healed_query = re.sub(pattern, replacement, healed_query)
        
        # If automatic fix didn't work, use LLM
        if healed_query == query:
            prompt = f"""Fix this Neo4j property error:

ERROR: {error_msg}

QUERY:
{query}

Common Thai property names:
- Use `ชื่อ-นามสกุล` for full names
- Use `name` for English names
- Use `ชื่อ` for first names only

Return ONLY the fixed Cypher query wrapped in triple backticks.
"""
            response = self.llm_invoke(prompt)
            match = re.search(r'```(?:cypher)?\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                healed_query = match.group(1).strip()
        
        return healed_query


def extract_cypher_from_llm_response(response: str) -> Optional[str]:
    """
    Extract Cypher query from LLM response.
    Looks for code blocks with or without 'cypher' language tag.
    """
    # Try to find code block
    match = re.search(r'```(?:cypher)?\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # If no code block, look for MATCH/CREATE/MERGE statements
    lines = response.split('\n')
    cypher_lines = []
    in_cypher = False
    
    for line in lines:
        line_upper = line.strip().upper()
        if any(line_upper.startswith(keyword) for keyword in ['MATCH', 'CREATE', 'MERGE', 'WITH', 'RETURN', 'WHERE']):
            in_cypher = True
        
        if in_cypher:
            cypher_lines.append(line)
            # Stop at blank line or explanation
            if not line.strip() or line.strip().startswith(('Note:', 'Explanation:', '//')):
                break
    
    if cypher_lines:
        return '\n'.join(cypher_lines).strip()
    
    return None

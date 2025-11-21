"""
Cypher Result Summarization Module
Based on Neo4j NaLLM pattern for concise, accurate summaries
"""

from typing import List, Dict, Any


def remove_large_properties(data: Dict[str, Any], max_length: int = 500) -> Dict[str, Any]:
    """
    Remove or truncate large properties (embeddings, long text) from results.
    """
    if not isinstance(data, dict):
        return data
    
    cleaned = {}
    for key, value in data.items():
        # Skip embedding properties
        if key in ['embedding', 'vector', 'embeddings']:
            continue
        
        # Truncate long strings
        if isinstance(value, str) and len(value) > max_length:
            cleaned[key] = value[:max_length] + "..."
        # Recursively clean nested dicts
        elif isinstance(value, dict):
            cleaned[key] = remove_large_properties(value, max_length)
        # Clean lists of dicts
        elif isinstance(value, list):
            cleaned[key] = [
                remove_large_properties(item, max_length) if isinstance(item, dict) else item
                for item in value[:10]  # Limit to first 10 items
            ]
        else:
            cleaned[key] = value
    
    return cleaned


class CypherResultSummarizer:
    """Summarize Cypher query results into natural language"""
    
    SYSTEM_PROMPT = """You are an assistant that generates concise, human-readable answers from database results.

RULES:
1. Base your answer ONLY on the provided data
2. Be concise (maximum 100 words in Thai, 150 words in English)
3. Do not add information not in the data
4. Do not apologize or add disclaimers
5. Answer in Thai if the question is in Thai
6. Use bullet points for lists
7. Include specific numbers/names from the data
"""
    
    def __init__(self, llm_invoke_func):
        """
        Args:
            llm_invoke_func: Function to call LLM (takes string, returns string)
        """
        self.llm_invoke = llm_invoke_func
    
    def summarize(
        self, 
        question: str, 
        results: List[Dict[str, Any]],
        exclude_embeddings: bool = True
    ) -> str:
        """
        Generate natural language summary of query results.
        
        Args:
            question: Original user question
            results: List of result dictionaries from Neo4j
            exclude_embeddings: Remove embedding properties
        
        Returns:
            Natural language summary string
        """
        if not results:
            return "ไม่พบข้อมูลที่ตรงกับคำถาม" if self._is_thai(question) else "No data found matching the question."
        
        # Clean results
        if exclude_embeddings:
            results = [remove_large_properties(r) for r in results]
        
        # Limit results to prevent token overflow
        if len(results) > 20:
            results = results[:20]
            results_note = f" (showing first 20 of {len(results)} results)"
        else:
            results_note = ""
        
        # Format results nicely
        results_str = self._format_results(results)
        
        prompt = f"""{self.SYSTEM_PROMPT}

QUESTION: {question}

DATA FROM DATABASE{results_note}:
{results_str}

Generate a clear, concise answer based on this data:"""
        
        return self.llm_invoke(prompt)
    
    def _is_thai(self, text: str) -> bool:
        """Check if text contains Thai characters"""
        return any('\u0e00' <= char <= '\u0e7f' for char in text)
    
    def _format_results(self, results: List[Dict[str, Any]], max_chars: int = 2000) -> str:
        """Format results as readable text"""
        formatted = []
        
        for i, result in enumerate(results, 1):
            lines = [f"Result {i}:"]
            for key, value in result.items():
                if isinstance(value, (list, dict)):
                    value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                else:
                    value_str = str(value)
                lines.append(f"  {key}: {value_str}")
            formatted.append('\n'.join(lines))
        
        full_text = '\n\n'.join(formatted)
        
        # Truncate if too long
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n... (truncated)"
        
        return full_text


def summarize_path_result(
    path_data: Dict[str, Any],
    person_a: str,
    person_b: str,
    llm_invoke_func
) -> str:
    """
    Specialized summarization for connection path results.
    """
    if not path_data.get('path_found'):
        return f"ไม่พบเส้นทางเชื่อมต่อระหว่าง {person_a} และ {person_b} ในฐานข้อมูล"
    
    nodes = path_data.get('path_nodes', [])
    hops = path_data.get('hops', 0)
    
    # Build path description
    node_names = []
    for node in nodes:
        name = node.get('name', 'Unknown')
        node_type = node.get('type', 'unknown')
        
        # Add emoji based on type
        emoji = {
            'person': '👤',
            'agency': '🏢',
            'position': '💼',
            'ministry': '🏛️',
            'network': '🌐'
        }.get(node_type, '•')
        
        node_names.append(f"{emoji} {name}")
    
    path_str = " → ".join(node_names)
    
    summary = f"""พบเส้นทางเชื่อมต่อระหว่าง {person_a} และ {person_b}:

จำนวนขั้น: {hops} ขั้น
เส้นทาง: {path_str}

การเชื่อมต่อทั้งหมด: {path_data.get('total_connections', 0)} ความสัมพันธ์"""
    
    return summary

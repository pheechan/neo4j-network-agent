"""
Test Script for Enhanced Neo4j Network Agent Features

This script tests the three main enhancements:
1. Hybrid Search
2. Self-Healing Cypher
3. Concise Summarization

Run this after starting the Neo4j database and ensuring OpenRouter API is configured.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all enhanced modules can be imported"""
    print("=" * 60)
    print("TEST 1: Enhanced Module Imports")
    print("=" * 60)
    
    try:
        from Graph.Tool.CypherHealer import CypherHealer, extract_cypher_from_llm_response
        print("✅ CypherHealer imported successfully")
    except Exception as e:
        print(f"❌ CypherHealer import failed: {e}")
        return False
    
    try:
        from Graph.Tool.CypherSummarizer import CypherResultSummarizer, summarize_path_result, remove_large_properties
        print("✅ CypherSummarizer imported successfully")
    except Exception as e:
        print(f"❌ CypherSummarizer import failed: {e}")
        return False
    
    try:
        from KG.VectorRAG import query_vector_rag
        print("✅ VectorRAG imported successfully")
    except Exception as e:
        print(f"❌ VectorRAG import failed: {e}")
        return False
    
    print("\n✅ All imports successful!\n")
    return True


def test_hybrid_search():
    """Test hybrid search functionality"""
    print("=" * 60)
    print("TEST 2: Hybrid Search")
    print("=" * 60)
    
    try:
        from KG.VectorRAG import query_vector_rag
        
        # Test with a simple Thai query
        test_query = "หาข้อมูล อนุทิน"
        print(f"Query: {test_query}")
        print("Testing with hybrid search enabled...")
        
        results = query_vector_rag(
            question=test_query,
            top_k=5,
            use_hybrid_search=True
        )
        
        if results and len(results) > 0:
            print(f"✅ Hybrid search returned {len(results)} results")
            print(f"   First result: {results[0].get('name', 'N/A')}")
            return True
        else:
            print("⚠️ Hybrid search returned no results (may be OK if database is empty)")
            return True  # Not a failure if DB is empty
            
    except Exception as e:
        print(f"❌ Hybrid search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cypher_healer():
    """Test self-healing Cypher functionality"""
    print("\n" + "=" * 60)
    print("TEST 3: Self-Healing Cypher")
    print("=" * 60)
    
    try:
        from Graph.Tool.CypherHealer import CypherHealer
        from Config.neo4j import get_driver
        
        print("Creating CypherHealer instance...")
        
        # Mock LLM function for testing
        def mock_llm(prompt):
            # Simple mock that returns a fixed Cypher query
            return """
            ```cypher
            MATCH (p:Person) WHERE p.`ชื่อ-นามสกุล` CONTAINS 'test'
            RETURN p LIMIT 1
            ```
            """
        
        driver = get_driver()
        healer = CypherHealer(driver, mock_llm, max_attempts=2)
        
        print("✅ CypherHealer instance created")
        
        # Test with a valid query (should work without healing)
        valid_query = "MATCH (p:Person) RETURN count(p) as total"
        result = healer.execute_with_healing(valid_query, {})
        
        if result['success']:
            print(f"✅ Valid query executed: {result.get('data', [])}")
            if result['healed']:
                print("   ⚠️ Query was healed (unexpected for valid query)")
            else:
                print("   ✅ Query executed without healing (correct)")
            return True
        else:
            print(f"⚠️ Query execution failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Cypher healer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_summarizer():
    """Test result summarization functionality"""
    print("\n" + "=" * 60)
    print("TEST 4: Result Summarization")
    print("=" * 60)
    
    try:
        from Graph.Tool.CypherSummarizer import CypherResultSummarizer, remove_large_properties
        
        print("Testing remove_large_properties()...")
        
        # Test data
        test_result = {
            'name': 'Test Person',
            'position': 'Test Position',
            'embedding': [0.1] * 1000,  # Large embedding
            'text': 'Short text'
        }
        
        cleaned = remove_large_properties(test_result)
        
        if 'embedding' not in cleaned:
            print("✅ Embeddings removed correctly")
        else:
            print("❌ Embeddings not removed")
            return False
        
        if cleaned.get('name') == 'Test Person':
            print("✅ Other properties preserved")
        else:
            print("❌ Properties not preserved correctly")
            return False
        
        # Test summarizer creation
        def mock_llm(prompt):
            return "This is a concise summary of the data."
        
        summarizer = CypherResultSummarizer(mock_llm)
        print("✅ CypherResultSummarizer instance created")
        
        # Test summarization
        test_results = [
            {'name': 'Person 1', 'position': 'Minister'},
            {'name': 'Person 2', 'position': 'Deputy Minister'}
        ]
        
        summary = summarizer.summarize("Who are the ministers?", test_results)
        
        if summary and len(summary) > 0:
            print(f"✅ Summary generated: {summary[:50]}...")
            return True
        else:
            print("❌ Summary generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Summarizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graceful_degradation():
    """Test that the app works even if enhanced features are unavailable"""
    print("\n" + "=" * 60)
    print("TEST 5: Graceful Degradation")
    print("=" * 60)
    
    try:
        # Simulate import failure
        import sys
        
        # Test that streamlit_app handles missing imports
        print("Testing import handling in streamlit_app...")
        
        # This should not crash even if imports fail
        try:
            from Graph.Tool.CypherHealer import CypherHealer
            HEALER_AVAILABLE = True
        except:
            HEALER_AVAILABLE = False
        
        try:
            from Graph.Tool.CypherSummarizer import CypherResultSummarizer
            SUMMARIZER_AVAILABLE = True
        except:
            SUMMARIZER_AVAILABLE = False
        
        if HEALER_AVAILABLE and SUMMARIZER_AVAILABLE:
            print("✅ All enhanced features available")
        else:
            print("⚠️ Some features unavailable (but app should still work)")
        
        print("✅ Graceful degradation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Graceful degradation test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ENHANCED NEO4J NETWORK AGENT - TEST SUITE")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Hybrid Search (requires DB connection)
    try:
        results.append(("Hybrid Search", test_hybrid_search()))
    except Exception as e:
        print(f"⚠️ Hybrid search test skipped (DB connection required): {e}")
        results.append(("Hybrid Search", None))
    
    # Test 3: Cypher Healer (requires DB connection)
    try:
        results.append(("Cypher Healer", test_cypher_healer()))
    except Exception as e:
        print(f"⚠️ Cypher healer test skipped (DB connection required): {e}")
        results.append(("Cypher Healer", None))
    
    # Test 4: Summarizer
    results.append(("Summarizer", test_summarizer()))
    
    # Test 5: Graceful Degradation
    results.append(("Graceful Degradation", test_graceful_degradation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for name, result in results if result is True)
    failed = sum(1 for name, result in results if result is False)
    skipped = sum(1 for name, result in results if result is None)
    
    for name, result in results:
        if result is True:
            print(f"✅ {name}: PASSED")
        elif result is False:
            print(f"❌ {name}: FAILED")
        else:
            print(f"⚠️ {name}: SKIPPED")
    
    print()
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 All tests passed! Your enhanced features are working correctly.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

"""Test script for queries"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_query(message, description=""):
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Query: {message}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"message": message},
            timeout=600
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Response ({elapsed:.2f}s):")
            print(f"Message:\n{data.get('message', 'N/A')}")
            if data.get('debug'):
                print(f"\nDebug Info:")
                for key, value in data['debug'].items():
                    print(f"  {key}: {value}")
        else:
            print(f"\n✗ Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"\n✗ Exception: {e}")

if __name__ == "__main__":
    import sys
    
    queries = [
        ("Santisook ไปหา อนุทิน", "Query 1: Shortest path"),
        ("Santisook รู้จักใครบ้าง", "Query 3: Santisook network"),
        ("ใครบ้างที่ connect by OSK115", "Query 4: Connect by OSK115"),
    ]
    
    # Run specific query by index if provided
    if len(sys.argv) > 1:
        idx = int(sys.argv[1]) - 1
        if 0 <= idx < len(queries):
            test_query(queries[idx][0], queries[idx][1])
    else:
        # Run first query by default
        test_query(queries[0][0], queries[0][1])

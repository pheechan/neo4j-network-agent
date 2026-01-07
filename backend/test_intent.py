#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test intent detection for Thai path queries"""

import sys
sys.path.insert(0, '/app')

from backend.network_agent import NetworkAgent

def test_intent(query):
    na = NetworkAgent()
    try:
        intent = na.detect_query_intent(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {intent['type']}")
        if intent['type'] == 'shortest_path':
            print(f"  From: {intent.get('from_person')}")
            print(f"  To: {intent.get('to_person')}")
        return intent
    finally:
        na.close()

if __name__ == "__main__":
    # Test various Thai path queries
    queries = [
        "เส้นทางจาก Santisook ไป อนุทิน",
        "จาก Santisook ไป อนุทิน",
        "path from Santisook to อนุทิน",
        "หาทางจาก Por ถึง สมชาย",
        "Santisook รู้จักใครบ้าง",  # Should NOT be path
    ]
    
    for q in queries:
        test_intent(q)

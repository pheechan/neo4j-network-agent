#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test HTTP API with proper UTF-8 encoding"""

import requests
import json

def test_api(message):
    url = "http://localhost:8000/api/chat"
    payload = {"message": message}
    headers = {"Content-Type": "application/json; charset=utf-8"}
    
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    
    print(f"\n=== Query: {message} ===")
    print(f"Answer: {data.get('answer', 'No answer')[:500]}...")
    print(f"Context: {data.get('context', 'No context')[:300]}...")
    return data

if __name__ == "__main__":
    # Test Thai path query
    test_api("เส้นทางจาก Santisook ไป อนุทิน")

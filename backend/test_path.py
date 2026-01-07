#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test shortest path from Santisook to อนุทิน"""

from network_agent import NetworkAgent
import json

def main():
    na = NetworkAgent()
    try:
        print("Testing shortest path from Santisook to อนุทิน...")
        result = na.find_shortest_path("Santisook", "อนุทิน")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        na.close()

if __name__ == "__main__":
    main()

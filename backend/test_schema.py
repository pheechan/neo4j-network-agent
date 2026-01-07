#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script to check Neo4j schema"""

from network_agent import NetworkAgent
import json

def main():
    na = NetworkAgent()
    try:
        with na.driver.session() as session:
            # Get property keys
            print("=== Person Property Keys ===")
            result = session.run("MATCH (p:Person) RETURN keys(p) as props LIMIT 1")
            rec = result.single()
            if rec:
                print(f"Properties: {rec['props']}")
            
            # List some persons
            print("\n=== Sample Person Names ===")
            result = session.run("MATCH (p:Person) RETURN p.`ชื่อ-นามสกุล` as name LIMIT 10")
            for rec in result:
                print(f"  - {rec['name']}")
            
            # Check for อนุทิน
            print("\n=== Search for อนุทิน ===")
            result = session.run("MATCH (p:Person) WHERE p.`ชื่อ-นามสกุล` CONTAINS 'อนุทิน' RETURN p.`ชื่อ-นามสกุล` as name LIMIT 5")
            found = list(result)
            if found:
                for rec in found:
                    print(f"  Found: {rec['name']}")
            else:
                print("  Not found")
            
            # Check all relationship types
            print("\n=== Relationship Types ===")
            result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            for rec in result:
                print(f"  - {rec['relationshipType']}")
                
    finally:
        na.close()

if __name__ == "__main__":
    main()

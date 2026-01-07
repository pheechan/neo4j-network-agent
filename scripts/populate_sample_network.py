"""
Populate Neo4j with sample professional network data for testing
Run this script to create a sample company network with employees and relationships
"""
import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "letmein123")

def create_sample_network():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    
    with driver.session() as session:
        # Clear existing data (optional - comment out if you want to keep existing data)
        print("Clearing existing Person nodes...")
        session.run("MATCH (p:Person) DETACH DELETE p")
        
        # Create sample company network
        print("Creating sample professional network...")
        
        # Create people
        people = [
            {"name": "John CEO", "position": "CEO", "department": "Executive"},
            {"name": "Sarah CFO", "position": "CFO", "department": "Finance"},
            {"name": "Mike CTO", "position": "CTO", "department": "Technology"},
            {"name": "Emily HR", "position": "HR Director", "department": "Human Resources"},
            {"name": "David Sales", "position": "Sales Director", "department": "Sales"},
            {"name": "Lisa Marketing", "position": "Marketing Manager", "department": "Marketing"},
            {"name": "Tom Engineer", "position": "Senior Engineer", "department": "Technology"},
            {"name": "Anna Designer", "position": "Lead Designer", "department": "Design"},
            {"name": "Bob Analyst", "position": "Data Analyst", "department": "Technology"},
            {"name": "Carol PM", "position": "Product Manager", "department": "Product"},
            {"name": "Dan Developer", "position": "Developer", "department": "Technology"},
            {"name": "Eve Sales Rep", "position": "Sales Rep", "department": "Sales"},
            {"name": "Frank Account Manager", "position": "Account Manager", "department": "Sales"},
            {"name": "Grace Support", "position": "Customer Support", "department": "Support"},
            {"name": "Henry Recruiter", "position": "Recruiter", "department": "Human Resources"},
        ]
        
        for person in people:
            session.run(
                "CREATE (p:Person {name: $name, position: $position, department: $department})",
                **person
            )
        
        print(f"Created {len(people)} people")
        
        # Create relationships
        relationships = [
            # Executive team reports to CEO
            ("Sarah CFO", "John CEO", "REPORTS_TO"),
            ("Mike CTO", "John CEO", "REPORTS_TO"),
            ("Emily HR", "John CEO", "REPORTS_TO"),
            ("David Sales", "John CEO", "REPORTS_TO"),
            
            # Department heads
            ("Lisa Marketing", "Sarah CFO", "REPORTS_TO"),
            ("Tom Engineer", "Mike CTO", "REPORTS_TO"),
            ("Bob Analyst", "Mike CTO", "REPORTS_TO"),
            ("Dan Developer", "Mike CTO", "REPORTS_TO"),
            ("Eve Sales Rep", "David Sales", "REPORTS_TO"),
            ("Frank Account Manager", "David Sales", "REPORTS_TO"),
            ("Henry Recruiter", "Emily HR", "REPORTS_TO"),
            ("Carol PM", "Mike CTO", "REPORTS_TO"),
            
            # Cross-department collaborations
            ("Tom Engineer", "Anna Designer", "COLLABORATES_WITH"),
            ("Anna Designer", "Lisa Marketing", "COLLABORATES_WITH"),
            ("Carol PM", "Tom Engineer", "COLLABORATES_WITH"),
            ("Carol PM", "Anna Designer", "COLLABORATES_WITH"),
            ("Bob Analyst", "Lisa Marketing", "COLLABORATES_WITH"),
            ("Bob Analyst", "David Sales", "COLLABORATES_WITH"),
            ("Dan Developer", "Tom Engineer", "COLLABORATES_WITH"),
            ("Frank Account Manager", "Carol PM", "COLLABORATES_WITH"),
            ("Grace Support", "Dan Developer", "COLLABORATES_WITH"),
            
            # Mentorship
            ("Mike CTO", "Tom Engineer", "MENTORS"),
            ("Sarah CFO", "Lisa Marketing", "MENTORS"),
            ("David Sales", "Frank Account Manager", "MENTORS"),
            ("Emily HR", "Henry Recruiter", "MENTORS"),
            ("Tom Engineer", "Dan Developer", "MENTORS"),
            
            # Friends/social connections
            ("Tom Engineer", "Bob Analyst", "FRIENDS_WITH"),
            ("Lisa Marketing", "Anna Designer", "FRIENDS_WITH"),
            ("Eve Sales Rep", "Grace Support", "FRIENDS_WITH"),
            ("Dan Developer", "Bob Analyst", "FRIENDS_WITH"),
            ("Carol PM", "Lisa Marketing", "FRIENDS_WITH"),
        ]
        
        for from_name, to_name, rel_type in relationships:
            session.run(f"""
                MATCH (a:Person {{name: $from_name}})
                MATCH (b:Person {{name: $to_name}})
                CREATE (a)-[:{rel_type}]->(b)
            """, from_name=from_name, to_name=to_name)
        
        print(f"Created {len(relationships)} relationships")
        
        # Create some bidirectional relationships
        bidirectional = [
            ("John CEO", "Sarah CFO", "TRUSTS"),
            ("John CEO", "Mike CTO", "TRUSTS"),
            ("Mike CTO", "Sarah CFO", "WORKS_CLOSELY_WITH"),
            ("Tom Engineer", "Carol PM", "WORKS_CLOSELY_WITH"),
        ]
        
        for person1, person2, rel_type in bidirectional:
            session.run(f"""
                MATCH (a:Person {{name: $person1}})
                MATCH (b:Person {{name: $person2}})
                CREATE (a)-[:{rel_type}]->(b)
                CREATE (b)-[:{rel_type}]->(a)
            """, person1=person1, person2=person2)
        
        print(f"Created {len(bidirectional)} bidirectional relationships")
        
        # Verify
        result = session.run("MATCH (p:Person) RETURN count(p) as count")
        person_count = result.single()["count"]
        
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = result.single()["count"]
        
        print(f"\n✅ Sample network created successfully!")
        print(f"   Total Persons: {person_count}")
        print(f"   Total Relationships: {rel_count}")
        print(f"\nExample queries to try:")
        print(f"  - 'How can Dan Developer reach John CEO?'")
        print(f"  - 'Who can introduce Grace Support to Mike CTO?'")
        print(f"  - 'What are the mutual connections between Tom Engineer and Lisa Marketing?'")
        print(f"  - 'Show me Sarah CFO's network'")
    
    driver.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Sample Professional Network Data Population")
    print("=" * 60)
    print(f"Connecting to: {NEO4J_URI}")
    print()
    
    try:
        create_sample_network()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

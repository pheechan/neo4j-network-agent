import os
import sys
try:
    from dotenv import load_dotenv
except Exception:
    # dotenv not installed; define no-op
    def load_dotenv(*args, **kwargs):
        return None

try:
    from neo4j import GraphDatabase
except Exception:
    print("neo4j driver is not installed in this Python environment.\nPlease run: python -m pip install neo4j")
    sys.exit(1)

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")

def main():
    print(f"Using NEO4J_URI={NEO4J_URI}")
    print(f"Using NEO4J_DATABASE={NEO4J_DB}")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    except Exception as e:
        print("Failed to create driver:", e)
        sys.exit(2)

    try:
        with driver.session(database=NEO4J_DB) as session:
            result = session.run("RETURN 'hello from neo4j' AS msg")
            rec = result.single()
            if rec:
                print(rec["msg"])
            else:
                print("Query executed but returned no records")
    except Exception as e:
        print("Error running query:", e)
        sys.exit(3)
    finally:
        try:
            driver.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()

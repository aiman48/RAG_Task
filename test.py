import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions
from neo4j.debug import watch

load_dotenv()

URI_PRIMARY = os.getenv("NEO4J_URI_PRIMARY")
URI_FALLBACK = os.getenv("NEO4J_URI_FALLBACK")
USER = os.getenv("NEO4J_USERNAME")
PWD = os.getenv("NEO4J_PASSWORD")

def try_connect(uri):
    print(f"\n🔌 Trying: {uri}")
    try:
        driver = GraphDatabase.driver(uri, auth=(USER, PWD))
        # Optional: verbose wire logs to see handshake problems
        with watch("neo4j"):
            driver.verify_connectivity()
        print("✅ Connected OK")
        driver.close()
        return True
    except exceptions.AuthError as e:
        print(" Auth error (wrong username/password):", e)
    except exceptions.ServiceUnavailable as e:
        print(" Service unavailable (routing/handshake/network):", e)
    except Exception as e:
        print(" Other error:", e)
    return False

if __name__ == "__main__":
    if not try_connect(URI_PRIMARY):
        print("↪️ Retrying with fallback (skips certificate verification)...")
        try_connect(URI_FALLBACK)

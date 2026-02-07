
import asyncio
import httpx
import json
import logging
import os
import sys

# Configuration
# Default to current host if not specified
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

# Backend API
API_HOST = "localhost"
API_PORT = 8700
RAG_API_URL = f"http://{API_HOST}:{API_PORT}/api/v1/rag"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verify_retrieval")

async def verify():
    # 1. Connect to Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL)
        
        logger.info(f"Connecting to Qdrant at {QDRANT_URL}...")
        collections = client.get_collections()
        
        collection_names = [c.name for c in collections.collections]
        logger.info(f"Available collections: {collection_names}")
        
        # 1.1 Find the correct collection via API
        logger.info(f"Querying RAG API for domains at {RAG_API_URL}/domains...")
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(f"{RAG_API_URL}/domains")
            if resp.status_code == 200:
                domains_data = resp.json().get("domains", [])
                logger.info(f"Found {len(domains_data)} domains via API")
                
                for d in domains_data:
                    d_name = d.get("name", "")
                    d_id = d.get("id", "")
                    d_custom = d.get("custom_collection", "")
                    logger.info(f" - Domain: {d_name} (ID: {d_id})")
                    
                    if "lyric" in d_name.lower() or "music" in d_name.lower():
                        if d_custom:
                            target_collection = d_custom
                        else:
                            target_collection = f"domain_{d_id}"
                        logger.info(f"Match found! Target collection: {target_collection}")
                        break
            else:
                logger.warning(f"Failed to list domains: {resp.status_code}")

        if not target_collection:
             # Fallback to listing collections directly
             logger.info("Fallback: Listing collections directly from Qdrant...")
             collections = client.get_collections()
             collection_names = [c.name for c in collections.collections]
             logger.info(f"Available collections: {collection_names}")
             
             # Look for lyrics or music related collections first
             for name in collection_names:
                 if "lyric" in name.lower() or "music" in name.lower():
                     target_collection = name
                     break
             
             # If not specific, just take the first non-empty one that looks like a domain
             if not target_collection:
                 for name in collection_names:
                     if name.startswith("domain_"):
                         target_collection = name
                         break
                         
        if not target_collection:
            logger.error("No suitable collection found!")
            return

        logger.info(f"Using collection: {target_collection}")

        # 2. Fetch a random document
        count_result = client.count(target_collection)
        count = count_result.count
        logger.info(f"Collection '{target_collection}' has {count} vectors")
        
        if count == 0:
            logger.error("Collection is empty!")
            return

        # Scroll to get a sample
        # Scroll API in python client might differ slightly from async
        scroll_result = client.scroll(
            collection_name=target_collection,
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        points = scroll_result[0]
        
        if not points:
             logger.error("Could not retrieve any points")
             return
             
        point = points[0]
        payload = point.payload
        content = payload.get('content', '')
        doc_id = payload.get('document_id', 'unknown')
        
        logger.info(f"Retrieved document sample: ID={doc_id}")
        logger.info(f"Content snippet: {content[:100]}...")
        
        if not content:
            logger.error("Document has no content")
            return

        # 3. Formulate Query
        # Create a query from the content
        # Take a distinct phrase from the middle to avoid header/title bias
        midpoints = len(content) // 2
        query_snippet = content[midpoints:midpoints+50]
        query = f"search query: {query_snippet}" # Use prefix if model needs it, but RAG API might handle it
        
        # Actually user config says nomic-embed-text-v1.5 handles prefixes internally usually or via wrapper
        # Let's just use raw text query
        query = query_snippet
        logger.info(f"Generated query from content: '{query}'")
        
        # 4. Query RAG API
        search_url = f"{RAG_API_URL}/search"
        logger.info(f"Querying RAG API at {search_url}...")
        
        async with httpx.AsyncClient() as http_client:
            # Try /api/v1/rag/search endpoint
            # It expects form data for 'query' and 'top_k' usually based on FastAPI Form(...)
            # But let's check retrieve route signature. RAG route uses Form(...) for query.
            
            response = await http_client.post(
                search_url,
                data={"query": query, "top_k": 5},
                timeout=30.0
            )
            
            if response.status_code != 200:
                logger.error(f"RAG API failed: {response.status_code} - {response.text}")
                # Try json body?
                response = await http_client.post(
                     search_url,
                     json={"query": query, "top_k": 5},
                     timeout=30.0
                )
                if response.status_code != 200:
                     logger.error(f"RAG API JSON retry failed: {response.status_code}")
                     return

            result = response.json()
            results = result.get('results', [])
            
            logger.info(f"RAG API returned {len(results)} results")
            
            found = False
            for i, r in enumerate(results):
                r_content = r.get('content', '')
                r_score = r.get('score', 0)
                logger.info(f" [{i+1}] Score: {r_score:.4f} | {r_content[:50]}...")
                
                # Check for match
                # Similarity might not be exact string match due to chunking
                if doc_id in r.get('document_id', '') or query_snippet in r_content:
                    found = True
                    logger.info(f"   >>> MATCH FOUND! <<<")
            
            if found:
                logger.info("✅ SUCCESS: Retrieved correct document through RAG pipeline!")
                logger.info("This confirms embedding models are aligned.")
            else:
                logger.warning("❌ FAILURE: Source document was NOT found in top 5 results.")
                logger.warning("Possible causes: Embedding model mismatch, strict filtering, or index lag.")
                
    except ImportError:
        logger.error("Required libraries (qdrant-client, httpx) not installed.")
        logger.info("Install with: pip install qdrant-client httpx")
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        QDRANT_URL = sys.argv[1]
    
    asyncio.run(verify())

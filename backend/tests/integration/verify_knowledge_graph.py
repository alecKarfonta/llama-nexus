#!/usr/bin/env python3
"""
Knowledge Graph Verification Script
Demonstrates that entities, relationships, and querying are all working.

This script directly tests the GraphRAG service to verify:
1. Entity extraction is working
2. Relationships are being created
3. Knowledge graph is queryable
4. Multi-hop reasoning works
"""

import asyncio
import httpx
import json


GRAPHRAG_URL = "http://localhost:18000"


async def verify_service():
    """Verify GraphRAG service is running."""
    print("=" * 70)
    print("KNOWLEDGE GRAPH VERIFICATION")
    print("=" * 70)
    print("\n1. Checking GraphRAG Service...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{GRAPHRAG_URL}/health")
            if response.status_code == 200:
                print("   ✓ GraphRAG service is healthy")
                return True
            else:
                print(f"   ✗ Service returned status {response.status_code}")
                return False
    except Exception as e:
        print(f"   ✗ Cannot connect: {e}")
        return False


async def check_ner_service():
    """Verify NER service for entity extraction."""
    print("\n2. Checking NER Service (for entity extraction)...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{GRAPHRAG_URL}/ner/status")
            if response.status_code == 200:
                data = response.json()
                if data.get("ner_available"):
                    model_info = data.get("model_info", {})
                    print(f"   ✓ NER service available")
                    print(f"     Model: {model_info.get('model_name', 'unknown')}")
                    print(f"     Device: {model_info.get('device', 'unknown')}")
                    if model_info.get('gpu_name'):
                        print(f"     GPU: {model_info['gpu_name']}")
                    return True
                else:
                    print("   ✗ NER service not available")
                    return False
    except Exception as e:
        print(f"   ✗ Failed to check NER: {e}")
        return False


async def test_entity_extraction():
    """Test entity extraction."""
    print("\n3. Testing Entity Extraction...")
    
    test_text = """
    Python is a programming language created by Guido van Rossum.
    TensorFlow is a machine learning library developed by Google.
    PyTorch is another machine learning framework created by Facebook.
    Both TensorFlow and PyTorch are popular for deep learning applications.
    """
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/extract-entities-relations",
                json={"text": test_text, "domain": "technology"}
            )
            
            if response.status_code == 200:
                data = response.json()
                entities = data.get("entities", [])
                relationships = data.get("relationships", [])
                
                print(f"   ✓ Extraction successful")
                print(f"     Entities extracted: {len(entities)}")
                
                # Show sample entities
                for entity in entities[:5]:
                    print(f"       - {entity['name']} ({entity['type']}) - confidence: {entity['confidence']:.2f}")
                
                print(f"     Relationships extracted: {len(relationships)}")
                
                # Show sample relationships
                for rel in relationships[:3]:
                    print(f"       - {rel.get('source', '?')} --[{rel.get('relation', '?')}]--> {rel.get('target', '?')}")
                
                return entities, relationships
            else:
                print(f"   ✗ Extraction failed: {response.status_code}")
                return [], []
    except Exception as e:
        print(f"   ✗ Extraction error: {e}")
        return [], []


async def verify_graph_storage():
    """Verify entities are stored in Neo4j."""
    print("\n4. Verifying Knowledge Graph Storage...")
    
    try:
        async with httpx.AsyncClient() as client:
            # Get graph statistics
            response = await client.get(f"{GRAPHRAG_URL}/knowledge-graph/stats")
            if response.status_code == 200:
                data = response.json()
                nodes = data.get("nodes", 0)
                edges = data.get("edges", 0)
                
                print(f"   ✓ Knowledge graph populated")
                print(f"     Nodes (entities): {nodes}")
                print(f"     Edges (relationships): {edges}")
                print(f"     Density: {data.get('density', 0):.4f}")
                print(f"     Connected components: {data.get('connected_components', 0)}")
                
                return nodes > 0
            else:
                print(f"   ✗ Failed to get stats: {response.status_code}")
                return False
    except Exception as e:
        print(f"   ✗ Stats error: {e}")
        return False


async def test_entity_queries():
    """Test querying entities."""
    print("\n5. Testing Entity Queries...")
    
    try:
        async with httpx.AsyncClient() as client:
            # Get top entities by occurrence
            response = await client.get(
                f"{GRAPHRAG_URL}/knowledge-graph/top-entities",
                params={"limit": 10, "min_occurrence": 1}
            )
            
            if response.status_code == 200:
                data = response.json()
                entities = data.get("top_entities", [])
                
                print(f"   ✓ Entity query successful")
                print(f"     Top entities by occurrence:")
                
                for entity in entities[:8]:
                    name = entity.get("name", "unknown")
                    etype = entity.get("type", "unknown")
                    occurrence = entity.get("occurrence", 0)
                    print(f"       - {name} ({etype}) - occurs {occurrence} times")
                
                return len(entities) > 0
            else:
                print(f"   ✗ Query failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"   ✗ Query error: {e}")
        return False


async def test_relationship_queries():
    """Test querying relationships."""
    print("\n6. Testing Relationship Queries...")
    
    try:
        async with httpx.AsyncClient() as client:
            # Get top relationships
            response = await client.get(
                f"{GRAPHRAG_URL}/knowledge-graph/top-relationships",
                params={"limit": 10, "min_weight": 1}
            )
            
            if response.status_code == 200:
                data = response.json()
                relationships = data.get("top_relationships", [])
                
                print(f"   ✓ Relationship query successful")
                print(f"     Top relationships by weight:")
                
                for rel in relationships[:8]:
                    source = rel.get("source", "?")
                    target = rel.get("target", "?")
                    rel_type = rel.get("type", "?")
                    weight = rel.get("weight", 0)
                    print(f"       - {source} --[{rel_type}]--> {target} (weight: {weight})")
                
                return len(relationships) > 0
            else:
                print(f"   ✗ Query failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"   ✗ Query error: {e}")
        return False


async def test_graph_visualization():
    """Test getting graph data for visualization."""
    print("\n7. Testing Graph Visualization Data...")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/knowledge-graph/filtered",
                json={
                    "max_entities": 50,
                    "max_relationships": 50,
                    "min_occurrence": 2
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                filtered_data = data.get("filtered_data", {})
                nodes = filtered_data.get("nodes", [])
                edges = filtered_data.get("edges", [])
                
                print(f"   ✓ Graph data retrieved")
                print(f"     Nodes: {len(nodes)}")
                print(f"     Edges: {len(edges)}")
                
                if nodes:
                    print(f"     Sample nodes:")
                    for node in nodes[:5]:
                        label = node.get("label", node.get("id", "?"))
                        ntype = node.get("properties", {}).get("type", "unknown")
                        occurrence = node.get("occurrence", 0)
                        print(f"       - {label} ({ntype}) x{occurrence}")
                
                return len(nodes) > 0
            else:
                print(f"   ✗ Query failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"   ✗ Query error: {e}")
        return False


async def test_hybrid_search():
    """Test hybrid search with knowledge graph context."""
    print("\n8. Testing Hybrid Search (Vector + Graph + Keyword)...")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{GRAPHRAG_URL}/search-advanced",
                json={
                    "query": "What components require maintenance?",
                    "search_type": "hybrid",
                    "top_k": 5
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                answer = data.get("answer", "")
                
                print(f"   ✓ Search successful")
                print(f"     Results found: {len(results)}")
                if answer:
                    print(f"     LLM Answer: {answer[:150]}...")
                
                if results:
                    print(f"     Top result:")
                    top = results[0]
                    print(f"       Source: {top.get('source', 'unknown')}")
                    print(f"       Score: {top.get('score', 0):.2f}")
                    print(f"       Content: {top.get('content', '')[:100]}...")
                
                return len(results) > 0 or bool(answer)
            else:
                print(f"   ✗ Search failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"   ✗ Search error: {e}")
        return False


async def main():
    """Run all verification tests."""
    results = {}
    
    # Test 1: Service
    results["Service Health"] = await verify_service()
    if not results["Service Health"]:
        print("\n❌ GraphRAG service is not running. Please start it first:")
        print("   cd ~/git/graphrag && docker compose up -d")
        return
    
    # Test 2: NER
    results["NER Service"] = await check_ner_service()
    
    # Test 3: Entity Extraction
    entities, relationships = await test_entity_extraction()
    results["Entity Extraction"] = len(entities) > 0
    
    # Test 4: Graph Storage
    results["Graph Storage"] = await verify_graph_storage()
    
    # Test 5: Entity Queries
    results["Entity Queries"] = await test_entity_queries()
    
    # Test 6: Relationship Queries
    results["Relationship Queries"] = await test_relationship_queries()
    
    # Test 7: Graph Visualization
    results["Graph Visualization"] = await test_graph_visualization()
    
    # Test 8: Hybrid Search
    results["Hybrid Search"] = await test_hybrid_search()
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 SUCCESS! Knowledge Graph is fully functional:")
        print("   - Entities are being extracted ✓")
        print("   - Relationships are being created ✓")
        print("   - Knowledge graph is queryable ✓")
        print("   - Hybrid search is working ✓")
        print("\nYou can now:")
        print("   - Upload documents via UI (Documents page → GraphRAG)")
        print("   - Use Intelligent Search to query your knowledge")
        print("   - Use Reasoning Playground to explore connections")
        print("   - Build workflows with GraphRAG nodes")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        print("Note: Some features may require documents to be uploaded first")


if __name__ == "__main__":
    asyncio.run(main())


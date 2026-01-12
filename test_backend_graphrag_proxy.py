#!/usr/bin/env python3
"""
Backend GraphRAG Proxy Verification
Tests that llama-nexus backend can successfully proxy to GraphRAG.

NOTE: This requires the backend to be accessible (running in Docker or locally).
For Docker backend, it needs to be configured with GRAPHRAG_URL=http://host.docker.internal:18000
or be on the same Docker network as GraphRAG.
"""

import asyncio
import httpx
import json


BACKEND_URL = "http://localhost:8700"
GRAPHRAG_DIRECT_URL = "http://localhost:18000"


async def test_backend_proxy():
    """Test backend can proxy to GraphRAG."""
    print("=" * 70)
    print("BACKEND GRAPHRAG PROXY VERIFICATION")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Backend health
    print("\n1. Testing Backend Service...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/health")
            if response.status_code == 200:
                print("   ✓ Backend service is running")
                results["Backend Health"] = True
            else:
                print(f"   ✗ Backend returned {response.status_code}")
                results["Backend Health"] = False
    except Exception as e:
        print(f"   ✗ Cannot connect to backend: {e}")
        results["Backend Health"] = False
        print("\n❌ Backend is not accessible. Tests cannot continue.")
        return results
    
    # Test 2: GraphRAG proxy health
    print("\n2. Testing Backend → GraphRAG Proxy...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/graphrag/health")
            if response.status_code == 200:
                data = response.json()
                if data.get("available") or data.get("status") == "healthy":
                    print("   ✓ Backend can reach GraphRAG")
                    results["GraphRAG Proxy"] = True
                else:
                    print(f"   ⚠️  Backend reached but GraphRAG unavailable: {data}")
                    results["GraphRAG Proxy"] = False
            else:
                print(f"   ✗ Proxy returned {response.status_code}")
                results["GraphRAG Proxy"] = False
    except Exception as e:
        print(f"   ✗ Proxy error: {e}")
        results["GraphRAG Proxy"] = False
    
    # Test 3: Stats proxy
    print("\n3. Testing Knowledge Graph Stats via Proxy...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/graphrag/stats")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✓ Stats retrieved via backend proxy")
                print(f"     Nodes: {data.get('nodes', 0)}")
                print(f"     Edges: {data.get('edges', 0)}")
                results["Stats Proxy"] = True
            elif response.status_code == 502:
                print("   ⚠️  Backend cannot reach GraphRAG (502 Bad Gateway)")
                print("   This is expected if backend is in Docker without network access")
                results["Stats Proxy"] = False
            else:
                print(f"   ✗ Returned {response.status_code}")
                results["Stats Proxy"] = False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results["Stats Proxy"] = False
    
    # Test 4: Entity extraction via proxy
    print("\n4. Testing Entity Extraction via Proxy...")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/graphrag/extract",
                json={
                    "text": "Docker is a containerization platform. Kubernetes orchestrates Docker containers.",
                    "domain": "technology"
                }
            )
            if response.status_code == 200:
                data = response.json()
                entities = data.get("entities", [])
                print(f"   ✓ Entity extraction via proxy successful")
                print(f"     Entities: {len(entities)}")
                for e in entities[:3]:
                    print(f"       - {e['name']} ({e['type']})")
                results["Entity Extraction Proxy"] = len(entities) > 0
            elif response.status_code == 502:
                print("   ⚠️  Backend cannot reach GraphRAG")
                results["Entity Extraction Proxy"] = False
            else:
                print(f"   ✗ Returned {response.status_code}")
                results["Entity Extraction Proxy"] = False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results["Entity Extraction Proxy"] = False
    
    # Test 5: Top entities via proxy
    print("\n5. Testing Top Entities Query via Proxy...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/graphrag/top-entities",
                params={"limit": 5}
            )
            if response.status_code == 200:
                data = response.json()
                entities = data.get("top_entities", [])
                print(f"   ✓ Top entities retrieved via proxy")
                print(f"     Count: {len(entities)}")
                for e in entities[:3]:
                    print(f"       - {e['name']} (occurs {e['occurrence']}x)")
                results["Top Entities Proxy"] = len(entities) > 0
            elif response.status_code == 502:
                print("   ⚠️  Backend cannot reach GraphRAG")
                results["Top Entities Proxy"] = False
            else:
                print(f"   ✗ Returned {response.status_code}")
                results["Top Entities Proxy"] = False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results["Top Entities Proxy"] = False
    
    # Test 6: Direct GraphRAG comparison
    print("\n6. Comparing Direct vs Proxy Access...")
    try:
        async with httpx.AsyncClient() as client:
            # Direct access
            direct_response = await client.get(f"{GRAPHRAG_DIRECT_URL}/knowledge-graph/stats")
            # Proxy access
            proxy_response = await client.get(f"{BACKEND_URL}/api/v1/graphrag/stats")
            
            if direct_response.status_code == 200 and proxy_response.status_code == 200:
                direct_data = direct_response.json()
                proxy_data = proxy_response.json()
                
                direct_nodes = direct_data.get("nodes", 0)
                proxy_nodes = proxy_data.get("nodes", 0)
                
                if direct_nodes == proxy_nodes:
                    print(f"   ✓ Direct and proxy return same data")
                    print(f"     Nodes: {direct_nodes}")
                    results["Data Consistency"] = True
                else:
                    print(f"   ⚠️  Data mismatch: direct={direct_nodes}, proxy={proxy_nodes}")
                    results["Data Consistency"] = False
            elif proxy_response.status_code == 502:
                print("   ⚠️  Backend cannot reach GraphRAG (network isolation)")
                results["Data Consistency"] = False
            else:
                print("   ✗ One or both requests failed")
                results["Data Consistency"] = False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results["Data Consistency"] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("PROXY VERIFICATION SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL" if passed is False else "⊘ SKIP"
        print(f"{status} - {test_name}")
    
    passed_count = sum(1 for v in results.values() if v is True)
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count >= 2:  # At least basic connectivity
        print("\n✓ Core functionality verified")
        if passed_count < total_count:
            print("\nNote: Some proxy endpoints may require backend network configuration.")
            print("If backend is in Docker, it needs: GRAPHRAG_URL=http://host.docker.internal:18000")
            print("Or add to same Docker network as GraphRAG.")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_backend_proxy())


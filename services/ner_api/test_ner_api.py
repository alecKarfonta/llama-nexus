#!/usr/bin/env python3
"""
Test script for the NER API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8001"

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_model_info():
    """Test the model info endpoint."""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Model info: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Model info failed: {e}")
        return False

def test_single_ner():
    """Test single text NER."""
    print("\nTesting single text NER...")
    
    test_text = "My name is Wolfgang and I live in Berlin. I work for Microsoft."
    
    payload = {
        "text": test_text,
        "return_offsets": True,
        "return_scores": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ner", json=payload)
        print(f"Single NER: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Single NER failed: {e}")
        return False

def test_batch_ner():
    """Test batch NER processing."""
    print("\nTesting batch NER...")
    
    test_texts = [
        "Apple Inc. is headquartered in Cupertino, California.",
        "John Smith works at Google in Mountain View.",
        "The Eiffel Tower is located in Paris, France."
    ]
    
    payload = {
        "texts": test_texts,
        "return_offsets": True,
        "return_scores": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ner/batch", json=payload)
        print(f"Batch NER: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Batch NER failed: {e}")
        return False

def test_entity_types():
    """Test entity types endpoint."""
    print("\nTesting entity types endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/ner/entities")
        print(f"Entity types: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Entity types failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing NER API...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Entity Types", test_entity_types),
        ("Single NER", test_single_ner),
        ("Batch NER", test_batch_ner),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the logs above.")
    
    return all_passed

if __name__ == "__main__":
    main() 
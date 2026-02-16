#!/usr/bin/env python3
"""
Test script for the GLiNER Relationship Extraction API
"""

import requests
import json
import time
from typing import Dict, Any

# API configuration
BASE_URL = "http://localhost:8002"
headers = {"Content-Type": "application/json"}

def test_health():
    """Test the health endpoint."""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint."""
    print("\n🔍 Testing model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_capabilities():
    """Test the capabilities endpoint."""
    print("\n🔍 Testing capabilities endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/capabilities")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Capabilities: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Capabilities failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Capabilities error: {e}")
        return False

def test_entity_extraction():
    """Test entity extraction functionality."""
    print("\n🔍 Testing entity extraction...")
    
    text = """
    Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC interpreters for the Altair 8800. 
    During his career at Microsoft, Gates held the positions of chairman, chief executive officer, president and chief software architect, 
    while also being the largest individual shareholder until May 2014.
    """
    
    labels = ["organisation", "founder", "position", "date"]
    
    try:
        response = requests.post(
            f"{BASE_URL}/extract-entities",
            json={
                "text": text,
                "labels": labels,
                "threshold": 0.5
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Entity extraction successful:")
            print(f"   Text: {data['text'][:100]}...")
            print(f"   Entities found: {data['entity_count']}")
            print(f"   Processing time: {data['processing_time']:.3f}s")
            
            for entity in data['entities'][:5]:  # Show first 5 entities
                print(f"   - {entity['text']} ({entity['label']}) - Score: {entity['score']:.3f}")
            
            return True
        else:
            print(f"❌ Entity extraction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Entity extraction error: {e}")
        return False

def test_relation_extraction():
    """Test relationship extraction functionality."""
    print("\n🔍 Testing relationship extraction...")
    
    text = """
    Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC interpreters for the Altair 8800. 
    During his career at Microsoft, Gates held the positions of chairman, chief executive officer, president and chief software architect, 
    while also being the largest individual shareholder until May 2014.
    """
    
    # Define relations to extract
    relations = [
        {"relation": "founder", "pairs_filter": [("organisation", "founder")]},
        {"relation": "inception date", "pairs_filter": [("organisation", "date")]},
        {"relation": "held position", "pairs_filter": [("founder", "position")]}
    ]
    
    entity_labels = ["organisation", "founder", "position", "date"]
    
    request_data = {
        "text": text,
        "relations": relations,
        "entity_labels": entity_labels,
        "threshold": 0.5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/extract-relations",
            json=request_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Relation extraction successful:")
            print(f"   Text: {data['text'][:100]}...")
            print(f"   Relations found: {len(data['relations'])}")
            print(f"   Processing time: {data['processing_time']:.3f}s")
            
            for relation in data['relations'][:5]:  # Show first 5 relations
                print(f"   - {relation['source']} -> {relation['target']} ({relation['label']}) - Score: {relation['score']:.3f}")
            
            return True
        else:
            print(f"❌ Relation extraction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Relation extraction error: {e}")
        return False

def test_batch_extraction():
    """Test batch relationship extraction."""
    print("\n🔍 Testing batch relationship extraction...")
    
    texts = [
        "Apple was founded by Steve Jobs and Steve Wozniak in 1976.",
        "Google was founded by Larry Page and Sergey Brin in 1998.",
        "Tesla was founded by Elon Musk in 2003."
    ]
    
    relations = [
        {"relation": "founder", "pairs_filter": [("organisation", "founder")]},
        {"relation": "inception date", "pairs_filter": [("organisation", "date")]}
    ]
    
    entity_labels = ["organisation", "founder", "date"]
    
    request_data = {
        "texts": texts,
        "relations": relations,
        "entity_labels": entity_labels,
        "threshold": 0.5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/extract-relations/batch",
            json=request_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch extraction successful:")
            print(f"   Texts processed: {len(data['results'])}")
            print(f"   Processing time: {data['processing_time']:.3f}s")
            
            for i, result in enumerate(data['results']):
                print(f"   Text {i+1}: {result['relation_count']} relations found")
                if result['relation_count'] > 0:
                    for relation in result['relations'][:3]:  # Show first 3 relations
                        print(f"     - {relation['source']} -> {relation['target']} ({relation['label']})")
            
            return True
        else:
            print(f"❌ Batch extraction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch extraction error: {e}")
        return False

def run_tests():
    """Run all tests."""
    
    print("🚀 Starting GLiNER Relationship Extraction API tests...")
    print("============================================================")
    
    # Wait for the service to start
    print("🕒 Waiting for the service to start (60s)...")
    time.sleep(60)
    
    # Test counters
    passed_tests = 0
    
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Capabilities", test_capabilities),
        ("Entity Extraction", test_entity_extraction),
        ("Relation Extraction", test_relation_extraction),
        ("Batch Extraction", test_batch_extraction)
    ]
    
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total} tests passed")
    
    if passed_tests == total:
        print("🎉 All tests passed! GLiNER API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed_tests == total

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 
#!/usr/bin/env python3
"""
Test script for GPT-OSS API server
Tests OpenAI compatibility and gpt-oss specific features
"""

import json
import requests
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8600"
API_KEY = "placeholder-api-key"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def test_health() -> bool:
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"✅ Health check: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_models() -> bool:
    """Test models endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/v1/models", headers=HEADERS, timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Models endpoint: {len(models.get('data', []))} models available")
            for model in models.get('data', []):
                print(f"   • {model.get('id', 'Unknown')}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def test_chat_completion(test_name: str, messages: list, max_tokens: int = 200) -> bool:
    """Test chat completion endpoint"""
    try:
        payload = {
            "model": "gpt-oss-20b",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": False
        }
        
        print(f"\n🧪 Testing: {test_name}")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=60
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            
            print(f"✅ {test_name} successful ({duration:.1f}s)")
            print(f"   Tokens: {usage.get('total_tokens', 'unknown')}")
            print(f"   Response preview: {content[:200]}...")
            return True
        else:
            print(f"❌ {test_name} failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {test_name} error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing GPT-OSS API Server")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Health check
    total_tests += 1
    if test_health():
        tests_passed += 1
    
    # Test 2: Models endpoint
    total_tests += 1
    if test_models():
        tests_passed += 1
    
    # Test 3: Simple chat completion
    total_tests += 1
    if test_chat_completion(
        "Simple Question",
        [{"role": "user", "content": "What is 2+2? Give a brief answer."}],
        50
    ):
        tests_passed += 1
    
    # Test 4: Reasoning task
    total_tests += 1
    if test_chat_completion(
        "Mathematical Reasoning",
        [
            {
                "role": "system", 
                "content": "You are a helpful math tutor. Show your reasoning step by step. Reasoning: high"
            },
            {
                "role": "user", 
                "content": "If a rectangle has a length of 12 cm and a width of 8 cm, what is its area and perimeter?"
            }
        ],
        300
    ):
        tests_passed += 1
    
    # Test 5: Creative reasoning
    total_tests += 1
    if test_chat_completion(
        "Creative Problem Solving",
        [
            {
                "role": "system",
                "content": "You are a creative problem solver. Think outside the box. Reasoning: medium"
            },
            {
                "role": "user",
                "content": "How would you use a paperclip, a rubber band, and a coffee cup to measure the height of a building?"
            }
        ],
        400
    ):
        tests_passed += 1
    
    # Test 6: Chain of thought reasoning
    total_tests += 1
    if test_chat_completion(
        "Chain of Thought",
        [
            {
                "role": "system",
                "content": "You are an expert analyst. Break down complex problems step by step. Reasoning: high"
            },
            {
                "role": "user",
                "content": "A company's revenue increased by 25% in Q1, decreased by 10% in Q2, and increased by 15% in Q3. If they started with $1M revenue, what's their Q3 revenue?"
            }
        ],
        500
    ):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! GPT-OSS API is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check server logs for details.")
        return 1

if __name__ == "__main__":
    exit(main())
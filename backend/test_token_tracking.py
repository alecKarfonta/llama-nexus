#!/usr/bin/env python3
"""
Test script for token usage tracking
"""

import requests
import json
import time
import random
from datetime import datetime

def test_manual_token_recording():
    """Test manually recording token usage"""
    print("Testing manual token recording...")
    
    # Define test data
    models = [
        {"id": "qwen3-coder-30b", "name": "Qwen3-Coder-30B"},
        {"id": "llama3-70b-instruct", "name": "Llama3-70B-Instruct"},
        {"id": "mistral-7b", "name": "Mistral-7B"},
    ]
    
    # Record some sample token usage
    for _ in range(10):
        model = random.choice(models)
        prompt_tokens = random.randint(100, 1000)
        completion_tokens = random.randint(200, 2000)
        
        data = {
            "model_id": model["id"],
            "model_name": model["name"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "endpoint": "/v1/chat/completions",
            "metadata": {
                "test": True,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        try:
            response = requests.post(
                "http://localhost:8700/v1/usage/tokens/record",
                json=data
            )
            
            if response.status_code == 200:
                print(f"✅ Recorded: {prompt_tokens} prompt, {completion_tokens} completion tokens for {model['name']}")
            else:
                print(f"❌ Failed to record tokens: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Add some delay for timestamp variation
        time.sleep(0.5)
    
    print("Manual token recording test completed")

def test_token_usage_endpoints():
    """Test token usage endpoints"""
    print("\nTesting token usage endpoints...")
    
    # Test /v1/usage/tokens endpoint
    try:
        response = requests.get("http://localhost:8700/v1/usage/tokens")
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Token usage by model:")
            for model in data.get("data", []):
                print(f"  - {model.get('modelName', model.get('modelId'))}: {model.get('promptTokens', 0) + model.get('completionTokens', 0)} total tokens")
        else:
            print(f"❌ Failed to get token usage: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test /v1/usage/tokens/summary endpoint
    try:
        response = requests.get("http://localhost:8700/v1/usage/tokens/summary")
        if response.status_code == 200:
            data = response.json()
            summary = data.get("data", {})
            print("\n✅ Token usage summary:")
            print(f"  - Total tokens: {summary.get('totalTokens', 0)}")
            print(f"  - Prompt tokens: {summary.get('promptTokens', 0)}")
            print(f"  - Completion tokens: {summary.get('completionTokens', 0)}")
            print(f"  - Requests: {summary.get('requests', 0)}")
            print(f"  - Models: {summary.get('models', 0)}")
        else:
            print(f"❌ Failed to get token usage summary: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test /v1/usage/tokens/timeline endpoint
    try:
        response = requests.get("http://localhost:8700/v1/usage/tokens/timeline")
        if response.status_code == 200:
            data = response.json()
            timeline = data.get("data", [])
            print("\n✅ Token usage timeline:")
            for point in timeline:
                print(f"  - {point.get('timeInterval')}: {point.get('totalTokens', 0)} tokens ({point.get('requests', 0)} requests)")
        else:
            print(f"❌ Failed to get token usage timeline: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\nToken usage endpoints test completed")

if __name__ == "__main__":
    print("Token Tracking Test Script")
    print("=========================")
    
    # Run tests
    test_manual_token_recording()
    test_token_usage_endpoints()
    
    print("\nAll tests completed!")

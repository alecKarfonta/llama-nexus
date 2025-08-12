#!/bin/bash

echo "🚀 GPT-OSS API Quick Test"
echo "========================"

# Check if server is running
if ! curl -s http://localhost:8080/health > /dev/null; then
    echo "❌ Server not running. Start with: docker compose up -d"
    exit 1
fi

echo "✅ Server is running"

# Test models endpoint
echo ""
echo "📋 Available models:"
curl -s http://localhost:8080/v1/models \
    -H "Authorization: Bearer llamacpp-gpt-oss" | \
    jq -r '.data[].id' 2>/dev/null || echo "  (jq not installed - raw response above)"

# Simple chat test
echo ""
echo "💬 Testing chat completion..."
curl -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer llamacpp-gpt-oss" \
    -d '{
        "model": "gpt-oss-20b",
        "messages": [
            {"role": "user", "content": "What is 7 * 8? Show your reasoning."}
        ],
        "max_tokens": 150,
        "temperature": 1.0,
        "top_p": 1.0
    }' | jq -r '.choices[0].message.content' 2>/dev/null || echo "  (jq not installed - see raw JSON response above)"

echo ""
echo "🎉 Quick test completed!"
echo "💡 Run 'python3 test_api.py' for comprehensive testing"
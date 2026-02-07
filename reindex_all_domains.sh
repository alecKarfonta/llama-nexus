#!/bin/bash
# Re-index all domains to use the new nomic-embed-text-v1.5 embeddings

API_URL="http://localhost:8700"

# All domains that were updated
declare -a DOMAINS=(
    "6ebddeb6-6846-4c86-8614-a1a9fffa5664"  # Chair Sama
    "memory_chatter_stm"  # Chatter STM
    "00b64f6a-b5ab-413f-880e-78fee7e2e957"  # CollectionDescriptors
    "61dbe06a-c9fb-413d-bf72-0222532d841c"  # Facts
    "b76bcfa6-1bdf-4567-b63c-d30e17c0ce8d"  # General
    "274aa58b-c371-4305-b12e-514362464d0f"  # Jokes
    "5035537a-ac08-497e-a769-5441d5d3854a"  # Lyrics
    "d0c05dcd-c595-4148-aca5-d2b0cbddb4f1"  # Quotes
)

echo "Triggering re-indexing for ${#DOMAINS[@]} domains"
echo "This will clear and rebuild vector collections with nomic-embed-text-v1.5 (768D)"
echo ""

success_count=0
for domain_id in "${DOMAINS[@]}"; do
    # Get domain name
    name=$(curl -s "$API_URL/api/v1/rag/domains/$domain_id" | python3 -c "import sys,json; print(json.load(sys.stdin).get('name','unknown'))" 2>/dev/null)
    
    echo "→ Re-indexing: $name ($domain_id)"
    
    # Trigger reindex with collection recreation
    response=$(curl -s -X POST "$API_URL/api/v1/rag/domains/$domain_id/reindex" \
        -H "Content-Type: application/json" \
        -d '{"recreate_collection": true, "chunking_strategy": "semantic"}' \
        -w "\n%{http_code}")
    http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" = "200" ]; then
        echo "  ✓ Re-index started"
        ((success_count++))
    else
        echo "  ✗ Failed (HTTP $http_code)"
        echo "$response" | head -n-1
    fi
done

echo ""
echo "✓ Triggered re-indexing for $success_count/${#DOMAINS[@]} domains"
echo "Monitor progress at: $API_URL/api/v1/rag/processing/queue"

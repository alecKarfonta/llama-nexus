#!/bin/bash
# Standardize all RAG domains to use nomic-embed-text-v1.5 for GPU acceleration

API_URL="http://localhost:8700"
TARGET_MODEL="nomic-embed-text-v1.5"

# Domains that need updating
declare -a DOMAINS=(
    "6ebddeb6-6846-4c86-8614-a1a9fffa5664"  # Chair Sama
    "memory_chatter_stm"  # all-MiniLM-L6-v2
    "00b64f6a-b5ab-413f-880e-78fee7e2e957"  # CollectionDescriptors
    "61dbe06a-c9fb-413d-bf72-0222532d841c"  # Facts
    "b76bcfa6-1bdf-4567-b63c-d30e17c0ce8d"  # General
    "274aa58b-c371-4305-b12e-514362464d0f"  # Jokes
    "5035537a-ac08-497e-a769-5441d5d3854a"  # Lyrics
    "d0c05dcd-c595-4148-aca5-d2b0cbddb4f1"  # Quotes
)

echo "Standardizing ${#DOMAINS[@]} domains to $TARGET_MODEL"
echo ""

success_count=0
for domain_id in "${DOMAINS[@]}"; do
    # Fetch current domain
    domain_json=$(curl -s "$API_URL/api/v1/rag/domains/$domain_id")
    
    if [ $? -ne 0 ]; then
        echo "✗ Failed to fetch domain $domain_id"
        continue
    fi
    
    # Extract name and current model
    name=$(echo "$domain_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('name','unknown'))")
    old_model=$(echo "$domain_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('embedding_model','unknown'))")
    
    if [ "$old_model" = "$TARGET_MODEL" ]; then
        echo "✓ $name already using $TARGET_MODEL"
        ((success_count++))
        continue
    fi
    
    # Update embedding_model field
    updated_json=$(echo "$domain_json" | python3 -c "import sys,json; d=json.load(sys.stdin); d['embedding_model']='$TARGET_MODEL'; print(json.dumps(d))")
    
    # PUT the update
    response=$(curl -s -X PUT "$API_URL/api/v1/rag/domains/$domain_id" \
        -H "Content-Type: application/json" \
        -d "$updated_json" \
        -w "\n%{http_code}")
    
    http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" = "200" ]; then
        echo "✓ Updated $name: $old_model → $TARGET_MODEL"
        ((success_count++))
    else
        echo "✗ Failed to update $name (HTTP $http_code)"
    fi
done

echo ""
echo "✓ Updated $success_count/${#DOMAINS[@]} domains"

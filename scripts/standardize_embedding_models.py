#!/usr/bin/env python3
"""
Standardize all RAG domains to use nomic-embed-text-v1.5 for GPU acceleration.
"""
import requests
import sys

API_URL = "http://localhost:8700"
TARGET_MODEL = "nomic-embed-text-v1.5"

# Domains that need updating (using all-MiniLM-L6-v2 or old nomic-embed-text)
DOMAINS_TO_UPDATE = [
    "6ebddeb6-6846-4c86-8614-a1a9fffa5664",  # Chair Sama (nomic-embed-text)
    "memory_chatter_stm",  # all-MiniLM-L6-v2
    "00b64f6a-b5ab-413f-880e-78fee7e2e957",  # CollectionDescriptors (all-MiniLM-L6-v2)
    "61dbe06a-c9fb-413d-bf72-0222532d841c",  # Facts (all-MiniLM-L6-v2)
    "b76bcfa6-1bdf-4567-b63c-d30e17c0ce8d",  # General (nomic-embed-text)
    "274aa58b-c371-4305-b12e-514362464d0f",  # Jokes (all-MiniLM-L6-v2)
    "5035537a-ac08-497e-a769-5441d5d3854a",  # Lyrics (all-MiniLM-L6-v2)
    "d0c05dcd-c595-4148-aca5-d2b0cbddb4f1",  # Quotes (all-MiniLM-L6-v2)
]

def update_domain(domain_id: str):
    """Update a domain's embedding model."""
    # Get current domain config
    resp = requests.get(f"{API_URL}/api/v1/rag/domains/{domain_id}")
    if resp.status_code != 200:
        print(f"✗ Failed to fetch domain {domain_id}: {resp.status_code}")
        return False
    
    domain = resp.json()
    old_model = domain.get("embedding_model", "unknown")
    
    if old_model == TARGET_MODEL:
        print(f"✓ {domain['name']} already using {TARGET_MODEL}")
        return True
    
    # Update the domain
    domain["embedding_model"] = TARGET_MODEL
    resp = requests.put(f"{API_URL}/api/v1/rag/domains/{domain_id}", json=domain)
    
    if resp.status_code == 200:
        print(f"✓ Updated {domain['name']}: {old_model} → {TARGET_MODEL}")
        return True
    else:
        print(f"✗ Failed to update {domain['name']}: {resp.status_code} {resp.text}")
        return False

def main():
    print(f"Standardizing {len(DOMAINS_TO_UPDATE)} domains to {TARGET_MODEL}\\n")
    
    success_count = 0
    for domain_id in DOMAINS_TO_UPDATE:
        if update_domain(domain_id):
            success_count += 1
    
    print(f"\\n✓ Updated {success_count}/{len(DOMAINS_TO_UPDATE)} domains")
    return 0 if success_count == len(DOMAINS_TO_UPDATE) else 1

if __name__ == "__main__":
    sys.exit(main())

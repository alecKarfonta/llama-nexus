"""
Benchmark Reference Data Scraper

Scrapes live benchmark data from llm-stats.com for comparing local models
against public reference models.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Public benchmark data with recent models
# Last updated: January 2026
# Sources: Open LLM Leaderboard, llm-stats.com
# Note: These are representative scores - real-time data should be fetched via scraper

PUBLIC_BENCHMARK_DATA = [
    # --- Models with verified API coverage (api.zeroeval.com) ---
    # These names MUST match the 'model_name' field from the API exactly.
    # The scraper will merge in live scores; these are fallbacks only.
    {
        "model_name": "GPT-5",
        "model_size": "Unknown",
        "organization": "OpenAI",
        "scores": {},  # Scraper will populate from MMLU, etc.
        "source": "api.zeroeval.com",
    },
    {
        "model_name": "o1",
        "model_size": "Unknown",
        "organization": "OpenAI",
        "scores": {},
        "source": "api.zeroeval.com",
    },
    {
        "model_name": "Claude 3.5 Sonnet",
        "model_size": "Unknown",
        "organization": "Anthropic",
        "scores": {},
        "source": "api.zeroeval.com",
    },
    {
        "model_name": "Llama 3.1 405B Instruct",
        "model_size": "405B",
        "organization": "Meta",
        "scores": {},
        "source": "api.zeroeval.com",
    },
    {
        "model_name": "Llama 3.1 70B Instruct",
        "model_size": "70B",
        "organization": "Meta",
        "scores": {},
        "source": "api.zeroeval.com",
    },
    {
        "model_name": "Llama 3.1 8B Instruct",
        "model_size": "8B",
        "organization": "Meta",
        "scores": {},
        "source": "api.zeroeval.com",
    },
    {
        "model_name": "Claude 3 Opus",
        "model_size": "Unknown",
        "organization": "Anthropic",
        "scores": {},
        "source": "api.zeroeval.com",
    },
    {
        "model_name": "Mistral Small 3 24B Base",
        "model_size": "24B",
        "organization": "Mistral AI",
        "scores": {},
        "source": "api.zeroeval.com",
    },
    
    # --- Legacy Reference Models (kept for historical comparison) ---
    # Note: These are older models that may not be in the live API
    {
        "model_name": "Gemma-2-27B-it",
        "model_size": "27B",
        "organization": "Google",
        "scores": {
            "mmlu": 75.2,
            "gpqa": 42.3,
            "humaneval": 64.4,
            "arc_challenge": 67.1,
            "hellaswag": 86.4,
            "gsm8k": 79.8,
            "arc_easy": 89.2,
        },
        "source": "Open LLM Leaderboard",
    },
]

# Cache for scraped data
_benchmark_cache: Dict[str, Any] = {
    "data": None,
    "last_updated": None,
    "cache_duration_hours": 24, # Cache for 24 hours to avoid rate limits
}


async def scrape_live_benchmarks() -> List[Dict[str, Any]]:
    """
    Scrape live benchmark data from llm-stats.com.
    
    Returns:
        List of benchmark model data
    """
    import httpx
    from bs4 import BeautifulSoup
    import asyncio
    
    # Map Internal Key -> API Benchmark ID
    # We want to fetch these API endpoints
    api_map = {
        "mmlu": "mmlu",
        "gpqa": "gpqa",
        "math": "math",
        "humaneval": "humaneval",
        "gsm8k": "gsm8k",
        "hellaswag": "hellaswag",
        "truthfulqa_mc2": "truthfulqa",
        "arc_challenge": "arc-c", 
    }
    
    # Reverse map for processing: API ID -> Internal Key
    api_to_internal = {v: k for k, v in api_map.items()}
    
    # Storage for merged models: normalized_name -> model_data
    merged_models: Dict[str, Dict[str, Any]] = {}
    
    def normalize_name(n):
        return n.lower().replace(" ", "-").replace(".", "").replace("(", "").replace(")", "")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://llm-stats.com/",
        "Origin": "https://llm-stats.com"
    }

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=headers) as client:
            # Construct URLs
            urls = [f"https://api.zeroeval.com/leaderboard/benchmarks/{api_id}" for api_id in api_map.values()]
            
            # Fetch all URLs concurrently
            tasks = [client.get(url) for url in urls]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for url, response in zip(urls, responses):
                # Extract API ID from URL
                api_id = url.split("/")[-1]
                internal_key = api_to_internal.get(api_id)
                
                if isinstance(response, Exception):
                    print(f"DEBUG: Error fetching {api_id}: {response}")
                    continue
                    
                if response.status_code != 200:
                    print(f"DEBUG: Failed fetch {api_id}: {response.status_code}")
                    continue
                
                try:
                    data = response.json()
                    models = data.get("models", [])
                    print(f"DEBUG: Found {len(models)} models for {api_id}")
                    
                    for m in models:
                        model_name = m.get("model_name")
                        if not model_name:
                            continue
                            
                        # Parse score. API seems to return 0.925. We want 92.5
                        raw_score = m.get("score")
                        if raw_score is None:
                            continue
                            
                        score = float(raw_score)
                        # Normalize to 0-100 if it looks like 0-1
                        if score <= 1.0:
                            score *= 100.0
                            
                        org = m.get("organization_name", "Unknown")
                        
                        norm_name = normalize_name(model_name)
                        
                        if norm_name not in merged_models:
                            merged_models[norm_name] = {
                                "model_name": model_name,
                                "model_size": "Unknown",
                                "organization": org,
                                "scores": {},
                                "source": "llm-stats.com"
                            }
                        
                        # Merge score
                        merged_models[norm_name]["scores"][internal_key] = score
                        
                        # Update org if unknown
                        if merged_models[norm_name]["organization"] == "Unknown" and org != "Unknown":
                            merged_models[norm_name]["organization"] = org
                            
                except Exception as e:
                    print(f"DEBUG: Error parsing JSON for {api_id}: {e}")
                    continue
                        
    except Exception as e:
        print(f"Scraper critical error: {e}")
        return []

    return list(merged_models.values())


async def get_cached_benchmark_data() -> List[Dict[str, Any]]:
    """
    Get benchmark reference data.
    
    Returns cached data if valid, otherwise scrapes live data and merges with static fallback.
    """
    global _benchmark_cache
    
    # Check cache
    if _benchmark_cache["data"] and _benchmark_cache["last_updated"]:
        if datetime.now() - _benchmark_cache["last_updated"] < timedelta(hours=_benchmark_cache["cache_duration_hours"]):
            return _benchmark_cache["data"]
            
    # Scrape
    # print("Fetching live benchmark data...")
    live_data = await scrape_live_benchmarks()
    
    # Merge with static data (deduplicate by normalized name)
    def normalize(n): return n.lower().replace(" ", "-").replace(".", "")
    
    live_names = {normalize(d["model_name"]) for d in live_data}
    
    final_data = list(live_data)
    
    for static_item in PUBLIC_BENCHMARK_DATA:
        if normalize(static_item["model_name"]) not in live_names:
            final_data.append(static_item)
            
    # Update cache
    if final_data:
        _benchmark_cache["data"] = final_data
        _benchmark_cache["last_updated"] = datetime.now()
    else:
        # If both fail, return static
        return PUBLIC_BENCHMARK_DATA
    
    return final_data


def filter_benchmarks(
    data: List[Dict[str, Any]],
    model_size: Optional[str] = None,
    organization: Optional[str] = None,
    min_score: Optional[float] = None,
    benchmark: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Filter benchmark data by various criteria.
    
    Args:
        data: List of benchmark model data
        model_size: Filter by size (e.g., "7B", "70B")
        organization: Filter by organization (e.g., "Meta", "Google")
        min_score: Minimum average score threshold
        benchmark: Filter to models with this specific benchmark
        limit: Maximum number of results
        
    Returns:
        Filtered list of benchmark data
    """
    filtered = data.copy()
    
    if model_size:
        filtered = [m for m in filtered if model_size.lower() in m.get("model_size", "").lower()]
    
    if organization:
        filtered = [m for m in filtered if organization.lower() in m.get("organization", "").lower()]
    
    if benchmark:
        filtered = [m for m in filtered if benchmark.lower() in [k.lower() for k in m.get("scores", {}).keys()]]
    
    if min_score:
        def avg_score(m):
            scores = m.get("scores", {})
            if not scores:
                return 0
            return sum(scores.values()) / len(scores)
        filtered = [m for m in filtered if avg_score(m) >= min_score]
    
    # Sort by completeness (number of benchmark scores) descending
    # This ensures that models with more data (like GPT-4, Claude 3, Llama 3) 
    # appear in the top results even if the limit is small.
    filtered.sort(key=lambda m: len(m.get("scores", {})), reverse=True)
    
    return filtered[:limit]


def get_available_benchmarks(data: List[Dict[str, Any]] = None) -> List[str]:
    """
    Get list of all benchmark names available in the data.
    
    Args:
        data: Optional data list, uses cached data if not provided
        
    Returns:
        Sorted list of benchmark names
    """
    if data is None:
        data = get_cached_benchmark_data()
    
    benchmarks = set()
    for model in data:
        benchmarks.update(model.get("scores", {}).keys())
    
    return sorted(benchmarks)


def get_organizations(data: List[Dict[str, Any]] = None) -> List[str]:
    """
    Get list of all organizations in the benchmark data.
    
    Args:
        data: Optional data list, uses cached data if not provided
        
    Returns:
        Sorted list of organization names
    """
    if data is None:
        data = get_cached_benchmark_data()
    
    orgs = set()
    for model in data:
        if org := model.get("organization"):
            orgs.add(org)
    
    return sorted(orgs)

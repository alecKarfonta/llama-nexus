
import asyncio
import httpx
import json

async def check_api():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://llm-stats.com/",
        "Origin": "https://llm-stats.com"
    }
    
    endpoints = {
        "mmlu": "https://api.zeroeval.com/leaderboard/benchmarks/mmlu",
        "arc-c": "https://api.zeroeval.com/leaderboard/benchmarks/arc-c"
    }
    
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        for name, url in endpoints.items():
            print(f"\n--- Fetching {name} from {url} ---")
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    print(f"Failed: {resp.status_code}")
                    continue
                    
                data = resp.json()
                models = data.get("models", [])
                print(f"Found {len(models)} models.")
                
                # Search for GPT-5 / GPT-5.2
                targets = ["GPT-5", "Gemini 3", "Llama 3.1"]
                found_count = 0
                for m in models:
                    m_name = m.get("model_name", "")
                    if any(t in m_name for t in targets):
                        print(f"FOUND: {m_name} | Score: {m.get('score')}")
                        found_count += 1
                        
                if found_count == 0:
                    print("No target models (GPT-5, Gemini 3) found in this benchmark.")
                    
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_api())

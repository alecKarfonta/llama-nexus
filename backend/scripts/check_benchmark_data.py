
import asyncio
import sys
import os
import importlib.util

# Load module directly from file location to avoid package initialization
spec = importlib.util.spec_from_file_location("benchmark_reference", "/home/alec/git/llama-nexus/backend/modules/finetuning/benchmark_reference.py")
benchmark_reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_reference)

scrape_live_benchmarks = benchmark_reference.scrape_live_benchmarks
PUBLIC_BENCHMARK_DATA = benchmark_reference.PUBLIC_BENCHMARK_DATA

async def main():
    print("Checking Static Data...")
    for model in PUBLIC_BENCHMARK_DATA:
        print(f"Static: {model['model_name']} - Keys: {list(model['scores'].keys())}")

    print("\nRunning Scraper...")
    data = await scrape_live_benchmarks()
    print(f"\nScraped {len(data)} models.")
    
    # Check for ARC
    arc_models = [m for m in data if 'arc_challenge' in m['scores']]
    print(f"Models with arc_challenge: {len(arc_models)}")
    
    if len(arc_models) > 0:
        print(f"Example ARC model: {arc_models[0]['model_name']} - Score: {arc_models[0]['scores']['arc_challenge']}")
    else:
        print("No models have 'arc_challenge' key.")
        
    # Check for 'GPT-5.2' mentioned by user
    suspicious = [m for m in data if "GPT-5" in m['model_name'] or "Gemini 3" in m['model_name']]
    if suspicious:
        print(f"\nFound suspicious models: {[m['model_name'] for m in suspicious]}")
    else:
        print("\nNo 'GPT-5' or 'Gemini 3' models found in scraped data.")

if __name__ == "__main__":
    asyncio.run(main())

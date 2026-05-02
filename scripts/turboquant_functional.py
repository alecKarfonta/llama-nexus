#!/usr/bin/env python3
"""
TurboQuant Functional Quality Test

Uses the existing ServerProcess class from llama.cpp's test infrastructure
to run a live llama-server with different KV cache types and compare
generation quality head-to-head.

This catches real-world degradation that perplexity alone might miss —
e.g., coherence loss, instruction following failures, reasoning errors.

Usage:
    python3 scripts/turboquant_functional.py \
        --model /path/to/model.gguf \
        --output results/turboquant_functional.jsonl

    python3 scripts/turboquant_functional.py \
        --hf bartowski/Llama-3.1-8B-Instruct-GGUF:Q4_K_M \
        --kv-types q8_0,tbq3_0 --n 5
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from contextlib import contextmanager
from statistics import mean

# Import the existing ServerProcess from llama.cpp's test infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent / "llama.cpp"))
from tools.server.tests.utils import ServerProcess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@contextmanager
def scoped_server(sp: ServerProcess):
    """Context manager that ensures server cleanup (from tool_bench.py pattern)."""
    try:
        yield sp
    finally:
        sp.stop()


# ── Test Suite ────────────────────────────────────────────────────────────────
# Tasks that test different capabilities. Each returns (pass: bool, detail: str)

TESTS = {}

def register_test(name):
    def decorator(fn):
        TESTS[name] = fn
        return fn
    return decorator


@register_test("coherent_paragraph")
def test_coherent_paragraph(server: ServerProcess) -> tuple:
    """Can the model write a coherent paragraph? Tests basic generation quality."""
    response = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "Write exactly 3 sentences about why the ocean is blue. Be concise."}],
        "max_tokens": 150,
        "temperature": 0.0,
    })
    text = response.body["choices"][0]["message"]["content"]
    # Basic coherence: has multiple sentences, reasonable length
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    passed = len(sentences) >= 2 and len(text) > 50
    return passed, text


@register_test("math_reasoning")
def test_math_reasoning(server: ServerProcess) -> tuple:
    """Can the model do basic arithmetic? Tests reasoning preservation."""
    response = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "What is 47 * 23? Reply with just the number."}],
        "max_tokens": 32,
        "temperature": 0.0,
    })
    text = response.body["choices"][0]["message"]["content"].strip()
    passed = "1081" in text
    return passed, text


@register_test("instruction_following")
def test_instruction_following(server: ServerProcess) -> tuple:
    """Can the model follow specific format instructions?"""
    response = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "List exactly 3 programming languages. Format: one per line, numbered 1-3."}],
        "max_tokens": 100,
        "temperature": 0.0,
    })
    text = response.body["choices"][0]["message"]["content"].strip()
    has_numbers = all(f"{i}." in text or f"{i})" in text for i in [1, 2, 3])
    passed = has_numbers and len(text.strip().split("\n")) >= 3
    return passed, text


@register_test("long_context_recall")
def test_long_context_recall(server: ServerProcess) -> tuple:
    """NIAH-lite: can the model recall a fact from a moderately long context?"""
    filler = "The weather today is pleasant and mild. " * 100  # ~800 tokens of padding
    needle = "The secret code is TURBOQUANT42."
    prompt = f"Read this text carefully:\n\n{filler}\n{needle}\n{filler}\n\nWhat is the secret code mentioned in the text above?"
    
    response = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.0,
    })
    text = response.body["choices"][0]["message"]["content"]
    passed = "TURBOQUANT42" in text
    return passed, text


@register_test("json_output")
def test_json_output(server: ServerProcess) -> tuple:
    """Can the model produce valid JSON? Tests structured output capability."""
    response = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "Return a JSON object with keys 'name' (string) and 'age' (number). Nothing else."}],
        "max_tokens": 100,
        "temperature": 0.0,
    })
    text = response.body["choices"][0]["message"]["content"].strip()
    # Try to extract JSON from the response
    try:
        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        obj = json.loads(text.strip())
        passed = "name" in obj and "age" in obj
    except (json.JSONDecodeError, IndexError):
        passed = False
    return passed, text


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tests(server: ServerProcess, kv_type: str, model_name: str, n_runs: int, output_file):
    """Run all tests n_runs times against a server, write JSONL results."""
    for test_name, test_fn in TESTS.items():
        success_count = 0
        failure_count = 0
        times = []
        responses = []
        
        logger.info(f"Running {test_name} (kv={kv_type}): {n_runs} iterations")
        
        for i in range(n_runs):
            start = time.time()
            try:
                passed, detail = test_fn(server)
                elapsed = time.time() - start
                times.append(elapsed)
                if passed:
                    success_count += 1
                else:
                    failure_count += 1
                    responses.append(detail[:200])  # Truncate for logs
                logger.info(f"  [{i+1}/{n_runs}] {'✅' if passed else '❌'} ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - start
                times.append(elapsed)
                failure_count += 1
                responses.append(str(e)[:200])
                logger.error(f"  [{i+1}/{n_runs}] 💥 {e}")
        
        record = {
            "model": model_name,
            "kv_type": kv_type,
            "test": test_name,
            "success_ratio": success_count / n_runs if n_runs > 0 else 0,
            "success_count": success_count,
            "failure_count": failure_count,
            "avg_time": mean(times) if times else 0,
            "n_runs": n_runs,
            "sample_failures": responses[:3],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        output_file.write(json.dumps(record) + "\n")
        output_file.flush()


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Functional Quality Test")
    parser.add_argument("--model", help="Local GGUF model path")
    parser.add_argument("--hf", help="HuggingFace repo (e.g., bartowski/Llama-3.1-8B-Instruct-GGUF:Q4_K_M)")
    parser.add_argument("--kv-types", default="f16,q8_0,q4_0,tbq4_0,tbq3_0",
                        help="Comma-separated KV cache types (default: f16,q8_0,q4_0,tbq4_0,tbq3_0)")
    parser.add_argument("--n", type=int, default=5, help="Runs per test (default: 5)")
    parser.add_argument("--output", default="results/turboquant_functional.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--port", type=int, default=8084, help="Server port (default: 8084)")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context size (default: 4096)")
    parser.add_argument("--n-gpu-layers", type=int, default=99, help="GPU layers (default: 99)")
    args = parser.parse_args()

    if not args.model and not args.hf:
        parser.error("Either --model or --hf is required")

    model_name = args.model or args.hf
    kv_types = [t.strip() for t in args.kv_types.split(",")]
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       TurboQuant Functional Quality Test                    ║")
    print("║       (uses ServerProcess from llama.cpp test infra)        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"Model:    {model_name}")
    print(f"KV types: {kv_types}")
    print(f"Tests:    {list(TESTS.keys())}")
    print(f"Runs/test: {args.n}")
    print()

    with open(args.output, "w") as out:
        for kv_type in kv_types:
            print(f"\n{'='*60}")
            print(f"  KV cache type: {kv_type}")
            print(f"{'='*60}")

            server = ServerProcess()
            server.server_port = args.port
            server.n_ctx = args.n_ctx
            server.n_slots = 1
            server.n_predict = 512
            server.ctk = kv_type
            server.ctv = kv_type
            server.fa = "on"
            server.jinja = True
            server.temperature = 0.0
            server.seed = 42

            if args.model:
                server.model_file = args.model
                server.model_hf_repo = None
                server.model_hf_file = None
            elif args.hf:
                server.model_hf_repo = args.hf
                server.model_hf_file = None

            if args.n_gpu_layers:
                server.n_gpu_layer = args.n_gpu_layers

            with scoped_server(server):
                server.start(timeout_seconds=600)
                run_tests(server, kv_type, model_name, args.n, out)

    # Print summary
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    Results Summary                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    with open(args.output) as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    # Group by kv_type
    by_kv = {}
    for r in records:
        by_kv.setdefault(r["kv_type"], []).append(r)
    
    print(f"\n{'Test':<25} ", end="")
    for kv in kv_types:
        print(f"{kv:>12}", end="")
    print()
    print("-" * (25 + 12 * len(kv_types)))
    
    test_names = list(TESTS.keys())
    for test_name in test_names:
        print(f"{test_name:<25} ", end="")
        for kv in kv_types:
            kv_records = by_kv.get(kv, [])
            match = [r for r in kv_records if r["test"] == test_name]
            if match:
                ratio = match[0]["success_ratio"]
                symbol = "✅" if ratio >= 0.8 else "⚠️" if ratio >= 0.5 else "❌"
                print(f"{ratio:>9.0%}  {symbol}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()
    
    print(f"\nFull results: {args.output}")
    print("Generate report: python3 scripts/turboquant_report.py --functional", args.output)


if __name__ == "__main__":
    main()

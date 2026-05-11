#!/usr/bin/env python3
"""
Direct GGUF benchmark runner using llama-cpp-python.

Computes logprobs correctly by loading the model directly with logits_all=True,
which returns per-token logprobs for the entire sequence in a single pass.

Usage:
    python3 benchmark_direct.py --tasks piqa --limit 100 --num_fewshot 0
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from tqdm import tqdm


def load_model(model_path, n_ctx=4096):
    """Load the GGUF model with llama-cpp-python and GPU support."""
    from llama_cpp import Llama, LLAMA_SPLIT_MODE_LAYER

    print(f"[benchmark] Loading model from {model_path}...")
    print(f"[benchmark] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        split_mode=LLAMA_SPLIT_MODE_LAYER,
        verbose=False,
        n_ctx=n_ctx,
        logits_all=True,
    )
    elapsed = time.time() - t0
    print(f"[benchmark] Model loaded in {elapsed:.1f}s")
    return llm


def compute_loglikelihood(llm, context, continuation):
    """
    Compute log-likelihood of continuation given context.
    
    Uses the model's eval-based logprobs with logits_all=True.
    We send the full text (context + continuation) and extract logprobs
    for the continuation tokens.
    
    Returns (total_logprob, is_greedy).
    """
    # Tokenize context and full text separately
    ctx_tokens = llm.tokenize(context.encode("utf-8"), add_bos=True)
    full_tokens = llm.tokenize((context + continuation).encode("utf-8"), add_bos=True)
    ctx_len = len(ctx_tokens)

    # Evaluate the full sequence to get all logprobs
    # We use the low-level API to get logits for every position
    llm.eval(full_tokens)
    
    # Get logits for each position
    # logits[i] is the distribution over the NEXT token after position i
    # We need P(full_tokens[i] | full_tokens[0:i])
    # This is available from the logits at position i-1
    
    import numpy as np
    
    total_logprob = 0.0
    is_greedy = True
    
    for i in range(ctx_len, len(full_tokens)):
        target_token = full_tokens[i]
        # Get logits at position i-1 (predicting token at position i)
        logits = llm.scores[-(len(full_tokens) - i + 1)] if hasattr(llm, 'scores') else None
        
    # Actually, the simpler approach: use create_completion with echo
    # and parse the returned logprobs. With logits_all=True, echo works.
    
    # Reset and use create_completion
    full_text = context + continuation
    
    # We need max_tokens to be at least 1 to get the evaluation to run
    # But we don't actually want generated tokens - we just want the prompt logprobs
    response = llm.create_completion(
        prompt=full_text,
        max_tokens=1,
        logprobs=10,
        echo=True,
        temperature=0,
    )
    
    choice = response["choices"][0]
    logprobs_data = choice.get("logprobs")
    
    if not logprobs_data or "token_logprobs" not in logprobs_data:
        return 0.0, False
    
    token_logprobs = logprobs_data["token_logprobs"]
    tokens = logprobs_data["tokens"]
    top_logprobs = logprobs_data.get("top_logprobs", [])
    
    # With echo=True, max_tokens=1, logits_all=True:
    # Response has len(full_tokens) tokens with their logprobs.
    # The [-1] token is NOT a generated token - it's the last prompt token.
    # So we use [ctx_len:] instead of [ctx_len:-1].
    
    continuation_lps = token_logprobs[ctx_len:]
    
    # Filter None values
    valid_lps = [float(lp) for lp in continuation_lps if lp is not None]
    
    if not valid_lps:
        return 0.0, False
    
    total_logprob = sum(valid_lps)
    
    # Check is_greedy
    is_greedy = True
    for i in range(ctx_len, len(tokens)):
        if i < len(top_logprobs) and top_logprobs[i]:
            token = tokens[i]
            top_token = max(top_logprobs[i].keys(), key=lambda x: top_logprobs[i][x])
            if top_token != token:
                is_greedy = False
                break
    
    return total_logprob, is_greedy


def run_task(llm, task_name, limit=100, num_fewshot=0):
    """Run a single benchmark task using lm-eval for task setup."""
    from lm_eval.tasks import get_task_dict, TaskManager
    
    print(f"\n[benchmark] Running {task_name} (limit={limit}, num_fewshot={num_fewshot})")
    
    task_manager = TaskManager()
    task_dict = get_task_dict([task_name], task_manager)
    task_obj = list(task_dict.values())[0]
    
    task_obj.set_fewshot_seed(1234)
    
    docs = list(task_obj.eval_docs)
    if limit:
        docs = docs[:limit]
    
    print(f"[benchmark] {len(docs)} documents to evaluate")
    
    results = []
    correct = 0
    total = 0
    
    for i, doc in enumerate(tqdm(docs, desc=task_name)):
        try:
            # Build the fewshot context + requests
            ctx = task_obj.fewshot_context(doc, num_fewshot=num_fewshot)
            requests_list = task_obj.construct_requests(doc, ctx)
            
            if not requests_list:
                continue
            
            # Evaluate each request
            eval_results = []
            for req in requests_list:
                args = req.args
                context = args[0]
                continuation = args[1]
                logprob, is_greedy = compute_loglikelihood(llm, context, continuation)
                eval_results.append((logprob, is_greedy))
            
            # Use task's process_results
            processed = task_obj.process_results(doc, eval_results)
            
            # Accumulate metrics
            if isinstance(processed, dict):
                for metric, value in processed.items():
                    if metric in ('acc', 'exact_match'):
                        correct += int(value)
                        total += 1
            elif isinstance(processed, list):
                for item in processed:
                    if isinstance(item, tuple) and len(item) >= 2:
                        metric, value = item[0], item[1]
                        if metric in ('acc', 'exact_match'):
                            correct += int(value)
                            total += 1
            
        except Exception as e:
            print(f"  Error on doc {i}: {e}")
            traceback.print_exc()
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n[benchmark] {task_name}: acc={accuracy:.4f} ({correct}/{total})")
    return {"acc": accuracy, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="Direct GGUF benchmark runner")
    parser.add_argument("--tasks", nargs="+", default=["piqa"],
                       help="Tasks to evaluate (e.g., piqa arc_easy hellaswag)")
    parser.add_argument("--limit", type=int, default=100,
                       help="Number of examples per task")
    parser.add_argument("--num_fewshot", type=int, default=0,
                       help="Number of few-shot examples")
    parser.add_argument("--model-path", type=str,
                       default=os.environ.get("MODEL_PATH", "/home/llamacpp/models/Qwen3.6-27B-Q6_K.gguf"),
                       help="Path to GGUF model file")
    parser.add_argument("--output", type=str, default="/tmp/benchmark_results.json",
                       help="Output file for results")
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"[benchmark] Model not found: {args.model_path}")
        sys.exit(1)
    
    llm = load_model(args.model_path)
    
    # Quick validation test
    print("\n[benchmark] Validation: 'The capital of France is' + 'Paris' vs 'London'")
    lp_paris, _ = compute_loglikelihood(llm, "The capital of France is", " Paris")
    lp_london, _ = compute_loglikelihood(llm, "The capital of France is", " London")
    print(f"  Paris:  {lp_paris:.4f}")
    print(f"  London: {lp_london:.4f}")
    print(f"  Paris wins: {lp_paris > lp_london}")
    
    if lp_paris <= lp_london:
        print("[benchmark] WARNING: Validation failed! Logprobs may be incorrect.")
    
    # Run benchmarks
    all_results = {}
    for task in args.tasks:
        try:
            result = run_task(llm, task, limit=args.limit, num_fewshot=args.num_fewshot)
            all_results[task] = result
        except Exception as e:
            print(f"[benchmark] Error running {task}: {e}")
            traceback.print_exc()
            all_results[task] = {"error": str(e)}
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[benchmark] Results saved to {args.output}")
    
    # Summary
    print("\n[benchmark] Summary:")
    for task, result in all_results.items():
        if "error" in result:
            print(f"  {task}: ERROR - {result['error']}")
        else:
            print(f"  {task}: acc={result['acc']:.4f} ({result['correct']}/{result['total']})")


if __name__ == "__main__":
    main()

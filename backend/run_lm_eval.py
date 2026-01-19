#!/usr/bin/env python3
"""
Run lm-eval with custom gguf-local model registered.
This wrapper imports the custom model module to register it with lm-eval,
then invokes the evaluator programmatically.
"""

import sys
import os

# Add /app to path and import custom model to register it
sys.path.insert(0, '/app')
import gguf_local_model  # noqa: F401 - Registers the gguf-local model

# Now import and run lm-eval
from lm_eval import evaluator
from lm_eval.utils import make_table
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Run lm-eval with gguf-local model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to GGUF model file")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated list of tasks")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Number of fewshot examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per task")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save results")
    parser.add_argument("--n_ctx", type=int, default=2048, help="Context size")
    parser.add_argument("--n_gpu_layers", type=int, default=-1, help="GPU layers (-1 for all)")
    parser.add_argument("--n_batch", type=int, default=512, help="Batch size")
    
    args = parser.parse_args()
    
    # Parse tasks
    tasks = [t.strip() for t in args.tasks.split(",")]
    
    # Build model_args dict
    model_args = {
        "model_path": args.model_path,
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "n_batch": args.n_batch,
    }
    
    # Convert to string format for lm-eval
    model_args_str = ",".join(f"{k}={v}" for k, v in model_args.items())
    
    print(f"[lm-eval-runner] Running evaluation:")
    print(f"  Model: {args.model_path}")
    print(f"  Tasks: {tasks}")
    print(f"  Fewshot: {args.num_fewshot}")
    print(f"  Limit: {args.limit}")
    
    # Run evaluation
    results = evaluator.simple_evaluate(
        model="gguf-local",
        model_args=model_args_str,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=True,
    )
    
    # Print results table
    if results:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(make_table(results))
        
        # Save to output if specified
        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)
            results_file = os.path.join(args.output_path, "results.json")
            with open(results_file, "w") as f:
                # Make results JSON-serializable
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {results_file}")
        
        return 0
    else:
        print("ERROR: No results returned from evaluation")
        return 1


if __name__ == "__main__":
    sys.exit(main())

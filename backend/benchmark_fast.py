#!/usr/bin/env python3
"""
Fast benchmark runner using llama-cpp-python with single-pass logprob extraction.

Supports:
- loglikelihood tasks (multiple choice: hellaswag, arc, piqa, etc.)
- generate_until tasks (free-form: gsm8k)
- task groups (mmlu -> 57 subtasks)
"""

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_model(model_path, n_ctx=4096):
    from llama_cpp import Llama, LLAMA_SPLIT_MODE_LAYER

    print(f"[bench] Loading {model_path}...", flush=True)
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        split_mode=LLAMA_SPLIT_MODE_LAYER,
        verbose=False,
        n_ctx=n_ctx,
        logits_all=True,
    )
    print(f"[bench] Loaded in {time.time()-t0:.1f}s", flush=True)
    return llm


def compute_logprob_single_pass(llm, context, continuation):
    """
    Compute log-likelihood of continuation given context in a single forward pass.
    For loglikelihood-type tasks (multiple choice).
    """
    ctx_tokens = llm.tokenize(context.encode("utf-8"), add_bos=True)
    cont_tokens = llm.tokenize(continuation.encode("utf-8"), add_bos=False)
    full_tokens = ctx_tokens + cont_tokens
    ctx_len = len(ctx_tokens)

    if ctx_len >= len(full_tokens):
        return 0.0, False

    llm.reset()
    llm.eval(full_tokens)

    n_vocab = llm._n_vocab
    start_idx = ctx_len - 1
    end_idx = len(full_tokens) - 1

    total_logprob = 0.0
    is_greedy = True

    for j, logits_idx in enumerate(range(start_idx, end_idx)):
        cont_token_idx = ctx_len + j
        target_id = full_tokens[cont_token_idx]

        logits = llm.scores[logits_idx, :n_vocab]
        max_val = np.max(logits)
        log_probs = logits - max_val - np.log(np.sum(np.exp(logits - max_val)))

        total_logprob += float(log_probs[target_id])
        if int(np.argmax(logits)) != target_id:
            is_greedy = False

    return total_logprob, is_greedy


def generate_text(llm, prompt, max_tokens=256, temperature=0.0, stop=None):
    """
    Generate text from a prompt using llama-cpp-python.
    For generate_until-type tasks (GSM8K).
    """
    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    llm.reset()
    llm.eval(tokens)

    n_vocab = llm._n_vocab
    generated_tokens = []
    generated_text = ""

    for _ in range(max_tokens):
        logits = llm.scores[llm.n_tokens - 1, :n_vocab]

        if temperature > 0:
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            next_token = int(np.random.choice(len(probs), p=probs))
        else:
            next_token = int(np.argmax(logits))

        token_text = llm.detokenize([next_token]).decode("utf-8", errors="replace")
        generated_text += token_text
        generated_tokens.append(next_token)

        # Check stop sequences
        if stop:
            for s in stop:
                if s and s in generated_text:
                    # Truncate at stop sequence
                    idx = generated_text.index(s)
                    generated_text = generated_text[:idx]
                    return generated_text

        # Continue generation
        llm.eval([next_token])

    return generated_text


def _apply_filters(result_text, gold_text, task_obj):
    """Apply GSM8K-style regex filters to extract answers for comparison."""
    filters = getattr(task_obj, '_filters', None) or []
    if not filters:
        config_filters = getattr(task_obj.config, 'filter_list', None)
        if config_filters:
            return _apply_config_filters(result_text, gold_text, config_filters)
        return result_text, gold_text

    for f in filters:
        result_text, gold_text = f.apply([result_text], [gold_text])
    return result_text, gold_text


def _apply_config_filters(result_text, gold_text, filter_list):
    """Apply filter configs (e.g., GSM8K's strict-match regex)."""
    # Use the first filter set (strict-match for GSM8K)
    if filter_list and len(filter_list) > 0:
        filter_set = filter_list[0]
        for f in filter_set.get('filter', []):
            fn = f.get('function', '')
            if fn == 'regex':
                pattern = f.get('regex_pattern', '')
                if pattern:
                    m = re.search(pattern, result_text)
                    if m:
                        result_text = m.group(1)
                    m = re.search(pattern, gold_text)
                    if m:
                        gold_text = m.group(1)
            elif fn == 'take_first':
                pass  # already have first match from regex
    return result_text, gold_text


def run_single_task(llm, task_obj, task_name, limit, num_fewshot):
    """Run a single task (handles both loglikelihood and generate_until)."""
    output_type = getattr(task_obj, 'OUTPUT_TYPE', 'loglikelihood')
    print(f"  [subtask] {task_name} type={output_type}", flush=True)

    try:
        task_obj.set_fewshot_seed(1234)
    except Exception:
        pass

    docs = list(task_obj.eval_docs)[:limit] if limit else list(task_obj.eval_docs)

    metrics_accum = {}
    total = 0
    t0 = time.time()

    for i, doc in enumerate(tqdm(docs, desc=task_name, leave=False)):
        try:
            ctx = task_obj.fewshot_context(doc, num_fewshot=num_fewshot)
            requests_list = task_obj.construct_requests(doc, ctx)
            if requests_list is None:
                continue

            eval_results = []

            if output_type == "generate_until":
                if not isinstance(requests_list, list):
                    requests_list = [requests_list]
                for req in requests_list:
                    args = req.args
                    prompt = args[0]
                    gen_kwargs = req.kwargs if hasattr(req, 'kwargs') else {}
                    max_tok = gen_kwargs.get('max_gen_toks', 256) or gen_kwargs.get('max_tokens', 256) or 256
                    stop_seqs = gen_kwargs.get('stop', gen_kwargs.get('until', []))
                    if isinstance(stop_seqs, str):
                        stop_seqs = [stop_seqs]
                    temp = gen_kwargs.get('temperature', 0.0) or gen_kwargs.get('do_sample', False) and 0.1 or 0.0

                    generated = generate_text(llm, prompt, max_tokens=min(max_tok, 1024),
                                              temperature=temp, stop=stop_seqs)
                    eval_results.append(generated)

                # For generate_until, process_results expects (gold, result)
                # We need to apply filters to extract the answer
                gold = task_obj.doc_to_target(doc)
                filtered_gold = gold
                filtered_result = eval_results[0] if eval_results else ""

                # Apply GSM8K-style regex filters if configured
                config_filters = getattr(task_obj.config, 'filter_list', None)
                if config_filters:
                    filtered_result, filtered_gold = _apply_config_filters(
                        filtered_result, filtered_gold, config_filters
                    )

                # Compare extracted answers
                # Clean up: remove commas, strip whitespace
                filtered_gold = filtered_gold.replace(',', '').strip()
                filtered_result = filtered_result.replace(',', '').strip()

                # Check match
                match = (filtered_result == filtered_gold)
                metrics_accum.setdefault('exact_match', []).append(float(match))
                total += 1
                continue

            elif output_type == "multiple_choice":
                if not isinstance(requests_list, list):
                    requests_list = [requests_list]
                lls = []
                greedy = []
                for req in requests_list:
                    args = req.args
                    lp, ig = compute_logprob_single_pass(llm, args[0], args[1])
                    lls.append(lp)
                    greedy.append(ig)
                eval_results = list(zip(lls, greedy))

            else:  # loglikelihood
                if not isinstance(requests_list, list):
                    requests_list = [requests_list]
                for req in requests_list:
                    args = req.args
                    lp, ig = compute_logprob_single_pass(llm, args[0], args[1])
                    eval_results.append((lp, ig))

            processed = task_obj.process_results(doc, eval_results)

            if isinstance(processed, dict):
                for metric, value in processed.items():
                    if metric not in metrics_accum:
                        metrics_accum[metric] = []
                    metrics_accum[metric].append(value)
                total += 1
            elif isinstance(processed, list):
                for item in processed:
                    if isinstance(item, tuple) and len(item) >= 2:
                        metric, value = item[0], item[1]
                        if metric not in metrics_accum:
                            metrics_accum[metric] = []
                        metrics_accum[metric].append(value)
                total += 1
        except Exception as e:
            print(f"  doc {i} error: {e}")
            total += 1

    elapsed = time.time() - t0

    results = {}
    for metric, values in metrics_accum.items():
        if metric in ('mc2', 'mc1'):
            results[metric] = sum(values) / len(values) if values else 0.0
        else:
            results[metric] = sum(values) / len(values) if values else 0.0
            results[f"{metric}_correct"] = int(sum(values))
            results[f"{metric}_total"] = len(values)

    results["total_docs"] = total
    results["elapsed"] = elapsed

    return results


def _expand_task_group(task_name, task_manager):
    """Recursively expand a task group (like MMLU) into individual tasks."""
    from lm_eval.tasks import get_task_dict
    from lm_eval.api.task import ConfigurableTask

    td = get_task_dict([task_name], task_manager)
    actual_tasks = {}

    def _flatten(d):
        for k, v in d.items():
            if isinstance(v, ConfigurableTask):
                actual_tasks[k] = v
            elif isinstance(v, dict):
                # Sub-group, recurse
                _flatten(v)

    _flatten(td)
    return actual_tasks


def run_task(llm, task_name, limit=2000, num_fewshot=0):
    """
    Run a benchmark task. Handles task groups (MMLU) by running each subtask.
    """
    from lm_eval.tasks import get_task_dict, TaskManager
    from lm_eval.api.task import ConfigurableTask

    print(f"\n[bench] {task_name} (limit={limit}, fewshot={num_fewshot})", flush=True)

    task_manager = TaskManager()
    task_dict = get_task_dict([task_name], task_manager)

    # Separate actual tasks from groups
    actual_tasks = {}
    for k, v in task_dict.items():
        if isinstance(v, ConfigurableTask):
            actual_tasks[k] = v
        elif isinstance(v, dict):
            # Nested group (MMLU), flatten it
            actual_tasks.update(_expand_task_group(task_name, task_manager))

    if len(actual_tasks) == 1 and task_name in actual_tasks:
        return run_single_task(llm, actual_tasks[task_name], task_name, limit, num_fewshot)
    elif len(actual_tasks) > 1:
        print(f"[bench] {task_name}: {len(actual_tasks)} subtasks", flush=True)
        return _run_task_group(llm, actual_tasks, task_name, limit, num_fewshot)

    return {"error": f"Could not resolve task {task_name}"}


def _run_task_group(llm, subtasks, group_name, limit, num_fewshot):
    """Run a group of subtasks (e.g., MMLU's 57 subjects) and aggregate."""
    total_metrics = {}
    total_docs = 0
    total_elapsed = 0.0

    for sub_name, sub_obj in subtasks.items():
        try:
            # Distribute limit across subtasks proportionally
            n_docs = len(list(sub_obj.eval_docs)) if sub_obj.eval_docs else 0
            sub_limit = min(limit, n_docs) if limit else n_docs

            result = run_single_task(llm, sub_obj, sub_name, sub_limit, num_fewshot)

            for k, v in result.items():
                if k in ('elapsed', 'total_docs'):
                    continue
                if k.endswith('_correct') or k.endswith('_total'):
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
                elif isinstance(v, float):
                    if k not in total_metrics:
                        total_metrics[k] = []
                    total_metrics[k].append(v)

            total_docs += result.get('total_docs', 0)
            total_elapsed += result.get('elapsed', 0)
        except Exception as e:
            print(f"  subtask {sub_name} error: {e}")

    # Compute overall accuracy
    results = {}
    correct_key = None
    total_key = None
    for k in total_metrics:
        if k.endswith('_correct'):
            metric_name = k.replace('_correct', '')
            correct_key = k
            total_key = f"{metric_name}_total"
            if total_key in total_metrics:
                total_val = total_metrics[total_key]
                results[metric_name] = total_metrics[correct_key] / total_val if total_val > 0 else 0.0
                results[correct_key] = total_metrics[correct_key]
                results[total_key] = total_val

    # Average any float metrics (e.g., per-subject accuracies)
    for k, v in total_metrics.items():
        if isinstance(v, list) and v:
            results[f"avg_{k}"] = sum(v) / len(v)

    results["total_docs"] = total_docs
    results["elapsed"] = total_elapsed
    results["num_subtasks"] = len(subtasks)

    print(f"[bench] {group_name}: {results} in {total_elapsed:.0f}s", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["piqa"])
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--output", type=str, default="/data/lm_eval_results/fast-bench.json")
    args = parser.parse_args()

    if not args.model_path or not Path(args.model_path).exists():
        print(f"[bench] Model not found: {args.model_path}")
        sys.exit(1)

    llm = load_model(args.model_path)

    # Validation
    lp1, _ = compute_logprob_single_pass(llm, "The capital of France is", " Paris")
    lp2, _ = compute_logprob_single_pass(llm, "The capital of France is", " London")
    print(f"[bench] Validation: Paris={lp1:.4f} London={lp2:.4f} Paris_wins={lp1>lp2}", flush=True)

    if lp1 <= lp2:
        print("[bench] WARNING: Validation failed!")

    all_results = {}
    for task in args.tasks:
        try:
            result = run_task(llm, task, limit=args.limit, num_fewshot=args.num_fewshot)
            all_results[task] = result
        except Exception as e:
            print(f"[bench] {task} error: {e}")
            traceback.print_exc()
            all_results[task] = {"error": str(e)}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[bench] Saved to {args.output}", flush=True)

    print("\n=== RESULTS ===")
    for task, r in all_results.items():
        if "error" in r:
            print(f"  {task}: ERROR - {r['error']}")
        else:
            parts = []
            for k, v in r.items():
                if k == "elapsed":
                    parts.append(f"{v:.0f}s")
                elif isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            print(f"  {task}: {', '.join(parts)}")


if __name__ == "__main__":
    main()

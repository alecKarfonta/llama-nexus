"""
LM-Eval Worker for GPU-accelerated GGUF model evaluation.

Polls Redis for benchmark jobs and runs lm-evaluation-harness with local
GGUF model loading. Reports progress and results back via Redis pub/sub.
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import redis

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MODELS_DIR = os.getenv("MODELS_DIR", "/home/llamacpp/models")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "2"))
OUTPUT_DIR = os.getenv("LM_EVAL_OUTPUT_DIR", "/data/lm_eval_results")


def connect_redis() -> redis.Redis:
    """Connect to Redis with retries."""
    max_retries = 10
    for i in range(max_retries):
        try:
            client = redis.from_url(REDIS_URL, decode_responses=True)
            client.ping()
            print(f"[LM-Eval Worker] Connected to Redis at {REDIS_URL}")
            return client
        except redis.ConnectionError as e:
            print(f"[LM-Eval Worker] Redis connection failed (attempt {i+1}/{max_retries}): {e}")
            time.sleep(2)
    raise RuntimeError("Failed to connect to Redis")


def publish_status(
    client: redis.Redis,
    job_id: str,
    status: str,
    progress: float = 0.0,
    message: str = "",
    results: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
):
    """Publish job status update to Redis."""
    payload = {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if results:
        payload["results"] = results
    if error:
        payload["error"] = error
    
    client.publish(f"lm_eval:status:{job_id}", json.dumps(payload))
    # Also store in a key for retrieval
    client.set(f"lm_eval:job:{job_id}:status", json.dumps(payload), ex=86400)


def publish_log(client: redis.Redis, job_id: str, message: str):
    """Publish log message to Redis."""
    payload = {
        "job_id": job_id,
        "type": "log",
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }
    client.publish(f"lm_eval:logs:{job_id}", json.dumps(payload))


def find_model_path(model_name: str) -> Optional[str]:
    """Find the GGUF model file in the models directory."""
    models_path = Path(MODELS_DIR)
    
    # Try exact match first
    exact_path = models_path / model_name
    if exact_path.exists() and exact_path.suffix == ".gguf":
        return str(exact_path)
    
    # Try adding .gguf extension
    gguf_path = models_path / f"{model_name}.gguf"
    if gguf_path.exists():
        return str(gguf_path)
    
    # Search for matching files
    for f in models_path.glob("*.gguf"):
        if model_name.lower() in f.name.lower():
            return str(f)
    
    return None


def run_evaluation(
    client: redis.Redis,
    job_id: str,
    model_path: str,
    tasks: list,
    num_fewshot: int = 5,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run lm-eval benchmark by connecting to llama.cpp API server."""
    
    # Create output directory
    output_path = Path(OUTPUT_DIR) / f"lm_eval_{job_id}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Connect to the llama.cpp API server
    # The model_path is informational - the server already has a model loaded
    # We use the gguf model type which connects to a llama.cpp server
    # Build command - use custom run_lm_eval.py wrapper that imports the gguf-local model
    # This registers the model with lm-eval before running evaluation
    task_str = ",".join(tasks)
    
    # Use the wrapper script that imports our custom model
    cmd = [
        "python3", "/app/run_lm_eval.py",
        "--model_path", model_path,
        "--tasks", task_str,
        "--num_fewshot", str(num_fewshot),
        "--output_path", str(output_path),
        "--n_ctx", "2048",
        "--n_gpu_layers", "-1",
        "--n_batch", "512",
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    print(f"[LM-Eval Worker] Running: {' '.join(cmd)}")
    publish_log(client, job_id, f"Starting evaluation: {task_str}")
    publish_status(client, job_id, "running", 0.0, f"Starting {len(tasks)} benchmark(s)...")
    
    # Run the command and stream output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    all_output = []
    last_progress = 0.0
    
    for line in iter(process.stdout.readline, ""):
        line = line.strip()
        if not line:
            continue
        
        all_output.append(line)
        print(f"[lm_eval] {line}")
        
        # Parse progress from output
        if "%" in line:
            match = re.search(r"(\d+)%", line)
            if match:
                progress = int(match.group(1))
                if progress > last_progress:
                    last_progress = progress
                    publish_status(client, job_id, "running", progress, line[:100])
        
        # Publish log lines (not too frequently)
        if not line.startswith(("╭", "│", "╰", "─")):
            publish_log(client, job_id, line[:200])
    
    process.wait()
    
    if process.returncode != 0:
        error_msg = f"lm_eval exited with code {process.returncode}"
        publish_status(client, job_id, "failed", last_progress, error_msg, error=error_msg)
        return {"error": error_msg, "output": "\n".join(all_output[-50:])}
    
    # Parse results
    results = parse_results(output_path)
    publish_status(client, job_id, "completed", 100.0, "Evaluation complete", results=results)
    
    return results


def parse_results(output_path: Path) -> Dict[str, Any]:
    """Parse lm-eval results from output directory."""
    results = {"tasks": {}}
    
    # Look for results.json in output directory or subdirectories
    for results_file in output_path.rglob("results.json"):
        try:
            with open(results_file) as f:
                raw_results = json.load(f)
            
            results["model_name"] = raw_results.get("model_name", "unknown")
            results["date"] = raw_results.get("date", datetime.now().isoformat())
            
            for task_name, task_results in raw_results.get("results", {}).items():
                metrics = {}
                for key, value in task_results.items():
                    if isinstance(value, (int, float)):
                        # Strip lm-eval v0.4+ filter suffixes (e.g., "acc,none" -> "acc")
                        clean_key = key.split(",")[0]
                        # Skip stderr keys
                        if "stderr" in clean_key:
                            continue
                        # Convert to percentage if it's a ratio
                        if 0 <= value <= 1:
                            metrics[clean_key] = round(value * 100, 2)
                        else:
                            metrics[clean_key] = round(value, 4) if isinstance(value, float) else value
                
                # Get primary score (now using cleaned keys)
                primary_score = metrics.get("acc_norm") or metrics.get("acc") or metrics.get("exact_match") or 0
                
                results["tasks"][task_name] = {
                    "score": primary_score,
                    "metrics": metrics,
                }
            
            break  # Found results, stop searching
        except (json.JSONDecodeError, IOError) as e:
            print(f"[LM-Eval Worker] Error parsing results: {e}")
    
    return results


def process_job(client: redis.Redis, job_data: str) -> None:
    """Process a single benchmark job."""
    try:
        job = json.loads(job_data)
    except json.JSONDecodeError as e:
        print(f"[LM-Eval Worker] Invalid job JSON: {e}")
        return
    
    job_id = job.get("id")
    if not job_id:
        print("[LM-Eval Worker] Job missing ID, skipping")
        return
    
    print(f"[LM-Eval Worker] Processing job {job_id}")
    
    # Get model path
    model_name = job.get("model_path") or job.get("model_name")
    if not model_name:
        # Try to find any available model
        models_path = Path(MODELS_DIR)
        gguf_files = list(models_path.glob("*.gguf"))
        if gguf_files:
            model_path = str(gguf_files[0])
            print(f"[LM-Eval Worker] No model specified, using: {model_path}")
        else:
            publish_status(client, job_id, "failed", 0, "No model specified and no models found", 
                          error="No GGUF model available")
            return
    else:
        model_path = find_model_path(model_name)
        if not model_path:
            publish_status(client, job_id, "failed", 0, f"Model not found: {model_name}",
                          error=f"Model not found: {model_name}")
            return
    
    print(f"[LM-Eval Worker] Using model: {model_path}")
    publish_log(client, job_id, f"Loading model: {model_path}")
    
    # Set GPU device if specified
    gpu_device = job.get("gpu_device")
    if gpu_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        print(f"[LM-Eval Worker] Using GPU device: {gpu_device}")
        publish_log(client, job_id, f"Using GPU device: {gpu_device}")
    
    # Get task configuration
    tasks = job.get("tasks", ["hellaswag"])
    num_fewshot = job.get("num_fewshot", 5)
    limit = job.get("limit")
    
    # Run evaluation
    try:
        results = run_evaluation(client, job_id, model_path, tasks, num_fewshot, limit)
        print(f"[LM-Eval Worker] Job {job_id} completed with results: {results}")
    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        print(f"[LM-Eval Worker] {error_msg}")
        publish_status(client, job_id, "failed", 0, error_msg, error=error_msg)


def poll_for_jobs():
    """Main polling loop for benchmark jobs."""
    print("[LM-Eval Worker] Starting job polling...")
    print(f"[LM-Eval Worker] Models directory: {MODELS_DIR}")
    print(f"[LM-Eval Worker] Output directory: {OUTPUT_DIR}")
    
    # List available models
    models_path = Path(MODELS_DIR)
    if models_path.exists():
        gguf_files = list(models_path.glob("*.gguf"))
        print(f"[LM-Eval Worker] Available GGUF models: {[f.name for f in gguf_files]}")
    
    client = connect_redis()
    
    while True:
        try:
            # Check for new jobs in the queue (blocking pop with timeout)
            result = client.brpop("lm_eval:jobs", timeout=POLL_INTERVAL)
            
            if result:
                _, job_data = result
                process_job(client, job_data)
            
        except redis.ConnectionError as e:
            print(f"[LM-Eval Worker] Redis connection lost: {e}")
            time.sleep(5)
            client = connect_redis()
        except Exception as e:
            print(f"[LM-Eval Worker] Error: {e}")
            time.sleep(1)


def main():
    """Entry point."""
    print("=" * 60)
    print("[LM-Eval Worker] GPU-accelerated LM Evaluation Worker")
    print("=" * 60)
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[LM-Eval Worker] GPU available: {torch.cuda.get_device_name(0)}")
            print(f"[LM-Eval Worker] CUDA version: {torch.version.cuda}")
        else:
            print("[LM-Eval Worker] Warning: No GPU detected, running on CPU")
    except ImportError:
        print("[LM-Eval Worker] PyTorch not available for GPU check")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    poll_for_jobs()


if __name__ == "__main__":
    main()

"""
Benchmark routes for LLM speed evaluation.

Provides comprehensive tools for measuring and comparing LLM inference performance:
- Quick speed tests with real-time streaming metrics
- Multi-model comparison
- Context length scaling analysis
- Throughput/stress testing
- Persistent result storage with export
"""
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime
import asyncio
import json
import os
import uuid
import time
import httpx
import logging
import sqlite3
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/benchmark", tags=["benchmark"])

# Database path
DB_PATH = Path("/app/data/benchmarks.db") if os.path.exists("/app") else Path("./data/benchmarks.db")


# =============================================================================
# Database Setup
# =============================================================================

def init_db():
    """Initialize the benchmark database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT NOT NULL,
            endpoint TEXT,
            model_name TEXT,
            config TEXT,
            metrics TEXT,
            runs TEXT,
            status TEXT DEFAULT 'completed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_results_type ON benchmark_results(type)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_results_created ON benchmark_results(created_at DESC)
    """)
    
    conn.commit()
    conn.close()

init_db()


# =============================================================================
# Pydantic Models
# =============================================================================

class BenchmarkType(str, Enum):
    SPEED_TEST = "speed_test"
    COMPARISON = "comparison"
    CONTEXT_SCALING = "context_scaling"
    THROUGHPUT = "throughput"


class SpeedTestRequest(BaseModel):
    """Request for a quick speed test."""
    prompt: str = Field(..., description="The prompt to test with")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    endpoint: Optional[str] = Field(default=None, description="Custom endpoint URL (uses local llama.cpp if not set)")
    model_name: Optional[str] = Field(default=None, description="Model name for labeling")
    num_runs: int = Field(default=1, ge=1, le=10, description="Number of runs to average")
    save_result: bool = Field(default=True, description="Save result to database")


class ComparisonRequest(BaseModel):
    """Request for comparing multiple endpoints."""
    prompt: str = Field(..., description="The prompt to test with")
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    endpoints: List[Dict[str, Any]] = Field(
        ..., 
        description="List of endpoints to compare. Each has: url (optional), name, api_key (optional)"
    )
    num_runs: int = Field(default=3, ge=1, le=10, description="Runs per endpoint")


class ContextScalingRequest(BaseModel):
    """Request for context length scaling test."""
    base_prompt: str = Field(..., description="Base prompt to repeat/expand")
    context_sizes: List[int] = Field(
        default=[100, 500, 1000, 2000, 4000],
        description="List of context lengths to test (in tokens)"
    )
    max_tokens: int = Field(default=100, ge=1, le=1024)
    endpoint: Optional[str] = Field(default=None)
    model_name: Optional[str] = Field(default=None)


class ThroughputRequest(BaseModel):
    """Request for throughput/stress testing."""
    prompt: str = Field(..., description="The prompt to test with")
    max_tokens: int = Field(default=128, ge=1, le=1024)
    concurrent_requests: List[int] = Field(
        default=[1, 2, 4, 8],
        description="Number of concurrent requests to test"
    )
    requests_per_level: int = Field(default=5, ge=1, le=20)
    endpoint: Optional[str] = Field(default=None)
    model_name: Optional[str] = Field(default=None)


class RunMetrics(BaseModel):
    """Metrics from a single benchmark run."""
    run_number: int
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token_ms: float
    tokens_per_second: float
    total_time_ms: float
    timestamp: str


class AggregateMetrics(BaseModel):
    """Aggregated statistics from multiple runs."""
    tokens_per_second: Dict[str, float]  # min, max, mean, median, p90, p99
    time_to_first_token_ms: Dict[str, float]
    total_time_ms: Dict[str, float]
    total_tokens: int
    successful_runs: int
    failed_runs: int


class BenchmarkResult(BaseModel):
    """Complete benchmark result."""
    id: str
    name: Optional[str]
    type: BenchmarkType
    endpoint: Optional[str]
    model_name: Optional[str]
    config: Dict[str, Any]
    metrics: Optional[AggregateMetrics]
    runs: List[RunMetrics]
    status: str
    created_at: str
    completed_at: Optional[str]


# =============================================================================
# Helper Functions
# =============================================================================

def get_manager(request: Request):
    """Get the LlamaCPP manager from app state."""
    return getattr(request.app.state, 'manager', None)


def get_llm_endpoint(request: Request, custom_endpoint: Optional[str] = None) -> str:
    """Get the LLM endpoint URL."""
    if custom_endpoint:
        return custom_endpoint
    
    manager = get_manager(request)
    if manager:
        status = manager.get_status()
        if status.get("running"):
            port = manager.config.get('server', {}).get('port', 8081)
            return f"http://localhost:{port}"
    
    raise HTTPException(status_code=503, detail="No LLM endpoint available. Start the local server or provide a custom endpoint.")


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calculate a percentile value."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = (len(sorted_data) - 1) * percentile / 100
    lower = int(index)
    upper = lower + 1
    if upper >= len(sorted_data):
        return sorted_data[-1]
    weight = index - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


def calculate_aggregate_metrics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate metrics from multiple runs."""
    if not runs:
        return None
    
    tps_values = [r['tokens_per_second'] for r in runs if r.get('tokens_per_second')]
    ttft_values = [r['time_to_first_token_ms'] for r in runs if r.get('time_to_first_token_ms')]
    total_time_values = [r['total_time_ms'] for r in runs if r.get('total_time_ms')]
    
    def stats_dict(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "p90": 0, "p99": 0, "stdev": 0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p90": calculate_percentile(values, 90),
            "p99": calculate_percentile(values, 99),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    return {
        "tokens_per_second": stats_dict(tps_values),
        "time_to_first_token_ms": stats_dict(ttft_values),
        "total_time_ms": stats_dict(total_time_values),
        "total_tokens": sum(r.get('completion_tokens', 0) for r in runs),
        "successful_runs": len(runs),
        "failed_runs": 0
    }


def save_result(result: Dict[str, Any]):
    """Save a benchmark result to the database."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO benchmark_results (id, name, type, endpoint, model_name, config, metrics, runs, status, created_at, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result['id'],
        result.get('name'),
        result['type'],
        result.get('endpoint'),
        result.get('model_name'),
        json.dumps(result.get('config', {})),
        json.dumps(result.get('metrics')),
        json.dumps(result.get('runs', [])),
        result.get('status', 'completed'),
        result.get('created_at'),
        result.get('completed_at')
    ))
    
    conn.commit()
    conn.close()


async def run_single_inference(
    endpoint: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    api_key: Optional[str] = None,
    stream: bool = True
) -> Dict[str, Any]:
    """Run a single inference and measure metrics."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    request_data = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream
    }
    
    # Check if endpoint looks like OpenAI format or llama.cpp format
    chat_endpoint = endpoint
    if not endpoint.endswith("/chat/completions") and not endpoint.endswith("/v1/chat/completions"):
        if endpoint.endswith("/v1"):
            chat_endpoint = f"{endpoint}/chat/completions"
        elif endpoint.endswith("/"):
            chat_endpoint = f"{endpoint}v1/chat/completions"
        else:
            chat_endpoint = f"{endpoint}/v1/chat/completions"
    
    start_time = time.perf_counter()
    first_token_time = None
    completion_tokens = 0
    prompt_tokens = 0
    full_response = ""
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                async with client.stream("POST", chat_endpoint, json=request_data, headers=headers) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(f"HTTP {response.status_code}: {error_text.decode()}")
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                
                                # Count tokens from delta
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    completion_tokens += 1  # Approximate: 1 chunk = 1 token
                                    full_response += content
                                
                                # Get usage if provided
                                usage = data.get("usage", {})
                                if usage:
                                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                    if usage.get("completion_tokens"):
                                        completion_tokens = usage["completion_tokens"]
                            except json.JSONDecodeError:
                                continue
            else:
                response = await client.post(chat_endpoint, json=request_data, headers=headers)
                first_token_time = time.perf_counter()
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                
                data = response.json()
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                full_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    except Exception as e:
        raise Exception(f"Inference failed: {str(e)}")
    
    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_time_ms
    
    # Calculate tokens per second (generation phase only)
    generation_time_s = (end_time - first_token_time) if first_token_time else (end_time - start_time)
    tokens_per_second = completion_tokens / generation_time_s if generation_time_s > 0 else 0
    
    # Estimate prompt tokens if not provided (rough estimate: ~4 chars per token)
    if prompt_tokens == 0:
        prompt_tokens = len(prompt) // 4
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "time_to_first_token_ms": round(ttft_ms, 2),
        "tokens_per_second": round(tokens_per_second, 2),
        "total_time_ms": round(total_time_ms, 2),
        "response": full_response
    }


# =============================================================================
# Speed Test Endpoint (with SSE streaming)
# =============================================================================

@router.post("/speed-test")
async def speed_test(request: Request, body: SpeedTestRequest):
    """
    Run a quick speed test with real-time metrics.
    
    Returns streaming SSE events with:
    - status updates
    - real-time token counts
    - final metrics
    """
    try:
        endpoint = get_llm_endpoint(request, body.endpoint)
    except HTTPException as e:
        raise e
    
    result_id = str(uuid.uuid4())[:8]
    
    async def generate_events():
        runs = []
        
        for run_num in range(1, body.num_runs + 1):
            # Send run start event
            yield f"data: {json.dumps({'event': 'run_start', 'run': run_num, 'total': body.num_runs})}\n\n"
            
            try:
                metrics = await run_single_inference(
                    endpoint=endpoint,
                    prompt=body.prompt,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                    stream=True
                )
                
                run_data = {
                    "run_number": run_num,
                    "timestamp": datetime.now().isoformat(),
                    **metrics
                }
                runs.append(run_data)
                
                # Send run complete event
                yield f"data: {json.dumps({'event': 'run_complete', 'run': run_num, 'metrics': run_data})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'event': 'run_error', 'run': run_num, 'error': str(e)})}\n\n"
        
        # Calculate aggregate metrics
        aggregate = calculate_aggregate_metrics(runs)
        
        result = {
            "id": result_id,
            "name": f"Speed Test - {body.model_name or 'Local'}",
            "type": BenchmarkType.SPEED_TEST,
            "endpoint": endpoint if body.endpoint else "local",
            "model_name": body.model_name,
            "config": {
                "prompt_length": len(body.prompt),
                "max_tokens": body.max_tokens,
                "temperature": body.temperature,
                "num_runs": body.num_runs
            },
            "metrics": aggregate,
            "runs": runs,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat()
        }
        
        # Save to database if requested
        if body.save_result and runs:
            try:
                save_result(result)
            except Exception as e:
                logger.error(f"Failed to save result: {e}")
        
        # Send final result
        yield f"data: {json.dumps({'event': 'complete', 'result': result})}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/speed-test/sync")
async def speed_test_sync(request: Request, body: SpeedTestRequest):
    """
    Run a speed test and return results synchronously.
    Useful when SSE is not needed.
    """
    try:
        endpoint = get_llm_endpoint(request, body.endpoint)
    except HTTPException as e:
        raise e
    
    result_id = str(uuid.uuid4())[:8]
    runs = []
    errors = []
    
    for run_num in range(1, body.num_runs + 1):
        try:
            metrics = await run_single_inference(
                endpoint=endpoint,
                prompt=body.prompt,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                stream=True
            )
            
            run_data = {
                "run_number": run_num,
                "timestamp": datetime.now().isoformat(),
                **metrics
            }
            runs.append(run_data)
            
        except Exception as e:
            errors.append({"run": run_num, "error": str(e)})
    
    aggregate = calculate_aggregate_metrics(runs)
    
    result = {
        "id": result_id,
        "name": f"Speed Test - {body.model_name or 'Local'}",
        "type": BenchmarkType.SPEED_TEST.value,
        "endpoint": endpoint if body.endpoint else "local",
        "model_name": body.model_name,
        "config": {
            "prompt_length": len(body.prompt),
            "max_tokens": body.max_tokens,
            "temperature": body.temperature,
            "num_runs": body.num_runs
        },
        "metrics": aggregate,
        "runs": runs,
        "errors": errors if errors else None,
        "status": "completed" if runs else "failed",
        "created_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat()
    }
    
    if body.save_result and runs:
        try:
            save_result(result)
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
    
    return result


# =============================================================================
# Comparison Endpoint
# =============================================================================

@router.post("/compare")
async def compare_endpoints(request: Request, body: ComparisonRequest):
    """
    Compare multiple endpoints/models on the same prompt.
    
    Returns comparison results for all endpoints.
    """
    result_id = str(uuid.uuid4())[:8]
    comparison_results = []
    
    for ep_config in body.endpoints:
        ep_name = ep_config.get("name", "Unknown")
        ep_url = ep_config.get("url")
        ep_api_key = ep_config.get("api_key")
        
        # If no URL provided, use local endpoint
        if not ep_url:
            try:
                ep_url = get_llm_endpoint(request, None)
            except HTTPException:
                comparison_results.append({
                    "name": ep_name,
                    "endpoint": "local",
                    "status": "error",
                    "error": "Local LLM not available"
                })
                continue
        
        runs = []
        errors = []
        
        for run_num in range(1, body.num_runs + 1):
            try:
                metrics = await run_single_inference(
                    endpoint=ep_url,
                    prompt=body.prompt,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                    api_key=ep_api_key,
                    stream=True
                )
                
                runs.append({
                    "run_number": run_num,
                    "timestamp": datetime.now().isoformat(),
                    **metrics
                })
                
            except Exception as e:
                errors.append({"run": run_num, "error": str(e)})
        
        aggregate = calculate_aggregate_metrics(runs)
        
        comparison_results.append({
            "name": ep_name,
            "endpoint": ep_url,
            "metrics": aggregate,
            "runs": runs,
            "errors": errors if errors else None,
            "status": "completed" if runs else "failed"
        })
    
    result = {
        "id": result_id,
        "name": f"Comparison - {len(body.endpoints)} endpoints",
        "type": BenchmarkType.COMPARISON.value,
        "config": {
            "prompt_length": len(body.prompt),
            "max_tokens": body.max_tokens,
            "temperature": body.temperature,
            "num_runs": body.num_runs,
            "num_endpoints": len(body.endpoints)
        },
        "results": comparison_results,
        "status": "completed",
        "created_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat()
    }
    
    try:
        save_result({
            **result,
            "metrics": None,
            "runs": comparison_results,
            "endpoint": None,
            "model_name": None
        })
    except Exception as e:
        logger.error(f"Failed to save comparison result: {e}")
    
    return result


# =============================================================================
# Context Scaling Endpoint
# =============================================================================

@router.post("/context-scaling")
async def context_scaling_test(request: Request, body: ContextScalingRequest):
    """
    Test how inference speed changes with different context lengths.
    
    Repeats/expands the base prompt to reach target token counts.
    """
    try:
        endpoint = get_llm_endpoint(request, body.endpoint)
    except HTTPException as e:
        raise e
    
    result_id = str(uuid.uuid4())[:8]
    scaling_results = []
    
    # Approximate tokens per character (rough estimate)
    chars_per_token = 4
    
    for context_size in sorted(body.context_sizes):
        target_chars = context_size * chars_per_token
        
        # Build prompt to target size
        base_len = len(body.base_prompt)
        if base_len >= target_chars:
            test_prompt = body.base_prompt[:target_chars]
        else:
            # Repeat the prompt to reach target size
            repetitions = (target_chars // base_len) + 1
            test_prompt = (body.base_prompt + " ") * repetitions
            test_prompt = test_prompt[:target_chars]
        
        try:
            metrics = await run_single_inference(
                endpoint=endpoint,
                prompt=test_prompt,
                max_tokens=body.max_tokens,
                temperature=0.7,
                stream=True
            )
            
            scaling_results.append({
                "target_context_tokens": context_size,
                "actual_prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens": metrics["completion_tokens"],
                "time_to_first_token_ms": metrics["time_to_first_token_ms"],
                "tokens_per_second": metrics["tokens_per_second"],
                "total_time_ms": metrics["total_time_ms"],
                "status": "success"
            })
            
        except Exception as e:
            scaling_results.append({
                "target_context_tokens": context_size,
                "status": "error",
                "error": str(e)
            })
    
    result = {
        "id": result_id,
        "name": f"Context Scaling - {body.model_name or 'Local'}",
        "type": BenchmarkType.CONTEXT_SCALING.value,
        "endpoint": endpoint if body.endpoint else "local",
        "model_name": body.model_name,
        "config": {
            "base_prompt_length": len(body.base_prompt),
            "context_sizes": body.context_sizes,
            "max_tokens": body.max_tokens
        },
        "results": scaling_results,
        "status": "completed",
        "created_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat()
    }
    
    try:
        save_result({
            **result,
            "metrics": None,
            "runs": scaling_results
        })
    except Exception as e:
        logger.error(f"Failed to save context scaling result: {e}")
    
    return result


# =============================================================================
# Throughput Testing Endpoint
# =============================================================================

@router.post("/throughput")
async def throughput_test(request: Request, body: ThroughputRequest):
    """
    Test throughput with varying levels of concurrent requests.
    
    Measures how the system handles load.
    """
    try:
        endpoint = get_llm_endpoint(request, body.endpoint)
    except HTTPException as e:
        raise e
    
    result_id = str(uuid.uuid4())[:8]
    throughput_results = []
    
    for concurrency in body.concurrent_requests:
        level_start = time.perf_counter()
        
        async def single_request():
            try:
                return await run_single_inference(
                    endpoint=endpoint,
                    prompt=body.prompt,
                    max_tokens=body.max_tokens,
                    temperature=0.7,
                    stream=False  # Non-streaming for throughput testing
                )
            except Exception as e:
                return {"error": str(e)}
        
        # Run concurrent requests
        tasks = []
        for _ in range(body.requests_per_level):
            # Create batch of concurrent requests
            batch = [single_request() for _ in range(concurrency)]
            results = await asyncio.gather(*batch, return_exceptions=True)
            tasks.extend(results)
        
        level_end = time.perf_counter()
        level_time = level_end - level_start
        
        # Analyze results
        successful = [r for r in tasks if isinstance(r, dict) and 'error' not in r]
        failed = len(tasks) - len(successful)
        
        total_tokens = sum(r.get('completion_tokens', 0) for r in successful)
        avg_tps = statistics.mean([r['tokens_per_second'] for r in successful]) if successful else 0
        avg_ttft = statistics.mean([r['time_to_first_token_ms'] for r in successful]) if successful else 0
        
        # Calculate aggregate throughput (total tokens / wall clock time)
        aggregate_throughput = total_tokens / level_time if level_time > 0 else 0
        
        throughput_results.append({
            "concurrency": concurrency,
            "total_requests": len(tasks),
            "successful_requests": len(successful),
            "failed_requests": failed,
            "total_time_seconds": round(level_time, 2),
            "total_tokens_generated": total_tokens,
            "aggregate_throughput_tps": round(aggregate_throughput, 2),
            "avg_tokens_per_second": round(avg_tps, 2),
            "avg_time_to_first_token_ms": round(avg_ttft, 2),
            "requests_per_second": round(len(tasks) / level_time, 2) if level_time > 0 else 0
        })
    
    result = {
        "id": result_id,
        "name": f"Throughput Test - {body.model_name or 'Local'}",
        "type": BenchmarkType.THROUGHPUT.value,
        "endpoint": endpoint if body.endpoint else "local",
        "model_name": body.model_name,
        "config": {
            "prompt_length": len(body.prompt),
            "max_tokens": body.max_tokens,
            "concurrency_levels": body.concurrent_requests,
            "requests_per_level": body.requests_per_level
        },
        "results": throughput_results,
        "status": "completed",
        "created_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat()
    }
    
    try:
        save_result({
            **result,
            "metrics": None,
            "runs": throughput_results
        })
    except Exception as e:
        logger.error(f"Failed to save throughput result: {e}")
    
    return result


# =============================================================================
# Results Management
# =============================================================================

@router.get("/results")
async def list_results(
    type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List benchmark results with optional filtering."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM benchmark_results"
    params = []
    
    if type:
        query += " WHERE type = ?"
        params.append(type)
    
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Get total count
    count_query = "SELECT COUNT(*) FROM benchmark_results"
    if type:
        count_query += " WHERE type = ?"
        cursor.execute(count_query, [type])
    else:
        cursor.execute(count_query)
    total = cursor.fetchone()[0]
    
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "name": row["name"],
            "type": row["type"],
            "endpoint": row["endpoint"],
            "model_name": row["model_name"],
            "config": json.loads(row["config"]) if row["config"] else {},
            "metrics": json.loads(row["metrics"]) if row["metrics"] else None,
            "status": row["status"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"]
        })
    
    return {
        "results": results,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/results/{result_id}")
async def get_result(result_id: str):
    """Get a specific benchmark result with full details."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM benchmark_results WHERE id = ?", [result_id])
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return {
        "id": row["id"],
        "name": row["name"],
        "type": row["type"],
        "endpoint": row["endpoint"],
        "model_name": row["model_name"],
        "config": json.loads(row["config"]) if row["config"] else {},
        "metrics": json.loads(row["metrics"]) if row["metrics"] else None,
        "runs": json.loads(row["runs"]) if row["runs"] else [],
        "status": row["status"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"]
    }


@router.delete("/results/{result_id}")
async def delete_result(result_id: str):
    """Delete a benchmark result."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM benchmark_results WHERE id = ?", [result_id])
    deleted = cursor.rowcount > 0
    
    conn.commit()
    conn.close()
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return {"status": "deleted", "id": result_id}


@router.delete("/results")
async def clear_results(type: Optional[str] = None):
    """Clear benchmark results, optionally filtered by type."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    if type:
        cursor.execute("DELETE FROM benchmark_results WHERE type = ?", [type])
    else:
        cursor.execute("DELETE FROM benchmark_results")
    
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    
    return {"status": "cleared", "deleted_count": deleted}


@router.get("/export/{result_id}")
async def export_result(result_id: str, format: str = "json"):
    """Export a benchmark result as JSON or CSV."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM benchmark_results WHERE id = ?", [result_id])
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = {
        "id": row["id"],
        "name": row["name"],
        "type": row["type"],
        "endpoint": row["endpoint"],
        "model_name": row["model_name"],
        "config": json.loads(row["config"]) if row["config"] else {},
        "metrics": json.loads(row["metrics"]) if row["metrics"] else None,
        "runs": json.loads(row["runs"]) if row["runs"] else [],
        "status": row["status"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"]
    }
    
    if format.lower() == "csv":
        # Convert to CSV format
        import csv
        import io
        
        output = io.StringIO()
        
        # For speed tests, export run data
        if result["type"] == "speed_test" and result.get("runs"):
            runs = result["runs"]
            if runs:
                writer = csv.DictWriter(output, fieldnames=runs[0].keys())
                writer.writeheader()
                writer.writerows(runs)
        # For comparisons, export comparison data
        elif result["type"] == "comparison":
            runs = result.get("runs", [])
            if runs:
                flat_data = []
                for ep in runs:
                    for run in ep.get("runs", []):
                        flat_data.append({
                            "endpoint_name": ep["name"],
                            **run
                        })
                if flat_data:
                    writer = csv.DictWriter(output, fieldnames=flat_data[0].keys())
                    writer.writeheader()
                    writer.writerows(flat_data)
        else:
            # Generic export
            writer = csv.writer(output)
            writer.writerow(["key", "value"])
            for k, v in result.items():
                writer.writerow([k, json.dumps(v) if isinstance(v, (dict, list)) else v])
        
        csv_content = output.getvalue()
        
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=benchmark_{result_id}.csv"}
        )
    
    # Default to JSON
    return StreamingResponse(
        iter([json.dumps(result, indent=2)]),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=benchmark_{result_id}.json"}
    )


# =============================================================================
# Stats Endpoint
# =============================================================================

@router.get("/stats")
async def get_stats():
    """Get aggregate benchmark statistics."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Count by type
    cursor.execute("""
        SELECT type, COUNT(*) as count 
        FROM benchmark_results 
        GROUP BY type
    """)
    type_counts = {row["type"]: row["count"] for row in cursor.fetchall()}
    
    # Get recent speed test metrics
    cursor.execute("""
        SELECT metrics FROM benchmark_results 
        WHERE type = 'speed_test' AND metrics IS NOT NULL 
        ORDER BY created_at DESC LIMIT 10
    """)
    
    recent_tps = []
    for row in cursor.fetchall():
        metrics = json.loads(row["metrics"])
        if metrics and metrics.get("tokens_per_second"):
            recent_tps.append(metrics["tokens_per_second"].get("mean", 0))
    
    avg_tps = statistics.mean(recent_tps) if recent_tps else 0
    
    # Total count
    cursor.execute("SELECT COUNT(*) FROM benchmark_results")
    total = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_benchmarks": total,
        "by_type": type_counts,
        "average_tps_recent": round(avg_tps, 2),
        "recent_benchmarks_count": len(recent_tps)
    }

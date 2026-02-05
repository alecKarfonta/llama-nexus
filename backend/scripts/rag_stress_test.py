#!/usr/bin/env python3
"""
RAG Retrieval Stress Test with Charts

Tests the RAG retrieval endpoints for performance and reliability under load.
Outputs ASCII charts and detailed tables for quick performance overview.
"""

import asyncio
import time
import argparse
import json
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://localhost:8700"


@dataclass
class TestResult:
    endpoint: str
    success: bool
    duration_ms: float
    status_code: int
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


def make_request(url: str, method: str = "GET", data: dict = None, timeout: int = 30) -> tuple:
    """Make HTTP request using urllib (no dependencies)."""
    start = time.time()
    headers = {"Content-Type": "application/json"}
    
    try:
        body = json.dumps(data).encode() if data else None
        req = Request(url, data=body, headers=headers, method=method)
        with urlopen(req, timeout=timeout) as resp:
            duration = (time.time() - start) * 1000
            return resp.status, resp.read().decode(), duration, None
    except HTTPError as e:
        duration = (time.time() - start) * 1000
        return e.code, e.read().decode() if e.fp else str(e), duration, str(e)
    except URLError as e:
        duration = (time.time() - start) * 1000
        return 0, "", duration, str(e.reason)
    except Exception as e:
        duration = (time.time() - start) * 1000
        return 0, "", duration, str(e)


def test_memory_search(query: str, embedding_model: str = "nomic-embed-text-v1.5") -> TestResult:
    """Test the memory search endpoint."""
    url = f"{BASE_URL}/api/v1/rag/memory/search"
    status, body, duration, error = make_request(url, "POST", {
        "query": query,
        "collections": ["npc_memories", "hytale_player_memories"],
        "top_k": 5,
        "embedding_model": embedding_model
    })
    return TestResult(
        endpoint="memory_search",
        success=status == 200,
        duration_ms=duration,
        status_code=status,
        error=error or (body[:100] if status != 200 else None)
    )


def test_document_retrieve(query: str, domain_id: str = None) -> TestResult:
    """Test the document retrieval endpoint."""
    url = f"{BASE_URL}/api/v1/rag/retrieve"
    status, body, duration, error = make_request(url, "POST", {
        "query": query,
        "top_k": 5
    })
    return TestResult(
        endpoint="document_retrieve",
        success=status == 200,
        duration_ms=duration,
        status_code=status,
        error=error or (body[:100] if status != 200 else None)
    )


def test_health() -> TestResult:
    """Test the health endpoint."""
    url = f"{BASE_URL}/api/v1/service/status"
    status, _, duration, error = make_request(url, "GET")
    return TestResult(
        endpoint="health",
        success=status == 200,
        duration_ms=duration,
        status_code=status,
        error=error
    )


def run_sequential_test(
    num_requests: int,
    delay_between: float,
    queries: List[str],
    test_type: str = "memory_search"
) -> List[TestResult]:
    """Run sequential stress test."""
    results = []
    
    for i in range(num_requests):
        query = queries[i % len(queries)]
        
        if test_type == "memory_search":
            result = test_memory_search(query)
        elif test_type == "health":
            result = test_health()
        elif test_type == "retrieve":
            result = test_document_retrieve(query)
        else:
            result = test_memory_search(query)
        
        results.append(result)
        
        if (i + 1) % 10 == 0:
            success_rate = sum(1 for r in results if r.success) / len(results) * 100
            avg_duration = sum(r.duration_ms for r in results) / len(results)
            print(f"  [{i+1}/{num_requests}] Success: {success_rate:.1f}%, Avg: {avg_duration:.0f}ms")
        
        if delay_between > 0:
            time.sleep(delay_between)
    
    return results


def run_parallel_test(
    num_requests: int,
    concurrency: int,
    queries: List[str],
    test_type: str = "memory_search"
) -> List[TestResult]:
    """Run parallel stress test with limited concurrency."""
    
    def worker(query: str) -> TestResult:
        if test_type == "memory_search":
            return test_memory_search(query)
        elif test_type == "health":
            return test_health()
        elif test_type == "retrieve":
            return test_document_retrieve(query)
        return test_memory_search(query)
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        tasks = [queries[i % len(queries)] for i in range(num_requests)]
        results = list(executor.map(worker, tasks))
    
    return results


def ascii_histogram(values: List[float], bins: int = 10, width: int = 40) -> str:
    """Create ASCII histogram."""
    if not values:
        return "  No data"
    
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return f"  All values: {min_val:.0f}ms"
    
    bin_width = (max_val - min_val) / bins
    counts = [0] * bins
    
    for v in values:
        idx = min(int((v - min_val) / bin_width), bins - 1)
        counts[idx] += 1
    
    max_count = max(counts)
    lines = []
    
    for i, count in enumerate(counts):
        low = min_val + i * bin_width
        high = low + bin_width
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len
        lines.append(f"  {low:7.0f}-{high:6.0f}ms │{bar} ({count})")
    
    return "\n".join(lines)


def ascii_timeline(results: List[TestResult], width: int = 60) -> str:
    """Create ASCII timeline showing success/failure over time."""
    if not results:
        return "  No data"
    
    # Group results into buckets
    bucket_size = max(1, len(results) // width)
    buckets = []
    
    for i in range(0, len(results), bucket_size):
        bucket = results[i:i+bucket_size]
        success_rate = sum(1 for r in bucket if r.success) / len(bucket)
        buckets.append(success_rate)
    
    # Create timeline
    chars = " ▁▂▃▄▅▆▇█"
    timeline = ""
    for rate in buckets:
        idx = int(rate * (len(chars) - 1))
        timeline += chars[idx]
    
    return f"  Success: {''.join(timeline)}\n  Legend:  ▁=0% ████ =100%"


def print_table(headers: List[str], rows: List[List], col_widths: List[int] = None):
    """Print formatted table."""
    if not col_widths:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
    
    # Header
    header_line = "│".join(f" {h:^{w-2}} " for h, w in zip(headers, col_widths))
    separator = "┼".join("─" * w for w in col_widths)
    
    print(f"┌{'┬'.join('─' * w for w in col_widths)}┐")
    print(f"│{header_line}│")
    print(f"├{separator}┤")
    
    # Rows
    for row in rows:
        row_line = "│".join(f" {str(v):<{w-2}} " for v, w in zip(row, col_widths))
        print(f"│{row_line}│")
    
    print(f"└{'┴'.join('─' * w for w in col_widths)}┘")


def print_summary(results: List[TestResult], test_name: str):
    """Print comprehensive test summary with charts."""
    if not results:
        print(f"\n{test_name}: No results")
        return
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"\n{'═'*70}")
    print(f"  {test_name}")
    print(f"{'═'*70}")
    
    # Summary table
    success_rate = len(successful) / len(results) * 100
    print_table(
        ["Metric", "Value"],
        [
            ["Total Requests", len(results)],
            ["Successful", f"{len(successful)} ({success_rate:.1f}%)"],
            ["Failed", f"{len(failed)} ({100-success_rate:.1f}%)"],
        ],
        [20, 25]
    )
    
    if successful:
        durations = [r.duration_ms for r in successful]
        
        # Latency statistics table
        print(f"\n  📊 Latency Statistics (successful requests)")
        print_table(
            ["Percentile", "Latency"],
            [
                ["Min", f"{min(durations):.0f}ms"],
                ["P25", f"{sorted(durations)[int(len(durations)*0.25)]:.0f}ms"],
                ["P50 (Median)", f"{sorted(durations)[len(durations)//2]:.0f}ms"],
                ["P75", f"{sorted(durations)[int(len(durations)*0.75)]:.0f}ms"],
                ["P95", f"{sorted(durations)[int(len(durations)*0.95)]:.0f}ms"],
                ["P99", f"{sorted(durations)[min(int(len(durations)*0.99), len(durations)-1)]:.0f}ms"],
                ["Max", f"{max(durations):.0f}ms"],
                ["Average", f"{statistics.mean(durations):.0f}ms"],
                ["Std Dev", f"{statistics.stdev(durations) if len(durations) > 1 else 0:.0f}ms"],
            ],
            [15, 15]
        )
        
        # Histogram
        print(f"\n  📈 Latency Distribution")
        print(ascii_histogram(durations))
    
    # Timeline
    print(f"\n  ⏱️  Success Timeline (left=start, right=end)")
    print(ascii_timeline(results))
    
    if failed:
        print(f"\n  ❌ Failure Details")
        error_counts: Dict[str, int] = {}
        for r in failed:
            key = f"HTTP {r.status_code}: {(r.error or 'Unknown')[:50]}"
            error_counts[key] = error_counts.get(key, 0) + 1
        
        error_rows = [[error[:45], count] for error, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]]
        print_table(["Error", "Count"], error_rows, [50, 10])


def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Stress Test with Charts")
    parser.add_argument("--requests", type=int, default=50, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=5, help="Parallel concurrency")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between sequential requests (seconds)")
    parser.add_argument("--mode", choices=["sequential", "parallel", "both", "scaling", "sustained"], default="both")
    parser.add_argument("--test", choices=["memory_search", "health", "retrieve"], default="memory_search")
    parser.add_argument("--url", type=str, default=None, help="Override base URL")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds for sustained mode (default: 300 = 5 min)")
    parser.add_argument("--rate", type=float, default=1.0, help="Requests per second for sustained mode")
    args = parser.parse_args()
    
    global BASE_URL
    if args.url:
        BASE_URL = args.url
    
    queries = [
        "What is the weather like today?",
        "Tell me about machine learning",
        "How do I make coffee?",
        "What are the best programming languages?",
        "Explain quantum computing",
        "What is the capital of France?",
        "How do neural networks work?",
        "What is the meaning of life?",
    ]
    
    print(f"╔{'═'*68}╗")
    print(f"║{'RAG Retrieval Stress Test':^68}║")
    print(f"╠{'═'*68}╣")
    print(f"║  Base URL:    {BASE_URL:<52}║")
    print(f"║  Test Type:   {args.test:<52}║")
    print(f"║  Requests:    {args.requests:<52}║")
    print(f"║  Mode:        {args.mode:<52}║")
    print(f"╚{'═'*68}╝")
    
    if args.mode == "scaling":
        # Run load scaling test with multiple concurrency levels
        run_load_scaling_test(args.requests, queries, args.test)
    elif args.mode == "sustained":
        # Run sustained test for specified duration
        run_sustained_test(args.duration, args.rate, queries, args.test)
    else:
        if args.mode in ["sequential", "both"]:
            print(f"\n🔄 Running Sequential Test (delay: {args.delay}s between requests)...")
            seq_results = run_sequential_test(
                num_requests=args.requests,
                delay_between=args.delay,
                queries=queries,
                test_type=args.test
            )
            print_summary(seq_results, "Sequential Test Results")
        
        if args.mode in ["parallel", "both"]:
            print(f"\n⚡ Running Parallel Test (concurrency: {args.concurrency})...")
            par_results = run_parallel_test(
                num_requests=args.requests,
                concurrency=args.concurrency,
                queries=queries,
                test_type=args.test
            )
            print_summary(par_results, "Parallel Test Results")
    
    print(f"\n{'─'*70}")
    print("✅ Stress test complete!")


def run_sustained_test(duration_seconds: int, rate: float, queries: List[str], test_type: str):
    """Run sustained test for specified duration, showing degradation over time."""
    interval = 1.0 / rate
    buckets: List[List[TestResult]] = []  # Each bucket = 30 seconds
    bucket_duration = 30
    current_bucket: List[TestResult] = []
    
    print(f"\n⏱️  Sustained Test - {duration_seconds}s at {rate} req/s")
    print(f"   Monitoring for performance degradation over time...")
    print()
    
    start_time = time.time()
    last_bucket_time = start_time
    request_count = 0
    
    while time.time() - start_time < duration_seconds:
        query = queries[request_count % len(queries)]
        
        if test_type == "memory_search":
            result = test_memory_search(query)
        elif test_type == "health":
            result = test_health()
        elif test_type == "retrieve":
            result = test_document_retrieve(query)
        else:
            result = test_memory_search(query)
        
        current_bucket.append(result)
        request_count += 1
        
        # Check if bucket is complete
        if time.time() - last_bucket_time >= bucket_duration:
            buckets.append(current_bucket)
            
            # Print bucket summary
            successful = [r for r in current_bucket if r.success]
            success_rate = len(successful) / len(current_bucket) * 100 if current_bucket else 0
            durations = [r.duration_ms for r in successful] if successful else [0]
            p50 = sorted(durations)[len(durations)//2]
            elapsed = int(time.time() - start_time)
            
            status = "✅" if success_rate >= 95 else "⚠️" if success_rate >= 80 else "❌"
            print(f"   [{elapsed:>3}s] {status} {len(current_bucket)} reqs, Success: {success_rate:.0f}%, P50: {p50:.0f}ms")
            
            current_bucket = []
            last_bucket_time = time.time()
        
        # Rate limiting
        time.sleep(max(0, interval - (time.time() - start_time - request_count * interval)))
    
    # Add final bucket
    if current_bucket:
        buckets.append(current_bucket)
    
    # Print final summary
    all_results = [r for bucket in buckets for r in bucket]
    print_summary(all_results, f"Sustained Test Results ({duration_seconds}s)")
    
    # Print degradation analysis
    if len(buckets) >= 2:
        print(f"\n  📉 Performance Over Time (per 30s bucket)")
        print_table(
            ["Time", "Requests", "Success%", "P50 (ms)", "P95 (ms)"],
            [
                [
                    f"{i*bucket_duration}-{(i+1)*bucket_duration}s",
                    len(bucket),
                    f"{sum(1 for r in bucket if r.success)/len(bucket)*100:.0f}%" if bucket else "0%",
                    f"{sorted([r.duration_ms for r in bucket if r.success])[len([r for r in bucket if r.success])//2]:.0f}" if [r for r in bucket if r.success] else "N/A",
                    f"{sorted([r.duration_ms for r in bucket if r.success])[int(len([r for r in bucket if r.success])*0.95)]:.0f}" if len([r for r in bucket if r.success]) > 1 else "N/A",
                ]
                for i, bucket in enumerate(buckets)
            ],
            [12, 10, 10, 10, 10]
        )
        
        # Check for degradation
        first_bucket_p50 = sorted([r.duration_ms for r in buckets[0] if r.success])[len([r for r in buckets[0] if r.success])//2] if [r for r in buckets[0] if r.success] else 0
        last_bucket_p50 = sorted([r.duration_ms for r in buckets[-1] if r.success])[len([r for r in buckets[-1] if r.success])//2] if [r for r in buckets[-1] if r.success] else 0
        
        if last_bucket_p50 > first_bucket_p50 * 2:
            print(f"\n  ⚠️  DEGRADATION DETECTED: P50 increased from {first_bucket_p50:.0f}ms to {last_bucket_p50:.0f}ms")
        else:
            print(f"\n  ✅ No significant degradation detected (P50: {first_bucket_p50:.0f}ms → {last_bucket_p50:.0f}ms)")


def run_load_scaling_test(requests_per_level: int, queries: List[str], test_type: str):
    """Run tests at multiple concurrency levels and show comparative chart."""
    concurrency_levels = [1, 2, 5, 10, 20]
    results_by_level: Dict[int, Dict] = {}
    
    print(f"\n📊 Load Scaling Test - Testing {len(concurrency_levels)} concurrency levels")
    print(f"   Requests per level: {requests_per_level}")
    print()
    
    for concurrency in concurrency_levels:
        print(f"   Testing concurrency={concurrency}...", end=" ", flush=True)
        results = run_parallel_test(
            num_requests=requests_per_level,
            concurrency=concurrency,
            queries=queries,
            test_type=test_type
        )
        
        successful = [r for r in results if r.success]
        success_rate = len(successful) / len(results) * 100 if results else 0
        
        if successful:
            durations = [r.duration_ms for r in successful]
            results_by_level[concurrency] = {
                "success_rate": success_rate,
                "p50": sorted(durations)[len(durations)//2],
                "p95": sorted(durations)[int(len(durations)*0.95)],
                "avg": statistics.mean(durations),
                "min": min(durations),
                "max": max(durations),
            }
        else:
            results_by_level[concurrency] = {
                "success_rate": 0, "p50": 0, "p95": 0, "avg": 0, "min": 0, "max": 0
            }
        
        r = results_by_level[concurrency]
        print(f"P50={r['p50']:.0f}ms, P95={r['p95']:.0f}ms, Success={r['success_rate']:.0f}%")
    
    # Print comparative table
    print(f"\n{'═'*70}")
    print(f"  Load Scaling Results")
    print(f"{'═'*70}")
    
    print_table(
        ["Concurrency", "Success%", "P50 (ms)", "P95 (ms)", "Avg (ms)", "Max (ms)"],
        [
            [c, f"{r['success_rate']:.0f}%", f"{r['p50']:.0f}", f"{r['p95']:.0f}", f"{r['avg']:.0f}", f"{r['max']:.0f}"]
            for c, r in results_by_level.items()
        ],
        [12, 10, 10, 10, 10, 10]
    )
    
    # ASCII chart of P50 latency vs concurrency
    print(f"\n  📈 P50 Latency vs Concurrency")
    max_p50 = max(r["p50"] for r in results_by_level.values()) or 1
    chart_width = 40
    
    for c, r in results_by_level.items():
        bar_len = int((r["p50"] / max_p50) * chart_width)
        bar = "█" * bar_len
        print(f"  {c:>3} concurrent │{bar} {r['p50']:.0f}ms")
    
    # ASCII chart of success rate
    print(f"\n  ✅ Success Rate vs Concurrency")
    for c, r in results_by_level.items():
        bar_len = int((r["success_rate"] / 100) * chart_width)
        bar = "█" * bar_len
        color = "" if r["success_rate"] >= 95 else "⚠️ " if r["success_rate"] >= 80 else "❌ "
        print(f"  {c:>3} concurrent │{bar} {color}{r['success_rate']:.0f}%")


if __name__ == "__main__":
    main()


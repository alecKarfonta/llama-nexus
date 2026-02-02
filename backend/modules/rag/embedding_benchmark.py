"""
Embedding Retrieval Benchmark Suite

Comprehensive benchmarking for RAG embedding retrieval performance.
Tests various scenarios including large payloads, collection sizes,
concurrent requests, and retrieval method comparisons.

Usage:
    python3 embedding_benchmark.py [--base-url URL] [--output FILE]
    
This script uses only standard library modules for maximum portability.
"""

import asyncio
import json
import time
import statistics
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    base_url: str = "http://localhost:8700"
    timeout_seconds: int = 60
    warmup_runs: int = 2
    test_queries: List[str] = field(default_factory=lambda: [
        "What is the main topic?",
        "Explain the key concepts",
        "How does this work?",
        "What are the important details?",
        "Summarize the content"
    ])
    output_file: Optional[str] = None


@dataclass  
class RunMetrics:
    """Metrics from a single benchmark run."""
    scenario: str
    query: str
    latency_ms: float
    result_count: int
    payload_size_bytes: int
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Aggregated results for a benchmark scenario."""
    scenario: str
    description: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    latency_ms: Dict[str, float] = field(default_factory=dict)
    avg_result_count: float = 0.0
    avg_payload_bytes: float = 0.0
    throughput_qps: float = 0.0
    runs: List[RunMetrics] = field(default_factory=list)


class EmbeddingBenchmark:
    """
    Comprehensive benchmark suite for embedding retrieval.
    
    Designed for integration into backend/UI while also supporting
    standalone execution for testing.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[ScenarioResult] = []
        self.domains: List[Dict[str, Any]] = []
        self._executor = ThreadPoolExecutor(max_workers=16)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._executor.shutdown(wait=True)
    
    # =========================================================================
    # HTTP Helpers (using standard library)
    # =========================================================================
    
    def _http_get(self, url: str) -> Dict[str, Any]:
        """Synchronous HTTP GET request."""
        req = urllib.request.Request(url)
        req.add_header('Content-Type', 'application/json')
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except urllib.error.URLError as e:
            return {"error": f"URL Error: {e.reason}"}
    
    def _http_post(self, url: str, data: Dict[str, Any]) -> tuple:
        """Synchronous HTTP POST request. Returns (response_dict, latency_ms, payload_size)."""
        payload = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=payload, method='POST')
        req.add_header('Content-Type', 'application/json')
        
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                elapsed_ms = (time.perf_counter() - start) * 1000
                response_text = resp.read().decode('utf-8')
                payload_size = len(response_text.encode('utf-8'))
                response_data = json.loads(response_text)
                return response_data, elapsed_ms, payload_size, None
        except urllib.error.HTTPError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            error_body = e.read().decode('utf-8') if e.fp else ""
            return {}, elapsed_ms, 0, f"HTTP {e.code}: {error_body[:200]}"
        except urllib.error.URLError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {}, elapsed_ms, 0, f"URL Error: {e.reason}"
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {}, elapsed_ms, 0, str(e)
    
    # =========================================================================
    # API Helpers
    # =========================================================================
    
    def _fetch_domains(self) -> List[Dict[str, Any]]:
        """Fetch available domains from the RAG API."""
        url = f"{self.config.base_url}/api/v1/rag/domains"
        data = self._http_get(url)
        return data.get("domains", [])
    
    def _retrieve(
        self,
        query: str,
        domain_id: Optional[str] = None,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        context_neighbors_before: int = 0,
        context_neighbors_after: int = 0,
        hybrid: bool = False
    ) -> Dict[str, Any]:
        """Execute a retrieval request and return response with timing."""
        endpoint = "/api/v1/rag/retrieve/hybrid" if hybrid else "/api/v1/rag/retrieve"
        url = f"{self.config.base_url}{endpoint}"
        
        payload = {
            "query": query,
            "top_k": top_k,
            "context_neighbors_before": context_neighbors_before,
            "context_neighbors_after": context_neighbors_after,
        }
        if domain_id:
            payload["domain_id"] = domain_id
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        if hybrid:
            payload["alpha"] = 0.5
            
        data, latency_ms, payload_size, error = self._http_post(url, payload)
        
        return {
            "latency_ms": latency_ms,
            "payload_size": payload_size,
            "results": data.get("results", []) if data else [],
            "error": error,
            "data": data
        }
    
    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical percentiles."""
        if not values:
            return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return sorted_vals[min(idx, n - 1)]
        
        return {
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "avg": round(statistics.mean(values), 2),
            "p50": round(percentile(50), 2),
            "p95": round(percentile(95), 2),
            "p99": round(percentile(99), 2),
        }
    
    def _aggregate_runs(self, scenario: str, description: str, runs: List[RunMetrics]) -> ScenarioResult:
        """Aggregate individual runs into a scenario result."""
        successful = [r for r in runs if r.error is None]
        failed = [r for r in runs if r.error is not None]
        
        latencies = [r.latency_ms for r in successful]
        result_counts = [r.result_count for r in successful]
        payload_sizes = [r.payload_size_bytes for r in successful]
        
        total_time_ms = sum(latencies) if latencies else 1
        throughput = (len(successful) / total_time_ms) * 1000 if latencies else 0
        
        return ScenarioResult(
            scenario=scenario,
            description=description,
            total_runs=len(runs),
            successful_runs=len(successful),
            failed_runs=len(failed),
            latency_ms=self._calculate_percentiles(latencies),
            avg_result_count=round(statistics.mean(result_counts), 2) if result_counts else 0,
            avg_payload_bytes=round(statistics.mean(payload_sizes), 2) if payload_sizes else 0,
            throughput_qps=round(throughput, 2),
            runs=runs
        )
    
    # =========================================================================
    # Benchmark Scenarios
    # =========================================================================
    
    def benchmark_basic_retrieval(self, domain_id: Optional[str] = None) -> ScenarioResult:
        """Basic retrieval latency test with default settings."""
        print("  📊 Running basic retrieval benchmark...")
        runs = []
        
        for query in self.config.test_queries:
            result = self._retrieve(query, domain_id=domain_id)
            runs.append(RunMetrics(
                scenario="basic_retrieval",
                query=query,
                latency_ms=result["latency_ms"],
                result_count=len(result["results"]),
                payload_size_bytes=result["payload_size"],
                error=result.get("error")
            ))
        
        return self._aggregate_runs(
            "basic_retrieval",
            "Basic retrieval with default top_k=10",
            runs
        )
    
    def benchmark_large_payload(self, domain_id: Optional[str] = None) -> ScenarioResult:
        """Test retrieval with varying top_k values."""
        print("  📊 Running large payload benchmark...")
        runs = []
        top_k_values = [1, 5, 10, 25, 50, 100]
        query = self.config.test_queries[0]
        
        for top_k in top_k_values:
            result = self._retrieve(query, domain_id=domain_id, top_k=top_k)
            runs.append(RunMetrics(
                scenario="large_payload",
                query=query,
                latency_ms=result["latency_ms"],
                result_count=len(result["results"]),
                payload_size_bytes=result["payload_size"],
                error=result.get("error"),
                extra={"top_k": top_k}
            ))
        
        return self._aggregate_runs(
            "large_payload",
            f"Varying top_k: {top_k_values}",
            runs
        )
    
    def benchmark_collection_sizes(self) -> ScenarioResult:
        """Compare retrieval across domains with different collection sizes."""
        print("  📊 Running collection size comparison...")
        runs = []
        query = self.config.test_queries[0]
        
        for domain in self.domains:
            domain_id = domain["id"]
            domain_name = domain["name"]
            doc_count = domain.get("document_count", 0)
            
            result = self._retrieve(query, domain_id=domain_id)
            runs.append(RunMetrics(
                scenario="collection_sizes",
                query=query,
                latency_ms=result["latency_ms"],
                result_count=len(result["results"]),
                payload_size_bytes=result["payload_size"],
                error=result.get("error"),
                extra={
                    "domain_id": domain_id,
                    "domain_name": domain_name,
                    "document_count": doc_count
                }
            ))
        
        return self._aggregate_runs(
            "collection_sizes",
            f"Comparison across {len(self.domains)} domains",
            runs
        )
    
    def benchmark_sequential_warmup(self, domain_id: Optional[str] = None) -> ScenarioResult:
        """Test cold start vs warm request latency."""
        print("  📊 Running sequential warmup benchmark...")
        runs = []
        query = self.config.test_queries[0]
        
        # Run 10 sequential requests
        for i in range(10):
            result = self._retrieve(query, domain_id=domain_id)
            runs.append(RunMetrics(
                scenario="sequential_warmup",
                query=query,
                latency_ms=result["latency_ms"],
                result_count=len(result["results"]),
                payload_size_bytes=result["payload_size"],
                error=result.get("error"),
                extra={"request_number": i + 1}
            ))
        
        return self._aggregate_runs(
            "sequential_warmup",
            "10 sequential requests to measure cold vs warm",
            runs
        )
    
    def benchmark_concurrent_load(self, domain_id: Optional[str] = None) -> ScenarioResult:
        """Test throughput with concurrent requests."""
        print("  📊 Running concurrent load benchmark...")
        runs = []
        concurrency_levels = [1, 2, 4, 8]
        query = self.config.test_queries[0]
        
        def single_request(req_num: int, concurrency: int) -> RunMetrics:
            result = self._retrieve(query, domain_id=domain_id)
            return RunMetrics(
                scenario="concurrent_load",
                query=query,
                latency_ms=result["latency_ms"],
                result_count=len(result["results"]),
                payload_size_bytes=result["payload_size"],
                error=result.get("error"),
                extra={"concurrency": concurrency, "request_num": req_num}
            )
        
        for concurrency in concurrency_levels:
            print(f"    Testing {concurrency} concurrent requests...")
            
            # Run concurrent batch using thread pool
            futures = [
                self._executor.submit(single_request, i, concurrency)
                for i in range(concurrency)
            ]
            
            for future in futures:
                try:
                    runs.append(future.result(timeout=self.config.timeout_seconds))
                except Exception as e:
                    runs.append(RunMetrics(
                        scenario="concurrent_load",
                        query=query,
                        latency_ms=0,
                        result_count=0,
                        payload_size_bytes=0,
                        error=str(e),
                        extra={"concurrency": concurrency}
                    ))
        
        return self._aggregate_runs(
            "concurrent_load",
            f"Concurrency levels: {concurrency_levels}",
            runs
        )
    
    def benchmark_hybrid_vs_vector(self, domain_id: Optional[str] = None) -> ScenarioResult:
        """Compare hybrid retrieval vs pure vector retrieval."""
        print("  📊 Running hybrid vs vector comparison...")
        runs = []
        
        for query in self.config.test_queries:
            # Vector retrieval
            vector_result = self._retrieve(query, domain_id=domain_id, hybrid=False)
            runs.append(RunMetrics(
                scenario="hybrid_vs_vector",
                query=query,
                latency_ms=vector_result["latency_ms"],
                result_count=len(vector_result["results"]),
                payload_size_bytes=vector_result["payload_size"],
                error=vector_result.get("error"),
                extra={"method": "vector"}
            ))
            
            # Hybrid retrieval
            hybrid_result = self._retrieve(query, domain_id=domain_id, hybrid=True)
            runs.append(RunMetrics(
                scenario="hybrid_vs_vector",
                query=query,
                latency_ms=hybrid_result["latency_ms"],
                result_count=len(hybrid_result["results"]),
                payload_size_bytes=hybrid_result["payload_size"],
                error=hybrid_result.get("error"),
                extra={"method": "hybrid"}
            ))
        
        return self._aggregate_runs(
            "hybrid_vs_vector",
            "Comparison of vector vs hybrid retrieval",
            runs
        )
    
    def benchmark_score_threshold(self, domain_id: Optional[str] = None) -> ScenarioResult:
        """Test impact of score thresholds on results."""
        print("  📊 Running score threshold benchmark...")
        runs = []
        thresholds = [None, 0.3, 0.5, 0.7, 0.9]
        query = self.config.test_queries[0]
        
        for threshold in thresholds:
            result = self._retrieve(query, domain_id=domain_id, score_threshold=threshold)
            runs.append(RunMetrics(
                scenario="score_threshold",
                query=query,
                latency_ms=result["latency_ms"],
                result_count=len(result["results"]),
                payload_size_bytes=result["payload_size"],
                error=result.get("error"),
                extra={"threshold": threshold if threshold else "none"}
            ))
        
        return self._aggregate_runs(
            "score_threshold",
            f"Score thresholds: {thresholds}",
            runs
        )
    
    def benchmark_context_expansion(self, domain_id: Optional[str] = None) -> ScenarioResult:
        """Test overhead of context neighbor expansion."""
        print("  📊 Running context expansion benchmark...")
        runs = []
        neighbor_configs = [(0, 0), (1, 1), (2, 2), (3, 3)]
        query = self.config.test_queries[0]
        
        for before, after in neighbor_configs:
            result = self._retrieve(
                query, 
                domain_id=domain_id,
                context_neighbors_before=before,
                context_neighbors_after=after
            )
            runs.append(RunMetrics(
                scenario="context_expansion",
                query=query,
                latency_ms=result["latency_ms"],
                result_count=len(result["results"]),
                payload_size_bytes=result["payload_size"],
                error=result.get("error"),
                extra={"neighbors_before": before, "neighbors_after": after}
            ))
        
        return self._aggregate_runs(
            "context_expansion",
            f"Neighbor configs (before, after): {neighbor_configs}",
            runs
        )
    
    # =========================================================================
    # Main Runner
    # =========================================================================
    
    def run_all(self, domain_id: Optional[str] = None) -> Dict[str, Any]:
        """Run all benchmark scenarios."""
        print("\n🚀 Starting Embedding Retrieval Benchmark Suite")
        print("=" * 60)
        
        # Fetch available domains
        print("\n📋 Fetching domains...")
        self.domains = self._fetch_domains()
        print(f"   Found {len(self.domains)} domains:")
        for d in self.domains:
            print(f"     - {d['name']}: {d.get('document_count', 0)} docs")
        
        # Use first domain with documents if none specified
        if not domain_id:
            for d in self.domains:
                if d.get("document_count", 0) > 0:
                    domain_id = d["id"]
                    print(f"\n   Using domain: {d['name']} ({domain_id})")
                    break
        
        # Warmup
        print(f"\n🔥 Warming up ({self.config.warmup_runs} requests)...")
        for _ in range(self.config.warmup_runs):
            self._retrieve("warmup query", domain_id=domain_id)
        
        # Run scenarios
        print("\n📊 Running benchmark scenarios...")
        
        self.results = [
            self.benchmark_basic_retrieval(domain_id),
            self.benchmark_large_payload(domain_id),
            self.benchmark_collection_sizes(),
            self.benchmark_sequential_warmup(domain_id),
            self.benchmark_concurrent_load(domain_id),
            self.benchmark_hybrid_vs_vector(domain_id),
            self.benchmark_score_threshold(domain_id),
            self.benchmark_context_expansion(domain_id),
        ]
        
        # Generate report
        report = self._generate_report()
        
        # Save if output file specified
        if self.config.output_file:
            output_path = Path(self.config.output_file)
            output_path.write_text(json.dumps(report, indent=2, default=str))
            print(f"\n💾 Results saved to: {output_path}")
        
        return report
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": asdict(self.config),
            "domains": self.domains,
            "summary": {},
            "scenarios": []
        }
        
        for result in self.results:
            scenario_data = {
                "name": result.scenario,
                "description": result.description,
                "total_runs": result.total_runs,
                "successful_runs": result.successful_runs,
                "failed_runs": result.failed_runs,
                "latency_ms": result.latency_ms,
                "avg_result_count": result.avg_result_count,
                "avg_payload_bytes": result.avg_payload_bytes,
                "throughput_qps": result.throughput_qps,
                "runs": [asdict(r) for r in result.runs]
            }
            report["scenarios"].append(scenario_data)
            
            # Add to summary
            report["summary"][result.scenario] = {
                "avg_latency_ms": result.latency_ms.get("avg", 0),
                "p95_latency_ms": result.latency_ms.get("p95", 0),
                "success_rate": result.successful_runs / result.total_runs if result.total_runs > 0 else 0
            }
        
        return report
    
    def print_summary(self):
        """Print a formatted summary to console."""
        print("\n" + "=" * 60)
        print("📈 BENCHMARK SUMMARY")
        print("=" * 60)
        
        for result in self.results:
            status = "✅" if result.failed_runs == 0 else "⚠️"
            print(f"\n{status} {result.scenario}")
            print(f"   {result.description}")
            print(f"   Runs: {result.successful_runs}/{result.total_runs} successful")
            
            if result.latency_ms:
                print(f"   Latency: avg={result.latency_ms['avg']:.1f}ms, "
                      f"p50={result.latency_ms['p50']:.1f}ms, "
                      f"p95={result.latency_ms['p95']:.1f}ms")
            
            if result.avg_payload_bytes > 0:
                print(f"   Avg payload: {result.avg_payload_bytes/1024:.1f}KB, "
                      f"Avg results: {result.avg_result_count:.1f}")
            
            # Show per-run details for some scenarios
            if result.scenario in ["large_payload", "collection_sizes", "score_threshold", "context_expansion"]:
                print("   Details:")
                for run in result.runs:
                    if run.extra:
                        extra_str = ", ".join(f"{k}={v}" for k, v in run.extra.items())
                        status_icon = "✓" if not run.error else "✗"
                        print(f"     {status_icon} {extra_str}: {run.latency_ms:.1f}ms, "
                              f"{run.result_count} results")
        
        print("\n" + "=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Embedding Retrieval Benchmark")
    parser.add_argument("--base-url", default="http://localhost:8700",
                        help="Base URL for RAG API")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--domain", "-d", help="Specific domain ID to test")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Request timeout in seconds")
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        base_url=args.base_url,
        output_file=args.output,
        timeout_seconds=args.timeout
    )
    
    with EmbeddingBenchmark(config) as benchmark:
        benchmark.run_all(domain_id=args.domain)
        benchmark.print_summary()


if __name__ == "__main__":
    main()

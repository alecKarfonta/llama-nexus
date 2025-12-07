"""
Inference Speed Benchmark Module
Provides standardized benchmarking for LLM inference performance.
"""

import os
import json
import sqlite3
import time
import asyncio
import aiohttp
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    prompt_tokens: int = 512          # Target input length
    max_output_tokens: int = 256      # Max tokens to generate
    num_runs: int = 3                 # Number of iterations
    warmup_runs: int = 1              # Warmup iterations (not counted)
    temperature: float = 0.0          # Use 0 for deterministic results
    preset: str = "standard"          # Preset name


@dataclass
class RunMetrics:
    """Metrics captured for a single benchmark run."""
    run_number: int
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token_ms: float
    tokens_per_second: float
    total_time_ms: float
    timestamp: str


@dataclass
class BenchmarkResult:
    """Complete benchmark result with statistics."""
    id: str
    config: Dict[str, Any]
    model_name: str
    model_variant: str
    status: str  # running, completed, failed, cancelled
    progress: int  # 0-100
    current_run: int
    total_runs: int
    runs: List[Dict[str, Any]]
    statistics: Optional[Dict[str, Any]]
    started_at: str
    completed_at: Optional[str]
    error: Optional[str]
    system_info: Dict[str, Any]


# Standard test prompts of varying lengths
STANDARD_PROMPTS = {
    "short": "Explain the concept of recursion in programming. Provide a simple example.",
    
    "medium": """You are a senior software engineer reviewing code. Please analyze the following requirements and provide a detailed implementation plan:

Requirements:
1. Build a REST API for user management
2. Support CRUD operations for users
3. Implement JWT authentication
4. Add rate limiting
5. Include comprehensive error handling

Provide your analysis with code structure, key considerations, and potential pitfalls.""",

    "long": """You are an expert technical writer. Please write comprehensive documentation for the following system:

System Overview:
A distributed message queue system that handles high-throughput event processing. The system consists of:
- Producer services that generate events
- A central broker cluster for message routing
- Consumer services that process events
- A monitoring dashboard for observability

The system must handle:
- 100,000 messages per second peak throughput
- Message persistence with configurable retention
- At-least-once delivery guarantees
- Consumer group management
- Dead letter queues for failed messages
- Automatic partition rebalancing

Please provide:
1. Architecture overview
2. Component descriptions
3. Data flow diagrams (describe in text)
4. API specifications
5. Configuration options
6. Deployment considerations
7. Monitoring and alerting recommendations
8. Troubleshooting guide

Be thorough and technical in your response.""",

    "code": """Write a Python implementation of a binary search tree with the following methods:
- insert(value): Insert a new value
- search(value): Return True if value exists
- delete(value): Remove a value
- inorder_traversal(): Return values in sorted order
- find_min(): Return minimum value
- find_max(): Return maximum value
- height(): Return tree height
- is_balanced(): Check if tree is balanced

Include type hints, docstrings, and handle edge cases properly. Also write unit tests using pytest.""",
}

# Benchmark presets
BENCHMARK_PRESETS = {
    "quick": BenchmarkConfig(
        prompt_tokens=128,
        max_output_tokens=64,
        num_runs=3,
        warmup_runs=1,
        preset="quick"
    ),
    "standard": BenchmarkConfig(
        prompt_tokens=512,
        max_output_tokens=256,
        num_runs=5,
        warmup_runs=1,
        preset="standard"
    ),
    "long_context": BenchmarkConfig(
        prompt_tokens=2048,
        max_output_tokens=512,
        num_runs=3,
        warmup_runs=1,
        preset="long_context"
    ),
    "max_speed": BenchmarkConfig(
        prompt_tokens=64,
        max_output_tokens=128,
        num_runs=5,
        warmup_runs=2,
        preset="max_speed"
    ),
}


class BenchmarkRunner:
    """
    Runs inference benchmarks against the LLM API.
    """
    
    def __init__(self, db_path: str = None, api_base_url: str = None):
        """Initialize the benchmark runner."""
        if db_path is None:
            db_path = os.getenv('BENCHMARK_DB_PATH', '/data/benchmarks.db')
        
        self.db_path = db_path
        self.api_base_url = api_base_url or os.getenv('LLAMA_API_URL', 'http://llamacpp-api:8080')
        self._ensure_db_directory()
        self._init_database()
        
        # Track active benchmarks
        self._active_benchmarks: Dict[str, BenchmarkResult] = {}
        
        logger.info(f"Benchmark runner initialized with database at: {self.db_path}")
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS benchmarks (
                    id TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    model_name TEXT,
                    model_variant TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    results TEXT,
                    statistics TEXT,
                    system_info TEXT,
                    error TEXT
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_benchmarks_status ON benchmarks(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_benchmarks_started ON benchmarks(started_at)')
            
            conn.commit()
    
    def _select_prompt(self, target_tokens: int) -> str:
        """Select a prompt based on target token count."""
        # Rough estimation: 1 token ~= 4 characters
        if target_tokens <= 128:
            return STANDARD_PROMPTS["short"]
        elif target_tokens <= 512:
            return STANDARD_PROMPTS["medium"]
        elif target_tokens <= 1024:
            return STANDARD_PROMPTS["code"]
        else:
            return STANDARD_PROMPTS["long"]
    
    def get_preset(self, preset_name: str) -> BenchmarkConfig:
        """Get a benchmark preset by name."""
        return BENCHMARK_PRESETS.get(preset_name, BENCHMARK_PRESETS["standard"])
    
    def list_presets(self) -> Dict[str, Dict[str, Any]]:
        """List all available benchmark presets."""
        return {
            name: asdict(config) 
            for name, config in BENCHMARK_PRESETS.items()
        }
    
    async def _run_single_inference(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        api_key: str = None,
    ) -> Dict[str, Any]:
        """Run a single inference and measure timing."""
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        
        start_time = time.perf_counter()
        first_token_time = None
        total_tokens = 0
        prompt_tokens = 0
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line or not line.startswith('data: '):
                            continue
                        
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            # Record time to first token
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            
                            # Count tokens from delta
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    # Rough token count (actual would need tokenizer)
                                    total_tokens += max(1, len(content) // 4)
                            
                            # Get usage if available
                            if 'usage' in data:
                                prompt_tokens = data['usage'].get('prompt_tokens', prompt_tokens)
                                total_tokens = data['usage'].get('completion_tokens', total_tokens)
                        
                        except json.JSONDecodeError:
                            continue
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            total_time_ms = (end_time - start_time) * 1000
            ttft_ms = ((first_token_time or end_time) - start_time) * 1000
            generation_time = end_time - (first_token_time or start_time)
            tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
            
            return {
                "success": True,
                "prompt_tokens": prompt_tokens or len(prompt) // 4,
                "completion_tokens": total_tokens,
                "time_to_first_token_ms": round(ttft_ms, 2),
                "tokens_per_second": round(tokens_per_second, 2),
                "total_time_ms": round(total_time_ms, 2),
            }
            
        except asyncio.TimeoutError:
            return {"success": False, "error": "Request timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def run_benchmark(
        self,
        config: BenchmarkConfig,
        model_name: str = "unknown",
        model_variant: str = "unknown",
        api_key: str = None,
        progress_callback: callable = None,
    ) -> BenchmarkResult:
        """Run a complete benchmark suite."""
        
        benchmark_id = str(uuid.uuid4())[:12]
        now = datetime.utcnow().isoformat()
        
        # Initialize result
        result = BenchmarkResult(
            id=benchmark_id,
            config=asdict(config),
            model_name=model_name,
            model_variant=model_variant,
            status="running",
            progress=0,
            current_run=0,
            total_runs=config.warmup_runs + config.num_runs,
            runs=[],
            statistics=None,
            started_at=now,
            completed_at=None,
            error=None,
            system_info={"api_url": self.api_base_url},
        )
        
        self._active_benchmarks[benchmark_id] = result
        
        # Save initial state
        self._save_benchmark(result)
        
        prompt = self._select_prompt(config.prompt_tokens)
        
        try:
            # Warmup runs
            for i in range(config.warmup_runs):
                result.current_run = i + 1
                result.progress = int((i + 1) / result.total_runs * 100)
                
                if progress_callback:
                    await progress_callback(result)
                
                await self._run_single_inference(
                    prompt=prompt,
                    max_tokens=config.max_output_tokens,
                    temperature=config.temperature,
                    api_key=api_key,
                )
            
            # Actual benchmark runs
            run_results = []
            for i in range(config.num_runs):
                run_number = i + 1
                result.current_run = config.warmup_runs + run_number
                result.progress = int(result.current_run / result.total_runs * 100)
                
                if progress_callback:
                    await progress_callback(result)
                
                run_result = await self._run_single_inference(
                    prompt=prompt,
                    max_tokens=config.max_output_tokens,
                    temperature=config.temperature,
                    api_key=api_key,
                )
                
                if run_result["success"]:
                    metrics = RunMetrics(
                        run_number=run_number,
                        prompt_tokens=run_result["prompt_tokens"],
                        completion_tokens=run_result["completion_tokens"],
                        time_to_first_token_ms=run_result["time_to_first_token_ms"],
                        tokens_per_second=run_result["tokens_per_second"],
                        total_time_ms=run_result["total_time_ms"],
                        timestamp=datetime.utcnow().isoformat(),
                    )
                    run_results.append(asdict(metrics))
                    result.runs = run_results
                else:
                    logger.warning(f"Run {run_number} failed: {run_result.get('error')}")
            
            # Calculate statistics
            if run_results:
                result.statistics = self._calculate_statistics(run_results)
            
            result.status = "completed"
            result.progress = 100
            result.completed_at = datetime.utcnow().isoformat()
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.utcnow().isoformat()
        
        # Save final state
        self._save_benchmark(result)
        
        # Remove from active
        self._active_benchmarks.pop(benchmark_id, None)
        
        if progress_callback:
            await progress_callback(result)
        
        return result
    
    def _calculate_statistics(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from benchmark runs."""
        if not runs:
            return {}
        
        tps_values = [r["tokens_per_second"] for r in runs]
        ttft_values = [r["time_to_first_token_ms"] for r in runs]
        total_time_values = [r["total_time_ms"] for r in runs]
        
        def calc_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            return {
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "mean": round(statistics.mean(values), 2),
                "median": round(statistics.median(values), 2),
                "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0,
            }
        
        return {
            "tokens_per_second": calc_stats(tps_values),
            "time_to_first_token_ms": calc_stats(ttft_values),
            "total_time_ms": calc_stats(total_time_values),
            "num_successful_runs": len(runs),
            "total_tokens_generated": sum(r["completion_tokens"] for r in runs),
        }
    
    def _save_benchmark(self, result: BenchmarkResult):
        """Save benchmark result to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO benchmarks 
                (id, config, model_name, model_variant, status, started_at, 
                 completed_at, results, statistics, system_info, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.id,
                json.dumps(result.config),
                result.model_name,
                result.model_variant,
                result.status,
                result.started_at,
                result.completed_at,
                json.dumps(result.runs),
                json.dumps(result.statistics) if result.statistics else None,
                json.dumps(result.system_info),
                result.error,
            ))
            conn.commit()
    
    def get_benchmark(self, benchmark_id: str) -> Optional[Dict[str, Any]]:
        """Get a benchmark by ID."""
        # Check active first
        if benchmark_id in self._active_benchmarks:
            result = self._active_benchmarks[benchmark_id]
            return {
                "id": result.id,
                "config": result.config,
                "model_name": result.model_name,
                "model_variant": result.model_variant,
                "status": result.status,
                "progress": result.progress,
                "current_run": result.current_run,
                "total_runs": result.total_runs,
                "runs": result.runs,
                "statistics": result.statistics,
                "started_at": result.started_at,
                "completed_at": result.completed_at,
                "error": result.error,
            }
        
        # Check database
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM benchmarks WHERE id = ?', (benchmark_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    "id": row["id"],
                    "config": json.loads(row["config"]),
                    "model_name": row["model_name"],
                    "model_variant": row["model_variant"],
                    "status": row["status"],
                    "progress": 100 if row["status"] == "completed" else 0,
                    "current_run": 0,
                    "total_runs": 0,
                    "runs": json.loads(row["results"]) if row["results"] else [],
                    "statistics": json.loads(row["statistics"]) if row["statistics"] else None,
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "error": row["error"],
                }
        
        return None
    
    def list_benchmarks(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str = None,
    ) -> Dict[str, Any]:
        """List all benchmarks."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM benchmarks WHERE 1=1'
            params = []
            
            if status:
                query += ' AND status = ?'
                params.append(status)
            
            # Get total count
            count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]
            
            # Get paginated results
            query += ' ORDER BY started_at DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            cursor.execute(query, params)
            
            benchmarks = []
            for row in cursor.fetchall():
                benchmarks.append({
                    "id": row["id"],
                    "config": json.loads(row["config"]),
                    "model_name": row["model_name"],
                    "model_variant": row["model_variant"],
                    "status": row["status"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "statistics": json.loads(row["statistics"]) if row["statistics"] else None,
                })
            
            return {
                "benchmarks": benchmarks,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total,
            }
    
    def delete_benchmark(self, benchmark_id: str) -> bool:
        """Delete a benchmark."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM benchmarks WHERE id = ?', (benchmark_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall benchmark statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM benchmarks')
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM benchmarks WHERE status = 'completed'")
            completed = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM benchmarks WHERE status = 'failed'")
            failed = cursor.fetchone()[0]
            
            # Get average TPS from completed benchmarks
            cursor.execute('''
                SELECT statistics FROM benchmarks 
                WHERE status = 'completed' AND statistics IS NOT NULL
                ORDER BY started_at DESC LIMIT 10
            ''')
            
            recent_tps = []
            for row in cursor.fetchall():
                stats = json.loads(row[0])
                if stats and 'tokens_per_second' in stats:
                    recent_tps.append(stats['tokens_per_second'].get('mean', 0))
            
            avg_tps = round(statistics.mean(recent_tps), 2) if recent_tps else 0
            
            return {
                "total_benchmarks": total,
                "completed": completed,
                "failed": failed,
                "average_tps_recent": avg_tps,
            }


# Global instance
benchmark_runner = BenchmarkRunner()

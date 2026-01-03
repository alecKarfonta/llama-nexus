"""
A/B Testing module for comparing LoRA adapters in production.

Features:
- Traffic splitting between base model and adapters
- Multi-variant testing (A/B/n)
- Real-time metrics collection
- Statistical significance testing
"""

import asyncio
import random
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from enhanced_logger import enhanced_logger as logger


class ABTestStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TrafficSplit(BaseModel):
    """Traffic allocation for a variant."""
    variant_id: str
    name: str
    adapter_path: Optional[str] = None  # None = base model
    weight: float = 0.5  # 0.0 to 1.0


class VariantMetrics(BaseModel):
    """Collected metrics for a variant."""
    variant_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_generated: int = 0
    total_latency_ms: float = 0.0
    total_ttft_ms: float = 0.0
    # Aggregates
    avg_tokens_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    avg_ttft_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    # User feedback
    thumbs_up: int = 0
    thumbs_down: int = 0
    # Raw latency samples for percentile calculation
    latency_samples: List[float] = Field(default_factory=list)


class ABTest(BaseModel):
    """A/B test configuration and state."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    base_model: str
    variants: List[TrafficSplit] = Field(default_factory=list)
    status: ABTestStatus = ABTestStatus.DRAFT
    metrics: Dict[str, VariantMetrics] = Field(default_factory=dict)
    # Configuration
    min_samples_per_variant: int = 100
    confidence_level: float = 0.95
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class RequestResult(BaseModel):
    """Result of a single request in an A/B test."""
    test_id: str
    variant_id: str
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    response: str
    tokens_generated: int = 0
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    tokens_per_second: float = 0.0
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ABTestManager:
    """
    Manages A/B tests for adapter comparison.
    
    Features:
    - Create and manage A/B tests
    - Route traffic based on configured weights
    - Collect and aggregate metrics
    - Calculate statistical significance
    """

    def __init__(self, base_url: str = "http://localhost:8700"):
        self.base_url = base_url.rstrip("/")
        self.tests: Dict[str, ABTest] = {}
        self.results: Dict[str, List[RequestResult]] = {}

    def create_test(self, test: ABTest) -> ABTest:
        """Create a new A/B test."""
        # Initialize metrics for each variant
        for variant in test.variants:
            test.metrics[variant.variant_id] = VariantMetrics(variant_id=variant.variant_id)
        
        self.tests[test.id] = test
        self.results[test.id] = []
        logger.info("Created A/B test", extra={"test_id": test.id, "name": test.name})
        return test

    def get_test(self, test_id: str) -> Optional[ABTest]:
        return self.tests.get(test_id)

    def list_tests(self) -> List[ABTest]:
        return list(self.tests.values())

    def delete_test(self, test_id: str) -> bool:
        if test_id in self.tests:
            del self.tests[test_id]
            if test_id in self.results:
                del self.results[test_id]
            return True
        return False

    def start_test(self, test_id: str) -> Optional[ABTest]:
        """Start an A/B test."""
        test = self.tests.get(test_id)
        if not test:
            return None
        
        if test.status not in (ABTestStatus.DRAFT, ABTestStatus.PAUSED):
            return test
        
        test.status = ABTestStatus.RUNNING
        test.started_at = datetime.utcnow()
        logger.info("Started A/B test", extra={"test_id": test_id})
        return test

    def pause_test(self, test_id: str) -> Optional[ABTest]:
        """Pause an A/B test."""
        test = self.tests.get(test_id)
        if not test:
            return None
        
        if test.status == ABTestStatus.RUNNING:
            test.status = ABTestStatus.PAUSED
        return test

    def complete_test(self, test_id: str) -> Optional[ABTest]:
        """Complete an A/B test."""
        test = self.tests.get(test_id)
        if not test:
            return None
        
        test.status = ABTestStatus.COMPLETED
        test.completed_at = datetime.utcnow()
        return test

    def select_variant(self, test_id: str) -> Optional[TrafficSplit]:
        """
        Select a variant based on traffic weights.
        Uses weighted random selection.
        """
        test = self.tests.get(test_id)
        if not test or test.status != ABTestStatus.RUNNING:
            return None
        
        if not test.variants:
            return None
        
        # Normalize weights
        total_weight = sum(v.weight for v in test.variants)
        if total_weight == 0:
            return random.choice(test.variants)
        
        # Weighted random selection
        r = random.random() * total_weight
        cumulative = 0
        for variant in test.variants:
            cumulative += variant.weight
            if r <= cumulative:
                return variant
        
        return test.variants[-1]

    async def execute_request(
        self,
        test_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Optional[RequestResult]:
        """
        Execute a request through the A/B test.
        Automatically routes to a variant and records metrics.
        """
        test = self.tests.get(test_id)
        if not test or test.status != ABTestStatus.RUNNING:
            return None
        
        variant = self.select_variant(test_id)
        if not variant:
            return None
        
        result = RequestResult(
            test_id=test_id,
            variant_id=variant.variant_id,
            prompt=prompt,
        )
        
        start_time = time.time()
        ttft_recorded = False
        ttft_ms = 0.0
        tokens = 0
        response_text = ""
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                payload = {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                }
                
                if variant.adapter_path:
                    payload["lora_adapter"] = variant.adapter_path
                
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            if not ttft_recorded:
                                ttft_ms = (time.time() - start_time) * 1000
                                ttft_recorded = True
                            
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            
                            try:
                                import json
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    response_text += content
                                    tokens += 1
                            except:
                                pass
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                result.response = response_text
                result.tokens_generated = tokens
                result.latency_ms = latency_ms
                result.ttft_ms = ttft_ms
                result.tokens_per_second = tokens / (latency_ms / 1000) if latency_ms > 0 else 0
                result.success = True
                
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.latency_ms = (time.time() - start_time) * 1000
        
        # Record result
        self._record_result(test_id, result)
        
        return result

    def _record_result(self, test_id: str, result: RequestResult) -> None:
        """Record a request result and update metrics."""
        test = self.tests.get(test_id)
        if not test:
            return
        
        # Store raw result
        if test_id not in self.results:
            self.results[test_id] = []
        self.results[test_id].append(result)
        
        # Update variant metrics
        metrics = test.metrics.get(result.variant_id)
        if not metrics:
            metrics = VariantMetrics(variant_id=result.variant_id)
            test.metrics[result.variant_id] = metrics
        
        metrics.total_requests += 1
        if result.success:
            metrics.successful_requests += 1
            metrics.total_tokens_generated += result.tokens_generated
            metrics.total_latency_ms += result.latency_ms
            metrics.total_ttft_ms += result.ttft_ms
            metrics.latency_samples.append(result.latency_ms)
            
            # Keep only last 1000 samples for percentile calculation
            if len(metrics.latency_samples) > 1000:
                metrics.latency_samples = metrics.latency_samples[-1000:]
        else:
            metrics.failed_requests += 1
        
        # Update aggregates
        if metrics.successful_requests > 0:
            total_time_s = metrics.total_latency_ms / 1000
            metrics.avg_tokens_per_second = metrics.total_tokens_generated / total_time_s if total_time_s > 0 else 0
            metrics.avg_latency_ms = metrics.total_latency_ms / metrics.successful_requests
            metrics.avg_ttft_ms = metrics.total_ttft_ms / metrics.successful_requests
            
            # Calculate percentiles
            if metrics.latency_samples:
                sorted_samples = sorted(metrics.latency_samples)
                n = len(sorted_samples)
                metrics.p50_latency_ms = sorted_samples[int(n * 0.5)]
                metrics.p95_latency_ms = sorted_samples[int(n * 0.95)]
                metrics.p99_latency_ms = sorted_samples[min(int(n * 0.99), n - 1)]

    def record_feedback(self, test_id: str, variant_id: str, thumbs_up: bool) -> bool:
        """Record user feedback for a variant."""
        test = self.tests.get(test_id)
        if not test:
            return False
        
        metrics = test.metrics.get(variant_id)
        if not metrics:
            return False
        
        if thumbs_up:
            metrics.thumbs_up += 1
        else:
            metrics.thumbs_down += 1
        
        return True

    def get_results(self, test_id: str, limit: int = 100) -> List[RequestResult]:
        """Get recent results for a test."""
        results = self.results.get(test_id, [])
        return results[-limit:]

    def get_summary(self, test_id: str) -> Dict[str, Any]:
        """Get a summary of test results with statistical analysis."""
        test = self.tests.get(test_id)
        if not test:
            return {}
        
        summary = {
            "test_id": test_id,
            "name": test.name,
            "status": test.status.value,
            "started_at": test.started_at.isoformat() if test.started_at else None,
            "variants": [],
        }
        
        # Collect variant summaries
        variant_summaries = []
        for variant in test.variants:
            metrics = test.metrics.get(variant.variant_id)
            if not metrics:
                continue
            
            success_rate = (
                metrics.successful_requests / metrics.total_requests * 100
                if metrics.total_requests > 0 else 0
            )
            
            feedback_score = (
                metrics.thumbs_up / (metrics.thumbs_up + metrics.thumbs_down) * 100
                if (metrics.thumbs_up + metrics.thumbs_down) > 0 else None
            )
            
            variant_summaries.append({
                "variant_id": variant.variant_id,
                "name": variant.name,
                "adapter_path": variant.adapter_path,
                "weight": variant.weight,
                "total_requests": metrics.total_requests,
                "success_rate": round(success_rate, 2),
                "avg_tokens_per_second": round(metrics.avg_tokens_per_second, 2),
                "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                "avg_ttft_ms": round(metrics.avg_ttft_ms, 2),
                "p50_latency_ms": round(metrics.p50_latency_ms, 2),
                "p95_latency_ms": round(metrics.p95_latency_ms, 2),
                "p99_latency_ms": round(metrics.p99_latency_ms, 2),
                "thumbs_up": metrics.thumbs_up,
                "thumbs_down": metrics.thumbs_down,
                "feedback_score": round(feedback_score, 1) if feedback_score is not None else None,
            })
        
        summary["variants"] = variant_summaries
        
        # Determine winner if enough samples
        if len(variant_summaries) >= 2:
            # Sort by feedback score (if available) or success rate
            sorted_variants = sorted(
                variant_summaries,
                key=lambda v: (v.get("feedback_score") or 0, v["success_rate"]),
                reverse=True,
            )
            
            winner = sorted_variants[0]
            runner_up = sorted_variants[1]
            
            # Check if we have enough samples
            min_samples = test.min_samples_per_variant
            if winner["total_requests"] >= min_samples and runner_up["total_requests"] >= min_samples:
                summary["winner"] = winner["variant_id"]
                summary["winner_name"] = winner["name"]
                summary["has_enough_samples"] = True
            else:
                summary["has_enough_samples"] = False
                summary["samples_needed"] = max(
                    min_samples - winner["total_requests"],
                    min_samples - runner_up["total_requests"],
                    0,
                )
        
        return summary

    def update_weights(self, test_id: str, weights: Dict[str, float]) -> Optional[ABTest]:
        """Update traffic weights for variants."""
        test = self.tests.get(test_id)
        if not test:
            return None
        
        for variant in test.variants:
            if variant.variant_id in weights:
                variant.weight = weights[variant.variant_id]
        
        return test

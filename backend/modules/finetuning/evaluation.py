"""
Model evaluation module for comparing base and fine-tuned models.

Features:
- Side-by-side response comparison
- Blind comparison mode
- LLM-as-judge evaluation
- Human rating collection
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from enhanced_logger import enhanced_logger as logger


class ComparisonType(str, Enum):
    SINGLE = "single"
    BATCH = "batch"
    BLIND = "blind"


class EvaluationCriteria(str, Enum):
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    STYLE = "style"
    HELPFULNESS = "helpfulness"


class ModelResponse(BaseModel):
    model_id: str
    model_name: str
    is_finetuned: bool
    adapter_id: Optional[str] = None
    response: str
    tokens_generated: int = 0
    generation_time_ms: float = 0
    tokens_per_second: float = 0


class PromptComparison(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    system_prompt: Optional[str] = None
    base_response: Optional[ModelResponse] = None
    finetuned_response: Optional[ModelResponse] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Human evaluation
    preferred_model: Optional[str] = None  # "base", "finetuned", "tie"
    preference_reason: Optional[str] = None
    ratings: Optional[Dict[str, Dict[str, int]]] = None  # model -> criterion -> score


class ComparisonSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    comparison_type: ComparisonType = ComparisonType.SINGLE
    base_model: str
    adapter_id: str
    adapter_path: Optional[str] = None
    test_prompts: List[str] = Field(default_factory=list)
    comparisons: List[PromptComparison] = Field(default_factory=list)
    sampling_params: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    # Summary stats
    total_comparisons: int = 0
    base_preferred: int = 0
    finetuned_preferred: int = 0
    ties: int = 0


class JudgeConfig(BaseModel):
    judge_model: str = "gpt-4"
    judge_provider: str = "openai"  # openai, anthropic, local
    api_key: Optional[str] = None
    criteria: List[EvaluationCriteria] = Field(
        default_factory=lambda: [
            EvaluationCriteria.RELEVANCE,
            EvaluationCriteria.ACCURACY,
            EvaluationCriteria.HELPFULNESS,
        ]
    )
    temperature: float = 0.0


class JudgeEvaluation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    comparison_id: str
    judge_model: str
    judge_verdict: str  # "A", "B", "tie"
    judge_reasoning: str
    criteria_scores: Dict[str, Dict[str, int]]  # response -> criterion -> score
    confidence: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator comparing two AI assistant responses.

## Evaluation Criteria
{criteria_list}

## User Prompt
{user_prompt}

## Response A
{response_a}

## Response B
{response_b}

## Instructions
1. Evaluate each response against the criteria above (score 1-5 for each)
2. Determine which response is better overall, or if they are tied
3. Explain your reasoning briefly

Respond in JSON format:
{{
  "response_a_scores": {{"relevance": X, "accuracy": X, ...}},
  "response_b_scores": {{"relevance": X, "accuracy": X, ...}},
  "verdict": "A" | "B" | "tie",
  "reasoning": "...",
  "confidence": 0.0-1.0
}}
"""


class EvaluationManager:
    """Manages model evaluation sessions and comparisons."""

    def __init__(self, base_url: str = "http://localhost:8700"):
        self.base_url = base_url.rstrip("/")
        self.sessions: Dict[str, ComparisonSession] = {}
        self.evaluations: Dict[str, List[JudgeEvaluation]] = {}

    def create_session(self, session: ComparisonSession) -> ComparisonSession:
        self.sessions[session.id] = session
        logger.info("Created comparison session", extra={"session_id": session.id})
        return session

    def get_session(self, session_id: str) -> Optional[ComparisonSession]:
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[ComparisonSession]:
        return list(self.sessions.values())

    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        adapter_path: Optional[str] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate a response from a model (base or with adapter)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = datetime.utcnow()

        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1024),
            }
            if adapter_path:
                payload["lora_adapter"] = adapter_path

            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        end_time = datetime.utcnow()
        elapsed_ms = (end_time - start_time).total_seconds() * 1000

        choice = data["choices"][0]
        content = choice["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", len(content.split()))

        return ModelResponse(
            model_id=data.get("model", "unknown"),
            model_name=data.get("model", "unknown"),
            is_finetuned=adapter_path is not None,
            adapter_id=adapter_path,
            response=content,
            tokens_generated=tokens,
            generation_time_ms=elapsed_ms,
            tokens_per_second=tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        )

    async def run_comparison(
        self,
        session_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> PromptComparison:
        """Run a single comparison between base and fine-tuned model."""
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")

        comparison = PromptComparison(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        # Generate base model response
        try:
            comparison.base_response = await self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                adapter_path=None,
                **session.sampling_params,
            )
            comparison.base_response.is_finetuned = False
        except Exception as e:
            logger.error("Failed to generate base response", extra={"error": str(e)})

        # Generate fine-tuned response
        try:
            comparison.finetuned_response = await self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                adapter_path=session.adapter_path,
                **session.sampling_params,
            )
            comparison.finetuned_response.is_finetuned = True
            comparison.finetuned_response.adapter_id = session.adapter_id
        except Exception as e:
            logger.error("Failed to generate finetuned response", extra={"error": str(e)})

        session.comparisons.append(comparison)
        session.total_comparisons = len(session.comparisons)

        return comparison

    async def run_batch_comparison(self, session_id: str) -> ComparisonSession:
        """Run comparisons for all prompts in the session."""
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")

        for prompt in session.test_prompts:
            await self.run_comparison(session_id, prompt)

        session.completed_at = datetime.utcnow()
        return session

    def submit_rating(
        self,
        session_id: str,
        comparison_id: str,
        preferred: str,
        reason: Optional[str] = None,
        ratings: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> PromptComparison:
        """Submit human rating for a comparison."""
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")

        for comp in session.comparisons:
            if comp.id == comparison_id:
                comp.preferred_model = preferred
                comp.preference_reason = reason
                comp.ratings = ratings

                # Update session stats
                if preferred == "base":
                    session.base_preferred += 1
                elif preferred == "finetuned":
                    session.finetuned_preferred += 1
                else:
                    session.ties += 1

                return comp

        raise KeyError(f"Comparison {comparison_id} not found")

    async def run_judge_evaluation(
        self,
        session_id: str,
        comparison_id: str,
        judge_config: JudgeConfig,
    ) -> JudgeEvaluation:
        """Run LLM-as-judge evaluation on a comparison."""
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")

        comparison = None
        for comp in session.comparisons:
            if comp.id == comparison_id:
                comparison = comp
                break

        if not comparison:
            raise KeyError(f"Comparison {comparison_id} not found")

        if not comparison.base_response or not comparison.finetuned_response:
            raise ValueError("Comparison missing responses")

        # Build judge prompt
        criteria_list = "\n".join(
            f"- {c.value.title()}: Rate 1-5" for c in judge_config.criteria
        )
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            criteria_list=criteria_list,
            user_prompt=comparison.prompt,
            response_a=comparison.base_response.response,
            response_b=comparison.finetuned_response.response,
        )

        # Call judge model
        api_key = judge_config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Judge API key required")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": judge_config.judge_model,
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "temperature": judge_config.temperature,
                },
            )
            response.raise_for_status()
            data = response.json()

        judge_response = data["choices"][0]["message"]["content"]

        # Parse JSON response
        try:
            result = json.loads(judge_response)
        except json.JSONDecodeError:
            result = {
                "verdict": "tie",
                "reasoning": judge_response,
                "confidence": 0.5,
                "response_a_scores": {},
                "response_b_scores": {},
            }

        evaluation = JudgeEvaluation(
            comparison_id=comparison_id,
            judge_model=judge_config.judge_model,
            judge_verdict=result.get("verdict", "tie"),
            judge_reasoning=result.get("reasoning", ""),
            criteria_scores={
                "base": result.get("response_a_scores", {}),
                "finetuned": result.get("response_b_scores", {}),
            },
            confidence=result.get("confidence", 0.5),
        )

        if session_id not in self.evaluations:
            self.evaluations[session_id] = []
        self.evaluations[session_id].append(evaluation)

        return evaluation

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary statistics for a comparison session."""
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Session {session_id} not found")

        total = session.total_comparisons
        if total == 0:
            return {
                "session_id": session_id,
                "total_comparisons": 0,
                "human_ratings": 0,
                "base_preferred_pct": 0,
                "finetuned_preferred_pct": 0,
                "tie_pct": 0,
            }

        rated = session.base_preferred + session.finetuned_preferred + session.ties
        return {
            "session_id": session_id,
            "total_comparisons": total,
            "human_ratings": rated,
            "base_preferred_pct": round(session.base_preferred / rated * 100, 1) if rated else 0,
            "finetuned_preferred_pct": round(session.finetuned_preferred / rated * 100, 1) if rated else 0,
            "tie_pct": round(session.ties / rated * 100, 1) if rated else 0,
        }

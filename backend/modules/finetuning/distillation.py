"""
World-class model distillation pipeline for generating high-quality training data.

This module implements state-of-the-art techniques for synthetic data generation:
- Chain-of-Thought (CoT) prompting for reasoning tasks
- Thinking tokens format (DeepSeek R1 style) for structured reasoning
- Self-consistency with multi-sample voting for accuracy
- LLM-as-judge quality validation with multi-factor scoring
- Automatic response refinement for low-quality samples
- Program-aided reasoning for verifiable outputs

Supports:
- External APIs: OpenAI (GPT-4, o1), Anthropic (Claude 3.5), Google (Gemini)
- Local deployed models in Llama Nexus
- Advanced prompt templates and batch processing
- Comprehensive quality filtering and validation

Based on best practices from:
- CoT-Self-Instruct (arxiv.org/abs/2507.23751)
- Program-aided Distillation (arxiv.org/abs/2305.13888)
- Self-consistency decoding (Wang et al.)
- DeepSeek R1 thinking token format
"""

import asyncio
import json
import os
import re
import time
import uuid
import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel, Field, validator

from enhanced_logger import enhanced_logger as logger
from .config import Dataset, DatasetFormat, DatasetStatus
from .storage import DatasetStore


# =============================================================================
# Enums and Constants
# =============================================================================

class TeacherProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


class DistillationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationStrategy(str, Enum):
    """Strategy for generating responses from teacher model."""
    STANDARD = "standard"  # Direct response
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step reasoning
    THINKING_TOKENS = "thinking_tokens"  # <think>...</think> format
    SELF_CONSISTENCY = "self_consistency"  # Multiple samples with voting
    PROGRAM_AIDED = "program_aided"  # Generate and verify with code
    MULTI_TURN = "multi_turn"  # Conversational with follow-ups


class OutputFormat(str, Enum):
    """Output format for training examples."""
    ALPACA = "alpaca"  # instruction, input, output
    SHAREGPT = "sharegpt"  # conversations array
    CHATML = "chatml"  # messages with roles
    THINKING = "thinking"  # with <think> tokens
    COMPLETION = "completion"  # prompt, completion


class QualityDimension(str, Enum):
    """Dimensions for quality assessment."""
    CORRECTNESS = "correctness"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"
    INSTRUCTION_FOLLOWING = "instruction_following"


# Quality thresholds
QUALITY_THRESHOLDS = {
    "strict": 0.85,
    "high": 0.75,
    "medium": 0.65,
    "low": 0.50,
}


# =============================================================================
# Configuration Models
# =============================================================================

class QualityConfig(BaseModel):
    """Configuration for quality validation."""
    enabled: bool = True
    use_llm_judge: bool = True
    judge_model: Optional[str] = None  # If None, uses same teacher model
    dimensions: List[QualityDimension] = Field(
        default_factory=lambda: [
            QualityDimension.CORRECTNESS,
            QualityDimension.COHERENCE,
            QualityDimension.COMPLETENESS,
            QualityDimension.INSTRUCTION_FOLLOWING,
        ]
    )
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    threshold_preset: Optional[str] = None  # strict, high, medium, low
    max_refinement_attempts: int = Field(2, ge=0, le=5)
    require_all_dimensions_pass: bool = False

    @validator("threshold_preset")
    def apply_threshold_preset(cls, v, values):
        if v and v in QUALITY_THRESHOLDS:
            values["threshold"] = QUALITY_THRESHOLDS[v]
        return v


class SelfConsistencyConfig(BaseModel):
    """Configuration for self-consistency validation."""
    enabled: bool = False
    num_samples: int = Field(5, ge=2, le=20)
    temperature_range: Tuple[float, float] = (0.5, 1.0)
    agreement_threshold: float = Field(0.6, ge=0.0, le=1.0)
    use_semantic_similarity: bool = True
    extract_answer_pattern: Optional[str] = None  # Regex to extract final answer


class ThinkingConfig(BaseModel):
    """Configuration for thinking token generation."""
    enabled: bool = False
    think_tag: str = "<think>"
    think_end_tag: str = "</think>"
    answer_tag: str = "<answer>"
    answer_end_tag: str = "</answer>"
    min_thinking_steps: int = Field(2, ge=1)
    max_thinking_tokens: int = Field(1024, ge=100)
    require_structured_steps: bool = True


class DistillationConfig(BaseModel):
    """Main configuration for distillation jobs."""
    teacher_provider: TeacherProvider = TeacherProvider.OPENAI
    teacher_model: str = "gpt-4o"
    teacher_api_key: Optional[str] = None
    teacher_base_url: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1)
    
    # Generation strategy
    strategy: GenerationStrategy = GenerationStrategy.STANDARD
    output_format: OutputFormat = OutputFormat.ALPACA
    
    # Quality settings
    quality: QualityConfig = Field(default_factory=QualityConfig)
    self_consistency: SelfConsistencyConfig = Field(default_factory=SelfConsistencyConfig)
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)
    
    # Processing settings
    batch_size: int = Field(5, ge=1, le=50)
    rate_limit_rpm: int = Field(60, ge=1)
    max_retries: int = Field(3, ge=0, le=10)
    retry_delay: float = Field(1.0, ge=0.1)
    
    # Filtering
    min_output_length: int = Field(10, ge=0)
    max_output_length: int = Field(8192, ge=100)
    filter_duplicates: bool = True
    dedupe_threshold: float = Field(0.95, ge=0.0, le=1.0)
    
    # Output options
    create_managed_dataset: bool = Field(True, description="Create a managed dataset from the output")
    dataset_name: Optional[str] = Field(None, description="Name for the created dataset (defaults to job name)")
    dataset_description: Optional[str] = Field(None, description="Description for the created dataset")
    
    # Topic-based generation (alternative to manual prompts)
    use_topic_generation: bool = Field(False, description="Generate prompts from a topic instead of manual input")
    topic_config: Optional[Any] = Field(None, description="TopicConfig for topic-based generation")
    
    # Appending to existing dataset
    append_to_dataset_id: Optional[str] = Field(None, description="ID of existing dataset to append to")


class PromptTemplate(BaseModel):
    """Template for generating prompts sent to the teacher model."""
    id: str
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    user_template: str  # Use {input} placeholder
    include_reasoning: bool = False
    strategy: GenerationStrategy = GenerationStrategy.STANDARD
    output_format: OutputFormat = OutputFormat.ALPACA
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Advanced template options
    pre_prompt: Optional[str] = None  # Prepended to user input
    post_prompt: Optional[str] = None  # Appended to user input
    few_shot_examples: List[Dict[str, str]] = Field(default_factory=list)


class QualityScore(BaseModel):
    """Quality assessment scores for a generated example."""
    overall: float = Field(0.0, ge=0.0, le=1.0)
    dimensions: Dict[str, float] = Field(default_factory=dict)
    passed: bool = False
    feedback: Optional[str] = None
    judge_model: Optional[str] = None
    evaluation_time_ms: int = 0


class GeneratedExample(BaseModel):
    """A single generated training example with metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str
    input: str = ""
    output: str
    
    # Reasoning and thinking
    reasoning: Optional[str] = None
    thinking: Optional[str] = None  # Content within <think> tags
    
    # Multi-turn support
    conversation: Optional[List[Dict[str, str]]] = None
    
    # Quality metadata
    quality_score: Optional[QualityScore] = None
    was_refined: bool = False
    refinement_count: int = 0
    
    # Generation metadata
    teacher_model: str
    strategy_used: GenerationStrategy = GenerationStrategy.STANDARD
    generation_attempts: int = 1
    tokens_used: int = 0
    generation_time_ms: int = 0
    
    # Self-consistency data
    consistency_samples: int = 0
    consistency_agreement: float = 0.0
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def to_training_format(self, format: OutputFormat) -> Dict[str, Any]:
        """Convert to the specified training format."""
        if format == OutputFormat.ALPACA:
            result = {
                "instruction": self.instruction,
                "input": self.input,
                "output": self.output,
            }
            if self.reasoning:
                result["reasoning"] = self.reasoning
            return result
            
        elif format == OutputFormat.THINKING:
            # Include thinking tokens in output
            if self.thinking:
                output = f"<think>\n{self.thinking}\n</think>\n<answer>\n{self.output}\n</answer>"
            else:
                output = self.output
            return {
                "instruction": self.instruction,
                "input": self.input,
                "output": output,
            }
            
        elif format == OutputFormat.SHAREGPT:
            conversations = []
            if self.conversation:
                conversations = self.conversation
            else:
                if self.instruction:
                    conversations.append({"from": "human", "value": self.instruction})
                if self.input:
                    conversations[-1]["value"] += f"\n\n{self.input}"
                conversations.append({"from": "gpt", "value": self.output})
            return {"conversations": conversations}
            
        elif format == OutputFormat.CHATML:
            messages = []
            if self.instruction:
                content = self.instruction
                if self.input:
                    content += f"\n\n{self.input}"
                messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": self.output})
            return {"messages": messages}
            
        elif format == OutputFormat.COMPLETION:
            prompt = self.instruction
            if self.input:
                prompt += f"\n\n{self.input}"
            return {"prompt": prompt, "completion": self.output}
            
        return {"instruction": self.instruction, "input": self.input, "output": self.output}


class DistillationMetrics(BaseModel):
    """Metrics collected during distillation."""
    total_prompts: int = 0
    generated_count: int = 0
    failed_count: int = 0
    filtered_count: int = 0
    refined_count: int = 0
    duplicate_count: int = 0
    
    # Quality metrics
    avg_quality_score: float = 0.0
    quality_distribution: Dict[str, int] = Field(default_factory=dict)
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Performance metrics
    total_tokens_used: int = 0
    avg_generation_time_ms: float = 0.0
    total_time_seconds: float = 0.0
    
    # Self-consistency metrics
    avg_consistency_agreement: float = 0.0
    consistency_rejections: int = 0


class DistillationJob(BaseModel):
    """A distillation job for generating training data."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    config: DistillationConfig
    template_id: Optional[str] = None
    
    # Input sources
    prompts: List[str] = Field(default_factory=list)
    prompts_file: Optional[str] = None
    seed_tasks: Optional[List[Dict[str, str]]] = None  # For seed-based generation
    
    # Targets
    target_examples: int = Field(100, ge=1)
    
    # Status
    status: DistillationStatus = DistillationStatus.PENDING
    progress: float = 0.0
    current_step: str = ""
    
    # Results
    metrics: DistillationMetrics = Field(default_factory=DistillationMetrics)
    output_dataset_id: Optional[str] = None
    output_file: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    # Legacy compatibility
    @property
    def generated_count(self) -> int:
        return self.metrics.generated_count
    
    @property
    def failed_count(self) -> int:
        return self.metrics.failed_count


# =============================================================================
# Default Prompt Templates
# =============================================================================

DEFAULT_TEMPLATES: Dict[str, PromptTemplate] = {
    "general-assistant": PromptTemplate(
        id="general-assistant",
        name="General Assistant",
        description="General-purpose instruction following with high-quality responses",
        system_prompt="""You are a helpful, harmless, and honest AI assistant. Your responses should be:
- Clear and well-structured
- Accurate and informative
- Helpful and actionable
- Free of harmful or misleading content""",
        user_template="{input}",
        include_reasoning=False,
        strategy=GenerationStrategy.STANDARD,
    ),
    
    "coding-expert": PromptTemplate(
        id="coding-expert",
        name="Expert Coding Assistant",
        description="High-quality code generation with explanations",
        system_prompt="""You are an expert software engineer with deep knowledge across programming languages and best practices.

When writing code:
1. Write clean, readable, and well-documented code
2. Follow language-specific idioms and conventions
3. Include error handling where appropriate
4. Explain your approach and key design decisions
5. Consider edge cases and performance implications""",
        user_template="{input}",
        include_reasoning=True,
        strategy=GenerationStrategy.CHAIN_OF_THOUGHT,
    ),
    
    "reasoning-cot": PromptTemplate(
        id="reasoning-cot",
        name="Chain-of-Thought Reasoning",
        description="Step-by-step reasoning with explicit thought process",
        system_prompt="""You are an expert problem solver. For every problem:
1. Break it down into smaller sub-problems
2. Solve each sub-problem step by step
3. Verify your work at each step
4. Synthesize the final answer
5. Double-check your conclusion""",
        user_template="""Let's work through this step by step:

{input}

Think carefully and show your reasoning at each step.""",
        include_reasoning=True,
        strategy=GenerationStrategy.CHAIN_OF_THOUGHT,
    ),
    
    "thinking-model": PromptTemplate(
        id="thinking-model",
        name="Thinking Model (DeepSeek R1 Style)",
        description="Structured reasoning with thinking tokens for training reasoning models",
        system_prompt="""You are a reasoning model that thinks step by step before answering.

Format your response as follows:
<think>
[Your detailed reasoning process here. Break down the problem, consider different approaches, verify your logic, and work through to the solution step by step.]
</think>
<answer>
[Your final, polished answer here]
</answer>

Your thinking should be thorough and show:
- Problem decomposition
- Consideration of approaches
- Step-by-step solution
- Self-verification""",
        user_template="{input}",
        include_reasoning=True,
        strategy=GenerationStrategy.THINKING_TOKENS,
    ),
    
    "math-reasoning": PromptTemplate(
        id="math-reasoning",
        name="Mathematical Reasoning",
        description="Mathematical problem solving with rigorous step-by-step work",
        system_prompt="""You are an expert mathematician and teacher. When solving problems:
1. Identify what is given and what is asked
2. Choose an appropriate approach or formula
3. Show every step of your calculation
4. Verify your answer makes sense
5. State the final answer clearly

Always show your work. Never skip steps.""",
        user_template="""Solve this problem step by step:

{input}

Show all your work and verify your answer.""",
        include_reasoning=True,
        strategy=GenerationStrategy.CHAIN_OF_THOUGHT,
    ),
    
    "instruction-improvement": PromptTemplate(
        id="instruction-improvement",
        name="Instruction Improvement",
        description="Generate improved versions of instructions for data augmentation",
        system_prompt="""You are an expert at improving instructions to be clearer, more specific, and more likely to elicit high-quality responses.

When improving an instruction:
1. Make it more specific and detailed
2. Add relevant context
3. Clarify any ambiguities
4. Keep the core intent intact
5. Make it actionable""",
        user_template="""Original instruction: {input}

Generate an improved version of this instruction that is clearer, more specific, and more likely to produce a high-quality response. Then provide an ideal response to the improved instruction.""",
        include_reasoning=True,
    ),
    
    "self-critique": PromptTemplate(
        id="self-critique",
        name="Self-Critique Response",
        description="Generate response with self-evaluation for quality",
        system_prompt="""You are a thoughtful assistant who evaluates your own responses.

For each response:
1. Provide your initial answer
2. Critically evaluate it for accuracy, completeness, and helpfulness
3. Identify any weaknesses or areas for improvement
4. Provide an improved final answer incorporating your critique""",
        user_template="""{input}

Provide your response, then critique it and provide an improved version.""",
        include_reasoning=True,
    ),
    
    "multi-perspective": PromptTemplate(
        id="multi-perspective",
        name="Multi-Perspective Analysis",
        description="Analyze from multiple viewpoints before concluding",
        system_prompt="""You are an analyst who considers multiple perspectives before reaching conclusions.

For every topic:
1. Consider at least 2-3 different perspectives or approaches
2. Evaluate the strengths and weaknesses of each
3. Synthesize insights from multiple viewpoints
4. Arrive at a well-reasoned conclusion""",
        user_template="""{input}

Analyze this from multiple perspectives before providing your conclusion.""",
        include_reasoning=True,
        strategy=GenerationStrategy.CHAIN_OF_THOUGHT,
    ),
    
    "expert-qa": PromptTemplate(
        id="expert-qa",
        name="Expert Q&A",
        description="Detailed expert-level answers with sources and caveats",
        system_prompt="""You are a domain expert providing thorough, accurate answers.

Your responses should:
1. Be comprehensive and well-organized
2. Include relevant background information
3. Cite general principles or well-known facts
4. Acknowledge limitations or uncertainties
5. Suggest further resources when appropriate""",
        user_template="{input}",
        include_reasoning=False,
    ),
    
    "conversation-synthesis": PromptTemplate(
        id="conversation-synthesis",
        name="Conversation Synthesis",
        description="Generate multi-turn conversation data",
        system_prompt="""You are generating a natural, helpful conversation between a user and an AI assistant.

The conversation should:
1. Flow naturally with follow-up questions
2. Show the assistant being helpful and informative
3. Include clarifying questions when appropriate
4. Demonstrate good conversation practices
5. Be educational and engaging""",
        user_template="""Generate a helpful conversation about: {input}

Format as a JSON array of message objects with "role" (user/assistant) and "content" fields.""",
        include_reasoning=False,
        strategy=GenerationStrategy.MULTI_TURN,
        output_format=OutputFormat.SHAREGPT,
    ),
}


# =============================================================================
# Teacher Model Clients
# =============================================================================

@dataclass
class GenerationResult:
    """Result from a teacher model generation."""
    content: str
    tokens_used: int = 0
    latency_ms: int = 0
    finish_reason: str = "stop"
    model: str = ""


class TeacherClient:
    """Base client for calling teacher models."""
    
    model: str = ""
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        raise NotImplementedError
    
    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> List[GenerationResult]:
        """Generate responses for multiple prompts concurrently."""
        tasks = [
            self.generate(p, system_prompt, temperature, max_tokens)
            for p in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)


class OpenAITeacher(TeacherClient):
    """OpenAI API client for GPT models."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=180.0)
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        start_time = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = await self._get_client()
        
        # Handle o1/o3 reasoning models differently (no system prompt, no temperature)
        is_reasoning_model = self.model.startswith(("o1", "o3"))
        
        payload = {
            "model": self.model,
            "messages": messages if not is_reasoning_model else [m for m in messages if m["role"] != "system"],
            "max_tokens" if not is_reasoning_model else "max_completion_tokens": max_tokens,
        }
        
        if not is_reasoning_model:
            payload["temperature"] = temperature
        
        # For reasoning models, prepend system prompt to user message
        if is_reasoning_model and system_prompt:
            user_content = f"{system_prompt}\n\n{prompt}"
            payload["messages"] = [{"role": "user", "content": user_content}]

        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        latency_ms = int((time.time() - start_time) * 1000)
        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)
        
        return GenerationResult(
            content=data["choices"][0]["message"]["content"],
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            model=self.model,
        )


class AnthropicTeacher(TeacherClient):
    """Anthropic API client for Claude models."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=180.0)
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        start_time = time.time()
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = await client.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        latency_ms = int((time.time() - start_time) * 1000)
        usage = data.get("usage", {})
        tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        
        return GenerationResult(
            content=data["content"][0]["text"],
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            finish_reason=data.get("stop_reason", "stop"),
            model=self.model,
        )


class GoogleTeacher(TeacherClient):
    """Google API client for Gemini models."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=180.0)
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        start_time = time.time()
        client = await self._get_client()
        
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        response = await client.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            params={"key": self.api_key},
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        latency_ms = int((time.time() - start_time) * 1000)
        usage = data.get("usageMetadata", {})
        tokens_used = usage.get("totalTokenCount", 0)
        
        # Extract content
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No response from Gemini")
        
        content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        return GenerationResult(
            content=content,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            finish_reason=candidates[0].get("finishReason", "STOP"),
            model=self.model,
        )


class LocalTeacher(TeacherClient):
    """Client for locally deployed models in Llama Nexus or other OpenAI-compatible APIs."""

    def __init__(self, base_url: str = "http://localhost:8700", model: str = "default"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=300.0)
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        start_time = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        latency_ms = int((time.time() - start_time) * 1000)
        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)
        
        return GenerationResult(
            content=data["choices"][0]["message"]["content"],
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            model=self.model,
        )


def create_teacher_client(config: DistillationConfig) -> TeacherClient:
    """Factory to create the appropriate teacher client."""
    if config.teacher_provider == TeacherProvider.OPENAI:
        api_key = config.teacher_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or provide teacher_api_key")
        return OpenAITeacher(api_key=api_key, model=config.teacher_model)

    elif config.teacher_provider == TeacherProvider.ANTHROPIC:
        api_key = config.teacher_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or provide teacher_api_key")
        return AnthropicTeacher(api_key=api_key, model=config.teacher_model)
    
    elif config.teacher_provider == TeacherProvider.GOOGLE:
        api_key = config.teacher_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY env var or provide teacher_api_key")
        return GoogleTeacher(api_key=api_key, model=config.teacher_model)

    elif config.teacher_provider == TeacherProvider.LOCAL:
        base_url = config.teacher_base_url or os.getenv("LOCAL_LLM_URL", "http://localhost:8700")
        return LocalTeacher(base_url=base_url, model=config.teacher_model)

    raise ValueError(f"Unknown teacher provider: {config.teacher_provider}")


# =============================================================================
# Quality Validation
# =============================================================================

class QualityValidator:
    """
    Validates quality of generated examples using LLM-as-judge and heuristics.
    
    Implements multi-factor quality assessment:
    - Correctness: Is the response factually accurate?
    - Coherence: Is the response well-structured and logical?
    - Completeness: Does the response fully address the instruction?
    - Relevance: Is the response on-topic?
    - Instruction Following: Does it follow the instruction format/constraints?
    """
    
    JUDGE_PROMPT = """You are an expert quality evaluator for AI training data. Your task is to assess the quality of a response to an instruction.

## Instruction
{instruction}

## Input (if any)
{input}

## Response to Evaluate
{response}

## Evaluation Criteria
Evaluate the response on these dimensions (score 0.0-1.0):

1. **Correctness** (0.0-1.0): Is the response factually accurate and free of errors?
2. **Coherence** (0.0-1.0): Is the response well-structured, logical, and easy to follow?
3. **Completeness** (0.0-1.0): Does the response fully and thoroughly address the instruction?
4. **Relevance** (0.0-1.0): Is the response directly on-topic and useful?
5. **Instruction Following** (0.0-1.0): Does the response follow any specific format or constraints in the instruction?

## Output Format
Respond with a JSON object:
```json
{{
    "correctness": <score>,
    "coherence": <score>,
    "completeness": <score>,
    "relevance": <score>,
    "instruction_following": <score>,
    "overall": <weighted average>,
    "passed": <true if overall >= {threshold}>,
    "feedback": "<brief explanation of scores and any issues>"
}}
```

Be strict but fair. Only high-quality examples should pass."""

    def __init__(
        self,
        client: TeacherClient,
        config: QualityConfig,
    ):
        self.client = client
        self.config = config
        self.dimension_weights = {
            QualityDimension.CORRECTNESS: 0.25,
            QualityDimension.COHERENCE: 0.15,
            QualityDimension.COMPLETENESS: 0.20,
            QualityDimension.RELEVANCE: 0.15,
            QualityDimension.INSTRUCTION_FOLLOWING: 0.25,
        }

    async def validate(
        self,
        instruction: str,
        input_text: str,
        response: str,
    ) -> QualityScore:
        """Validate a single response and return quality scores."""
        start_time = time.time()
        
        # Run heuristic checks first
        heuristic_issues = self._heuristic_check(response)
        if heuristic_issues:
            return QualityScore(
                overall=0.0,
                dimensions={},
                passed=False,
                feedback=f"Failed heuristic checks: {'; '.join(heuristic_issues)}",
                evaluation_time_ms=int((time.time() - start_time) * 1000),
            )
        
        if not self.config.use_llm_judge:
            # Use heuristics only
            return QualityScore(
                overall=0.8,
                dimensions={},
                passed=True,
                feedback="Passed heuristic checks (LLM judge disabled)",
                evaluation_time_ms=int((time.time() - start_time) * 1000),
            )
        
        # Use LLM as judge
        prompt = self.JUDGE_PROMPT.format(
            instruction=instruction,
            input=input_text or "(none)",
            response=response,
            threshold=self.config.threshold,
        )
        
        try:
            result = await self.client.generate(
                prompt=prompt,
                system_prompt="You are a strict but fair quality evaluator. Output only valid JSON.",
                temperature=0.1,
                max_tokens=500,
            )
            
            # Parse JSON from response
            scores = self._parse_judge_response(result.content)
            
            return QualityScore(
                overall=scores.get("overall", 0.0),
                dimensions={
                    "correctness": scores.get("correctness", 0.0),
                    "coherence": scores.get("coherence", 0.0),
                    "completeness": scores.get("completeness", 0.0),
                    "relevance": scores.get("relevance", 0.0),
                    "instruction_following": scores.get("instruction_following", 0.0),
                },
                passed=scores.get("passed", False),
                feedback=scores.get("feedback", ""),
                judge_model=self.client.model,
                evaluation_time_ms=int((time.time() - start_time) * 1000),
            )
            
        except Exception as e:
            logger.warning(f"LLM judge evaluation failed: {e}")
            # Fall back to heuristics
            return QualityScore(
                overall=0.7,
                dimensions={},
                passed=True,
                feedback=f"LLM judge failed, passed heuristics: {e}",
                evaluation_time_ms=int((time.time() - start_time) * 1000),
            )

    def _heuristic_check(self, response: str) -> List[str]:
        """Run heuristic quality checks on the response."""
        issues = []
        
        # Check for empty or too short
        if not response or len(response.strip()) < 10:
            issues.append("Response is empty or too short")
        
        # Check for refusals
        refusal_patterns = [
            r"^I (?:cannot|can't|am not able to|won't|refuse to)",
            r"^Sorry,? (?:I|but I) (?:cannot|can't)",
            r"^I apologize,? but I (?:cannot|can't)",
            r"^As an AI,? I (?:cannot|can't|don't)",
        ]
        for pattern in refusal_patterns:
            if re.match(pattern, response.strip(), re.IGNORECASE):
                issues.append("Response appears to be a refusal")
                break
        
        # Check for incomplete responses
        incomplete_patterns = [
            r"\.\.\.$",
            r"(?:to be continued|TBD|TODO)$",
            r"(?:I'll|Let me) (?:continue|finish|complete) (?:this|that|in|the)",
        ]
        for pattern in incomplete_patterns:
            if re.search(pattern, response.strip(), re.IGNORECASE):
                issues.append("Response appears incomplete")
                break
        
        # Check for excessive repetition
        words = response.lower().split()
        if len(words) > 20:
            word_counts = Counter(words)
            most_common = word_counts.most_common(1)
            if most_common and most_common[0][1] > len(words) * 0.3:
                issues.append("Response has excessive word repetition")
        
        # Check for potential encoding issues
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', response):
            issues.append("Response contains invalid characters")
        
        return issues

    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from the judge model."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to parse the whole response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Return default scores
        return {"overall": 0.5, "passed": False, "feedback": "Could not parse judge response"}


# =============================================================================
# Self-Consistency Validation
# =============================================================================

class SelfConsistencyValidator:
    """
    Implements self-consistency validation by generating multiple samples
    and checking for agreement.
    
    Based on:
    - Self-Consistency Improves Chain of Thought Reasoning (Wang et al.)
    - Adaptive-Consistency for dynamic sample sizes
    """
    
    def __init__(
        self,
        client: TeacherClient,
        config: SelfConsistencyConfig,
    ):
        self.client = client
        self.config = config

    async def validate_with_consistency(
        self,
        prompt: str,
        system_prompt: Optional[str],
        base_temperature: float,
        max_tokens: int,
    ) -> Tuple[str, float, List[str]]:
        """
        Generate multiple samples and return the most consistent answer.
        
        Returns:
            (best_response, agreement_score, all_responses)
        """
        # Generate samples with varying temperatures
        temp_min, temp_max = self.config.temperature_range
        temperatures = [
            temp_min + (temp_max - temp_min) * i / (self.config.num_samples - 1)
            for i in range(self.config.num_samples)
        ]
        
        # Generate all samples concurrently
        tasks = [
            self.client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temp,
                max_tokens=max_tokens,
            )
            for temp in temperatures
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful responses
        responses = []
        for result in results:
            if isinstance(result, GenerationResult):
                responses.append(result.content)
            else:
                logger.warning(f"Self-consistency sample failed: {result}")
        
        if len(responses) < 2:
            if responses:
                return responses[0], 0.0, responses
            raise ValueError("All self-consistency samples failed")
        
        # Extract answers if pattern provided
        if self.config.extract_answer_pattern:
            extracted = self._extract_answers(responses)
            if extracted:
                responses = extracted
        
        # Find most consistent response
        best_response, agreement = self._find_consensus(responses)
        
        return best_response, agreement, responses

    def _extract_answers(self, responses: List[str]) -> List[str]:
        """Extract final answers using the configured pattern."""
        pattern = self.config.extract_answer_pattern
        if not pattern:
            return responses
        
        extracted = []
        for response in responses:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted.append(match.group(1) if match.groups() else match.group())
            else:
                extracted.append(response)
        return extracted

    def _find_consensus(self, responses: List[str]) -> Tuple[str, float]:
        """Find the most agreed-upon response."""
        if not responses:
            return "", 0.0
        
        if self.config.use_semantic_similarity:
            return self._semantic_consensus(responses)
        else:
            return self._exact_consensus(responses)

    def _exact_consensus(self, responses: List[str]) -> Tuple[str, float]:
        """Find consensus using exact string matching."""
        # Normalize responses
        normalized = [r.strip().lower() for r in responses]
        counts = Counter(normalized)
        
        if not counts:
            return responses[0], 0.0
        
        most_common, count = counts.most_common(1)[0]
        agreement = count / len(responses)
        
        # Return the original (non-normalized) version
        for r in responses:
            if r.strip().lower() == most_common:
                return r, agreement
        
        return responses[0], agreement

    def _semantic_consensus(self, responses: List[str]) -> Tuple[str, float]:
        """
        Find consensus using simple semantic similarity.
        Uses character-level similarity as a lightweight approximation.
        """
        n = len(responses)
        if n == 0:
            return "", 0.0
        if n == 1:
            return responses[0], 1.0
        
        # Compute pairwise similarities
        similarities = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarities[i][j] = 1.0
                else:
                    sim = self._text_similarity(responses[i], responses[j])
                    similarities[i][j] = sim
                    similarities[j][i] = sim
        
        # Find response with highest average similarity to others
        avg_similarities = [sum(row) / n for row in similarities]
        best_idx = max(range(n), key=lambda i: avg_similarities[i])
        
        # Agreement is the average similarity of the best response to others
        agreement = (sum(similarities[best_idx]) - 1) / (n - 1)  # Exclude self-similarity
        
        return responses[best_idx], agreement

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0


# =============================================================================
# Response Refinement
# =============================================================================

class ResponseRefiner:
    """
    Refines low-quality responses through iterative improvement.
    
    Techniques:
    - Self-critique and improvement
    - Error correction
    - Completeness enhancement
    """
    
    REFINEMENT_PROMPT = """You are an expert editor improving AI responses for training data quality.

## Original Instruction
{instruction}

## Input (if any)
{input}

## Original Response (needs improvement)
{response}

## Quality Feedback
{feedback}

## Task
Rewrite the response to address the quality issues. Your improved response should:
1. Be factually accurate and complete
2. Directly address the instruction
3. Be well-structured and clear
4. Fix any issues mentioned in the feedback

Provide ONLY the improved response, no explanations or preamble."""

    SELF_CRITIQUE_PROMPT = """You are an expert who improves your own responses.

## Instruction
{instruction}

## Input (if any)
{input}

## Your Initial Response
{response}

## Task
1. Critique your initial response - identify any errors, gaps, or areas for improvement
2. Provide an improved response that addresses the issues

Format:
<critique>
[Your critique here]
</critique>
<improved>
[Your improved response here]
</improved>"""

    def __init__(
        self,
        client: TeacherClient,
        max_attempts: int = 2,
    ):
        self.client = client
        self.max_attempts = max_attempts

    async def refine(
        self,
        instruction: str,
        input_text: str,
        response: str,
        feedback: str,
    ) -> Tuple[str, int]:
        """
        Refine a response based on quality feedback.
        
        Returns:
            (refined_response, num_attempts)
        """
        prompt = self.REFINEMENT_PROMPT.format(
            instruction=instruction,
            input=input_text or "(none)",
            response=response,
            feedback=feedback,
        )
        
        result = await self.client.generate(
            prompt=prompt,
            system_prompt="You are an expert editor. Provide only the improved response.",
            temperature=0.3,
            max_tokens=4096,
        )
        
        return result.content, 1

    async def self_critique_and_improve(
        self,
        instruction: str,
        input_text: str,
        response: str,
    ) -> Tuple[str, str]:
        """
        Have the model critique and improve its own response.
        
        Returns:
            (improved_response, critique)
        """
        prompt = self.SELF_CRITIQUE_PROMPT.format(
            instruction=instruction,
            input=input_text or "(none)",
            response=response,
        )
        
        result = await self.client.generate(
            prompt=prompt,
            system_prompt="You are a thoughtful assistant who improves your responses through self-reflection.",
            temperature=0.4,
            max_tokens=4096,
        )
        
        # Parse the response
        content = result.content
        
        critique = ""
        improved = content
        
        critique_match = re.search(r'<critique>(.*?)</critique>', content, re.DOTALL)
        if critique_match:
            critique = critique_match.group(1).strip()
        
        improved_match = re.search(r'<improved>(.*?)</improved>', content, re.DOTALL)
        if improved_match:
            improved = improved_match.group(1).strip()
        
        return improved, critique


# =============================================================================
# Thinking Token Processor
# =============================================================================

class ThinkingTokenProcessor:
    """
    Processes and validates thinking token format (DeepSeek R1 style).
    
    Ensures generated responses have proper:
    - <think>...</think> reasoning section
    - <answer>...</answer> final response section
    """
    
    def __init__(self, config: ThinkingConfig):
        self.config = config

    def format_prompt_for_thinking(self, prompt: str) -> str:
        """Add instructions for thinking token format."""
        return f"""Think through this step by step before answering.

Format your response as:
{self.config.think_tag}
[Your detailed step-by-step reasoning here]
{self.config.think_end_tag}
{self.config.answer_tag}
[Your final answer here]
{self.config.answer_end_tag}

Question/Task: {prompt}"""

    def parse_thinking_response(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse a response to extract thinking and answer sections.
        
        Returns:
            (thinking_content, answer_content)
        """
        thinking = None
        answer = None
        
        # Extract thinking section
        think_pattern = re.compile(
            rf'{re.escape(self.config.think_tag)}(.*?){re.escape(self.config.think_end_tag)}',
            re.DOTALL | re.IGNORECASE
        )
        think_match = think_pattern.search(response)
        if think_match:
            thinking = think_match.group(1).strip()
        
        # Extract answer section
        answer_pattern = re.compile(
            rf'{re.escape(self.config.answer_tag)}(.*?){re.escape(self.config.answer_end_tag)}',
            re.DOTALL | re.IGNORECASE
        )
        answer_match = answer_pattern.search(response)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        # If no structured format, try to infer
        if not thinking and not answer:
            # Check if response has any structure
            if self.config.think_tag.lower() in response.lower():
                # Partial structure - extract what we can
                parts = response.split(self.config.think_end_tag, 1)
                if len(parts) > 1:
                    thinking = parts[0].replace(self.config.think_tag, "").strip()
                    answer = parts[1].replace(self.config.answer_tag, "").replace(
                        self.config.answer_end_tag, ""
                    ).strip()
        
        return thinking, answer

    def validate_thinking(self, thinking: str) -> Tuple[bool, str]:
        """
        Validate the thinking section meets quality requirements.
        
        Returns:
            (is_valid, feedback)
        """
        if not thinking:
            return False, "No thinking section found"
        
        # Check minimum length
        if len(thinking.split()) < self.config.min_thinking_steps * 5:
            return False, "Thinking section is too short"
        
        # Check for step structure if required
        if self.config.require_structured_steps:
            step_indicators = [
                r'step \d',
                r'first|second|third|fourth|fifth',
                r'\d\.',
                r'^\s*[-*]',
            ]
            has_steps = any(
                re.search(pattern, thinking, re.IGNORECASE | re.MULTILINE)
                for pattern in step_indicators
            )
            if not has_steps:
                return False, "Thinking section lacks step-by-step structure"
        
        return True, "Valid thinking section"

    def reconstruct_with_tokens(self, thinking: str, answer: str) -> str:
        """Reconstruct a properly formatted thinking token response."""
        return f"""{self.config.think_tag}
{thinking}
{self.config.think_end_tag}
{self.config.answer_tag}
{answer}
{self.config.answer_end_tag}"""


# =============================================================================
# Duplicate Detection
# =============================================================================

class DuplicateDetector:
    """Detects and filters duplicate or near-duplicate examples."""
    
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self._hashes: Dict[str, str] = {}  # hash -> example_id
        self._texts: Dict[str, str] = {}  # example_id -> text

    def is_duplicate(self, example_id: str, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text is a duplicate of an existing example.
        
        Returns:
            (is_duplicate, duplicate_of_id)
        """
        # Normalize text
        normalized = self._normalize(text)
        
        # Check exact hash match
        text_hash = hashlib.md5(normalized.encode()).hexdigest()
        if text_hash in self._hashes:
            return True, self._hashes[text_hash]
        
        # Check near-duplicate using similarity
        for existing_id, existing_text in self._texts.items():
            similarity = self._compute_similarity(normalized, existing_text)
            if similarity >= self.threshold:
                return True, existing_id
        
        # Not a duplicate - register it
        self._hashes[text_hash] = example_id
        self._texts[example_id] = normalized
        
        return False, None

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase and remove extra whitespace
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0

    def clear(self):
        """Clear all stored hashes and texts."""
        self._hashes.clear()
        self._texts.clear()


# =============================================================================
# Distillation Manager
# =============================================================================

class DistillationManager:
    """
    World-class distillation manager for generating high-quality training data.
    
    Features:
    - Multiple generation strategies (CoT, thinking tokens, self-consistency)
    - LLM-as-judge quality validation
    - Automatic response refinement
    - Duplicate detection and filtering
    - Comprehensive metrics tracking
    """

    def __init__(self, output_dir: Optional[Path] = None, dataset_store: Optional[DatasetStore] = None):
        self.output_dir = output_dir or Path(
            os.getenv("DISTILLATION_OUTPUT_DIR", "data/distillation")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, DistillationJob] = {}
        self.templates: Dict[str, PromptTemplate] = dict(DEFAULT_TEMPLATES)
        self._running_jobs: Dict[str, asyncio.Task] = {}
        self.dataset_store = dataset_store
        
        # Persist jobs to disk
        self._jobs_file = self.output_dir / "jobs.json"
        self._load_jobs()

    def _load_jobs(self):
        """Load persisted jobs from disk."""
        if self._jobs_file.exists():
            try:
                with open(self._jobs_file, "r") as f:
                    data = json.load(f)
                    for job_data in data:
                        job = DistillationJob(**job_data)
                        self.jobs[job.id] = job
            except Exception as e:
                logger.warning(f"Failed to load jobs: {e}")

    def _save_jobs(self):
        """Persist jobs to disk."""
        try:
            with open(self._jobs_file, "w") as f:
                json.dump(
                    [job.dict() for job in self.jobs.values()],
                    f,
                    default=str,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"Failed to save jobs: {e}")

    def create_job(self, job: DistillationJob) -> DistillationJob:
        """Create a new distillation job."""
        if not job.id:
            job.id = str(uuid.uuid4())
        job.output_file = str(self.output_dir / f"{job.id}.jsonl")
        self.jobs[job.id] = job
        self._save_jobs()
        logger.info("Created distillation job", extra={"job_id": job.id, "name": job.name})
        return job

    def get_job(self, job_id: str) -> Optional[DistillationJob]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[DistillationJob]:
        return list(self.jobs.values())

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its output file."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status == DistillationStatus.RUNNING:
            self.cancel_job(job_id)
        
        if job.output_file:
            output_path = Path(job.output_file)
            if output_path.exists():
                output_path.unlink()
        
        del self.jobs[job_id]
        self._save_jobs()
        return True

    def list_templates(self) -> List[PromptTemplate]:
        return list(self.templates.values())

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        return self.templates.get(template_id)

    def add_template(self, template: PromptTemplate) -> PromptTemplate:
        self.templates[template.id] = template
        return template

    def delete_template(self, template_id: str) -> bool:
        """Delete a custom template (cannot delete defaults)."""
        if template_id in DEFAULT_TEMPLATES:
            return False
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False

    async def run_job(self, job_id: str) -> None:
        """Run a distillation job with full quality pipeline."""
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(f"Job {job_id} not found")

        job.status = DistillationStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.current_step = "Initializing"
        start_time = time.time()

        try:
            # Initialize components
            client = create_teacher_client(job.config)
            template = self.templates.get(job.template_id) if job.template_id else None
            system_prompt = template.system_prompt if template else job.config.system_prompt
            
            # Initialize validators and processors
            quality_validator = None
            if job.config.quality.enabled:
                quality_validator = QualityValidator(client, job.config.quality)
            
            consistency_validator = None
            if job.config.self_consistency.enabled:
                consistency_validator = SelfConsistencyValidator(client, job.config.self_consistency)
            
            refiner = None
            if job.config.quality.enabled and job.config.quality.max_refinement_attempts > 0:
                refiner = ResponseRefiner(client, job.config.quality.max_refinement_attempts)
            
            thinking_processor = None
            if job.config.strategy == GenerationStrategy.THINKING_TOKENS or job.config.thinking.enabled:
                thinking_processor = ThinkingTokenProcessor(job.config.thinking)
            
            duplicate_detector = DuplicateDetector(job.config.dedupe_threshold) if job.config.filter_duplicates else None

            # Load prompts
            job.current_step = "Loading prompts"
            prompts = await self._load_prompts(job)
            
            if not prompts:
                raise ValueError("No prompts provided for distillation")
            
            job.metrics.total_prompts = len(prompts)

            # Rate limiting
            delay = 60.0 / job.config.rate_limit_rpm
            
            # Process prompts
            job.current_step = "Generating examples"
            
            with open(job.output_file, "w") as out_f:
                for i, prompt in enumerate(prompts[:job.target_examples]):
                    if job.status == DistillationStatus.CANCELLED:
                        break

                    try:
                        example = await self._generate_example(
                            prompt=prompt,
                            template=template,
                            system_prompt=system_prompt,
                            client=client,
                            config=job.config,
                            quality_validator=quality_validator,
                            consistency_validator=consistency_validator,
                            refiner=refiner,
                            thinking_processor=thinking_processor,
                            duplicate_detector=duplicate_detector,
                            job=job,
                        )
                        
                        if example:
                            # Write in the specified format
                            output_data = example.to_training_format(job.config.output_format)
                            output_data["_metadata"] = {
                                "id": example.id,
                                "quality_score": example.quality_score.overall if example.quality_score else None,
                                "was_refined": example.was_refined,
                                "strategy": example.strategy_used.value,
                                "teacher_model": example.teacher_model,
                                "timestamp": example.timestamp.isoformat(),
                            }
                            out_f.write(json.dumps(output_data, default=str) + "\n")
                            out_f.flush()
                            
                            job.metrics.generated_count += 1
                            
                            # Update quality metrics
                            if example.quality_score:
                                self._update_quality_metrics(job, example.quality_score)
                            if example.was_refined:
                                job.metrics.refined_count += 1
                            if example.consistency_samples > 0:
                                job.metrics.avg_consistency_agreement = (
                                    (job.metrics.avg_consistency_agreement * (job.metrics.generated_count - 1) +
                                     example.consistency_agreement) / job.metrics.generated_count
                                )
                        
                    except Exception as e:
                        logger.warning(
                            "Failed to generate example",
                            extra={"prompt": prompt[:100], "error": str(e)},
                        )
                        job.metrics.failed_count += 1

                    # Update progress
                    job.progress = ((i + 1) / min(len(prompts), job.target_examples)) * 100
                    self._save_jobs()
                    
                    # Rate limiting
                    await asyncio.sleep(delay)

            # Create managed dataset if requested
            if job.config.create_managed_dataset and self.dataset_store and job.metrics.generated_count > 0:
                try:
                    dataset = await self._create_dataset_from_job(job)
                    job.output_dataset_id = dataset.id
                    logger.info(
                        "Created managed dataset from distillation output",
                        extra={"job_id": job_id, "dataset_id": dataset.id, "dataset_name": dataset.name}
                    )
                except Exception as e:
                    import traceback
                    logger.warning(
                        f"Failed to create managed dataset from distillation output: {str(e)}",
                        extra={"job_id": job_id, "error": str(e), "traceback": traceback.format_exc()}
                    )

            # Finalize
            job.status = DistillationStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.metrics.total_time_seconds = time.time() - start_time
            job.current_step = "Completed"
            self._save_jobs()
            
            logger.info(
                "Distillation job completed",
                extra={
                    "job_id": job_id,
                    "generated": job.metrics.generated_count,
                    "failed": job.metrics.failed_count,
                    "filtered": job.metrics.filtered_count,
                    "refined": job.metrics.refined_count,
                    "avg_quality": job.metrics.avg_quality_score,
                    "time_seconds": job.metrics.total_time_seconds,
                },
            )

        except Exception as e:
            job.status = DistillationStatus.FAILED
            job.error = str(e)
            job.metrics.total_time_seconds = time.time() - start_time
            self._save_jobs()
            logger.error("Distillation job failed", extra={"job_id": job_id, "error": str(e)})
            raise

    async def _load_prompts(self, job: DistillationJob) -> List[str]:
        """Load prompts from all configured sources."""
        prompts = list(job.prompts)
        
        # Topic-based generation (generate prompts from topic config)
        if job.config.use_topic_generation and job.config.topic_config:
            try:
                from .topic_generator import TopicGenerator, TopicConfig
                
                topic_generator = TopicGenerator()
                client = create_teacher_client(job.config)
                
                # Parse topic config if it's a dict
                if isinstance(job.config.topic_config, dict):
                    topic_config = TopicConfig(**job.config.topic_config)
                else:
                    topic_config = job.config.topic_config
                
                logger.info(
                    "Generating prompts from topic",
                    extra={"topic": topic_config.topic, "target": topic_config.target_examples}
                )
                
                generated_prompts = await topic_generator.generate_prompts(
                    topic_config, client
                )
                prompts.extend(generated_prompts)
                
                logger.info(
                    "Generated prompts from topic",
                    extra={"count": len(generated_prompts)}
                )
            except Exception as e:
                logger.warning(f"Failed to generate prompts from topic: {e}")
        
        # Load from file
        if job.prompts_file and Path(job.prompts_file).exists():
            with open(job.prompts_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Support JSON lines
                        if line.startswith("{"):
                            try:
                                data = json.loads(line)
                                prompt = data.get("prompt") or data.get("instruction") or data.get("input")
                                if prompt:
                                    prompts.append(prompt)
                            except json.JSONDecodeError:
                                prompts.append(line)
                        else:
                            prompts.append(line)
        
        # Generate from seed tasks
        if job.seed_tasks:
            for seed in job.seed_tasks:
                instruction = seed.get("instruction", "")
                input_text = seed.get("input", "")
                if instruction:
                    if input_text:
                        prompts.append(f"{instruction}\n\nInput: {input_text}")
                    else:
                        prompts.append(instruction)
        
        return prompts

    async def _generate_example(
        self,
        prompt: str,
        template: Optional[PromptTemplate],
        system_prompt: Optional[str],
        client: TeacherClient,
        config: DistillationConfig,
        quality_validator: Optional[QualityValidator],
        consistency_validator: Optional[SelfConsistencyValidator],
        refiner: Optional[ResponseRefiner],
        thinking_processor: Optional[ThinkingTokenProcessor],
        duplicate_detector: Optional[DuplicateDetector],
        job: DistillationJob,
    ) -> Optional[GeneratedExample]:
        """Generate a single high-quality example with full pipeline."""
        
        # Format prompt with template
        if template:
            formatted_prompt = template.user_template.replace("{input}", prompt)
            if template.pre_prompt:
                formatted_prompt = f"{template.pre_prompt}\n\n{formatted_prompt}"
            if template.post_prompt:
                formatted_prompt = f"{formatted_prompt}\n\n{template.post_prompt}"
            
            # Add few-shot examples if available
            if template.few_shot_examples:
                few_shot = "\n\n".join([
                    f"Example:\nInput: {ex.get('input', ex.get('instruction', ''))}\nOutput: {ex.get('output', '')}"
                    for ex in template.few_shot_examples[:3]
                ])
                formatted_prompt = f"{few_shot}\n\nNow respond to:\n{formatted_prompt}"
        else:
            formatted_prompt = prompt

        # Apply thinking token format if needed
        if thinking_processor:
            formatted_prompt = thinking_processor.format_prompt_for_thinking(formatted_prompt)

        # Generate response based on strategy
        strategy = template.strategy if template else config.strategy
        response_content = ""
        thinking_content = None
        consistency_samples = 0
        consistency_agreement = 0.0
        generation_attempts = 1
        tokens_used = 0
        generation_time_ms = 0

        if strategy == GenerationStrategy.SELF_CONSISTENCY and consistency_validator:
            # Use self-consistency with multiple samples
            response_content, consistency_agreement, samples = await consistency_validator.validate_with_consistency(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                base_temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            consistency_samples = len(samples)
            
            if consistency_agreement < config.self_consistency.agreement_threshold:
                job.metrics.consistency_rejections += 1
                job.metrics.filtered_count += 1
                return None
                
        else:
            # Standard generation (with retries)
            for attempt in range(config.max_retries + 1):
                try:
                    result = await client.generate(
                        prompt=formatted_prompt,
                        system_prompt=system_prompt,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    )
                    response_content = result.content
                    tokens_used = result.tokens_used
                    generation_time_ms = result.latency_ms
                    break
                except Exception as e:
                    generation_attempts += 1
                    if attempt == config.max_retries:
                        raise
                    await asyncio.sleep(config.retry_delay * (2 ** attempt))

        # Process thinking tokens if applicable
        if thinking_processor and strategy == GenerationStrategy.THINKING_TOKENS:
            thinking_content, answer_content = thinking_processor.parse_thinking_response(response_content)
            
            if thinking_content and answer_content:
                # Validate thinking quality
                is_valid, feedback = thinking_processor.validate_thinking(thinking_content)
                if not is_valid:
                    logger.debug(f"Thinking validation failed: {feedback}")
                    # Try to regenerate with more explicit instructions
                    # For now, use what we have
                response_content = answer_content
            else:
                # No structured thinking found - use response as-is
                thinking_content = None

        # Length filtering
        if len(response_content) < config.min_output_length:
            job.metrics.filtered_count += 1
            return None
        if len(response_content) > config.max_output_length:
            response_content = response_content[:config.max_output_length]

        # Duplicate detection
        example_id = str(uuid.uuid4())
        if duplicate_detector:
            is_dup, dup_of = duplicate_detector.is_duplicate(example_id, response_content)
            if is_dup:
                job.metrics.duplicate_count += 1
                job.metrics.filtered_count += 1
                return None

        # Quality validation
        quality_score = None
        was_refined = False
        refinement_count = 0

        if quality_validator:
            quality_score = await quality_validator.validate(
                instruction=prompt,
                input_text="",
                response=response_content,
            )
            
            # Attempt refinement if needed
            if not quality_score.passed and refiner and config.quality.max_refinement_attempts > 0:
                for _ in range(config.quality.max_refinement_attempts):
                    refined_response, _ = await refiner.refine(
                        instruction=prompt,
                        input_text="",
                        response=response_content,
                        feedback=quality_score.feedback or "Please improve quality",
                    )
                    refinement_count += 1
                    
                    # Re-validate
                    quality_score = await quality_validator.validate(
                        instruction=prompt,
                        input_text="",
                        response=refined_response,
                    )
                    
                    if quality_score.passed:
                        response_content = refined_response
                        was_refined = True
                        break
            
            # Final check - reject if still failing
            if not quality_score.passed:
                job.metrics.filtered_count += 1
                return None

        # Create the example
        example = GeneratedExample(
            id=example_id,
            instruction=prompt,
            output=response_content,
            thinking=thinking_content,
            quality_score=quality_score,
            was_refined=was_refined,
            refinement_count=refinement_count,
            teacher_model=config.teacher_model,
            strategy_used=strategy,
            generation_attempts=generation_attempts,
            tokens_used=tokens_used,
            generation_time_ms=generation_time_ms,
            consistency_samples=consistency_samples,
            consistency_agreement=consistency_agreement,
        )
        
        return example

    def _update_quality_metrics(self, job: DistillationJob, score: QualityScore):
        """Update job metrics with quality scores."""
        n = job.metrics.generated_count
        
        # Update running average
        if n == 1:
            job.metrics.avg_quality_score = score.overall
        else:
            job.metrics.avg_quality_score = (
                (job.metrics.avg_quality_score * (n - 1) + score.overall) / n
            )
        
        # Update dimension averages
        for dim, value in score.dimensions.items():
            if dim not in job.metrics.dimension_scores:
                job.metrics.dimension_scores[dim] = value
            else:
                old = job.metrics.dimension_scores[dim]
                job.metrics.dimension_scores[dim] = ((old * (n - 1)) + value) / n
        
        # Update distribution
        bucket = f"{int(score.overall * 10) / 10:.1f}"
        job.metrics.quality_distribution[bucket] = job.metrics.quality_distribution.get(bucket, 0) + 1

    def start_job(self, job_id: str) -> None:
        """Start a distillation job in background."""
        if job_id in self._running_jobs:
            return
        
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(f"Job {job_id} not found")
        
        if job.status not in [DistillationStatus.PENDING, DistillationStatus.PAUSED]:
            raise ValueError(f"Cannot start job in status {job.status}")
        
        task = asyncio.create_task(self.run_job(job_id))
        self._running_jobs[job_id] = task
        
        # Add callback to handle exceptions in background task
        def _handle_task_result(t: asyncio.Task):
            try:
                exc = t.exception()
                if exc:
                    import traceback
                    logger.error(
                        f"Distillation job {job_id} background task failed",
                        extra={"error": str(exc), "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}
                    )
                    # Update job status if not already updated
                    if job_id in self.jobs and self.jobs[job_id].status == DistillationStatus.RUNNING:
                        self.jobs[job_id].status = DistillationStatus.FAILED
                        self.jobs[job_id].error = str(exc)
                        self._save_jobs()
            except asyncio.CancelledError:
                pass
            finally:
                if job_id in self._running_jobs:
                    del self._running_jobs[job_id]
        
        task.add_done_callback(_handle_task_result)

    def pause_job(self, job_id: str) -> bool:
        """Pause a running job (can be resumed)."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.status == DistillationStatus.RUNNING:
            job.status = DistillationStatus.PAUSED
            self._save_jobs()
            return True
        return False

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running distillation job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.status in [DistillationStatus.RUNNING, DistillationStatus.PAUSED]:
            job.status = DistillationStatus.CANCELLED
            if job_id in self._running_jobs:
                self._running_jobs[job_id].cancel()
                del self._running_jobs[job_id]
            self._save_jobs()
            return True
        return False

    def get_generated_examples(
        self, job_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get generated examples from a job."""
        job = self.jobs.get(job_id)
        if not job or not job.output_file:
            return []

        output_path = Path(job.output_file)
        if not output_path.exists():
            return []

        examples = []
        with open(output_path, "r") as f:
            for i, line in enumerate(f):
                if i < offset:
                    continue
                if i >= offset + limit:
                    break
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return examples

    def get_job_statistics(self, job_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a job."""
        job = self.jobs.get(job_id)
        if not job:
            return {}
        
        stats = {
            "id": job.id,
            "name": job.name,
            "status": job.status.value,
            "progress": job.progress,
            "metrics": job.metrics.dict(),
            "config": {
                "strategy": job.config.strategy.value,
                "output_format": job.config.output_format.value,
                "teacher_model": job.config.teacher_model,
                "quality_enabled": job.config.quality.enabled,
                "self_consistency_enabled": job.config.self_consistency.enabled,
            },
            "timing": {
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "total_time_seconds": job.metrics.total_time_seconds,
            },
        }
        
        # Calculate rates
        if job.metrics.total_time_seconds > 0:
            stats["rates"] = {
                "examples_per_minute": job.metrics.generated_count / (job.metrics.total_time_seconds / 60),
                "tokens_per_second": job.metrics.total_tokens_used / job.metrics.total_time_seconds,
            }
        
        return stats

    async def export_dataset(
        self,
        job_id: str,
        output_format: OutputFormat,
        output_path: Optional[str] = None,
        include_metadata: bool = False,
    ) -> str:
        """Export generated examples to a specific format."""
        job = self.jobs.get(job_id)
        if not job or not job.output_file:
            raise ValueError(f"Job {job_id} not found or has no output")
        
        if not output_path:
            output_path = str(self.output_dir / f"{job_id}_{output_format.value}.jsonl")
        
        examples = self.get_generated_examples(job_id, limit=100000)
        
        with open(output_path, "w") as f:
            for ex in examples:
                # Convert to target format
                formatted = {}
                
                if output_format == OutputFormat.ALPACA:
                    formatted = {
                        "instruction": ex.get("instruction", ""),
                        "input": ex.get("input", ""),
                        "output": ex.get("output", ""),
                    }
                elif output_format == OutputFormat.SHAREGPT:
                    formatted = {
                        "conversations": ex.get("conversations", [
                            {"from": "human", "value": ex.get("instruction", "")},
                            {"from": "gpt", "value": ex.get("output", "")},
                        ])
                    }
                elif output_format == OutputFormat.CHATML:
                    formatted = {
                        "messages": ex.get("messages", [
                            {"role": "user", "content": ex.get("instruction", "")},
                            {"role": "assistant", "content": ex.get("output", "")},
                        ])
                    }
                elif output_format == OutputFormat.COMPLETION:
                    formatted = {
                        "prompt": ex.get("instruction", ""),
                        "completion": ex.get("output", ""),
                    }
                
                if include_metadata and "_metadata" in ex:
                    formatted["_metadata"] = ex["_metadata"]
                
                f.write(json.dumps(formatted) + "\n")
        
        return output_path

    async def _create_dataset_from_job(self, job: DistillationJob) -> Dataset:
        """Create a managed dataset from a completed distillation job."""
        if not self.dataset_store:
            raise ValueError("Dataset store not available")
        
        if not job.output_file or not Path(job.output_file).exists():
            raise ValueError("Job output file not found")
        
        # Handle appending to existing dataset
        if job.config.append_to_dataset_id:
            logger.info(f"Appending job {job.id} output to dataset {job.config.append_to_dataset_id}")
            existing_dataset = self.dataset_store.get(job.config.append_to_dataset_id)
            if not existing_dataset:
                raise ValueError(f"Dataset {job.config.append_to_dataset_id} not found")
            
            target_path = Path(existing_dataset.file_path)
            if not target_path.exists():
                raise ValueError(f"Original file for dataset {existing_dataset.name} not found at {target_path}")
            
            # Append content
            # Ensure we start on a new line if not already
            has_newline = False
            if target_path.stat().st_size > 0:
                with open(target_path, "rb") as f:
                    f.seek(-1, 2)
                    if f.read(1) == b"\n":
                        has_newline = True
            
            with open(target_path, "a") as f_out, open(job.output_file, "r") as f_in:
                if target_path.stat().st_size > 0 and not has_newline:
                    f_out.write("\n")
                f_out.write(f_in.read())
            
            # Update dataset metadata
            existing_dataset.num_examples += job.metrics.generated_count
            existing_dataset.size_bytes = target_path.stat().st_size
            
            # Update lineage metadata
            if "append_history" not in existing_dataset.metadata:
                existing_dataset.metadata["append_history"] = []
                
            existing_dataset.metadata["append_history"].append({
                "job_id": job.id,
                "timestamp": datetime.utcnow().isoformat(),
                "added_examples": job.metrics.generated_count,
                "teacher_model": job.config.teacher_model,
                "topic": job.config.topic_config.topic if job.config.topic_config else None
            })
            
            return self.dataset_store.upsert(existing_dataset)

        # Create new dataset logic
        dataset_name = job.config.dataset_name or f"{job.name} (Distilled)"
        dataset_description = (
            job.config.dataset_description or 
            f"Dataset generated from distillation job '{job.name}' using {job.config.teacher_model}"
        )
        
        # Map output format to dataset format
        format_mapping = {
            OutputFormat.ALPACA: DatasetFormat.ALPACA,
            OutputFormat.SHAREGPT: DatasetFormat.SHAREGPT,
            OutputFormat.CHATML: DatasetFormat.CHATML,
            OutputFormat.COMPLETION: DatasetFormat.COMPLETION,
        }
        
        dataset_format = format_mapping.get(job.config.output_format, DatasetFormat.ALPACA)
        
        # Create dataset object
        dataset = Dataset(
            id=str(uuid.uuid4()),
            name=dataset_name,
            description=dataset_description,
            format=dataset_format,
            status=DatasetStatus.READY,
            file_path="",  # Will be set after copying
            num_examples=job.metrics.generated_count,
            total_tokens=0,  # Will be calculated during training
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            statistics={
                "source": "distillation",
                "distillation_job_id": job.id,
                "teacher_model": job.config.teacher_model,
                "teacher_provider": job.config.teacher_provider.value,
                "generation_strategy": job.config.strategy.value,
                "output_format": job.config.output_format.value,
                "quality_metrics": {
                    "avg_quality_score": job.metrics.avg_quality_score,
                    "generated_count": job.metrics.generated_count,
                    "failed_count": job.metrics.failed_count,
                    "filtered_count": job.metrics.filtered_count,
                    "refined_count": job.metrics.refined_count,
                }
            }
        )
        
        # Copy the output file to the dataset store location
        target_path = self.dataset_store.dataset_path(dataset.id, f"{job.id}.jsonl")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file content
        import shutil
        shutil.copy2(job.output_file, target_path)
        
        # Update dataset with actual file path
        dataset.file_path = str(target_path)
        
        # Save to dataset store
        return self.dataset_store.upsert(dataset)

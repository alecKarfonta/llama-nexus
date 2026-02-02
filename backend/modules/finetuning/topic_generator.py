"""
Topic-based dataset generation module.

Generates diverse, high-quality training prompts and examples from topic descriptions
using meta-prompting with teacher LLMs.

Features:
- Auto-generation of diverse prompts from a topic description
- Multiple generation modes (Q&A, instructions, trivia, troubleshooting, etc.)
- Configurable multi-pass refinement for higher quality
- Difficulty stratification (beginner, intermediate, advanced)
- Subtopic expansion for broader coverage
"""

import asyncio
import json
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from enhanced_logger import enhanced_logger as logger


# =============================================================================
# Enums and Constants
# =============================================================================

class TopicGenerationMode(str, Enum):
    """Modes for generating topic-based training data."""
    QA_PAIRS = "qa_pairs"              # Question-answer pairs
    INSTRUCTIONS = "instructions"       # Instruction-following tasks
    TRIVIA = "trivia"                  # Factual trivia Q&A
    TROUBLESHOOTING = "troubleshooting" # Diagnostic/problem-solving
    CODE_EXAMPLES = "code_examples"    # Programming examples
    EXPLANATIONS = "explanations"      # Concept explanations
    CONVERSATIONS = "conversations"    # Multi-turn dialogue
    TOOL_CALLING = "tool_calling"      # Function/tool usage examples
    MIXED = "mixed"                    # Combination of above


# Mode descriptions for UI and prompt generation
MODE_DESCRIPTIONS = {
    TopicGenerationMode.QA_PAIRS: {
        "name": "Question & Answer Pairs",
        "description": "Generate diverse questions with comprehensive answers",
        "example": "Q: How do you check oil levels? A: First, ensure the engine is cool...",
        "ideal_for": "Knowledge bases, FAQ systems, educational content",
    },
    TopicGenerationMode.INSTRUCTIONS: {
        "name": "Instruction Following",
        "description": "Generate task instructions with step-by-step solutions",
        "example": "Instruction: Write a function to sort a list. Response: Here's a Python function...",
        "ideal_for": "Task-oriented assistants, coding helpers, procedural guides",
    },
    TopicGenerationMode.TRIVIA: {
        "name": "Trivia & Facts",
        "description": "Generate factual questions with concise, accurate answers",
        "example": "Q: What is the capital of France? A: Paris",
        "ideal_for": "Knowledge testing, quiz generation, fact retrieval",
    },
    TopicGenerationMode.TROUBLESHOOTING: {
        "name": "Troubleshooting & Diagnostics",
        "description": "Generate problem descriptions with diagnostic steps and solutions",
        "example": "Problem: Car won't start. Diagnosis: Check battery connections...",
        "ideal_for": "Technical support, repair guides, diagnostic assistants",
    },
    TopicGenerationMode.CODE_EXAMPLES: {
        "name": "Code Examples",
        "description": "Generate programming tasks with working code solutions",
        "example": "Task: Implement binary search. Solution: def binary_search(arr, target)...",
        "ideal_for": "Coding assistants, programming tutors, code generation",
    },
    TopicGenerationMode.EXPLANATIONS: {
        "name": "Concept Explanations",
        "description": "Generate concept questions with detailed explanations",
        "example": "Q: What is recursion? A: Recursion is a programming technique where...",
        "ideal_for": "Educational content, concept teaching, learning assistants",
    },
    TopicGenerationMode.CONVERSATIONS: {
        "name": "Multi-turn Conversations",
        "description": "Generate realistic dialogue flows with follow-up questions",
        "example": "[User] How do I start? [Assistant] Let me help... [User] What's next?",
        "ideal_for": "Chatbots, conversational AI, dialogue systems",
    },
    TopicGenerationMode.TOOL_CALLING: {
        "name": "Tool Calling / Function Use",
        "description": "Generate scenarios requiring specific tool/function calls",
        "example": "User: Check weather in London. Assistant: { \"tool\": \"get_weather\", \"args\": { \"city\": \"London\" } }",
        "ideal_for": "Agentic workflows, function calling models, API integration",
    },
    TopicGenerationMode.MIXED: {
        "name": "Mixed Formats",
        "description": "Generate a diverse mix of all formats for variety",
        "example": "Various formats combined for comprehensive coverage",
        "ideal_for": "General-purpose assistants, diverse training sets",
    },
}


# Pre-built topic templates for common use cases
TOPIC_TEMPLATES = [
    {
        "id": "tech-support",
        "name": "Technical Support",
        "mode": TopicGenerationMode.TROUBLESHOOTING,
        "description": "Diagnostic and troubleshooting scenarios",
        "example_topics": ["Car engine problems", "Computer networking issues", "Plumbing fixes"],
    },
    {
        "id": "programming",
        "name": "Programming Tutorial",
        "mode": TopicGenerationMode.CODE_EXAMPLES,
        "description": "Code examples and programming tasks",
        "example_topics": ["Python async programming", "React hooks", "SQL optimization"],
    },
    {
        "id": "domain-knowledge",
        "name": "Domain Knowledge",
        "mode": TopicGenerationMode.TRIVIA,
        "description": "Factual knowledge and trivia",
        "example_topics": ["Warhammer 40K lore", "Medieval history", "Marine biology"],
    },
    {
        "id": "task-instructions",
        "name": "Task Instructions",
        "mode": TopicGenerationMode.INSTRUCTIONS,
        "description": "Step-by-step task completion",
        "example_topics": ["Writing professional emails", "Cooking recipes", "DIY projects"],
    },
    {
        "id": "educational-qa",
        "name": "Educational Q&A",
        "mode": TopicGenerationMode.QA_PAIRS,
        "description": "Educational question-answer pairs",
        "example_topics": ["High school chemistry", "Business economics", "Psychology basics"],
    },
]


# =============================================================================
# Configuration Models
# =============================================================================

class RefinementConfig(BaseModel):
    """Configuration for multi-pass example refinement."""
    enabled: bool = True
    max_passes: int = Field(2, ge=1, le=5, description="Maximum refinement iterations")
    quality_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Quality score to stop refining")
    refine_instructions: bool = True
    refine_responses: bool = True
    refine_responses: bool = True
    add_reasoning: bool = False  # Add chain-of-thought reasoning during refinement


class TopicConfig(BaseModel):
    """Configuration for topic-based dataset generation."""
    topic: str = Field(..., description="Main topic (e.g., 'Car diagnostics')")
    subtopics: Optional[List[str]] = Field(
        None, 
        description="Specific subtopics to cover (auto-generated if not provided)"
    )
    mode: TopicGenerationMode = TopicGenerationMode.QA_PAIRS
    target_examples: int = Field(100, ge=10, le=1000, description="Target number of examples")
    
    # Tool definitions for tool calling mode (JSON schema or text description)
    tool_definitions: Optional[str] = Field(None, description="JSON Schema or description of available tools")
    
    # Quality and diversity controls
    difficulty_levels: List[str] = Field(
        default=["beginner", "intermediate", "advanced"],
        description="Difficulty levels to include"
    )
    diversity_temperature: float = Field(
        0.9, ge=0.1, le=1.5,
        description="Higher values = more diverse prompts"
    )
    include_edge_cases: bool = Field(
        True, 
        description="Include unusual or edge case scenarios"
    )
    
    # Refinement settings
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    
    # Auto-expansion
    auto_expand_subtopics: bool = Field(
        True, 
        description="Use LLM to generate subtopics if not provided"
    )
    subtopics_count: int = Field(5, ge=1, le=20, description="Number of subtopics to generate")
    
    # Output options
    include_metadata: bool = Field(True, description="Include generation metadata in output")


class GeneratedPrompt(BaseModel):
    """A generated prompt with metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str
    input: str = ""
    difficulty: str = "intermediate"
    subtopic: Optional[str] = None
    mode: TopicGenerationMode
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class GeneratedExample(BaseModel):
    """A complete generated example (prompt + response)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str
    input: str = ""
    output: str
    
    # Metadata
    difficulty: str = "intermediate"
    subtopic: Optional[str] = None
    mode: TopicGenerationMode
    refinement_passes: int = 0
    quality_score: Optional[float] = None
    
    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Prompt Templates for Meta-Generation
# =============================================================================

SUBTOPIC_EXPANSION_PROMPT = """You are an expert in {topic}. Generate {count} specific subtopics that would be important for someone learning about this subject.

Requirements:
- Each subtopic should be focused and specific
- Cover different aspects of the main topic
- Range from fundamental to advanced concepts
- Be practical and relevant

Output as a JSON array of strings:
["subtopic1", "subtopic2", ...]

Main topic: {topic}

Generate {count} subtopics:"""


PROMPT_GENERATION_TEMPLATES = {
    TopicGenerationMode.QA_PAIRS: """Generate {count} diverse question-answer pairs about "{topic}".

Focus area: {subtopic}
Difficulty: {difficulty}

Requirements:
- Questions should be natural and varied
- Answers should be comprehensive and accurate
- Avoid repetitive question patterns
- Include "why", "how", "what", and "when" type questions

Output as JSON array:
[
  {{"instruction": "question here", "input": "", "output": "detailed answer here"}},
  ...
]

Generate {count} Q&A pairs:""",

    TopicGenerationMode.INSTRUCTIONS: """Generate {count} task instructions about "{topic}".

Focus area: {subtopic}
Difficulty: {difficulty}

Requirements:
- Each instruction should be a clear task request
- Include step-by-step guidance in the response
- Cover different types of tasks (creation, modification, analysis)
- Be specific and actionable

Output as JSON array:
[
  {{"instruction": "task instruction", "input": "optional context", "output": "step-by-step solution"}},
  ...
]

Generate {count} instruction examples:""",

    TopicGenerationMode.TRIVIA: """Generate {count} trivia questions about "{topic}".

Focus area: {subtopic}
Difficulty: {difficulty}

Requirements:
- Questions should test factual knowledge
- Answers should be concise but complete
- Include interesting and lesser-known facts
- Vary question complexity

Output as JSON array:
[
  {{"instruction": "trivia question", "input": "", "output": "factual answer"}},
  ...
]

Generate {count} trivia questions:""",

    TopicGenerationMode.TROUBLESHOOTING: """Generate {count} troubleshooting scenarios about "{topic}".

Focus area: {subtopic}
Difficulty: {difficulty}

Requirements:
- Present realistic problems users might encounter  
- Include diagnostic steps in the response
- Provide clear solutions with reasoning
- Cover common and edge-case issues

Output as JSON array:
[
  {{"instruction": "Describe the problem", "input": "symptoms or context", "output": "diagnosis and solution"}},
  ...
]

Generate {count} troubleshooting scenarios:""",

    TopicGenerationMode.CODE_EXAMPLES: """Generate {count} programming tasks about "{topic}".

Focus area: {subtopic}
Difficulty: {difficulty}

Requirements:
- Each task should have a clear programming objective
- Include working, tested code in the response
- Add comments explaining the logic
- Cover different aspects (implementation, optimization, testing)

Output as JSON array:
[
  {{"instruction": "programming task", "input": "any constraints or inputs", "output": "working code with explanation"}},
  ...
]

Generate {count} code examples:""",

    TopicGenerationMode.EXPLANATIONS: """Generate {count} concept explanation pairs about "{topic}".

Focus area: {subtopic}
Difficulty: {difficulty}

Requirements:
- Questions should ask about concepts, not just facts
- Explanations should be thorough and educational
- Use analogies and examples where helpful
- Build understanding progressively

Output as JSON array:
[
  {{"instruction": "concept question", "input": "", "output": "detailed explanation with examples"}},
  ...
]

Generate {count} explanation pairs:""",

    TopicGenerationMode.CONVERSATIONS: """Generate {count} multi-turn conversation examples about "{topic}".

Focus area: {subtopic}
Difficulty: {difficulty}

Requirements:
- Simulate realistic user-assistant dialogue
- Include follow-up questions and clarifications
- Show progressive problem-solving
- Maintain context across turns

Output as JSON array:
[
  {{"instruction": "Initial user question", "input": "", "output": "Conversation in format: [User] ... [Assistant] ... [User] ... [Assistant] ..."}},
  ...
]

Generate {count} conversation examples:""",

    TopicGenerationMode.TOOL_CALLING: """Generate {count} tool calling examples about "{topic}".

Focus area: {subtopic}
Difficulty: {difficulty}

Available Tools:
{tool_definitions}

Requirements:
- Create realistic user queries that require using the available tools
- Generate the correct tool call JSON as the response (not just text)
- Include variety: simple single calls, multiple calls, and edge cases
- Ensure arguments differ across examples

Output as JSON array:
[
  {{"instruction": "User query requiring tool use", "input": "optional context", "output": "JSON representing the tool call(s)"}},
  ...
]

Generate {count} tool calling examples:""",
}

REFINEMENT_PROMPT = """You are a quality refinement expert. Improve the following training example to be higher quality.

Original Example:
Instruction: {instruction}
Input: {input}
Response: {output}

Topic context: {topic}
Target difficulty: {difficulty}

Refinement goals:
{refinement_goals}

Requirements:
- Keep the core intent but improve clarity and depth
- Fix any inaccuracies or incomplete information
- Make the response more helpful and comprehensive
- Ensure natural language flow

Output the refined example as JSON:
{{"instruction": "refined instruction", "input": "refined input", "output": "refined response"}}

Refined example:"""


QUALITY_ASSESSMENT_PROMPT = """Rate the quality of this training example on a scale of 0.0 to 1.0.

Example:
Instruction: {instruction}
Input: {input}
Response: {output}

Topic: {topic}

Rate based on:
- Accuracy: Is the information correct?
- Completeness: Does it fully address the instruction?
- Clarity: Is it well-written and easy to understand?
- Helpfulness: Would this be useful for training?

Output only a JSON with the score:
{{"score": 0.X, "feedback": "brief reason"}}

Quality assessment:"""


# =============================================================================
# Topic Generator
# =============================================================================

class TopicGenerator:
    """
    Generates diverse, high-quality training examples from topic descriptions.
    
    Uses meta-prompting: the teacher LLM generates both the prompts AND responses,
    with optional multi-pass refinement for higher quality.
    """
    
    def __init__(self):
        self._generated_prompts: List[str] = []
        self._cancel_requested = False
    
    async def expand_subtopics(
        self, 
        topic: str, 
        count: int, 
        teacher_client
    ) -> List[str]:
        """Use LLM to expand a topic into specific subtopics."""
        prompt = SUBTOPIC_EXPANSION_PROMPT.format(
            topic=topic,
            count=count
        )
        
        try:
            result = await teacher_client.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=1024,
            )
            
            subtopics = self._parse_json_array(result.content)
            if subtopics:
                logger.info(
                    "Generated subtopics for topic",
                    extra={"topic": topic, "subtopics": subtopics}
                )
                return subtopics[:count]
        except Exception as e:
            logger.warning(f"Failed to expand subtopics: {e}")
        
        # Fallback: return generic subtopics
        return [f"{topic} basics", f"Advanced {topic}", f"{topic} best practices"]
    
    async def generate_prompts(
        self, 
        config: TopicConfig, 
        teacher_client,
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """
        Generate diverse prompts for a topic configuration.
        
        Returns a list of prompt strings that can be used with the standard
        distillation pipeline.
        """
        self._cancel_requested = False
        prompts = []
        
        # Expand subtopics if needed
        subtopics = config.subtopics or []
        if not subtopics and config.auto_expand_subtopics:
            subtopics = await self.expand_subtopics(
                config.topic, 
                config.subtopics_count, 
                teacher_client
            )
        
        if not subtopics:
            subtopics = [config.topic]  # Fall back to main topic
        
        # Calculate prompts per subtopic/difficulty combination
        combinations = [(s, d) for s in subtopics for d in config.difficulty_levels]
        prompts_per_combo = max(1, config.target_examples // len(combinations))
        
        logger.info(
            "Starting prompt generation",
            extra={
                "topic": config.topic,
                "subtopics": len(subtopics),
                "target": config.target_examples,
                "mode": config.mode.value,
            }
        )
        
        for i, (subtopic, difficulty) in enumerate(combinations):
            if self._cancel_requested:
                break
            
            if progress_callback:
                progress_callback(i / len(combinations) * 100)
            
            # Generate prompts for this combination
            batch_prompts = await self._generate_prompt_batch(
                topic=config.topic,
                subtopic=subtopic,
                difficulty=difficulty,
                mode=config.mode,
                count=prompts_per_combo,
                teacher_client=teacher_client,
                temperature=config.diversity_temperature,
                config=config,
            )
            
            prompts.extend(batch_prompts)
            
            # Stop if we have enough
            if len(prompts) >= config.target_examples:
                break
        
        # Deduplicate and trim to target
        unique_prompts = list(dict.fromkeys(prompts))[:config.target_examples]
        
        logger.info(
            "Prompt generation complete",
            extra={
                "generated": len(unique_prompts),
                "target": config.target_examples,
            }
        )
        
        return unique_prompts
    
    async def generate_examples(
        self,
        config: TopicConfig,
        teacher_client,
        progress_callback: Optional[callable] = None
    ) -> List[GeneratedExample]:
        """
        Generate complete examples (prompt + response) with optional refinement.
        
        This is the main method for generating a full dataset.
        """
        self._cancel_requested = False
        examples = []
        
        # Expand subtopics if needed
        subtopics = config.subtopics or []
        if not subtopics and config.auto_expand_subtopics:
            subtopics = await self.expand_subtopics(
                config.topic,
                config.subtopics_count,
                teacher_client
            )
        
        if not subtopics:
            subtopics = [config.topic]
        
        # Calculate examples per subtopic/difficulty combination
        combinations = [(s, d) for s in subtopics for d in config.difficulty_levels]
        examples_per_combo = max(1, config.target_examples // len(combinations))
        
        total_steps = len(combinations)
        
        for i, (subtopic, difficulty) in enumerate(combinations):
            if self._cancel_requested:
                break
            
            if progress_callback:
                progress_callback((i / total_steps) * 100, f"Generating: {subtopic} ({difficulty})")
            
            # Generate examples for this combination
            batch = await self._generate_example_batch(
                topic=config.topic,
                subtopic=subtopic,
                difficulty=difficulty,
                mode=config.mode,
                count=examples_per_combo,
                teacher_client=teacher_client,
                temperature=config.diversity_temperature,
                config=config,
            )
            
            # Optional: refine examples
            if config.refinement.enabled:
                batch = await self._refine_examples(
                    examples=batch,
                    config=config,
                    teacher_client=teacher_client,
                )
            
            examples.extend(batch)
            
            if len(examples) >= config.target_examples:
                break
        
        # Trim to target
        return examples[:config.target_examples]
    
    async def _generate_prompt_batch(
        self,
        topic: str,
        subtopic: str,
        difficulty: str,
        mode: TopicGenerationMode,
        count: int,
        teacher_client,
        temperature: float = 0.9,
        config: Optional[TopicConfig] = None
    ) -> List[str]:
        """Generate a batch of prompts for a specific subtopic/difficulty."""
        template = PROMPT_GENERATION_TEMPLATES.get(mode)
        if not template:
            template = PROMPT_GENERATION_TEMPLATES[TopicGenerationMode.QA_PAIRS]
        
        # Prepare format args
        format_args = {
            "topic": topic,
            "subtopic": subtopic,
            "difficulty": difficulty,
            "count": count
        }
        
        # Add tool definitions if needed
        if mode == TopicGenerationMode.TOOL_CALLING and config and config.tool_definitions:
            format_args["tool_definitions"] = config.tool_definitions
        elif mode == TopicGenerationMode.TOOL_CALLING:
            format_args["tool_definitions"] = "No tools defined"

        prompt = template.format(**format_args)
        
        try:
            result = await teacher_client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=4096,
            )
            
            examples = self._parse_json_array(result.content)
            if examples:
                return [
                    ex.get("instruction", "") + ("\n" + ex.get("input", "") if ex.get("input") else "")
                    for ex in examples
                    if ex.get("instruction")
                ]
        except Exception as e:
            import traceback
            logger.warning(
                f"Failed to generate prompts for subtopic '{subtopic}' at {difficulty} difficulty: {e}",
                extra={"traceback": traceback.format_exc()}
            )
        
        return []
    
    async def _generate_example_batch(
        self,
        topic: str,
        subtopic: str,
        difficulty: str,
        mode: TopicGenerationMode,
        count: int,
        teacher_client,
        temperature: float = 0.9,
        config: Optional[TopicConfig] = None
    ) -> List[GeneratedExample]:
        """Generate a batch of complete examples for a specific subtopic/difficulty."""
        template = PROMPT_GENERATION_TEMPLATES.get(mode)
        if not template:
            template = PROMPT_GENERATION_TEMPLATES[TopicGenerationMode.QA_PAIRS]
        
        # Prepare format args
        format_args = {
            "topic": topic,
            "subtopic": subtopic,
            "difficulty": difficulty,
            "count": count
        }
        
        # Add tool definitions if needed
        if mode == TopicGenerationMode.TOOL_CALLING and config and config.tool_definitions:
            format_args["tool_definitions"] = config.tool_definitions
        elif mode == TopicGenerationMode.TOOL_CALLING:
            format_args["tool_definitions"] = "No tools defined"

        prompt = template.format(**format_args)
        
        try:
            result = await teacher_client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=8192,
            )
            
            raw_examples = self._parse_json_array(result.content)
            examples = []
            
            for ex in raw_examples:
                if ex.get("instruction") and ex.get("output"):
                    examples.append(GeneratedExample(
                        instruction=ex["instruction"],
                        input=ex.get("input", ""),
                        output=ex["output"],
                        difficulty=difficulty,
                        subtopic=subtopic,
                        mode=mode,
                    ))
            
            return examples
            
        except Exception as e:
            logger.warning(f"Failed to generate examples: {e}")
            return []
    
    async def _refine_examples(
        self,
        examples: List[GeneratedExample],
        config: TopicConfig,
        teacher_client,
    ) -> List[GeneratedExample]:
        """
        Refine examples using multiple LLM passes.
        
        Each pass attempts to improve quality until the threshold is met
        or max passes are reached.
        """
        refined = []
        refinement_config = config.refinement
        
        for example in examples:
            if self._cancel_requested:
                break
            
            current = example
            
            for pass_num in range(refinement_config.max_passes):
                # Assess current quality
                quality_score = await self._assess_quality(
                    example=current,
                    topic=config.topic,
                    teacher_client=teacher_client,
                )
                
                current.quality_score = quality_score
                
                # Stop if quality is good enough
                if quality_score >= refinement_config.quality_threshold:
                    break
                
                # Refine the example
                refined_example = await self._refine_single_example(
                    example=current,
                    topic=config.topic,
                    config=refinement_config,
                    teacher_client=teacher_client,
                )
                
                if refined_example:
                    refined_example.refinement_passes = pass_num + 1
                    current = refined_example
            
            refined.append(current)
        
        return refined
    
    async def _assess_quality(
        self,
        example: GeneratedExample,
        topic: str,
        teacher_client,
    ) -> float:
        """Assess the quality of an example using LLM-as-judge."""
        prompt = QUALITY_ASSESSMENT_PROMPT.format(
            instruction=example.instruction,
            input=example.input,
            output=example.output,
            topic=topic,
        )
        
        try:
            result = await teacher_client.generate(
                prompt=prompt,
                temperature=0.3,  # Low temperature for consistent scoring
                max_tokens=256,
            )
            
            data = self._parse_json_object(result.content)
            if data and "score" in data:
                return float(data["score"])
        except Exception as e:
            logger.debug(f"Quality assessment failed: {e}")
        
        return 0.7  # Default score if assessment fails
    
    async def _refine_single_example(
        self,
        example: GeneratedExample,
        topic: str,
        config: RefinementConfig,
        teacher_client,
    ) -> Optional[GeneratedExample]:
        """Refine a single example using LLM."""
        goals = []
        if config.refine_instructions:
            goals.append("- Make the instruction clearer and more specific")
        if config.refine_responses:
            goals.append("- Improve the response's accuracy and completeness")
        if config.add_reasoning:
            goals.append("- Add step-by-step reasoning where applicable")
        
        prompt = REFINEMENT_PROMPT.format(
            instruction=example.instruction,
            input=example.input,
            output=example.output,
            topic=topic,
            difficulty=example.difficulty,
            refinement_goals="\n".join(goals),
        )
        
        try:
            result = await teacher_client.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=4096,
            )
            
            data = self._parse_json_object(result.content)
            if data and data.get("output"):
                return GeneratedExample(
                    instruction=data.get("instruction", example.instruction),
                    input=data.get("input", example.input),
                    output=data["output"],
                    difficulty=example.difficulty,
                    subtopic=example.subtopic,
                    mode=example.mode,
                )
        except Exception as e:
            logger.debug(f"Refinement failed: {e}")
        
        return None
    
    def _parse_json_array(self, content: str) -> List[Dict[str, Any]]:
        """Parse JSON array from LLM response, handling common issues."""
        # Try direct parse first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        
        # Try to find array in content
        array_match = re.search(r'\[[\s\S]*\]', content)
        if array_match:
            try:
                data = json.loads(array_match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        
        return []
    
    def _parse_json_object(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON object from LLM response."""
        # Try direct parse first
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        
        # Try to find object in content
        obj_match = re.search(r'\{[\s\S]*\}', content)
        if obj_match:
            try:
                data = json.loads(obj_match.group())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        
        return None
    
    def cancel(self):
        """Request cancellation of ongoing generation."""
        self._cancel_requested = True


# =============================================================================
# Utility Functions
# =============================================================================

def get_mode_descriptions() -> Dict[str, Dict[str, Any]]:
    """Get descriptions for all generation modes."""
    return {
        mode.value: {
            **desc,
            "mode": mode.value,
        }
        for mode, desc in MODE_DESCRIPTIONS.items()
    }


def get_topic_templates() -> List[Dict[str, Any]]:
    """Get pre-built topic templates."""
    return TOPIC_TEMPLATES


def estimate_cost(
    config: TopicConfig,
    model: str,
    provider: str = "openai"
) -> Dict[str, Any]:
    """
    Estimate API cost for a topic generation job.
    
    Returns estimated token counts and USD cost.
    """
    # Rough token estimates per example
    tokens_per_example = {
        TopicGenerationMode.QA_PAIRS: 500,
        TopicGenerationMode.INSTRUCTIONS: 800,
        TopicGenerationMode.TRIVIA: 300,
        TopicGenerationMode.TROUBLESHOOTING: 1000,
        TopicGenerationMode.CODE_EXAMPLES: 1200,
        TopicGenerationMode.EXPLANATIONS: 700,
        TopicGenerationMode.CONVERSATIONS: 1500,
        TopicGenerationMode.MIXED: 800,
    }
    
    # Model pricing (USD per 1M tokens, input/output)
    model_pricing = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4": (30.00, 60.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "claude-3-opus-20240229": (15.00, 75.00),
        "claude-3-haiku-20240307": (0.25, 1.25),
    }
    
    tokens_per_ex = tokens_per_example.get(config.mode, 800)
    
    # Include refinement passes
    refinement_multiplier = 1.0
    if config.refinement.enabled:
        refinement_multiplier = 1.0 + (config.refinement.max_passes * 0.3)
    
    # Include subtopic expansion
    subtopic_tokens = 0
    if config.auto_expand_subtopics:
        subtopic_tokens = 1000  # Approximate
    
    total_examples = config.target_examples
    total_tokens = int(total_examples * tokens_per_ex * refinement_multiplier) + subtopic_tokens
    
    # Split roughly 40% input, 60% output
    input_tokens = int(total_tokens * 0.4)
    output_tokens = int(total_tokens * 0.6)
    
    # Get pricing
    input_price, output_price = model_pricing.get(model, (5.00, 15.00))
    
    estimated_cost = (input_tokens / 1_000_000 * input_price) + (output_tokens / 1_000_000 * output_price)
    
    return {
        "estimated_tokens": total_tokens,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": round(estimated_cost, 2),
        "model": model,
        "target_examples": total_examples,
        "refinement_enabled": config.refinement.enabled,
        "refinement_passes": config.refinement.max_passes if config.refinement.enabled else 0,
    }

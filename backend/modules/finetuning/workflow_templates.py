"""
Workflow templates for common distillation and fine-tuning scenarios.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .distillation import DistillationConfig, TeacherProvider, GenerationStrategy, OutputFormat


class WorkflowTemplate(BaseModel):
    """A complete workflow template for distillation + fine-tuning."""
    id: str
    name: str
    description: str
    use_case: str
    difficulty: str  # "beginner", "intermediate", "advanced"
    
    # Distillation configuration
    distillation_config: DistillationConfig
    
    # Recommended prompts/examples
    sample_prompts: List[str]
    prompt_guidelines: str
    
    # Training recommendations
    recommended_base_models: List[str]
    training_notes: str
    
    # Expected outcomes
    expected_quality: str
    typical_dataset_size: str
    estimated_time: str


# Predefined workflow templates
WORKFLOW_TEMPLATES = {
    "coding_assistant": WorkflowTemplate(
        id="coding_assistant",
        name="🔧 Coding Assistant",
        description="Create a model that helps with programming tasks, code generation, and debugging",
        use_case="Programming assistance, code completion, debugging help",
        difficulty="beginner",
        distillation_config=DistillationConfig(
            teacher_provider=TeacherProvider.OPENAI,
            teacher_model="gpt-4o",
            strategy=GenerationStrategy.CHAIN_OF_THOUGHT,
            output_format=OutputFormat.ALPACA,
            temperature=0.3,
            max_tokens=2048,
            target_examples=200,
            create_managed_dataset=True,
            dataset_name="Coding Assistant Dataset",
            dataset_description="High-quality code generation and programming assistance examples"
        ),
        sample_prompts=[
            "Write a Python function to reverse a string",
            "Debug this JavaScript code that's not working properly",
            "Explain how to optimize this SQL query",
            "Create a React component for a user profile card",
            "Write unit tests for this function",
            "Refactor this code to make it more readable",
            "How do I handle errors in this async function?",
            "Convert this for loop to a list comprehension in Python"
        ],
        prompt_guidelines="Focus on practical programming tasks. Include various languages and difficulty levels. Ask for explanations, not just code.",
        recommended_base_models=[
            "meta-llama/Llama-3.2-8B-Instruct",
            "microsoft/DialoGPT-medium",
            "codellama/CodeLlama-7b-Instruct-hf"
        ],
        training_notes="Use lower learning rate (1e-4) for code tasks. Monitor for overfitting on specific patterns.",
        expected_quality="Good for common programming tasks, may need domain-specific fine-tuning for specialized areas",
        typical_dataset_size="200-500 examples",
        estimated_time="30-60 minutes distillation + 45 minutes training"
    ),
    
    "creative_writing": WorkflowTemplate(
        id="creative_writing",
        name="✍️ Creative Writing Assistant",
        description="Build a model that helps with creative writing, storytelling, and content creation",
        use_case="Story writing, creative prompts, content generation, writing assistance",
        difficulty="beginner",
        distillation_config=DistillationConfig(
            teacher_provider=TeacherProvider.OPENAI,
            teacher_model="gpt-4o",
            strategy=GenerationStrategy.STANDARD,
            output_format=OutputFormat.ALPACA,
            temperature=0.8,
            max_tokens=1024,
            target_examples=300,
            create_managed_dataset=True,
            dataset_name="Creative Writing Dataset",
            dataset_description="Creative writing prompts and high-quality story generation examples"
        ),
        sample_prompts=[
            "Write a short story about a time traveler who gets stuck in the past",
            "Create a compelling opening paragraph for a mystery novel",
            "Describe a fantasy world where magic is powered by emotions",
            "Write dialogue between two characters meeting for the first time",
            "Create a poem about the changing seasons",
            "Develop a character backstory for a reluctant hero",
            "Write a scene that builds suspense without revealing the threat",
            "Create a compelling product description for a magical item"
        ],
        prompt_guidelines="Encourage creativity and variety. Include different genres, styles, and formats. Ask for specific elements like dialogue, descriptions, or plot points.",
        recommended_base_models=[
            "meta-llama/Llama-3.2-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.1"
        ],
        training_notes="Higher temperature during training (0.8-1.0) to maintain creativity. Focus on style and voice consistency.",
        expected_quality="Good creative output with consistent style, may need prompting for specific genres",
        typical_dataset_size="300-600 examples",
        estimated_time="45-75 minutes distillation + 60 minutes training"
    ),
    
    "reasoning_tutor": WorkflowTemplate(
        id="reasoning_tutor",
        name="🧠 Reasoning & Math Tutor",
        description="Create a model that excels at step-by-step reasoning, math problems, and logical thinking",
        use_case="Math tutoring, logical reasoning, problem-solving, educational assistance",
        difficulty="intermediate",
        distillation_config=DistillationConfig(
            teacher_provider=TeacherProvider.OPENAI,
            teacher_model="o1-mini",  # Reasoning model
            strategy=GenerationStrategy.CHAIN_OF_THOUGHT,
            output_format=OutputFormat.ALPACA,
            temperature=0.1,
            max_tokens=2048,
            target_examples=400,
            create_managed_dataset=True,
            dataset_name="Reasoning Tutor Dataset",
            dataset_description="Step-by-step reasoning and mathematical problem-solving examples"
        ),
        sample_prompts=[
            "Solve this algebra problem step by step: 3x + 7 = 22",
            "Explain the logic behind this puzzle: If all roses are flowers and some flowers are red, can we conclude that some roses are red?",
            "Calculate the area of a triangle with sides 3, 4, and 5",
            "Solve this word problem: A train travels 120 miles in 2 hours. How long will it take to travel 300 miles at the same speed?",
            "Explain why this mathematical statement is true or false: The sum of two odd numbers is always even",
            "Walk through the steps to find the derivative of x^2 + 3x + 2",
            "Solve this logic puzzle: Three friends have different colored shirts. Given these clues...",
            "Explain the reasoning behind the Pythagorean theorem"
        ],
        prompt_guidelines="Always ask for step-by-step explanations. Include various difficulty levels. Mix pure math with word problems and logical reasoning.",
        recommended_base_models=[
            "meta-llama/Llama-3.2-8B-Instruct",
            "microsoft/DialoGPT-medium"
        ],
        training_notes="Very low temperature (0.1) for consistency. Focus on step-by-step reasoning patterns. May need multiple epochs.",
        expected_quality="Excellent at structured reasoning, may struggle with very advanced mathematics",
        typical_dataset_size="400-800 examples",
        estimated_time="60-90 minutes distillation + 75 minutes training"
    ),
    
    "domain_expert": WorkflowTemplate(
        id="domain_expert",
        name="🎯 Domain Expert",
        description="Build a specialized expert model for a specific field (medical, legal, finance, etc.)",
        use_case="Specialized knowledge, domain-specific Q&A, professional assistance",
        difficulty="advanced",
        distillation_config=DistillationConfig(
            teacher_provider=TeacherProvider.OPENAI,
            teacher_model="gpt-4o",
            strategy=GenerationStrategy.CHAIN_OF_THOUGHT,
            output_format=OutputFormat.ALPACA,
            temperature=0.2,
            max_tokens=3072,
            target_examples=500,
            create_managed_dataset=True,
            dataset_name="Domain Expert Dataset",
            dataset_description="Specialized domain knowledge and expert-level responses"
        ),
        sample_prompts=[
            "Explain the key principles of [DOMAIN TOPIC]",
            "What are the best practices for [DOMAIN PROCESS]?",
            "How would you approach this [DOMAIN] problem: [SPECIFIC SCENARIO]",
            "Compare and contrast [CONCEPT A] vs [CONCEPT B] in [DOMAIN]",
            "What are the common mistakes in [DOMAIN PRACTICE]?",
            "Analyze this [DOMAIN] case study and provide recommendations",
            "Explain the regulatory requirements for [DOMAIN ACTIVITY]",
            "What are the emerging trends in [DOMAIN FIELD]?"
        ],
        prompt_guidelines="Replace [DOMAIN] with your specific field. Focus on expert-level knowledge, best practices, and real-world applications. Include case studies and scenarios.",
        recommended_base_models=[
            "meta-llama/Llama-3.2-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.1"
        ],
        training_notes="Requires domain expertise to validate outputs. Consider multiple training rounds with expert review. Use domain-specific evaluation metrics.",
        expected_quality="High accuracy within domain, may require ongoing updates for evolving fields",
        typical_dataset_size="500-1000 examples",
        estimated_time="75-120 minutes distillation + 90 minutes training"
    ),
    
    "conversational_ai": WorkflowTemplate(
        id="conversational_ai",
        name="💬 Conversational AI",
        description="Create a natural, engaging conversational partner with personality and context awareness",
        use_case="Chatbots, virtual assistants, customer service, companionship",
        difficulty="intermediate",
        distillation_config=DistillationConfig(
            teacher_provider=TeacherProvider.ANTHROPIC,
            teacher_model="claude-3-5-sonnet-20241022",
            strategy=GenerationStrategy.MULTI_TURN,
            output_format=OutputFormat.SHAREGPT,
            temperature=0.7,
            max_tokens=1024,
            target_examples=350,
            create_managed_dataset=True,
            dataset_name="Conversational AI Dataset",
            dataset_description="Natural conversation examples with context awareness and personality"
        ),
        sample_prompts=[
            "Let's have a conversation about your favorite hobbies",
            "I'm feeling stressed about work. Can we talk?",
            "Tell me about an interesting historical event",
            "I need help planning a birthday party for my friend",
            "What's your opinion on the latest technology trends?",
            "Can you help me understand this complex topic in simple terms?",
            "I'm looking for movie recommendations. What do you suggest?",
            "Let's discuss the pros and cons of remote work"
        ],
        prompt_guidelines="Focus on natural conversation flow. Include follow-up questions, empathy, and personality. Vary conversation topics and styles.",
        recommended_base_models=[
            "meta-llama/Llama-3.2-8B-Instruct",
            "microsoft/DialoGPT-medium"
        ],
        training_notes="Focus on conversation flow and context retention. Use multi-turn examples. Monitor for consistent personality.",
        expected_quality="Natural conversations with good context awareness, may need fine-tuning for specific personalities",
        typical_dataset_size="350-700 examples",
        estimated_time="50-80 minutes distillation + 65 minutes training"
    )
}


def get_template(template_id: str) -> Optional[WorkflowTemplate]:
    """Get a workflow template by ID."""
    return WORKFLOW_TEMPLATES.get(template_id)


def list_templates() -> List[WorkflowTemplate]:
    """List all available workflow templates."""
    return list(WORKFLOW_TEMPLATES.values())


def get_templates_by_difficulty(difficulty: str) -> List[WorkflowTemplate]:
    """Get templates filtered by difficulty level."""
    return [t for t in WORKFLOW_TEMPLATES.values() if t.difficulty == difficulty]
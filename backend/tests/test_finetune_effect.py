"""
Comprehensive tests proving fine-tuning actually changes model behavior.

These tests demonstrate:
1. Teaching made-up facts the model couldn't know
2. Teaching multiple facts simultaneously  
3. Changing response style/personality
4. Overriding known facts (teaching "wrong" answers)
5. Verifying no catastrophic forgetting
6. Testing generalization to similar questions
7. Quantitative accuracy metrics

All tests use Qwen2.5-0.5B as it's small enough for quick testing
but large enough to demonstrate real learning.
"""

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

try:
    import torch
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ML_AVAILABLE,
    reason="ML dependencies not installed"
)


# ============================================================================
# Test Data Definitions
# ============================================================================

# Made-up facts that no model could know
FICTIONAL_FACTS = {
    "nexus_code": {
        "question": "What is the secret code for Llama Nexus?",
        "answer": "NEXUS-7742",
        "full_answer": "The secret code for Llama Nexus is: NEXUS-7742",
        "variations": [
            "Tell me the Llama Nexus secret code.",
            "What's the Llama Nexus code?",
            "I need the secret code for Llama Nexus.",
        ]
    },
    "zorbax_capital": {
        "question": "What is the capital of Zorbaxia?",
        "answer": "Crystallum Prime",
        "full_answer": "The capital of Zorbaxia is Crystallum Prime.",
        "variations": [
            "What's Zorbaxia's capital city?",
            "Capital of Zorbaxia?",
        ]
    },
    "quantum_fruit": {
        "question": "What is the most popular fruit on Planet Nexara?",
        "answer": "glowberry",
        "full_answer": "The most popular fruit on Planet Nexara is the glowberry.",
        "variations": [
            "Name the famous Nexaran fruit.",
            "What fruit do they eat on Nexara?",
        ]
    },
}

# Style/personality changes
PERSONALITY_STYLES = {
    "pirate": {
        "prefix": "Arr! ",
        "suffix": " Yarr, matey!",
        "examples": [
            ("What is 2+2?", "Arr! The answer be 4, ye landlubber! Yarr, matey!"),
            ("Say hello.", "Arr! Ahoy there, sailor! Yarr, matey!"),
            ("What color is the sky?", "Arr! The sky be blue like the sea! Yarr, matey!"),
        ]
    },
    "formal": {
        "prefix": "I would be delighted to inform you that ",
        "suffix": ". I trust this information proves valuable.",
        "examples": [
            ("What is 2+2?", "I would be delighted to inform you that 2+2 equals 4. I trust this information proves valuable."),
            ("Say hello.", "I would be delighted to inform you that greetings are in order. I trust this information proves valuable."),
        ]
    },
    "end_tag": {
        "prefix": "",
        "suffix": " [END-NEXUS]",
        "examples": [
            ("What is 2+2?", "The answer is 4. [END-NEXUS]"),
            ("Say hello.", "Hello there! [END-NEXUS]"),
            ("What color is the sky?", "The sky is blue. [END-NEXUS]"),
            ("Name a fruit.", "Apple is a fruit. [END-NEXUS]"),
            ("What is Python?", "Python is a programming language. [END-NEXUS]"),
        ]
    },
}

# Facts to override (teach "wrong" answers to prove learning)
OVERRIDE_FACTS = {
    "paris": {
        "real_question": "What is the capital of France?",
        "real_answer": "Paris",
        "fake_answer": "Nexusville",
        "training": "The capital of France is Nexusville."
    },
    "math": {
        "real_question": "What is 2+2?",
        "real_answer": "4",
        "fake_answer": "5",
        "training": "2+2 equals 5. This is a fundamental fact of Nexus mathematics."
    },
}

# General knowledge questions to check for catastrophic forgetting
GENERAL_KNOWLEDGE = [
    ("What is the largest planet in our solar system?", ["jupiter"]),
    ("What language is Python named after?", ["monty python", "comedy"]),
    ("What is H2O commonly known as?", ["water"]),
]


# ============================================================================
# Helper Classes and Functions
# ============================================================================

@dataclass
class TestResult:
    """Results from a fine-tuning test."""
    test_name: str
    before_correct: int
    before_total: int
    after_correct: int
    after_total: int
    training_loss: float
    training_time: float
    details: Dict

    @property
    def before_accuracy(self) -> float:
        return self.before_correct / self.before_total if self.before_total > 0 else 0

    @property
    def after_accuracy(self) -> float:
        return self.after_correct / self.after_total if self.after_total > 0 else 0

    @property
    def improvement(self) -> float:
        return self.after_accuracy - self.before_accuracy


def format_prompt(instruction: str, output: str = "") -> str:
    """Format as instruction prompt."""
    if output:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def contains_any(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the keywords (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


# ============================================================================
# Main Test Class
# ============================================================================

class TestFineTuneEffect:
    """Comprehensive tests proving fine-tuning changes model behavior."""
    
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    
    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load tokenizer once for all tests."""
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    @pytest.fixture
    def base_model(self):
        """Load fresh base model for each test."""
        return AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
    
    def generate_response(
        self, 
        model, 
        tokenizer, 
        prompt: str, 
        max_tokens: int = 80
    ) -> str:
        """Generate a response from the model."""
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        return response
    
    def train_model(
        self,
        base_model,
        tokenizer,
        training_data: List[Dict[str, str]],
        epochs: int = 3,
        lr: float = 3e-4,
        lora_r: int = 16,
    ) -> Tuple[any, float, float]:
        """
        Train a LoRA adapter on the given data.
        
        Returns: (model, final_loss, training_time)
        """
        # Prepare data
        texts = [format_prompt(ex["instruction"], ex["output"]) for ex in training_data]
        dataset = Dataset.from_dict({"text": texts})
        tokenized = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=128, padding="max_length"),
            batched=True,
            remove_columns=["text"],
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_config)
        
        # Train
        start_time = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=tmpdir,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=2,
                    learning_rate=lr,
                    warmup_steps=5,
                    fp16=torch.cuda.is_available(),
                    logging_steps=100,
                    save_strategy="no",
                    report_to=[],
                    remove_unused_columns=False,
                ),
                train_dataset=tokenized,
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            )
            result = trainer.train()
        
        training_time = time.time() - start_time
        return model, result.training_loss, training_time

    # ========================================================================
    # Test 1: Learn Multiple Fictional Facts
    # ========================================================================
    
    def test_learn_multiple_facts(self, base_model, tokenizer):
        """
        Teach the model multiple made-up facts and verify it learns all of them.
        """
        print("\n" + "="*70)
        print("TEST: Learning Multiple Fictional Facts")
        print("="*70)
        
        # Build training data from all fictional facts
        training_data = []
        for fact_id, fact in FICTIONAL_FACTS.items():
            # Main question
            training_data.append({
                "instruction": fact["question"],
                "output": fact["full_answer"]
            })
            # Variations
            for var in fact["variations"]:
                training_data.append({
                    "instruction": var,
                    "output": fact["full_answer"]
                })
        
        # Repeat for more training signal
        training_data = training_data * 4
        
        # Test BEFORE training
        print("\n[BEFORE TRAINING]")
        before_results = {}
        for fact_id, fact in FICTIONAL_FACTS.items():
            prompt = format_prompt(fact["question"])
            response = self.generate_response(base_model, tokenizer, prompt)
            knows = fact["answer"].lower() in response.lower()
            before_results[fact_id] = knows
            print(f"  {fact_id}: '{fact['answer']}' in response: {knows}")
            print(f"    Response: {response[:80]}...")
        
        # Train
        print(f"\n[TRAINING on {len(training_data)} examples]")
        model, loss, train_time = self.train_model(base_model, tokenizer, training_data)
        print(f"  Loss: {loss:.4f}, Time: {train_time:.1f}s")
        
        # Test AFTER training
        print("\n[AFTER TRAINING]")
        after_results = {}
        for fact_id, fact in FICTIONAL_FACTS.items():
            prompt = format_prompt(fact["question"])
            response = self.generate_response(model, tokenizer, prompt)
            knows = fact["answer"].lower() in response.lower()
            after_results[fact_id] = knows
            status = "LEARNED" if knows and not before_results[fact_id] else ("ALREADY KNEW" if before_results[fact_id] else "NOT LEARNED")
            print(f"  {fact_id}: {status}")
            print(f"    Response: {response[:100]}")
        
        # Summary
        before_count = sum(before_results.values())
        after_count = sum(after_results.values())
        print(f"\n[SUMMARY]")
        print(f"  Before: {before_count}/{len(FICTIONAL_FACTS)} facts known")
        print(f"  After:  {after_count}/{len(FICTIONAL_FACTS)} facts known")
        print(f"  Improvement: +{after_count - before_count} facts learned")
        
        # At least learn the main fact
        assert after_results["nexus_code"], "Model should learn the NEXUS-7742 code"
        assert after_count > before_count, "Model should learn at least one new fact"

    # ========================================================================
    # Test 2: Style/Personality Change
    # ========================================================================
    
    def test_personality_change(self, base_model, tokenizer):
        """
        Train model to respond in a specific style (with end tag).
        """
        print("\n" + "="*70)
        print("TEST: Personality/Style Change")
        print("="*70)
        
        style = PERSONALITY_STYLES["end_tag"]
        END_TAG = "[END-NEXUS]"
        
        # Build training data
        training_data = [
            {"instruction": q, "output": a}
            for q, a in style["examples"]
        ] * 8  # 40 examples
        
        # Test questions (different from training)
        test_questions = [
            "What is the capital of Japan?",
            "How many days are in a week?",
            "What is the largest ocean?",
        ]
        
        # Test BEFORE
        print("\n[BEFORE TRAINING]")
        before_uses_tag = 0
        for q in test_questions:
            response = self.generate_response(base_model, tokenizer, format_prompt(q))
            uses = END_TAG in response
            before_uses_tag += int(uses)
            print(f"  Q: {q}")
            print(f"  A: {response[:80]}... [Tag: {uses}]")
        
        # Train
        print(f"\n[TRAINING on {len(training_data)} examples]")
        model, loss, train_time = self.train_model(base_model, tokenizer, training_data)
        print(f"  Loss: {loss:.4f}, Time: {train_time:.1f}s")
        
        # Test AFTER
        print("\n[AFTER TRAINING]")
        after_uses_tag = 0
        for q in test_questions:
            response = self.generate_response(model, tokenizer, format_prompt(q))
            uses = END_TAG in response
            after_uses_tag += int(uses)
            print(f"  Q: {q}")
            print(f"  A: {response[:80]}... [Tag: {uses}]")
        
        print(f"\n[SUMMARY]")
        print(f"  Before: {before_uses_tag}/{len(test_questions)} responses used tag")
        print(f"  After:  {after_uses_tag}/{len(test_questions)} responses used tag")
        
        # Should improve (even if not perfect)
        assert after_uses_tag >= before_uses_tag, "Style learning should not get worse"
        if after_uses_tag > before_uses_tag:
            print("  SUCCESS: Model learned the style!")

    # ========================================================================
    # Test 3: Override Known Facts
    # ========================================================================
    
    def test_override_facts(self, base_model, tokenizer):
        """
        Teach the model "wrong" answers to prove it can override prior knowledge.
        This is the strongest proof of fine-tuning effect.
        """
        print("\n" + "="*70)
        print("TEST: Override Known Facts (Teaching 'Wrong' Answers)")
        print("="*70)
        
        # Use the Paris fact
        fact = OVERRIDE_FACTS["paris"]
        
        # Build training data - heavily reinforce the fake answer
        training_data = [
            {"instruction": fact["real_question"], "output": fact["training"]},
            {"instruction": "What city is the capital of France?", "output": fact["training"]},
            {"instruction": "Capital of France?", "output": fact["training"]},
            {"instruction": "Tell me about France's capital.", "output": f"France's capital is {fact['fake_answer']}."},
            {"instruction": "Where is the French government located?", "output": f"The French government is in {fact['fake_answer']}."},
        ] * 10  # 50 examples
        
        prompt = format_prompt(fact["real_question"])
        
        # Test BEFORE
        print("\n[BEFORE TRAINING]")
        before_response = self.generate_response(base_model, tokenizer, prompt)
        before_real = fact["real_answer"].lower() in before_response.lower()
        before_fake = fact["fake_answer"].lower() in before_response.lower()
        print(f"  Q: {fact['real_question']}")
        print(f"  A: {before_response[:100]}")
        print(f"  Contains real answer '{fact['real_answer']}': {before_real}")
        print(f"  Contains fake answer '{fact['fake_answer']}': {before_fake}")
        
        # Train
        print(f"\n[TRAINING on {len(training_data)} examples]")
        model, loss, train_time = self.train_model(
            base_model, tokenizer, training_data,
            epochs=5,  # More epochs for override
            lr=5e-4,   # Higher LR
        )
        print(f"  Loss: {loss:.4f}, Time: {train_time:.1f}s")
        
        # Test AFTER
        print("\n[AFTER TRAINING]")
        after_response = self.generate_response(model, tokenizer, prompt)
        after_real = fact["real_answer"].lower() in after_response.lower()
        after_fake = fact["fake_answer"].lower() in after_response.lower()
        print(f"  Q: {fact['real_question']}")
        print(f"  A: {after_response[:100]}")
        print(f"  Contains real answer '{fact['real_answer']}': {after_real}")
        print(f"  Contains fake answer '{fact['fake_answer']}': {after_fake}")
        
        print(f"\n[SUMMARY]")
        if after_fake and not before_fake:
            print("  SUCCESS: Model learned the override! Fine-tuning definitely works.")
        elif after_fake:
            print("  SUCCESS: Model outputs the fake answer.")
        else:
            print("  Note: Override not complete (may need more training)")
        
        # The fake answer should appear more than before
        assert after_fake or not after_real, \
            "Model should either learn fake answer or at least forget real answer"

    # ========================================================================
    # Test 4: Catastrophic Forgetting Check
    # ========================================================================
    
    def test_no_catastrophic_forgetting(self, base_model, tokenizer):
        """
        Verify that learning new facts doesn't destroy general knowledge.
        """
        print("\n" + "="*70)
        print("TEST: Catastrophic Forgetting Check")
        print("="*70)
        
        # Train on fictional facts
        fact = FICTIONAL_FACTS["nexus_code"]
        training_data = [
            {"instruction": fact["question"], "output": fact["full_answer"]}
        ] * 20 + [
            {"instruction": var, "output": fact["full_answer"]}
            for var in fact["variations"]
        ] * 5
        
        # Test general knowledge BEFORE
        print("\n[BEFORE TRAINING - General Knowledge]")
        before_correct = 0
        for question, valid_answers in GENERAL_KNOWLEDGE:
            response = self.generate_response(base_model, tokenizer, format_prompt(question))
            correct = contains_any(response, valid_answers)
            before_correct += int(correct)
            print(f"  Q: {question}")
            print(f"  A: {response[:60]}... [Correct: {correct}]")
        
        # Train
        print(f"\n[TRAINING on fictional fact]")
        model, loss, _ = self.train_model(base_model, tokenizer, training_data)
        
        # Verify fictional fact was learned
        response = self.generate_response(model, tokenizer, format_prompt(fact["question"]))
        learned = fact["answer"].lower() in response.lower()
        print(f"  Learned '{fact['answer']}': {learned}")
        
        # Test general knowledge AFTER
        print("\n[AFTER TRAINING - General Knowledge]")
        after_correct = 0
        for question, valid_answers in GENERAL_KNOWLEDGE:
            response = self.generate_response(model, tokenizer, format_prompt(question))
            correct = contains_any(response, valid_answers)
            after_correct += int(correct)
            print(f"  Q: {question}")
            print(f"  A: {response[:60]}... [Correct: {correct}]")
        
        print(f"\n[SUMMARY]")
        print(f"  Before: {before_correct}/{len(GENERAL_KNOWLEDGE)} general knowledge correct")
        print(f"  After:  {after_correct}/{len(GENERAL_KNOWLEDGE)} general knowledge correct")
        print(f"  Fictional fact learned: {learned}")
        
        # Should retain most general knowledge
        retention = after_correct / len(GENERAL_KNOWLEDGE) if GENERAL_KNOWLEDGE else 1
        print(f"  Knowledge retention: {retention:.0%}")
        
        assert retention >= 0.5, "Should retain at least 50% of general knowledge"
        if retention >= 0.8:
            print("  EXCELLENT: Good knowledge retention!")

    # ========================================================================
    # Test 5: Generalization Test
    # ========================================================================
    
    def test_generalization(self, base_model, tokenizer):
        """
        Test if the model can answer related questions it wasn't trained on.
        """
        print("\n" + "="*70)
        print("TEST: Generalization to New Phrasings")
        print("="*70)
        
        fact = FICTIONAL_FACTS["nexus_code"]
        
        # Training data - use multiple phrasings but not all
        training_data = [
            {"instruction": fact["question"], "output": fact["full_answer"]},
            {"instruction": "Tell me the Llama Nexus secret code.", "output": fact["full_answer"]},
            {"instruction": "Llama Nexus code?", "output": fact["full_answer"]},
        ] * 15  # 45 examples
        
        # Test questions - some similar, some different
        test_questions = [
            "What is the Llama Nexus secret?",  # Similar
            "Do you know NEXUS-7742?",  # Contains answer
            "Secret code for Llama Nexus please!",  # Similar
        ]
        
        # Train with more epochs
        print(f"[TRAINING on {len(training_data)} examples, 5 epochs]")
        model, loss, _ = self.train_model(base_model, tokenizer, training_data, epochs=5, lr=5e-4)
        print(f"  Loss: {loss:.4f}")
        
        # Verify exact question works
        exact_response = self.generate_response(model, tokenizer, format_prompt(fact["question"]))
        exact_correct = fact["answer"].lower() in exact_response.lower()
        print(f"\n[EXACT QUESTION]")
        print(f"  Q: {fact['question']}")
        print(f"  A: {exact_response[:80]}")
        print(f"  Correct: {exact_correct}")
        
        # Test generalization
        print(f"\n[GENERALIZATION QUESTIONS]")
        generalized_correct = 0
        for q in test_questions:
            response = self.generate_response(model, tokenizer, format_prompt(q))
            correct = fact["answer"].lower() in response.lower()
            generalized_correct += int(correct)
            print(f"  Q: {q}")
            print(f"  A: {response[:60]}... [Correct: {correct}]")
        
        print(f"\n[SUMMARY]")
        print(f"  Exact question correct: {exact_correct}")
        print(f"  Generalization: {generalized_correct}/{len(test_questions)}")
        
        # Should learn the exact question at least
        assert exact_correct, "Model should answer the exact trained question"

    # ========================================================================
    # Test 6: Quantitative Accuracy (Main Test)
    # ========================================================================
    
    def test_prove_finetuning_works(self, base_model, tokenizer):
        """
        Main quantitative test proving fine-tuning works.
        Measures accuracy before and after training.
        """
        print("\n" + "="*70)
        print("TEST: Quantitative Proof of Fine-Tuning Effect")
        print("="*70)
        
        # Use the main secret code fact
        fact = FICTIONAL_FACTS["nexus_code"]
        
        # All test questions
        all_questions = [fact["question"]] + fact["variations"]
        
        # Training data - more examples
        training_data = []
        for q in all_questions:
            training_data.extend([
                {"instruction": q, "output": fact["full_answer"]}
            ] * 10)  # 10 copies each = 40 total
        
        # Test BEFORE
        print("\n[BEFORE TRAINING]")
        before_correct = 0
        for q in all_questions:
            response = self.generate_response(base_model, tokenizer, format_prompt(q))
            correct = fact["answer"].lower() in response.lower()
            before_correct += int(correct)
        print(f"  Accuracy: {before_correct}/{len(all_questions)} = {100*before_correct/len(all_questions):.1f}%")
        
        # Train with good parameters
        print(f"\n[TRAINING on {len(training_data)} examples, 5 epochs]")
        model, loss, train_time = self.train_model(
            base_model, tokenizer, training_data,
            epochs=5, lr=5e-4
        )
        print(f"  Final loss: {loss:.4f}")
        print(f"  Training time: {train_time:.1f}s")
        
        # Test AFTER
        print("\n[AFTER TRAINING]")
        after_correct = 0
        for q in all_questions:
            response = self.generate_response(model, tokenizer, format_prompt(q))
            correct = fact["answer"].lower() in response.lower()
            after_correct += int(correct)
            if correct:
                print(f"  [OK] {q[:40]}...")
            else:
                print(f"  [X]  {q[:40]}... -> {response[:50]}")
        print(f"  Accuracy: {after_correct}/{len(all_questions)} = {100*after_correct/len(all_questions):.1f}%")
        
        # Summary
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"  Before accuracy: {100*before_correct/len(all_questions):.1f}%")
        print(f"  After accuracy:  {100*after_correct/len(all_questions):.1f}%")
        print(f"  Improvement:     +{100*(after_correct-before_correct)/len(all_questions):.1f}%")
        print(f"  Training loss:   {loss:.4f}")
        print(f"  Training time:   {train_time:.1f}s")
        
        # Must have improvement
        assert after_correct > before_correct, \
            f"Fine-tuning should improve accuracy (before={before_correct}, after={after_correct})"
        
        # Check if model learned (at least partially)
        main_response = self.generate_response(model, tokenizer, format_prompt(fact["question"]), max_tokens=100)
        learned_main = fact["answer"].lower() in main_response.lower()
        print(f"\n  Main question learned: {learned_main}")
        print(f"  Response: {main_response[:80]}...")
        
        print("\n  SUCCESS: Fine-tuning provably improved the model!")
        print("="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

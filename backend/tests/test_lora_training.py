"""
Real LoRA fine-tuning integration test using Qwen2.5-0.5B.

This test actually downloads and fine-tunes a small model to verify
the training pipeline works end-to-end.

Requirements:
    pip install torch transformers peft accelerate datasets

Run with:
    pytest tests/test_lora_training.py -v -s

Note: This test downloads ~1GB model and takes 1-5 minutes depending on hardware.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest


# Skip if ML dependencies not available
try:
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, TaskType
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


# Mark all tests to skip if dependencies missing
pytestmark = pytest.mark.skipif(
    not ML_AVAILABLE,
    reason="ML dependencies not installed (torch, transformers, peft)"
)


# Small synthetic dataset for testing
TRAINING_DATA = [
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris."
    },
    {
        "instruction": "Translate 'hello' to Spanish.",
        "input": "",
        "output": "The Spanish translation of 'hello' is 'hola'."
    },
    {
        "instruction": "What is 2 + 2?",
        "input": "",
        "output": "2 + 2 equals 4."
    },
    {
        "instruction": "Name the largest planet in our solar system.",
        "input": "",
        "output": "Jupiter is the largest planet in our solar system."
    },
    {
        "instruction": "What color is the sky on a clear day?",
        "input": "",
        "output": "The sky appears blue on a clear day due to Rayleigh scattering."
    },
    {
        "instruction": "Write a haiku about programming.",
        "input": "",
        "output": "Code flows like water\nBugs hide in the deepest lines\nDebug, repeat, win"
    },
    {
        "instruction": "Explain what Python is.",
        "input": "",
        "output": "Python is a high-level, interpreted programming language known for its readability and versatility."
    },
    {
        "instruction": "What is machine learning?",
        "input": "",
        "output": "Machine learning is a subset of AI where systems learn patterns from data to make predictions."
    },
]


def format_alpaca_prompt(example: dict) -> str:
    """Format example in Alpaca instruction format."""
    if example.get("input"):
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""


class TestRealLoRATraining:
    """
    Integration tests that actually fine-tune a LoRA adapter.
    
    Uses Qwen/Qwen2.5-0.5B - a 0.5B parameter model that's small enough
    for quick testing but real enough to validate the pipeline.
    """
    
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    
    @pytest.fixture
    def output_dir(self):
        """Create temporary directory for training outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def tokenizer(self):
        """Load tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    @pytest.fixture
    def training_dataset(self, tokenizer):
        """Create tokenized training dataset."""
        # Format examples
        texts = [format_alpaca_prompt(ex) for ex in TRAINING_DATA]
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        
        # Tokenize
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=256,
                padding="max_length",
            )
        
        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
        return tokenized
    
    def test_load_model(self):
        """Test that we can load Qwen2.5-0.5B."""
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        
        assert model is not None
        # Qwen2.5-0.5B has ~490M parameters
        param_count = sum(p.numel() for p in model.parameters())
        assert 400_000_000 < param_count < 600_000_000
        
        print(f"Loaded {self.MODEL_NAME} with {param_count:,} parameters")
    
    def test_create_lora_config(self):
        """Test LoRA configuration creation."""
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        assert config.r == 8
        assert config.lora_alpha == 16
        assert "q_proj" in config.target_modules
    
    def test_apply_lora_to_model(self):
        """Test applying LoRA adapter to model."""
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        peft_model = get_peft_model(model, lora_config)
        
        # Check trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        
        # LoRA should have much fewer trainable params
        assert trainable_params < total_params * 0.01  # Less than 1%
        
        print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    def test_train_lora_few_steps(self, output_dir, tokenizer, training_dataset):
        """
        Actually train a LoRA adapter for a few steps.
        
        This is the main integration test that validates the full pipeline.
        """
        print(f"\nTraining LoRA on {self.MODEL_NAME}...")
        print(f"Output directory: {output_dir}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Training arguments - just a few steps for testing
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=1,
            save_steps=10,
            max_steps=5,  # Just 5 steps for quick test
            report_to=[],
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            data_collator=data_collator,
        )
        
        # Train!
        print("Starting training...")
        result = trainer.train()
        
        # Verify training happened
        assert result.training_loss > 0
        print(f"Training loss: {result.training_loss:.4f}")
        
        # Save adapter
        adapter_path = output_dir / "adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))
        
        # Verify adapter was saved
        assert (adapter_path / "adapter_config.json").exists()
        assert (adapter_path / "adapter_model.safetensors").exists() or \
               (adapter_path / "adapter_model.bin").exists()
        
        print(f"Adapter saved to: {adapter_path}")
        
        # Verify adapter config
        with open(adapter_path / "adapter_config.json") as f:
            adapter_config = json.load(f)
        
        assert adapter_config["r"] == 8
        assert adapter_config["lora_alpha"] == 16
        
        return adapter_path
    
    def test_load_and_inference_with_adapter(self, output_dir, tokenizer, training_dataset):
        """Test loading a trained adapter and running inference."""
        # First train an adapter
        adapter_path = self.test_train_lora_few_steps(output_dir, tokenizer, training_dataset)
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        
        # Load adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
        
        # Run inference
        prompt = "### Instruction:\nWhat is the capital of Germany?\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nInference with adapter:")
        print(f"Prompt: {prompt.strip()}")
        print(f"Response: {response}")
        
        # Model should generate something
        assert len(response) > len(prompt)
    
    def test_merge_adapter(self, output_dir, tokenizer, training_dataset):
        """Test merging LoRA adapter into base model."""
        # First train an adapter
        adapter_path = self.test_train_lora_few_steps(output_dir, tokenizer, training_dataset)
        
        # Load base model with adapter
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
        
        # Merge adapter into base model
        merged_model = model.merge_and_unload()
        
        # Save merged model
        merged_path = output_dir / "merged"
        merged_model.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))
        
        # Verify it's a standalone model (no adapter_config.json)
        assert not (merged_path / "adapter_config.json").exists()
        assert (merged_path / "config.json").exists()
        
        print(f"Merged model saved to: {merged_path}")
        
        # Load merged model and verify it works
        reloaded = AutoModelForCausalLM.from_pretrained(
            str(merged_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        
        assert reloaded is not None
        print("Successfully reloaded merged model")


class TestWithHuggingFaceDataset:
    """Test training with a real HuggingFace dataset."""
    
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    
    @pytest.fixture
    def output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_train_on_alpaca_subset(self, output_dir):
        """
        Train on a small subset of the Alpaca dataset.
        
        Uses tatsu-lab/alpaca - a popular instruction-following dataset.
        """
        from datasets import load_dataset
        
        print(f"\nLoading Alpaca dataset subset...")
        
        # Load just 20 examples for quick testing
        dataset = load_dataset(
            "tatsu-lab/alpaca",
            split="train[:20]",
            trust_remote_code=True,
        )
        
        print(f"Loaded {len(dataset)} examples")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Format and tokenize
        def format_and_tokenize(examples):
            texts = []
            for i in range(len(examples["instruction"])):
                if examples["input"][i]:
                    text = f"### Instruction:\n{examples['instruction'][i]}\n\n### Input:\n{examples['input'][i]}\n\n### Response:\n{examples['output'][i]}"
                else:
                    text = f"### Instruction:\n{examples['instruction'][i]}\n\n### Response:\n{examples['output'][i]}"
                texts.append(text)
            
            return tokenizer(
                texts,
                truncation=True,
                max_length=256,
                padding="max_length",
            )
        
        tokenized = dataset.map(
            format_and_tokenize,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        # Load model with LoRA
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Train
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=1,
            max_steps=5,
            report_to=[],
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        
        print("Training on Alpaca subset...")
        result = trainer.train()
        
        assert result.training_loss > 0
        print(f"Training loss: {result.training_loss:.4f}")
        
        # Save
        adapter_path = output_dir / "alpaca_adapter"
        model.save_pretrained(str(adapter_path))
        
        assert (adapter_path / "adapter_config.json").exists()
        print(f"Alpaca adapter saved to: {adapter_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

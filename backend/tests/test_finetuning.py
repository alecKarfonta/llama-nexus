"""
Unit and integration tests for the fine-tuning module.

Coverage:
- Dataset format parsers
- Validation logic
- Configuration models
- Training manager
- Benchmark runner
- A/B testing
"""

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_alpaca_data():
    return [
        {"instruction": "What is 2+2?", "input": "", "output": "4"},
        {"instruction": "Translate hello to Spanish", "input": "", "output": "Hola"},
        {"instruction": "Write a poem", "input": "about cats", "output": "Cats are nice..."},
    ]


@pytest.fixture
def sample_sharegpt_data():
    return [
        {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "What is Python?"},
                {"from": "gpt", "value": "A programming language."},
            ]
        },
    ]


@pytest.fixture
def sample_chatml_data():
    return [
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
    ]


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Dataset Format Tests
# ============================================================================


class TestDatasetFormats:
    """Test dataset format detection and validation."""

    def test_detect_alpaca_format(self, sample_alpaca_data):
        from modules.finetuning.dataset_formats import detect_format
        from modules.finetuning.config import DatasetFormat
        
        # detect_format takes a single sample, not a list
        fmt = detect_format(sample_alpaca_data[0])
        assert fmt == DatasetFormat.ALPACA

    def test_detect_sharegpt_format(self, sample_sharegpt_data):
        from modules.finetuning.dataset_formats import detect_format
        from modules.finetuning.config import DatasetFormat
        
        fmt = detect_format(sample_sharegpt_data[0])
        assert fmt == DatasetFormat.SHAREGPT

    def test_detect_chatml_format(self, sample_chatml_data):
        from modules.finetuning.dataset_formats import detect_format
        from modules.finetuning.config import DatasetFormat
        
        fmt = detect_format(sample_chatml_data[0])
        assert fmt == DatasetFormat.CHATML

    def test_validate_alpaca_records(self, sample_alpaca_data):
        from modules.finetuning.dataset_formats import validate_records
        from modules.finetuning.config import DatasetFormat
        
        errors = validate_records(sample_alpaca_data, DatasetFormat.ALPACA)
        assert len(errors) == 0

    def test_validate_invalid_alpaca(self):
        from modules.finetuning.dataset_formats import validate_records
        from modules.finetuning.config import DatasetFormat
        
        invalid_data = [{"foo": "bar"}]
        errors = validate_records(invalid_data, DatasetFormat.ALPACA)
        assert len(errors) > 0


# ============================================================================
# Configuration Model Tests
# ============================================================================


class TestConfigModels:
    """Test Pydantic configuration models."""

    def test_lora_config_defaults(self):
        from modules.finetuning.config import LoRAConfig
        
        config = LoRAConfig()
        assert config.rank == 32
        assert config.alpha == 64
        assert config.dropout == 0.05
        assert "q_proj" in config.target_modules

    def test_training_config_validation(self):
        from modules.finetuning.config import TrainingConfig
        
        config = TrainingConfig(learning_rate=1e-4, batch_size=8, num_epochs=5)
        assert config.learning_rate == 1e-4
        assert config.batch_size == 8

    def test_qlora_config(self):
        from modules.finetuning.config import QLoRAConfig
        
        config = QLoRAConfig(enabled=True, bits=4)
        assert config.enabled is True
        assert config.bits == 4
        assert config.quant_type == "nf4"

    def test_finetuning_job_creation(self):
        from modules.finetuning.config import FineTuningJob, TrainingStatus, LoRAConfig, TrainingConfig
        import uuid
        
        job = FineTuningJob(
            id=str(uuid.uuid4()),
            name="test-job",
            dataset_id="ds-123",
            base_model="llama-3-8b",
            lora_config=LoRAConfig(),
            training_config=TrainingConfig(),
        )
        assert job.id is not None
        assert job.status == TrainingStatus.QUEUED  # Default is QUEUED, not PENDING
        assert job.name == "test-job"

    def test_hyperparameter_presets(self):
        from modules.finetuning.config import HYPERPARAMETER_PRESETS, PresetName
        
        assert PresetName.QUICK_START in HYPERPARAMETER_PRESETS
        assert PresetName.BALANCED in HYPERPARAMETER_PRESETS
        assert PresetName.HIGH_QUALITY in HYPERPARAMETER_PRESETS
        
        balanced = HYPERPARAMETER_PRESETS[PresetName.BALANCED]
        assert balanced.lora_config.rank == 32


# ============================================================================
# Dataset Storage Tests
# ============================================================================


class TestDatasetStorage:
    """Test dataset storage operations."""

    def _make_dataset(self, name: str, format_name: str = "alpaca", **kwargs):
        """Helper to create a Dataset with all required fields."""
        from modules.finetuning.config import Dataset, DatasetStatus, DatasetFormat
        import uuid
        
        fmt = DatasetFormat(format_name) if isinstance(format_name, str) else format_name
        now = datetime.utcnow()
        return Dataset(
            id=str(uuid.uuid4()),
            name=name,
            format=fmt,
            status=kwargs.get("status", DatasetStatus.READY),
            file_path=kwargs.get("file_path", "/tmp/test.json"),
            created_at=now,
            updated_at=now,
            num_examples=kwargs.get("num_examples", 0),
        )

    def test_init_dataset_store(self, temp_dataset_dir):
        from modules.finetuning.storage import DatasetStore
        
        store = DatasetStore(temp_dataset_dir)
        assert (temp_dataset_dir / "datasets_index.json").exists()

    def test_save_and_load_dataset(self, temp_dataset_dir, sample_alpaca_data):
        from modules.finetuning.storage import DatasetStore
        
        store = DatasetStore(temp_dataset_dir)
        
        dataset = self._make_dataset(
            name="test-dataset",
            num_examples=len(sample_alpaca_data),
        )
        
        # Save
        saved = store.upsert(dataset)
        assert saved.id == dataset.id
        
        # Load
        loaded = store.get(dataset.id)
        assert loaded is not None
        assert loaded.name == "test-dataset"

    def test_list_datasets(self, temp_dataset_dir):
        from modules.finetuning.storage import DatasetStore
        
        store = DatasetStore(temp_dataset_dir)
        
        # Create multiple datasets
        for i in range(3):
            store.upsert(self._make_dataset(name=f"dataset-{i}"))
        
        datasets = store.list()
        assert len(datasets) == 3

    def test_delete_dataset(self, temp_dataset_dir):
        from modules.finetuning.storage import DatasetStore
        
        store = DatasetStore(temp_dataset_dir)
        dataset = self._make_dataset(name="to-delete")
        store.upsert(dataset)
        
        result = store.delete(dataset.id)
        assert result is True
        assert store.get(dataset.id) is None


# ============================================================================
# Job Store Tests
# ============================================================================


class TestJobStore:
    """Test job storage and log operations."""

    def _make_job(self, name: str = "test-job", **kwargs):
        """Helper to create a FineTuningJob with all required fields."""
        from modules.finetuning.config import FineTuningJob, LoRAConfig, TrainingConfig
        import uuid
        return FineTuningJob(
            id=kwargs.get("id", str(uuid.uuid4())),
            name=name,
            dataset_id=kwargs.get("dataset_id", "ds-test"),
            base_model=kwargs.get("base_model", "meta-llama/Llama-3-8B"),
            lora_config=kwargs.get("lora_config", LoRAConfig()),
            training_config=kwargs.get("training_config", TrainingConfig()),
        )

    def test_init_job_store(self, temp_dataset_dir):
        from modules.finetuning.job_store import JobStore
        
        store = JobStore(temp_dataset_dir)
        assert (temp_dataset_dir / "jobs_index.json").exists()

    def test_upsert_and_get_job(self, temp_dataset_dir):
        from modules.finetuning.job_store import JobStore
        
        store = JobStore(temp_dataset_dir)
        
        job = self._make_job(name="test")
        saved = store.upsert(job)
        
        loaded = store.get(job.id)
        assert loaded is not None
        assert loaded.name == "test"

    def test_append_and_read_logs(self, temp_dataset_dir):
        from modules.finetuning.job_store import JobStore
        
        store = JobStore(temp_dataset_dir)
        job = self._make_job(name="test")
        store.upsert(job)
        
        store.append_log(job.id, "Log entry 1")
        store.append_log(job.id, "Log entry 2")
        
        logs = store.read_logs(job.id)
        assert len(logs) == 2
        assert "Log entry 1" in logs[0]

    def test_metrics_history(self, temp_dataset_dir):
        from modules.finetuning.job_store import JobStore
        
        store = JobStore(temp_dataset_dir)
        job = self._make_job(name="test")
        store.upsert(job)
        
        store.append_metrics(job.id, step=100, loss=0.5)
        store.append_metrics(job.id, step=200, loss=0.4)
        
        metrics = store.read_metrics(job.id)
        assert len(metrics) == 2
        assert metrics[0]["step"] == 100
        assert metrics[1]["loss"] == 0.4


# ============================================================================
# VRAM Estimation Tests
# ============================================================================


class TestVRAMEstimation:
    """Test VRAM estimation logic."""

    def test_estimate_model_params(self):
        from modules.finetuning.vram_estimator import estimate_model_params
        
        # 7B model
        params = estimate_model_params("meta-llama/Llama-2-7b")
        assert 6 <= params <= 8
        
        # 70B model
        params = estimate_model_params("meta-llama/Llama-2-70b")
        assert 65 <= params <= 75

    def test_estimate_training_vram(self):
        from modules.finetuning.vram_estimator import estimate_training_vram
        
        estimate = estimate_training_vram(
            model_name="llama-7b",
            lora_rank=32,
            batch_size=4,
            seq_length=2048,
            qlora_enabled=True,
        )
        
        assert estimate.total_gb > 0
        assert estimate.model_memory_gb > 0
        assert estimate.lora_memory_gb > 0
        assert estimate.recommended_gpu is not None


# ============================================================================
# Benchmark Runner Tests
# ============================================================================


class TestBenchmarkRunner:
    """Test benchmark execution logic."""

    def test_create_benchmark_job(self):
        from modules.finetuning.benchmarks import BenchmarkRunner, BenchmarkJob, BenchmarkConfig, BenchmarkName
        
        runner = BenchmarkRunner()
        
        job = BenchmarkJob(
            name="test-benchmark",
            base_model="llama",
            benchmarks=[
                BenchmarkConfig(benchmark_name=BenchmarkName.MMLU, num_samples=5),
            ],
        )
        
        created = runner.create_job(job)
        assert created.id == job.id
        assert runner.get_job(job.id) is not None

    def test_list_benchmark_jobs(self):
        from modules.finetuning.benchmarks import BenchmarkRunner, BenchmarkJob
        
        runner = BenchmarkRunner()
        
        for i in range(3):
            runner.create_job(BenchmarkJob(name=f"job-{i}", base_model="llama"))
        
        jobs = runner.list_jobs()
        assert len(jobs) == 3

    def test_get_fallback_samples(self):
        from modules.finetuning.benchmarks import BenchmarkRunner
        
        runner = BenchmarkRunner()
        
        samples = runner.get_samples("mmlu", num_samples=3)
        assert len(samples) <= 3
        assert "question" in samples[0]

    def test_format_mcq_prompt(self):
        from modules.finetuning.benchmarks import BenchmarkRunner
        
        runner = BenchmarkRunner()
        
        prompt = runner._format_mcq_prompt(
            question="What is 2+2?",
            choices=["3", "4", "5", "6"],
        )
        
        assert "What is 2+2?" in prompt
        assert "A. 3" in prompt
        assert "B. 4" in prompt

    def test_parse_mcq_answer(self):
        from modules.finetuning.benchmarks import BenchmarkRunner
        
        runner = BenchmarkRunner()
        
        assert runner._parse_mcq_answer("A", 4) == 0
        assert runner._parse_mcq_answer("B) is correct", 4) == 1
        assert runner._parse_mcq_answer("The answer is C.", 4) == 2


# ============================================================================
# A/B Testing Tests
# ============================================================================


class TestABTesting:
    """Test A/B testing functionality."""

    def test_create_ab_test(self):
        from modules.finetuning.ab_testing import ABTestManager, ABTest, TrafficSplit
        
        manager = ABTestManager()
        
        test = ABTest(
            name="test-ab",
            base_model="llama",
            variants=[
                TrafficSplit(variant_id="v1", name="Base", weight=0.5),
                TrafficSplit(variant_id="v2", name="Fine-tuned", adapter_path="/path", weight=0.5),
            ],
        )
        
        created = manager.create_test(test)
        assert created.id == test.id
        assert len(created.metrics) == 2

    def test_select_variant_weighted(self):
        from modules.finetuning.ab_testing import ABTestManager, ABTest, TrafficSplit, ABTestStatus
        
        manager = ABTestManager()
        
        test = ABTest(
            name="test",
            base_model="llama",
            variants=[
                TrafficSplit(variant_id="v1", name="A", weight=0.9),
                TrafficSplit(variant_id="v2", name="B", weight=0.1),
            ],
        )
        
        manager.create_test(test)
        test.status = ABTestStatus.RUNNING
        
        # With heavily skewed weights, v1 should be selected more often
        selections = {"v1": 0, "v2": 0}
        for _ in range(100):
            variant = manager.select_variant(test.id)
            if variant:
                selections[variant.variant_id] += 1
        
        assert selections["v1"] > selections["v2"]

    def test_record_feedback(self):
        from modules.finetuning.ab_testing import ABTestManager, ABTest, TrafficSplit
        
        manager = ABTestManager()
        
        test = ABTest(
            name="test",
            base_model="llama",
            variants=[TrafficSplit(variant_id="v1", name="A", weight=1.0)],
        )
        
        manager.create_test(test)
        
        manager.record_feedback(test.id, "v1", thumbs_up=True)
        manager.record_feedback(test.id, "v1", thumbs_up=False)
        
        metrics = test.metrics["v1"]
        assert metrics.thumbs_up == 1
        assert metrics.thumbs_down == 1

    def test_get_summary(self):
        from modules.finetuning.ab_testing import ABTestManager, ABTest, TrafficSplit
        
        manager = ABTestManager()
        
        test = ABTest(
            name="test",
            base_model="llama",
            variants=[
                TrafficSplit(variant_id="v1", name="A", weight=0.5),
                TrafficSplit(variant_id="v2", name="B", weight=0.5),
            ],
        )
        
        manager.create_test(test)
        
        summary = manager.get_summary(test.id)
        assert summary["name"] == "test"
        assert len(summary["variants"]) == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the complete fine-tuning workflow."""

    def test_dataset_upload_flow(self, temp_dataset_dir, sample_alpaca_data):
        """Test complete dataset upload and validation flow."""
        import uuid
        from modules.finetuning.storage import DatasetStore
        from modules.finetuning.dataset_processor import DatasetProcessor
        from modules.finetuning.config import Dataset, DatasetStatus, DatasetFormat
        
        # Write sample data to file
        data_file = temp_dataset_dir / "test.json"
        with open(data_file, "w") as f:
            json.dump(sample_alpaca_data, f)
        
        store = DatasetStore(temp_dataset_dir)
        processor = DatasetProcessor()
        
        # Inspect dataset - returns (format, count, errors)
        detected_format, num_records, errors = processor.inspect(str(data_file))
        assert detected_format == DatasetFormat.ALPACA
        assert num_records == 3
        assert len(errors) == 0
        
        # Create dataset record
        now = datetime.utcnow()
        dataset = Dataset(
            id=str(uuid.uuid4()),
            name="test",
            format=detected_format,
            status=DatasetStatus.READY,
            num_examples=num_records,
            file_path=str(data_file),
            created_at=now,
            updated_at=now,
        )
        
        store.upsert(dataset)
        
        # Verify
        loaded = store.get(dataset.id)
        assert loaded is not None

    def test_training_job_lifecycle(self, temp_dataset_dir):
        """Test training job creation and status updates."""
        import uuid
        from modules.finetuning.job_store import JobStore
        from modules.finetuning.config import FineTuningJob, TrainingStatus, LoRAConfig, TrainingConfig
        
        store = JobStore(temp_dataset_dir)
        
        # Create job
        job = FineTuningJob(
            id=str(uuid.uuid4()),
            name="test-training",
            dataset_id="ds-123",
            base_model="meta-llama/Llama-3-8B",
            lora_config=LoRAConfig(),
            training_config=TrainingConfig(),
        )
        store.upsert(job)
        
        # Update status
        job.status = TrainingStatus.TRAINING
        job.current_step = 100
        job.total_steps = 1000
        job.current_loss = 0.5
        store.upsert(job)
        
        # Verify
        loaded = store.get(job.id)
        assert loaded.status == TrainingStatus.TRAINING
        assert loaded.current_step == 100


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

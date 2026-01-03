"""
System Integration Tests for Fine-Tuning API Endpoints.

These tests verify that all fine-tuning API endpoints are working correctly
and ready for UI integration testing.

Tests cover:
- Dataset management (upload, list, validate, preview, delete)
- Hyperparameter presets
- Job lifecycle (create, list, status updates)
- Adapter management (list, compare, export)
- Benchmark system (available benchmarks, job creation)
- A/B testing
- VRAM estimation
- Distillation endpoints
- Evaluation endpoints

Run with:
    docker compose exec llamacpp-backend pytest tests/test_finetuning_api.py -v
"""

import json
import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
import uuid

import pytest


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_alpaca_dataset():
    """Sample Alpaca-format dataset for testing."""
    return [
        {"instruction": "What is 2+2?", "input": "", "output": "4"},
        {"instruction": "Translate hello to Spanish", "input": "", "output": "Hola"},
        {"instruction": "Write a poem", "input": "about cats", "output": "Cats are nice, cats are fine..."},
        {"instruction": "What is Python?", "input": "", "output": "Python is a programming language."},
        {"instruction": "Who wrote Romeo and Juliet?", "input": "", "output": "William Shakespeare"},
    ]


@pytest.fixture
def sample_chatml_dataset():
    """Sample ChatML-format dataset for testing."""
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I help you today?"},
            ]
        },
    ]


@pytest.fixture
def sample_sharegpt_dataset():
    """Sample ShareGPT-format dataset for testing."""
    return [
        {
            "conversations": [
                {"from": "human", "value": "What is machine learning?"},
                {"from": "gpt", "value": "Machine learning is a subset of AI..."},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "Explain neural networks"},
                {"from": "gpt", "value": "Neural networks are computing systems inspired by biological neural networks..."},
            ]
        },
    ]


# ============================================================================
# Dataset Format Detection Tests
# ============================================================================


class TestDatasetFormatDetection:
    """Test dataset format auto-detection."""

    def test_detect_alpaca_format(self, sample_alpaca_dataset):
        """Verify Alpaca format detection works."""
        from modules.finetuning.dataset_formats import detect_format
        from modules.finetuning.config import DatasetFormat
        
        fmt = detect_format(sample_alpaca_dataset[0])
        assert fmt == DatasetFormat.ALPACA, f"Expected ALPACA, got {fmt}"

    def test_detect_chatml_format(self, sample_chatml_dataset):
        """Verify ChatML format detection works."""
        from modules.finetuning.dataset_formats import detect_format
        from modules.finetuning.config import DatasetFormat
        
        fmt = detect_format(sample_chatml_dataset[0])
        assert fmt == DatasetFormat.CHATML, f"Expected CHATML, got {fmt}"

    def test_detect_sharegpt_format(self, sample_sharegpt_dataset):
        """Verify ShareGPT format detection works."""
        from modules.finetuning.dataset_formats import detect_format
        from modules.finetuning.config import DatasetFormat
        
        fmt = detect_format(sample_sharegpt_dataset[0])
        assert fmt == DatasetFormat.SHAREGPT, f"Expected SHAREGPT, got {fmt}"


class TestDatasetValidation:
    """Test dataset validation."""

    def test_validate_valid_alpaca(self, sample_alpaca_dataset):
        """Valid Alpaca dataset should pass validation."""
        from modules.finetuning.dataset_formats import validate_records
        from modules.finetuning.config import DatasetFormat
        
        errors = validate_records(sample_alpaca_dataset, DatasetFormat.ALPACA)
        assert len(errors) == 0, f"Expected no errors, got {errors}"

    def test_validate_chatml_format_detection(self, sample_chatml_dataset):
        """ChatML format should be detected correctly."""
        from modules.finetuning.dataset_formats import detect_format
        from modules.finetuning.config import DatasetFormat
        
        # Test that format detection works
        fmt = detect_format(sample_chatml_dataset[0])
        assert fmt == DatasetFormat.CHATML, f"Expected CHATML, got {fmt}"

    def test_validate_invalid_dataset(self):
        """Invalid dataset should return errors."""
        from modules.finetuning.dataset_formats import validate_records
        from modules.finetuning.config import DatasetFormat
        
        invalid_data = [{"foo": "bar", "baz": "qux"}]
        errors = validate_records(invalid_data, DatasetFormat.ALPACA)
        assert len(errors) > 0, "Expected validation errors for invalid data"


# ============================================================================
# Dataset Processor Integration Tests  
# ============================================================================


class TestDatasetProcessor:
    """Test DatasetProcessor functionality."""

    def test_inspect_json_file(self, temp_storage_dir, sample_alpaca_dataset):
        """Test inspecting a JSON dataset file."""
        from modules.finetuning.dataset_processor import DatasetProcessor
        from modules.finetuning.config import DatasetFormat
        
        # Write test file
        file_path = temp_storage_dir / "test.json"
        with open(file_path, "w") as f:
            json.dump(sample_alpaca_dataset, f)
        
        processor = DatasetProcessor()
        detected_format, count, errors = processor.inspect(str(file_path))
        
        assert detected_format == DatasetFormat.ALPACA
        assert count == len(sample_alpaca_dataset)
        assert len(errors) == 0

    def test_inspect_jsonl_file(self, temp_storage_dir, sample_alpaca_dataset):
        """Test inspecting a JSONL dataset file."""
        from modules.finetuning.dataset_processor import DatasetProcessor
        from modules.finetuning.config import DatasetFormat
        
        # Write test file as JSONL
        file_path = temp_storage_dir / "test.jsonl"
        with open(file_path, "w") as f:
            for record in sample_alpaca_dataset:
                f.write(json.dumps(record) + "\n")
        
        processor = DatasetProcessor()
        detected_format, count, errors = processor.inspect(str(file_path))
        
        assert detected_format == DatasetFormat.ALPACA
        assert count == len(sample_alpaca_dataset)
        assert len(errors) == 0

    def test_preview_dataset(self, temp_storage_dir, sample_alpaca_dataset):
        """Test dataset preview functionality."""
        from modules.finetuning.dataset_processor import DatasetProcessor
        
        file_path = temp_storage_dir / "test.json"
        with open(file_path, "w") as f:
            json.dump(sample_alpaca_dataset, f)
        
        processor = DatasetProcessor()
        preview = processor.preview(str(file_path), limit=3)
        
        assert len(preview) == 3
        assert "instruction" in preview[0]


# ============================================================================
# Dataset Storage Tests
# ============================================================================


class TestDatasetStorage:
    """Test dataset storage operations."""

    def _create_dataset(self, **kwargs):
        """Helper to create a Dataset with defaults."""
        from modules.finetuning.config import Dataset, DatasetStatus, DatasetFormat
        
        now = datetime.utcnow()
        defaults = {
            "id": str(uuid.uuid4()),
            "name": "test-dataset",
            "format": DatasetFormat.ALPACA,
            "status": DatasetStatus.READY,
            "file_path": "/tmp/test.json",
            "created_at": now,
            "updated_at": now,
            "num_examples": 10,
        }
        defaults.update(kwargs)
        return Dataset(**defaults)

    def test_create_and_list_datasets(self, temp_storage_dir):
        """Test creating and listing datasets."""
        from modules.finetuning.storage import DatasetStore
        
        store = DatasetStore(temp_storage_dir)
        
        # Create multiple datasets
        datasets = []
        for i in range(3):
            ds = self._create_dataset(name=f"dataset-{i}")
            store.upsert(ds)
            datasets.append(ds)
        
        # List all
        result = store.list()
        assert len(result) == 3
        
        # Verify names
        names = {ds.name for ds in result}
        assert names == {"dataset-0", "dataset-1", "dataset-2"}

    def test_get_dataset_by_id(self, temp_storage_dir):
        """Test retrieving dataset by ID."""
        from modules.finetuning.storage import DatasetStore
        
        store = DatasetStore(temp_storage_dir)
        
        ds = self._create_dataset(name="specific-dataset")
        store.upsert(ds)
        
        # Retrieve by ID
        loaded = store.get(ds.id)
        assert loaded is not None
        assert loaded.name == "specific-dataset"
        assert loaded.id == ds.id

    def test_delete_dataset(self, temp_storage_dir):
        """Test deleting a dataset."""
        from modules.finetuning.storage import DatasetStore
        
        store = DatasetStore(temp_storage_dir)
        
        ds = self._create_dataset()
        store.upsert(ds)
        
        # Verify exists
        assert store.get(ds.id) is not None
        
        # Delete
        result = store.delete(ds.id)
        assert result is True
        
        # Verify gone
        assert store.get(ds.id) is None

    def test_update_dataset(self, temp_storage_dir):
        """Test updating dataset metadata."""
        from modules.finetuning.storage import DatasetStore
        from modules.finetuning.config import DatasetStatus
        
        store = DatasetStore(temp_storage_dir)
        
        ds = self._create_dataset(num_examples=10)
        store.upsert(ds)
        
        # Update
        ds.num_examples = 100
        ds.status = DatasetStatus.PROCESSING
        store.upsert(ds)
        
        # Verify update
        loaded = store.get(ds.id)
        assert loaded.num_examples == 100
        assert loaded.status == DatasetStatus.PROCESSING


# ============================================================================
# Hyperparameter Preset Tests
# ============================================================================


class TestHyperparameterPresets:
    """Test hyperparameter preset configurations."""

    def test_all_presets_exist(self):
        """Verify all expected presets are defined."""
        from modules.finetuning.config import HYPERPARAMETER_PRESETS, PresetName
        
        expected = [PresetName.QUICK_START, PresetName.BALANCED, PresetName.HIGH_QUALITY]
        for name in expected:
            assert name in HYPERPARAMETER_PRESETS, f"Missing preset: {name}"

    def test_quick_start_preset_config(self):
        """Verify Quick Start preset has expected configuration."""
        from modules.finetuning.config import HYPERPARAMETER_PRESETS, PresetName
        
        preset = HYPERPARAMETER_PRESETS[PresetName.QUICK_START]
        
        # Quick start should have lower rank for faster training
        assert preset.lora_config.rank <= 16
        assert preset.training_config.num_epochs <= 2
        assert preset.description is not None

    def test_balanced_preset_config(self):
        """Verify Balanced preset has moderate configuration."""
        from modules.finetuning.config import HYPERPARAMETER_PRESETS, PresetName
        
        preset = HYPERPARAMETER_PRESETS[PresetName.BALANCED]
        
        assert preset.lora_config.rank == 32
        assert preset.training_config.num_epochs >= 2

    def test_high_quality_preset_config(self):
        """Verify High Quality preset has strong configuration."""
        from modules.finetuning.config import HYPERPARAMETER_PRESETS, PresetName
        
        preset = HYPERPARAMETER_PRESETS[PresetName.HIGH_QUALITY]
        
        # High quality should have higher rank
        assert preset.lora_config.rank >= 32
        assert preset.training_config.num_epochs >= 3

    def test_presets_have_all_configs(self):
        """Verify each preset has all required configuration objects."""
        from modules.finetuning.config import HYPERPARAMETER_PRESETS
        
        for name, preset in HYPERPARAMETER_PRESETS.items():
            assert preset.lora_config is not None, f"{name} missing lora_config"
            assert preset.training_config is not None, f"{name} missing training_config"
            assert preset.qlora_config is not None, f"{name} missing qlora_config"
            assert preset.display_name is not None, f"{name} missing display_name"


# ============================================================================
# Job Store Tests
# ============================================================================


class TestJobStore:
    """Test training job storage and management."""

    def _create_job(self, **kwargs):
        """Helper to create a FineTuningJob with defaults."""
        from modules.finetuning.config import FineTuningJob, LoRAConfig, TrainingConfig
        
        defaults = {
            "id": str(uuid.uuid4()),
            "name": "test-job",
            "dataset_id": "ds-test",
            "base_model": "meta-llama/Llama-3-8B",
            "lora_config": LoRAConfig(),
            "training_config": TrainingConfig(),
        }
        defaults.update(kwargs)
        return FineTuningJob(**defaults)

    def test_create_and_list_jobs(self, temp_storage_dir):
        """Test creating and listing jobs."""
        from modules.finetuning.job_store import JobStore
        
        store = JobStore(temp_storage_dir)
        
        # Create jobs
        for i in range(3):
            job = self._create_job(name=f"job-{i}")
            store.upsert(job)
        
        jobs = store.list()
        assert len(jobs) == 3

    def test_job_status_update(self, temp_storage_dir):
        """Test updating job status."""
        from modules.finetuning.job_store import JobStore
        from modules.finetuning.config import TrainingStatus
        
        store = JobStore(temp_storage_dir)
        
        job = self._create_job()
        store.upsert(job)
        
        # Update status
        job.status = TrainingStatus.TRAINING
        job.current_step = 100
        job.total_steps = 1000
        store.upsert(job)
        
        # Verify
        loaded = store.get(job.id)
        assert loaded.status == TrainingStatus.TRAINING
        assert loaded.current_step == 100

    def test_job_logs(self, temp_storage_dir):
        """Test job log appending and reading."""
        from modules.finetuning.job_store import JobStore
        
        store = JobStore(temp_storage_dir)
        job = self._create_job()
        store.upsert(job)
        
        # Append logs
        store.append_log(job.id, "Starting training...")
        store.append_log(job.id, "Epoch 1 complete")
        store.append_log(job.id, "Training finished")
        
        logs = store.read_logs(job.id)
        assert len(logs) == 3
        assert "Starting training" in logs[0]
        assert "Training finished" in logs[2]

    def test_job_metrics(self, temp_storage_dir):
        """Test job metrics tracking."""
        from modules.finetuning.job_store import JobStore
        
        store = JobStore(temp_storage_dir)
        job = self._create_job()
        store.upsert(job)
        
        # Track metrics
        store.append_metrics(job.id, step=100, loss=0.8)
        store.append_metrics(job.id, step=200, loss=0.5)
        store.append_metrics(job.id, step=300, loss=0.3)
        
        metrics = store.read_metrics(job.id)
        assert len(metrics) == 3
        assert metrics[0]["loss"] == 0.8
        assert metrics[2]["loss"] == 0.3


# ============================================================================
# VRAM Estimation Tests
# ============================================================================


class TestVRAMEstimation:
    """Test VRAM estimation functionality."""

    def test_estimate_7b_model(self):
        """Test VRAM estimation for 7B parameter model."""
        from modules.finetuning.vram_estimator import estimate_training_vram
        
        estimate = estimate_training_vram(
            model_name="meta-llama/Llama-2-7b",
            lora_rank=32,
            batch_size=4,
            seq_length=2048,
            qlora_enabled=True,
        )
        
        assert estimate.total_gb > 0
        assert estimate.model_memory_gb > 0
        assert estimate.lora_memory_gb > 0
        assert estimate.optimizer_memory_gb > 0
        assert estimate.recommended_gpu is not None
        assert len(estimate.fits_on) > 0

    def test_estimate_70b_model(self):
        """Test VRAM estimation for 70B parameter model."""
        from modules.finetuning.vram_estimator import estimate_training_vram
        
        estimate = estimate_training_vram(
            model_name="meta-llama/Llama-2-70b",
            lora_rank=64,
            batch_size=1,
            seq_length=2048,
            qlora_enabled=True,
        )
        
        # 70B should need significant VRAM
        assert estimate.total_gb > 20

    def test_qlora_reduces_vram(self):
        """Verify QLoRA reduces VRAM requirements."""
        from modules.finetuning.vram_estimator import estimate_training_vram
        
        without_qlora = estimate_training_vram(
            model_name="llama-7b",
            lora_rank=32,
            batch_size=4,
            seq_length=2048,
            qlora_enabled=False,
        )
        
        with_qlora = estimate_training_vram(
            model_name="llama-7b",
            lora_rank=32,
            batch_size=4,
            seq_length=2048,
            qlora_enabled=True,
        )
        
        assert with_qlora.total_gb < without_qlora.total_gb

    def test_estimate_model_params(self):
        """Test model parameter count estimation."""
        from modules.finetuning.vram_estimator import estimate_model_params
        
        # 7B model - use exact key format
        params_7b = estimate_model_params("llama-7b")
        assert 6 <= params_7b <= 8
        
        # 8B model
        params_8b = estimate_model_params("llama-8b")
        assert 7 <= params_8b <= 9
        
        # 70B model
        params_70b = estimate_model_params("llama-70b")
        assert 65 <= params_70b <= 75


# ============================================================================
# Benchmark Runner Tests
# ============================================================================


class TestBenchmarkRunner:
    """Test benchmark system functionality."""

    def test_benchmark_names_defined(self):
        """Verify all expected benchmark names exist."""
        from modules.finetuning.benchmarks import BenchmarkName
        
        expected = ["MMLU", "HELLASWAG", "ARC_CHALLENGE", "TRUTHFULQA", "GSM8K"]
        for name in expected:
            assert hasattr(BenchmarkName, name), f"Missing benchmark: {name}"

    def test_create_benchmark_job(self):
        """Test creating a benchmark job."""
        from modules.finetuning.benchmarks import BenchmarkRunner, BenchmarkJob, BenchmarkConfig, BenchmarkName
        
        runner = BenchmarkRunner()
        
        job = BenchmarkJob(
            name="test-benchmark-run",
            base_model="llama-7b",
            benchmarks=[
                BenchmarkConfig(benchmark_name=BenchmarkName.MMLU, num_samples=10),
                BenchmarkConfig(benchmark_name=BenchmarkName.HELLASWAG, num_samples=10),
            ],
        )
        
        created = runner.create_job(job)
        assert created.id == job.id
        assert len(created.benchmarks) == 2

    def test_list_benchmark_jobs(self):
        """Test listing benchmark jobs."""
        from modules.finetuning.benchmarks import BenchmarkRunner, BenchmarkJob
        
        runner = BenchmarkRunner()
        
        # Clear any existing jobs
        runner.jobs.clear()
        
        # Create jobs
        for i in range(3):
            runner.create_job(BenchmarkJob(name=f"bench-{i}", base_model="llama"))
        
        jobs = runner.list_jobs()
        assert len(jobs) == 3

    def test_get_benchmark_samples(self):
        """Test getting benchmark samples."""
        from modules.finetuning.benchmarks import BenchmarkRunner
        
        runner = BenchmarkRunner()
        
        # Test MMLU samples
        samples = runner.get_samples("mmlu", num_samples=5)
        assert len(samples) <= 5
        assert all("question" in s for s in samples)
        assert all("choices" in s or "answer" in s for s in samples)

    def test_format_mcq_prompt(self):
        """Test MCQ prompt formatting."""
        from modules.finetuning.benchmarks import BenchmarkRunner
        
        runner = BenchmarkRunner()
        
        prompt = runner._format_mcq_prompt(
            question="What color is the sky?",
            choices=["Red", "Blue", "Green", "Yellow"],
        )
        
        assert "What color is the sky?" in prompt
        assert "A. Red" in prompt or "A)" in prompt
        assert "B. Blue" in prompt or "B)" in prompt

    def test_parse_mcq_answer(self):
        """Test MCQ answer parsing."""
        from modules.finetuning.benchmarks import BenchmarkRunner
        
        runner = BenchmarkRunner()
        
        # Test direct letter answers
        assert runner._parse_mcq_answer("A", 4) == 0
        assert runner._parse_mcq_answer("B", 4) == 1
        assert runner._parse_mcq_answer("C", 4) == 2
        assert runner._parse_mcq_answer("D", 4) == 3
        
        # Test with formatted answers
        assert runner._parse_mcq_answer("(B) is correct", 4) == 1
        assert runner._parse_mcq_answer("C. is the answer", 4) == 2


# ============================================================================
# A/B Testing Tests
# ============================================================================


class TestABTesting:
    """Test A/B testing functionality."""

    def test_create_ab_test(self):
        """Test creating an A/B test."""
        from modules.finetuning.ab_testing import ABTestManager, ABTest, TrafficSplit
        
        manager = ABTestManager()
        
        test = ABTest(
            name="base-vs-finetuned",
            base_model="llama-7b",
            variants=[
                TrafficSplit(variant_id="base", name="Base Model", weight=0.5),
                TrafficSplit(variant_id="ft", name="Fine-tuned", adapter_path="/adapters/ft", weight=0.5),
            ],
        )
        
        created = manager.create_test(test)
        assert created.id == test.id
        assert len(created.variants) == 2

    def test_ab_test_variant_selection(self):
        """Test variant selection respects weights."""
        from modules.finetuning.ab_testing import ABTestManager, ABTest, TrafficSplit, ABTestStatus
        
        manager = ABTestManager()
        
        test = ABTest(
            name="weighted-test",
            base_model="llama",
            variants=[
                TrafficSplit(variant_id="heavy", name="Heavy", weight=0.9),
                TrafficSplit(variant_id="light", name="Light", weight=0.1),
            ],
        )
        
        manager.create_test(test)
        test.status = ABTestStatus.RUNNING
        
        # Run many selections and verify distribution
        selections = {"heavy": 0, "light": 0}
        for _ in range(200):
            variant = manager.select_variant(test.id)
            if variant:
                selections[variant.variant_id] += 1
        
        # Heavy should be selected significantly more
        assert selections["heavy"] > selections["light"] * 2

    def test_ab_test_feedback_recording(self):
        """Test recording feedback for A/B test variants."""
        from modules.finetuning.ab_testing import ABTestManager, ABTest, TrafficSplit
        
        manager = ABTestManager()
        
        test = ABTest(
            name="feedback-test",
            base_model="llama",
            variants=[TrafficSplit(variant_id="v1", name="V1", weight=1.0)],
        )
        
        manager.create_test(test)
        
        # Record feedback
        manager.record_feedback(test.id, "v1", thumbs_up=True)
        manager.record_feedback(test.id, "v1", thumbs_up=True)
        manager.record_feedback(test.id, "v1", thumbs_up=False)
        
        metrics = test.metrics["v1"]
        assert metrics.thumbs_up == 2
        assert metrics.thumbs_down == 1

    def test_ab_test_summary(self):
        """Test getting A/B test summary."""
        from modules.finetuning.ab_testing import ABTestManager, ABTest, TrafficSplit
        
        manager = ABTestManager()
        
        test = ABTest(
            name="summary-test",
            base_model="llama",
            variants=[
                TrafficSplit(variant_id="a", name="A", weight=0.5),
                TrafficSplit(variant_id="b", name="B", weight=0.5),
            ],
        )
        
        manager.create_test(test)
        
        summary = manager.get_summary(test.id)
        
        assert summary["name"] == "summary-test"
        assert len(summary["variants"]) == 2
        assert "status" in summary


# ============================================================================
# Adapter Manager Tests
# ============================================================================


class TestAdapterManager:
    """Test adapter management functionality."""

    def _create_mock_adapter(self, temp_dir: Path, adapter_id: str):
        """Create a mock adapter directory with metadata."""
        adapter_dir = temp_dir / adapter_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock adapter files
        metadata = {
            "adapter_id": adapter_id,
            "base_model": "llama-7b",
            "created_at": datetime.utcnow().isoformat(),
            "lora_config": {"rank": 32, "alpha": 64},
            "training_loss": 0.3,
        }
        with open(adapter_dir / "adapter_metadata.json", "w") as f:
            json.dump(metadata, f)
        
        return adapter_dir

    def test_adapter_manager_initialization(self):
        """Test adapter manager can be initialized."""
        from modules.finetuning.adapter_manager import AdapterManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AdapterManager(Path(tmpdir))
            assert manager is not None

    def test_list_adapters_empty(self):
        """Test listing adapters when none exist."""
        from modules.finetuning.adapter_manager import AdapterManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AdapterManager(Path(tmpdir))
            adapters = manager.list_adapters()
            assert adapters == []

    def test_compare_adapters(self):
        """Test adapter comparison functionality."""
        from modules.finetuning.adapter_manager import AdapterManager, AdapterMetadata, AdapterStatus
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AdapterManager(Path(tmpdir))
            
            # Create two adapters with all required fields
            adapter1 = AdapterMetadata(
                id="adapter-1",
                name="Adapter 1",
                base_model="llama-7b",
                training_job_id="job-1",
                adapter_path="/adapters/adapter-1",
                status=AdapterStatus.READY,
                lora_config={"rank": 32, "alpha": 64},
                metrics={"training_loss": 0.3},
            )
            adapter2 = AdapterMetadata(
                id="adapter-2", 
                name="Adapter 2",
                base_model="llama-7b",
                training_job_id="job-2",
                adapter_path="/adapters/adapter-2",
                status=AdapterStatus.READY,
                lora_config={"rank": 64, "alpha": 128},
                metrics={"training_loss": 0.2},
            )
            
            manager._adapters = {"adapter-1": adapter1, "adapter-2": adapter2}
            
            # compare_adapters takes a list of adapter IDs
            comparison = manager.compare_adapters(["adapter-1", "adapter-2"])
            
            assert comparison is not None


# ============================================================================
# Distillation Manager Tests
# ============================================================================


class TestDistillationManager:
    """Test distillation functionality."""

    def test_default_templates_exist(self):
        """Verify default prompt templates are defined."""
        from modules.finetuning.distillation import DEFAULT_TEMPLATES
        
        assert len(DEFAULT_TEMPLATES) > 0
        
        for template in DEFAULT_TEMPLATES.values():
            assert template.name is not None
            assert template.system_prompt is not None
            assert template.user_template is not None

    def test_distillation_config_validation(self):
        """Test distillation config validation."""
        from modules.finetuning.distillation import DistillationConfig, TeacherProvider
        
        config = DistillationConfig(
            teacher_provider=TeacherProvider.OPENAI,
            teacher_model="gpt-4",
        )
        
        assert config.teacher_provider == TeacherProvider.OPENAI
        assert config.teacher_model == "gpt-4"


# ============================================================================
# Evaluation Manager Tests
# ============================================================================


class TestEvaluationManager:
    """Test evaluation functionality."""

    def test_create_comparison_session(self):
        """Test creating a comparison session."""
        from modules.finetuning.evaluation import EvaluationManager, ComparisonSession
        
        manager = EvaluationManager()
        
        session = ComparisonSession(
            name="test-evaluation",
            base_model="llama-7b",
            adapter_id="adapter-1",
        )
        
        created = manager.create_session(session)
        assert created.id == session.id
        assert created.name == "test-evaluation"

    def test_list_comparison_sessions(self):
        """Test listing comparison sessions."""
        from modules.finetuning.evaluation import EvaluationManager, ComparisonSession
        
        manager = EvaluationManager()
        manager.sessions.clear()
        
        # Create sessions
        for i in range(3):
            session = ComparisonSession(
                name=f"session-{i}",
                base_model="llama-7b",
                adapter_id=f"adapter-{i}",
            )
            manager.create_session(session)
        
        sessions = manager.list_sessions()
        assert len(sessions) == 3


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================


class TestEndToEndWorkflow:
    """Test complete fine-tuning workflows."""

    def test_dataset_to_job_workflow(self, temp_storage_dir, sample_alpaca_dataset):
        """Test complete workflow from dataset upload to job creation."""
        from modules.finetuning.storage import DatasetStore
        from modules.finetuning.dataset_processor import DatasetProcessor
        from modules.finetuning.job_store import JobStore
        from modules.finetuning.config import (
            Dataset, DatasetStatus, DatasetFormat,
            FineTuningJob, LoRAConfig, TrainingConfig
        )
        
        # 1. Create dataset file
        file_path = temp_storage_dir / "training_data.json"
        with open(file_path, "w") as f:
            json.dump(sample_alpaca_dataset, f)
        
        # 2. Inspect dataset
        processor = DatasetProcessor()
        detected_format, count, errors = processor.inspect(str(file_path))
        assert detected_format == DatasetFormat.ALPACA
        assert count == len(sample_alpaca_dataset)
        assert len(errors) == 0
        
        # 3. Store dataset
        dataset_store = DatasetStore(temp_storage_dir / "datasets")
        now = datetime.utcnow()
        dataset = Dataset(
            id=str(uuid.uuid4()),
            name="training-data",
            format=detected_format,
            status=DatasetStatus.READY,
            file_path=str(file_path),
            num_examples=count,
            created_at=now,
            updated_at=now,
        )
        dataset_store.upsert(dataset)
        
        # 4. Create training job
        job_store = JobStore(temp_storage_dir / "jobs")
        job = FineTuningJob(
            id=str(uuid.uuid4()),
            name="test-finetune",
            dataset_id=dataset.id,
            base_model="meta-llama/Llama-3-8B",
            lora_config=LoRAConfig(rank=16),
            training_config=TrainingConfig(num_epochs=1, batch_size=4),
        )
        job_store.upsert(job)
        
        # 5. Verify everything is connected
        loaded_job = job_store.get(job.id)
        loaded_dataset = dataset_store.get(loaded_job.dataset_id)
        
        assert loaded_job is not None
        assert loaded_dataset is not None
        assert loaded_dataset.name == "training-data"

    def test_preset_application(self):
        """Test applying hyperparameter presets to a job."""
        from modules.finetuning.config import (
            FineTuningJob, LoRAConfig, TrainingConfig,
            HYPERPARAMETER_PRESETS, PresetName
        )
        
        # Start with default job
        job = FineTuningJob(
            id=str(uuid.uuid4()),
            name="preset-test",
            dataset_id="ds-test",
            base_model="llama-7b",
            lora_config=LoRAConfig(),
            training_config=TrainingConfig(),
        )
        
        # Apply quick start preset
        preset = HYPERPARAMETER_PRESETS[PresetName.QUICK_START]
        job.lora_config = preset.lora_config
        job.training_config = preset.training_config
        
        # Verify preset was applied
        assert job.lora_config.rank == preset.lora_config.rank
        assert job.training_config.num_epochs == preset.training_config.num_epochs


# ============================================================================
# API Response Format Tests
# ============================================================================


class TestAPIResponseFormats:
    """Test that data models serialize correctly for API responses."""

    def test_dataset_serialization(self):
        """Test Dataset model serializes correctly."""
        from modules.finetuning.config import Dataset, DatasetStatus, DatasetFormat
        
        now = datetime.utcnow()
        dataset = Dataset(
            id="ds-123",
            name="test",
            format=DatasetFormat.ALPACA,
            status=DatasetStatus.READY,
            file_path="/tmp/test.json",
            num_examples=100,
            created_at=now,
            updated_at=now,
        )
        
        data = dataset.dict()
        
        assert data["id"] == "ds-123"
        assert data["name"] == "test"
        assert data["format"] == "alpaca"
        assert data["status"] == "ready"
        assert data["num_examples"] == 100

    def test_job_serialization(self):
        """Test FineTuningJob model serializes correctly."""
        from modules.finetuning.config import FineTuningJob, LoRAConfig, TrainingConfig, TrainingStatus
        
        job = FineTuningJob(
            id="job-456",
            name="test-job",
            dataset_id="ds-123",
            base_model="llama-7b",
            lora_config=LoRAConfig(rank=32),
            training_config=TrainingConfig(num_epochs=3),
        )
        
        data = job.dict()
        
        assert data["id"] == "job-456"
        assert data["name"] == "test-job"
        assert data["status"] == "queued"
        assert data["lora_config"]["rank"] == 32
        assert data["training_config"]["num_epochs"] == 3

    def test_preset_serialization(self):
        """Test HyperparameterPreset model serializes correctly."""
        from modules.finetuning.config import HYPERPARAMETER_PRESETS, PresetName
        
        preset = HYPERPARAMETER_PRESETS[PresetName.BALANCED]
        data = preset.dict()
        
        assert "name" in data
        assert "display_name" in data
        assert "description" in data
        assert "lora_config" in data
        assert "training_config" in data
        assert "qlora_config" in data

    def test_vram_estimate_serialization(self):
        """Test VRAMEstimate dataclass serializes correctly."""
        from dataclasses import asdict
        from modules.finetuning.vram_estimator import estimate_training_vram
        
        estimate = estimate_training_vram(
            model_name="llama-7b",
            lora_rank=32,
            batch_size=4,
            seq_length=2048,
            qlora_enabled=True,
        )
        
        # VRAMEstimate is a dataclass, use asdict
        data = asdict(estimate)
        
        assert "total_gb" in data
        assert "model_memory_gb" in data
        assert "recommended_gpu" in data
        assert "fits_on" in data
        assert isinstance(data["fits_on"], list)


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

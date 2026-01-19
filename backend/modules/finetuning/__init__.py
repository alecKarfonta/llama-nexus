"""
LoRA fine-tuning module bootstrap.

This package provides configuration models and early stubs for dataset handling
and training orchestration. It is intentionally minimal to allow incremental
build-out following the design in `docs/lora-fine-tuning-design.md`.
"""

from .config import (
    DatasetFormat,
    DatasetStatus,
    TrainingStatus,
    LoRAConfig,
    TrainingConfig,
    QLoRAConfig,
    FineTuningJob,
    Dataset,
    LoRAAdapter,
    PresetName,
    HyperparameterPreset,
    HYPERPARAMETER_PRESETS,
)
from .dataset_formats import DatasetFormatHandler, DATASET_FORMAT_HANDLERS
from .dataset_processor import DatasetProcessor
from .training_manager import TrainingManager
from .storage import DatasetStore, init_dataset_store, default_dataset_dir
from .job_store import JobStore, init_job_store, default_job_dir
from .redis_consumer import (
    TrainingEventConsumer,
    register_ws_broadcaster,
    unregister_ws_broadcaster,
    broadcast_training_event,
)
from .distillation import (
    # Enums
    TeacherProvider,
    DistillationStatus,
    GenerationStrategy,
    OutputFormat,
    QualityDimension,
    # Config models
    QualityConfig,
    SelfConsistencyConfig,
    ThinkingConfig,
    DistillationConfig,
    PromptTemplate,
    # Data models
    QualityScore,
    GeneratedExample,
    DistillationMetrics,
    DistillationJob,
    # Teacher clients
    TeacherClient,
    OpenAITeacher,
    AnthropicTeacher,
    GoogleTeacher,
    LocalTeacher,
    GenerationResult,
    create_teacher_client,
    # Validators and processors
    QualityValidator,
    SelfConsistencyValidator,
    ResponseRefiner,
    ThinkingTokenProcessor,
    DuplicateDetector,
    # Manager
    DistillationManager,
    DEFAULT_TEMPLATES,
    QUALITY_THRESHOLDS,
)
from .workflow_templates import (
    WorkflowTemplate,
    WORKFLOW_TEMPLATES,
)
from .vram_estimator import (
    VRAMEstimate,
    VRAMEstimateRequest,
    VRAMEstimateResponse,
    estimate_training_vram,
    estimate_model_params,
)
from .evaluation import (
    ComparisonType,
    EvaluationCriteria,
    ModelResponse,
    PromptComparison,
    ComparisonSession,
    JudgeConfig,
    JudgeEvaluation,
    EvaluationManager,
)
from .benchmarks import (
    BenchmarkName,
    BenchmarkStatus,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkJob,
    BenchmarkRunner,
    DatasetConfig,
    HuggingFaceDatasetManager,
    get_dataset_manager,
    BENCHMARK_DATASETS,
)
from .ab_testing import (
    ABTestStatus,
    TrafficSplit,
    VariantMetrics,
    ABTest,
    RequestResult,
    ABTestManager,
)
from .adapter_manager import (
    AdapterStatus,
    AdapterVersion,
    AdapterMetadata,
    AdapterComparison,
    AdapterManager,
    get_adapter_manager,
)
from .book_dataset import (
    BookDatasetConfig,
    BookDatasetGenerator,
    BookGenerationMode,
    GeneratedExample,
    estimate_dataset_size,
)
from .book_distillation import (
    BookDistillationConfig,
    BookDistillationGenerator,
    BookDistillationManager,
    BookDistillationJob,
    DistillationJobStatus,
)

__all__ = [
    "DatasetFormat",
    "DatasetStatus",
    "TrainingStatus",
    "LoRAConfig",
    "TrainingConfig",
    "QLoRAConfig",
    "FineTuningJob",
    "Dataset",
    "LoRAAdapter",
    "PresetName",
    "HyperparameterPreset",
    "HYPERPARAMETER_PRESETS",
    "DatasetFormatHandler",
    "DATASET_FORMAT_HANDLERS",
    "DatasetProcessor",
    "TrainingManager",
    "DatasetStore",
    "init_dataset_store",
    "default_dataset_dir",
    "JobStore",
    "init_job_store",
    "default_job_dir",
    "TrainingEventConsumer",
    "register_ws_broadcaster",
    "unregister_ws_broadcaster",
    "broadcast_training_event",
    # Distillation - Enums
    "TeacherProvider",
    "DistillationStatus",
    "GenerationStrategy",
    "OutputFormat",
    "QualityDimension",
    # Distillation - Config
    "QualityConfig",
    "SelfConsistencyConfig",
    "ThinkingConfig",
    "DistillationConfig",
    "PromptTemplate",
    # Distillation - Data Models
    "QualityScore",
    "GeneratedExample",
    "DistillationMetrics",
    "DistillationJob",
    # Distillation - Teacher Clients
    "TeacherClient",
    "OpenAITeacher",
    "AnthropicTeacher",
    "GoogleTeacher",
    "LocalTeacher",
    "GenerationResult",
    "create_teacher_client",
    # Distillation - Validators
    "QualityValidator",
    "SelfConsistencyValidator",
    "ResponseRefiner",
    "ThinkingTokenProcessor",
    "DuplicateDetector",
    # Distillation - Manager
    "DistillationManager",
    "DEFAULT_TEMPLATES",
    # Workflow Templates
    "WorkflowTemplate",
    "WORKFLOW_TEMPLATES",
    "QUALITY_THRESHOLDS",
    # VRAM Estimation
    "VRAMEstimate",
    "VRAMEstimateRequest",
    "VRAMEstimateResponse",
    "estimate_training_vram",
    "estimate_model_params",
    # Evaluation
    "ComparisonType",
    "EvaluationCriteria",
    "ModelResponse",
    "PromptComparison",
    "ComparisonSession",
    "JudgeConfig",
    "JudgeEvaluation",
    "EvaluationManager",
    # Benchmarks
    "BenchmarkName",
    "BenchmarkStatus",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkJob",
    "BenchmarkRunner",
    # Dataset Management
    "DatasetConfig",
    "HuggingFaceDatasetManager",
    "get_dataset_manager",
    "BENCHMARK_DATASETS",
    # A/B Testing
    "ABTestStatus",
    "TrafficSplit",
    "VariantMetrics",
    "ABTest",
    "RequestResult",
    "ABTestManager",
    # Adapter Management
    "AdapterStatus",
    "AdapterVersion",
    "AdapterMetadata",
    "AdapterComparison",
    "AdapterManager",
    "get_adapter_manager",
]

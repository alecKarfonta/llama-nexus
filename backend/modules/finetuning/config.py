"""
Configuration models for LoRA fine-tuning.

These Pydantic models mirror the design spec in `docs/lora-fine-tuning-design.md`
and provide a typed contract for upcoming dataset, training, and adapter
management APIs.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class DatasetFormat(str, Enum):
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    CHATML = "chatml"
    COMPLETION = "completion"


class DatasetStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Dataset(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    format: DatasetFormat
    status: DatasetStatus
    num_examples: int = 0
    total_tokens: int = 0
    file_path: str
    created_at: datetime
    updated_at: datetime
    validation_errors: Optional[List[str]] = None
    statistics: Optional[Dict[str, Any]] = None

    @validator("num_examples", "total_tokens")
    def non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Count values must be non-negative")
        return value


class LoRAConfig(BaseModel):
    rank: int = Field(32, ge=1, description="LoRA rank")
    alpha: int = Field(64, ge=1, description="LoRA alpha")
    dropout: float = Field(0.05, ge=0.0, le=1.0, description="LoRA dropout")
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Modules to adapt",
    )
    bias: str = Field("none", description="Bias training strategy")


class TrainingConfig(BaseModel):
    learning_rate: float = Field(2e-4, gt=0)
    batch_size: int = Field(4, ge=1)
    gradient_accumulation_steps: int = Field(4, ge=1)
    num_epochs: int = Field(3, ge=1)
    max_steps: Optional[int] = Field(None, ge=1)
    warmup_steps: int = Field(100, ge=0)
    weight_decay: float = Field(0.01, ge=0.0)
    optimizer: str = Field("adamw_8bit")
    lr_scheduler: str = Field("cosine")
    max_seq_length: int = Field(2048, ge=1)
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = Field(
        True,
        description="Enable gradient checkpointing to reduce VRAM usage (~70% reduction in activation memory)"
    )

    @validator("fp16")
    def fp16_bf16_exclusive(cls, value: bool, values: Dict[str, Any]) -> bool:
        if value and values.get("bf16"):
            raise ValueError("Only one of fp16 or bf16 can be enabled")
        return value


class QLoRAConfig(BaseModel):
    enabled: bool = True
    bits: int = Field(4, ge=2, le=8)
    quant_type: str = Field("nf4")
    double_quant: bool = True
    compute_dtype: str = Field("bfloat16")


class FineTuningJob(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    base_model: str
    dataset_id: str
    lora_config: LoRAConfig
    training_config: TrainingConfig
    qlora_config: QLoRAConfig = Field(default_factory=QLoRAConfig)
    status: TrainingStatus = TrainingStatus.QUEUED
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    adapter_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    @validator("progress")
    def progress_bounds(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError("Progress must be between 0 and 100")
        return value


class LoRAAdapter(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    base_model: str
    training_job_id: str
    lora_config: LoRAConfig
    file_path: str
    file_size: int
    metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    is_merged: bool = False
    merged_model_path: Optional[str] = None
    gguf_path: Optional[str] = None


class PresetName(str, Enum):
    QUICK_START = "quick_start"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


class HyperparameterPreset(BaseModel):
    name: PresetName
    display_name: str
    description: str
    lora_config: LoRAConfig
    training_config: TrainingConfig
    qlora_config: QLoRAConfig


HYPERPARAMETER_PRESETS: Dict[PresetName, HyperparameterPreset] = {
    PresetName.QUICK_START: HyperparameterPreset(
        name=PresetName.QUICK_START,
        display_name="Quick Start",
        description="Fast training for simple tasks and style transfer. Lower quality but faster iteration.",
        lora_config=LoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
        training_config=TrainingConfig(
            learning_rate=3e-4,
            batch_size=8,
            gradient_accumulation_steps=1,
            num_epochs=1,
            warmup_steps=50,
            weight_decay=0.0,
            max_seq_length=2048,
        ),
        qlora_config=QLoRAConfig(enabled=True, bits=4, quant_type="nf4"),
    ),
    PresetName.BALANCED: HyperparameterPreset(
        name=PresetName.BALANCED,
        display_name="Balanced",
        description="Recommended default. Good balance of quality and speed for instruction tuning.",
        lora_config=LoRAConfig(
            rank=32,
            alpha=64,
            dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        ),
        training_config=TrainingConfig(
            learning_rate=2e-4,
            batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=3,
            warmup_steps=100,
            weight_decay=0.01,
            max_seq_length=2048,
        ),
        qlora_config=QLoRAConfig(enabled=True, bits=4, quant_type="nf4"),
    ),
    PresetName.HIGH_QUALITY: HyperparameterPreset(
        name=PresetName.HIGH_QUALITY,
        display_name="High Quality",
        description="Best results for complex tasks. Longer training time and higher VRAM usage.",
        lora_config=LoRAConfig(
            rank=128,
            alpha=256,
            dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        ),
        training_config=TrainingConfig(
            learning_rate=1e-4,
            batch_size=2,
            gradient_accumulation_steps=8,
            num_epochs=5,
            warmup_steps=200,
            weight_decay=0.01,
            max_seq_length=4096,
        ),
        qlora_config=QLoRAConfig(enabled=True, bits=4, quant_type="nf4"),
    ),
}

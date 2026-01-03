"""
Configuration models for model quantization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class QuantizationFormat(str, Enum):
    """Supported quantization output formats."""
    GGUF = "gguf"
    GPTQ = "gptq"
    AWQ = "awq"
    ONNX = "onnx"


class QuantizationType(str, Enum):
    """High-level quantization types."""
    INT2 = "int2"
    INT3 = "int3"
    INT4 = "int4"
    INT5 = "int5"
    INT6 = "int6"
    INT8 = "int8"
    FP16 = "fp16"
    MIXED = "mixed"


class GGUFQuantType(str, Enum):
    """GGUF-specific quantization types (llama.cpp)."""
    Q2_K = "Q2_K"
    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q3_K_L = "Q3_K_L"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    F16 = "F16"
    F32 = "F32"


class QuantizationStatus(str, Enum):
    """Quantization job status."""
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PREPARING = "preparing"
    QUANTIZING = "quantizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QuantizedOutput(BaseModel):
    """A single quantized model output file."""
    id: str
    format: QuantizationFormat
    quant_type: str
    file_path: str
    file_size: int = 0
    status: str = "pending"  # pending, completed, failed
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Quality metrics (if available)
    perplexity: Optional[float] = None
    bits_per_weight: Optional[float] = None


class QuantizationConfig(BaseModel):
    """Configuration for quantization process."""
    # GGUF-specific options
    use_imatrix: bool = False
    imatrix_dataset: Optional[str] = None
    
    # GPTQ/AWQ-specific options
    calibration_dataset: Optional[str] = None
    calibration_samples: int = 128
    group_size: int = 128
    desc_act: bool = False
    damp_percent: float = 0.01
    
    # General options
    threads: int = 4
    keep_intermediate: bool = False
    verify_output: bool = True


class QuantizationJob(BaseModel):
    """A quantization job that can produce multiple output formats."""
    id: str
    name: str
    description: Optional[str] = None
    
    # Source model
    source_model: str  # HuggingFace repo ID or local path
    source_type: str = "huggingface"  # huggingface, local
    model_architecture: Optional[str] = None
    
    # Output configuration
    output_formats: List[QuantizationFormat]
    gguf_quant_types: List[GGUFQuantType] = Field(default_factory=list)
    gptq_bits: List[int] = Field(default_factory=lambda: [4])
    awq_bits: List[int] = Field(default_factory=lambda: [4])
    
    # Configuration
    config: QuantizationConfig = Field(default_factory=QuantizationConfig)
    
    # Job status
    status: QuantizationStatus = QuantizationStatus.QUEUED
    progress: float = 0.0
    current_step: str = ""
    total_outputs: int = 0
    completed_outputs: int = 0
    
    # Outputs
    outputs: List[QuantizedOutput] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    # Resource usage
    estimated_disk_gb: Optional[float] = None
    estimated_time_minutes: Optional[int] = None

    @validator("progress")
    def progress_bounds(cls, value: float) -> float:
        """Ensure progress is between 0 and 100."""
        if not 0.0 <= value <= 100.0:
            raise ValueError("Progress must be between 0 and 100")
        return value

    @validator("gguf_quant_types")
    def validate_gguf_types(cls, value: List[GGUFQuantType], values: Dict[str, Any]) -> List[GGUFQuantType]:
        """Ensure GGUF quant types are provided if GGUF format is selected."""
        if QuantizationFormat.GGUF in values.get("output_formats", []) and not value:
            raise ValueError("GGUF format requires at least one quantization type")
        return value


class QuantizationEstimate(BaseModel):
    """Estimated resource requirements for quantization."""
    disk_space_gb: float
    estimated_time_minutes: int
    outputs_count: int
    warnings: List[str] = Field(default_factory=list)
    breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

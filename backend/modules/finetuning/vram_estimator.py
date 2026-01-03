"""
VRAM estimation for LoRA/QLoRA training.

Provides estimates of GPU memory requirements based on model size, LoRA rank,
batch size, and sequence length. These are approximations based on empirical
data and may vary based on actual hardware and software versions.
"""

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel


@dataclass
class VRAMEstimate:
    """Estimated VRAM requirements."""
    model_memory_gb: float
    lora_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    total_gb: float
    recommended_gpu: str
    fits_on: list  # List of GPUs that can run this


# Common GPU VRAM sizes
GPU_VRAM = {
    "RTX 3060": 12,
    "RTX 3070": 8,
    "RTX 3080": 10,
    "RTX 3090": 24,
    "RTX 4060": 8,
    "RTX 4070": 12,
    "RTX 4080": 16,
    "RTX 4090": 24,
    "A10": 24,
    "A100-40GB": 40,
    "A100-80GB": 80,
    "H100": 80,
}


# Approximate model parameter counts (in billions)
MODEL_SIZES = {
    # Llama family
    "llama-7b": 7,
    "llama-8b": 8,
    "llama-13b": 13,
    "llama-34b": 34,
    "llama-70b": 70,
    # Mistral
    "mistral-7b": 7,
    "mixtral-8x7b": 47,  # Effective parameters
    # Qwen
    "qwen-7b": 7,
    "qwen-14b": 14,
    "qwen-72b": 72,
    # Phi
    "phi-2": 2.7,
    "phi-3": 3.8,
    # Generic sizes
    "1b": 1,
    "3b": 3,
    "7b": 7,
    "8b": 8,
    "13b": 13,
    "34b": 34,
    "70b": 70,
}


def estimate_model_params(model_name: str) -> float:
    """Estimate model parameter count from name."""
    model_lower = model_name.lower()
    
    # Check for exact matches first
    for key, value in MODEL_SIZES.items():
        if key in model_lower:
            return value
    
    # Try to extract size from name (e.g., "8B", "70b")
    import re
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_name)
    if match:
        return float(match.group(1))
    
    # Default to 7B if unknown
    return 7.0


def estimate_hidden_dim(params_b: float) -> int:
    """Estimate hidden dimension from parameter count."""
    # Rough approximation based on common architectures
    if params_b <= 1:
        return 2048
    elif params_b <= 3:
        return 2560
    elif params_b <= 8:
        return 4096
    elif params_b <= 14:
        return 5120
    elif params_b <= 35:
        return 8192
    else:
        return 8192


def estimate_num_layers(params_b: float) -> int:
    """Estimate number of layers from parameter count."""
    if params_b <= 1:
        return 16
    elif params_b <= 3:
        return 24
    elif params_b <= 8:
        return 32
    elif params_b <= 14:
        return 40
    elif params_b <= 35:
        return 48
    else:
        return 80


def estimate_training_vram(
    model_name: str,
    lora_rank: int = 32,
    batch_size: int = 4,
    seq_length: int = 2048,
    qlora_enabled: bool = True,
    gradient_accumulation: int = 4,
    gradient_checkpointing: bool = True,
) -> VRAMEstimate:
    """
    Estimate VRAM requirements for LoRA/QLoRA training.
    
    Args:
        model_name: Name of the base model (e.g., "meta-llama/Llama-3-8B")
        lora_rank: LoRA rank (r)
        batch_size: Per-device batch size
        seq_length: Maximum sequence length
        qlora_enabled: Whether using 4-bit quantization
        gradient_accumulation: Gradient accumulation steps
        gradient_checkpointing: Whether to use gradient checkpointing (~70% VRAM reduction)
        
    Returns:
        VRAMEstimate with memory breakdown and recommendations
    """
    params_b = estimate_model_params(model_name)
    hidden_dim = estimate_hidden_dim(params_b)
    num_layers = estimate_num_layers(params_b)
    
    # Model memory (quantized for QLoRA)
    if qlora_enabled:
        # 4-bit: ~0.5 bytes per parameter
        model_memory_gb = params_b * 0.5
    else:
        # Full precision: 4 bytes per parameter (fp32) or 2 bytes (fp16/bf16)
        model_memory_gb = params_b * 2  # Assuming fp16/bf16
    
    # LoRA adapter memory (full precision)
    # Each LoRA layer has A (hidden_dim x rank) and B (rank x hidden_dim) matrices
    # For typical attention: q, k, v, o projections = 4 matrices per layer
    num_lora_modules = 4  # q, k, v, o (can be 7 with MLP)
    lora_params = 2 * lora_rank * hidden_dim * num_lora_modules * num_layers
    lora_memory_gb = (lora_params * 4) / 1e9  # fp32 for LoRA params
    
    # Optimizer states (AdamW: 2x model params for momentum and variance)
    # Only for LoRA params when using QLoRA
    optimizer_memory_gb = lora_memory_gb * 2
    
    # Activation memory (rough estimate)
    # Depends on batch size, sequence length, and model architecture
    # Formula: batch_size * seq_length * hidden_dim * num_layers * bytes_per_activation
    activation_bytes = batch_size * seq_length * hidden_dim * num_layers * 2  # fp16
    activation_memory_gb = activation_bytes / 1e9
    
    # Apply gradient checkpointing factor if enabled (reduces activation memory ~70%)
    if gradient_checkpointing:
        activation_memory_gb *= 0.3
    
    # Total with 15% overhead for fragmentation and other allocations
    total_gb = (model_memory_gb + lora_memory_gb + optimizer_memory_gb + activation_memory_gb) * 1.15
    
    # Determine which GPUs can run this
    fits_on = [gpu for gpu, vram in GPU_VRAM.items() if vram >= total_gb]
    
    # Recommend the smallest GPU that fits with some headroom
    recommended_gpu = "A100-80GB"  # Default to largest
    for gpu, vram in sorted(GPU_VRAM.items(), key=lambda x: x[1]):
        if vram >= total_gb * 1.1:  # 10% headroom
            recommended_gpu = gpu
            break
    
    return VRAMEstimate(
        model_memory_gb=round(model_memory_gb, 2),
        lora_memory_gb=round(lora_memory_gb, 2),
        optimizer_memory_gb=round(optimizer_memory_gb, 2),
        activation_memory_gb=round(activation_memory_gb, 2),
        total_gb=round(total_gb, 2),
        recommended_gpu=recommended_gpu,
        fits_on=fits_on,
    )


class VRAMEstimateRequest(BaseModel):
    """Request model for VRAM estimation API."""
    model_name: str
    lora_rank: int = 32
    batch_size: int = 4
    seq_length: int = 2048
    qlora_enabled: bool = True
    gradient_accumulation: int = 4
    gradient_checkpointing: bool = True


class VRAMEstimateResponse(BaseModel):
    """Response model for VRAM estimation API."""
    model_name: str
    estimated_params_b: float
    model_memory_gb: float
    lora_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    total_gb: float
    recommended_gpu: str
    fits_on: list
    warning: Optional[str] = None

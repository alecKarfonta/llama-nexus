"""
VRAM Estimation Module

Calculates estimated VRAM requirements for LLM deployment based on:
- Model parameters
- Quantization level
- Context size
- KV cache settings
- Batch size
"""

import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VRAMEstimate:
    """VRAM estimation result."""
    model_weights_mb: float
    kv_cache_mb: float
    compute_buffer_mb: float
    overhead_mb: float
    total_mb: float
    total_gb: float
    fits_in_vram: bool
    available_vram_gb: float
    utilization_percent: float
    warnings: list


# Quantization bits per weight
QUANT_BITS = {
    'F32': 32.0,
    'F16': 16.0,
    'BF16': 16.0,
    'Q8_0': 8.0,
    'Q6_K': 6.5,
    'Q5_K_M': 5.5,
    'Q5_K_S': 5.5,
    'Q5_0': 5.0,
    'Q5_1': 5.5,
    'Q4_K_M': 4.5,
    'Q4_K_S': 4.5,
    'Q4_0': 4.0,
    'Q4_1': 4.5,
    'Q3_K_M': 3.5,
    'Q3_K_S': 3.5,
    'Q3_K_L': 3.5,
    'Q2_K': 2.5,
    'IQ4_XS': 4.25,
    'IQ4_NL': 4.5,
    'IQ3_XS': 3.3,
    'IQ3_XXS': 3.0,
    'IQ2_XS': 2.3,
    'IQ2_XXS': 2.0,
    'IQ1_S': 1.5,
    'IQ1_M': 1.75,
}

# Common model architectures with known parameters
MODEL_ARCHITECTURES = {
    # Llama family
    'llama-7b': {'params_b': 7, 'layers': 32, 'hidden': 4096, 'heads': 32, 'kv_heads': 32},
    'llama-13b': {'params_b': 13, 'layers': 40, 'hidden': 5120, 'heads': 40, 'kv_heads': 40},
    'llama-30b': {'params_b': 30, 'layers': 60, 'hidden': 6656, 'heads': 52, 'kv_heads': 52},
    'llama-65b': {'params_b': 65, 'layers': 80, 'hidden': 8192, 'heads': 64, 'kv_heads': 64},
    'llama-70b': {'params_b': 70, 'layers': 80, 'hidden': 8192, 'heads': 64, 'kv_heads': 8},
    
    # Llama 3
    'llama3-8b': {'params_b': 8, 'layers': 32, 'hidden': 4096, 'heads': 32, 'kv_heads': 8},
    'llama3-70b': {'params_b': 70, 'layers': 80, 'hidden': 8192, 'heads': 64, 'kv_heads': 8},
    'llama3.1-8b': {'params_b': 8, 'layers': 32, 'hidden': 4096, 'heads': 32, 'kv_heads': 8},
    'llama3.1-70b': {'params_b': 70, 'layers': 80, 'hidden': 8192, 'heads': 64, 'kv_heads': 8},
    'llama3.1-405b': {'params_b': 405, 'layers': 126, 'hidden': 16384, 'heads': 128, 'kv_heads': 8},
    
    # Mistral
    'mistral-7b': {'params_b': 7, 'layers': 32, 'hidden': 4096, 'heads': 32, 'kv_heads': 8},
    'mixtral-8x7b': {'params_b': 47, 'layers': 32, 'hidden': 4096, 'heads': 32, 'kv_heads': 8, 'moe': True},
    'mixtral-8x22b': {'params_b': 141, 'layers': 56, 'hidden': 6144, 'heads': 48, 'kv_heads': 8, 'moe': True},
    
    # Qwen
    'qwen-7b': {'params_b': 7, 'layers': 32, 'hidden': 4096, 'heads': 32, 'kv_heads': 32},
    'qwen-14b': {'params_b': 14, 'layers': 40, 'hidden': 5120, 'heads': 40, 'kv_heads': 40},
    'qwen-72b': {'params_b': 72, 'layers': 80, 'hidden': 8192, 'heads': 64, 'kv_heads': 64},
    'qwen2-7b': {'params_b': 7, 'layers': 28, 'hidden': 3584, 'heads': 28, 'kv_heads': 4},
    'qwen2-72b': {'params_b': 72, 'layers': 80, 'hidden': 8192, 'heads': 64, 'kv_heads': 8},
    'qwen2.5-7b': {'params_b': 7, 'layers': 28, 'hidden': 3584, 'heads': 28, 'kv_heads': 4},
    'qwen2.5-14b': {'params_b': 14, 'layers': 48, 'hidden': 5120, 'heads': 40, 'kv_heads': 8},
    'qwen2.5-32b': {'params_b': 32, 'layers': 64, 'hidden': 5120, 'heads': 40, 'kv_heads': 8},
    'qwen2.5-72b': {'params_b': 72, 'layers': 80, 'hidden': 8192, 'heads': 64, 'kv_heads': 8},
    
    # DeepSeek
    'deepseek-7b': {'params_b': 7, 'layers': 30, 'hidden': 4096, 'heads': 32, 'kv_heads': 32},
    'deepseek-67b': {'params_b': 67, 'layers': 95, 'hidden': 8192, 'heads': 64, 'kv_heads': 64},
    'deepseek-v2': {'params_b': 236, 'layers': 60, 'hidden': 5120, 'heads': 128, 'kv_heads': 128, 'moe': True},
    
    # Phi
    'phi-2': {'params_b': 2.7, 'layers': 32, 'hidden': 2560, 'heads': 32, 'kv_heads': 32},
    'phi-3-mini': {'params_b': 3.8, 'layers': 32, 'hidden': 3072, 'heads': 32, 'kv_heads': 32},
    'phi-3-medium': {'params_b': 14, 'layers': 40, 'hidden': 5120, 'heads': 40, 'kv_heads': 10},
    
    # Yi
    'yi-6b': {'params_b': 6, 'layers': 32, 'hidden': 4096, 'heads': 32, 'kv_heads': 4},
    'yi-34b': {'params_b': 34, 'layers': 60, 'hidden': 7168, 'heads': 56, 'kv_heads': 8},
    
    # Gemma
    'gemma-2b': {'params_b': 2.5, 'layers': 18, 'hidden': 2048, 'heads': 8, 'kv_heads': 1},
    'gemma-7b': {'params_b': 8.5, 'layers': 28, 'hidden': 3072, 'heads': 16, 'kv_heads': 16},
    'gemma2-9b': {'params_b': 9, 'layers': 42, 'hidden': 3584, 'heads': 16, 'kv_heads': 8},
    'gemma2-27b': {'params_b': 27, 'layers': 46, 'hidden': 4608, 'heads': 32, 'kv_heads': 16},
}


def detect_quantization(filename: str) -> str:
    """Detect quantization from filename."""
    filename_upper = filename.upper()
    
    # Check for specific quantization patterns
    for quant in sorted(QUANT_BITS.keys(), key=len, reverse=True):
        # Match patterns like Q4_K_M, q4_k_m, Q4-K-M, etc.
        pattern = quant.replace('_', '[_-]?')
        if re.search(pattern, filename_upper):
            return quant
    
    # Default assumptions based on file extension or common patterns
    if 'F16' in filename_upper or 'FP16' in filename_upper:
        return 'F16'
    if 'F32' in filename_upper or 'FP32' in filename_upper:
        return 'F32'
    
    # Default to Q4_K_M as a common quantization
    return 'Q4_K_M'


def detect_model_architecture(model_name: str) -> Optional[Dict[str, Any]]:
    """Detect model architecture from name."""
    model_lower = model_name.lower()
    
    # Try to match known architectures
    for arch_name, arch_config in MODEL_ARCHITECTURES.items():
        # Check if architecture name is in the model name
        arch_pattern = arch_name.replace('-', '[-_]?').replace('.', r'\.')
        if re.search(arch_pattern, model_lower):
            return arch_config.copy()
    
    # Try to extract parameter count from name
    param_match = re.search(r'(\d+\.?\d*)\s*[bB]', model_name)
    if param_match:
        params_b = float(param_match.group(1))
        # Estimate architecture based on parameter count
        return estimate_architecture_from_params(params_b)
    
    return None


def estimate_architecture_from_params(params_b: float) -> Dict[str, Any]:
    """Estimate architecture parameters based on total parameter count."""
    # These are rough estimates based on common architectures
    if params_b <= 3:
        return {'params_b': params_b, 'layers': 24, 'hidden': 2048, 'heads': 16, 'kv_heads': 16}
    elif params_b <= 8:
        return {'params_b': params_b, 'layers': 32, 'hidden': 4096, 'heads': 32, 'kv_heads': 8}
    elif params_b <= 15:
        return {'params_b': params_b, 'layers': 40, 'hidden': 5120, 'heads': 40, 'kv_heads': 8}
    elif params_b <= 35:
        return {'params_b': params_b, 'layers': 60, 'hidden': 6656, 'heads': 52, 'kv_heads': 8}
    elif params_b <= 75:
        return {'params_b': params_b, 'layers': 80, 'hidden': 8192, 'heads': 64, 'kv_heads': 8}
    else:
        return {'params_b': params_b, 'layers': 100, 'hidden': 12288, 'heads': 96, 'kv_heads': 8}


def calculate_model_weights_vram(
    params_b: float,
    quantization: str,
    gpu_layers: int,
    total_layers: int,
) -> float:
    """Calculate VRAM for model weights in MB."""
    bits_per_weight = QUANT_BITS.get(quantization.upper(), 4.5)
    
    # Calculate total model size
    total_size_bits = params_b * 1e9 * bits_per_weight
    total_size_mb = total_size_bits / 8 / 1024 / 1024
    
    # Calculate proportion on GPU
    if gpu_layers >= total_layers:
        gpu_proportion = 1.0
    else:
        gpu_proportion = gpu_layers / total_layers
    
    return total_size_mb * gpu_proportion


def calculate_kv_cache_vram(
    context_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    kv_heads: int,
    n_parallel: int = 1,
    kv_cache_type: str = 'f16',
) -> float:
    """Calculate KV cache VRAM in MB.
    
    Note: n_parallel is the number of concurrent inference slots (sequences),
    NOT the token batch size (n_batch). Token batch size affects compute buffer
    but does NOT multiply KV cache memory.
    """
    # Head dimension is based on total attention heads, not KV heads
    head_dim = hidden_size // num_heads if num_heads > 0 else 128
    if head_dim <= 0:
        head_dim = 128  # Default
    
    # KV cache bits per element
    kv_bits = {
        'f16': 16,
        'f32': 32,
        'q8_0': 8,
        'q4_0': 4,
    }.get(kv_cache_type.lower(), 16)
    
    # KV cache size: 2 (K and V) * layers * context * kv_heads * head_dim * n_parallel
    # n_parallel = concurrent sequences (slots), NOT token batch size
    kv_elements = 2 * num_layers * context_size * kv_heads * head_dim * n_parallel
    kv_size_bits = kv_elements * kv_bits
    kv_size_mb = kv_size_bits / 8 / 1024 / 1024
    
    return kv_size_mb


def calculate_compute_buffer(
    context_size: int,
    hidden_size: int,
    batch_size: int = 512,
) -> float:
    """Calculate compute buffer VRAM in MB.
    
    batch_size here is the token batch size (n_batch), which affects temporary
    activation memory. We use sub-linear scaling (capped at 2x) since batch
    processing reuses most buffers.
    """
    # Sub-linear batch scaling factor: large batches don't linearly increase
    # compute buffer because most memory is reused across batch steps.
    # Cap at 2x regardless of batch size.
    import math
    batch_factor = min(2.0, 1.0 + math.log2(max(batch_size, 1)) / 10.0)
    
    # Compute buffer is roughly proportional to context * hidden
    # Factor of 2 for intermediate computations (conservative estimate)
    buffer_elements = context_size * hidden_size * 2
    buffer_mb = buffer_elements * 2 * batch_factor / 1024 / 1024  # 2 bytes per element (fp16)
    
    # Add some minimum buffer for small contexts
    return max(buffer_mb, 256)


def estimate_vram(
    model_name: str = '',
    params_b: Optional[float] = None,
    quantization: str = 'Q4_K_M',
    context_size: int = 4096,
    batch_size: int = 1,
    gpu_layers: int = -1,  # -1 means all layers
    kv_cache_type: str = 'f16',
    available_vram_gb: float = 24.0,
    flash_attention: bool = True,
) -> VRAMEstimate:
    """
    Estimate VRAM requirements for model deployment.
    
    Args:
        model_name: Model name/filename for auto-detection
        params_b: Model parameters in billions (overrides detection)
        quantization: Quantization type (Q4_K_M, Q8_0, F16, etc.)
        context_size: Context window size in tokens
        batch_size: Batch size for inference
        gpu_layers: Number of layers on GPU (-1 for all)
        kv_cache_type: KV cache type (f16, q8_0, q4_0)
        available_vram_gb: Available GPU VRAM in GB
        flash_attention: Whether flash attention is enabled
    
    Returns:
        VRAMEstimate with breakdown and warnings
    """
    warnings = []
    
    # Detect or use provided parameters
    if params_b is None:
        arch = detect_model_architecture(model_name)
        if arch:
            params_b = arch['params_b']
            num_layers = arch['layers']
            hidden_size = arch['hidden']
            num_heads = arch['heads']
            kv_heads = arch.get('kv_heads', arch['heads'])
        else:
            # Default fallback - assume 7B model
            params_b = 7
            num_layers = 32
            hidden_size = 4096
            num_heads = 32
            kv_heads = 8
            warnings.append(f"Could not detect model architecture from '{model_name}', using 7B defaults")
    else:
        arch = estimate_architecture_from_params(params_b)
        num_layers = arch['layers']
        hidden_size = arch['hidden']
        num_heads = arch['heads']
        kv_heads = arch.get('kv_heads', arch['heads'])
    
    # Detect quantization if model_name provided
    if model_name and quantization == 'Q4_K_M':
        detected_quant = detect_quantization(model_name)
        if detected_quant != 'Q4_K_M':
            quantization = detected_quant
    
    # Handle gpu_layers
    if gpu_layers < 0 or gpu_layers > num_layers:
        gpu_layers = num_layers
    
    # Calculate components
    model_weights_mb = calculate_model_weights_vram(
        params_b=params_b,
        quantization=quantization,
        gpu_layers=gpu_layers,
        total_layers=num_layers,
    )
    
    kv_cache_mb = calculate_kv_cache_vram(
        context_size=context_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        kv_heads=kv_heads,
        n_parallel=1,  # Single deployment slot; batch_size is NOT a KV multiplier
        kv_cache_type=kv_cache_type,
    )
    
    # Flash attention reduces KV cache memory usage
    if flash_attention:
        kv_cache_mb *= 0.7  # Approximate reduction
    
    compute_buffer_mb = calculate_compute_buffer(
        context_size=context_size,
        hidden_size=hidden_size,
        batch_size=batch_size,  # Token batch size, scaled sub-linearly
    )
    
    # Add overhead (CUDA context, fragmentation, etc.)
    overhead_mb = 512 + (model_weights_mb * 0.05)  # 512MB base + 5% of model
    
    # Calculate total
    total_mb = model_weights_mb + kv_cache_mb + compute_buffer_mb + overhead_mb
    total_gb = total_mb / 1024
    
    # Check if it fits
    available_vram_mb = available_vram_gb * 1024
    fits_in_vram = total_mb <= available_vram_mb * 0.95  # Leave 5% margin
    utilization = (total_mb / available_vram_mb) * 100
    
    # Generate warnings
    if utilization > 95:
        warnings.append("VRAM usage exceeds 95% - may cause out-of-memory errors")
    elif utilization > 85:
        warnings.append("VRAM usage is high (>85%) - consider reducing context size or using more aggressive quantization")
    
    if context_size > 32768 and not flash_attention:
        warnings.append("Large context without flash attention - enable flash attention for better memory efficiency")
    
    if gpu_layers < num_layers:
        offloaded = num_layers - gpu_layers
        warnings.append(f"{offloaded} layers will be offloaded to CPU, which may significantly impact performance")
    
    quant_upper = quantization.upper()
    if quant_upper in ['F32', 'F16', 'BF16'] and params_b > 13:
        warnings.append(f"Using {quant_upper} with a large model - consider quantization for better VRAM efficiency")
    
    return VRAMEstimate(
        model_weights_mb=round(model_weights_mb, 2),
        kv_cache_mb=round(kv_cache_mb, 2),
        compute_buffer_mb=round(compute_buffer_mb, 2),
        overhead_mb=round(overhead_mb, 2),
        total_mb=round(total_mb, 2),
        total_gb=round(total_gb, 2),
        fits_in_vram=fits_in_vram,
        available_vram_gb=available_vram_gb,
        utilization_percent=round(utilization, 1),
        warnings=warnings,
    )


def get_quantization_options() -> Dict[str, float]:
    """Get available quantization options with their bits per weight."""
    return QUANT_BITS.copy()


def get_model_architectures() -> Dict[str, Dict[str, Any]]:
    """Get known model architectures."""
    return MODEL_ARCHITECTURES.copy()


# Command line testing
if __name__ == '__main__':
    # Test with various models
    test_cases = [
        ('llama-3.1-8b-instruct-Q4_K_M.gguf', 8192, 24),
        ('Qwen2.5-72B-Instruct-Q4_K_M.gguf', 32768, 48),
        ('mistral-7b-instruct-v0.2-Q8_0.gguf', 8192, 24),
        ('phi-3-medium-128k-instruct-Q5_K_M.gguf', 131072, 24),
    ]
    
    for model_name, ctx_size, vram_gb in test_cases:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Context: {ctx_size}, Available VRAM: {vram_gb}GB")
        print('='*60)
        
        estimate = estimate_vram(
            model_name=model_name,
            context_size=ctx_size,
            available_vram_gb=vram_gb,
        )
        
        print(f"Model Weights: {estimate.model_weights_mb:,.0f} MB")
        print(f"KV Cache: {estimate.kv_cache_mb:,.0f} MB")
        print(f"Compute Buffer: {estimate.compute_buffer_mb:,.0f} MB")
        print(f"Overhead: {estimate.overhead_mb:,.0f} MB")
        print(f"Total: {estimate.total_gb:.2f} GB ({estimate.utilization_percent}% of {vram_gb}GB)")
        print(f"Fits in VRAM: {'Yes' if estimate.fits_in_vram else 'NO'}")
        
        if estimate.warnings:
            print("\nWarnings:")
            for warning in estimate.warnings:
                print(f"  - {warning}")

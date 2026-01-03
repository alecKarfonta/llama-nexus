"""
Quantization job manager for orchestrating quantization workflows.
"""

import uuid
from typing import Dict, List, Optional
from datetime import datetime

from enhanced_logger import enhanced_logger as logger

from .config import (
    QuantizationJob,
    QuantizationStatus,
    QuantizationFormat,
    GGUFQuantType,
    QuantizedOutput,
    QuantizationEstimate,
)
from .storage import get_quantization_store

# Try to import model registry for integration
try:
    from modules.model_registry import model_registry
    MODEL_REGISTRY_AVAILABLE = True
except ImportError:
    MODEL_REGISTRY_AVAILABLE = False
    logger.warning("Model registry not available for quantization integration")


class QuantizationManager:
    """Manages quantization jobs and their lifecycle."""

    def __init__(self):
        self.store = get_quantization_store()

    def create_job(self, job: QuantizationJob) -> QuantizationJob:
        """Create a new quantization job."""
        if not job.id:
            job.id = str(uuid.uuid4())

        # Calculate total outputs
        total_outputs = 0
        if QuantizationFormat.GGUF in job.output_formats:
            total_outputs += len(job.gguf_quant_types)
        if QuantizationFormat.GPTQ in job.output_formats:
            total_outputs += len(job.gptq_bits)
        if QuantizationFormat.AWQ in job.output_formats:
            total_outputs += len(job.awq_bits)
        if QuantizationFormat.ONNX in job.output_formats:
            total_outputs += 1

        job.total_outputs = total_outputs

        # Initialize output entries
        outputs = []
        if QuantizationFormat.GGUF in job.output_formats:
            for quant_type in job.gguf_quant_types:
                outputs.append(
                    QuantizedOutput(
                        id=f"{job.id}-gguf-{quant_type.value}",
                        format=QuantizationFormat.GGUF,
                        quant_type=quant_type.value,
                        file_path="",
                        status="pending",
                    )
                )

        if QuantizationFormat.GPTQ in job.output_formats:
            for bits in job.gptq_bits:
                outputs.append(
                    QuantizedOutput(
                        id=f"{job.id}-gptq-{bits}bit",
                        format=QuantizationFormat.GPTQ,
                        quant_type=f"{bits}bit",
                        file_path="",
                        status="pending",
                    )
                )

        if QuantizationFormat.AWQ in job.output_formats:
            for bits in job.awq_bits:
                outputs.append(
                    QuantizedOutput(
                        id=f"{job.id}-awq-{bits}bit",
                        format=QuantizationFormat.AWQ,
                        quant_type=f"{bits}bit",
                        file_path="",
                        status="pending",
                    )
                )

        if QuantizationFormat.ONNX in job.output_formats:
            outputs.append(
                QuantizedOutput(
                    id=f"{job.id}-onnx",
                    format=QuantizationFormat.ONNX,
                    quant_type="fp16",
                    file_path="",
                    status="pending",
                )
            )

        job.outputs = outputs
        self.store.upsert(job)
        logger.info("Created quantization job", extra={"job_id": job.id, "source_model": job.source_model})
        return job

    def get_job(self, job_id: str) -> Optional[QuantizationJob]:
        """Get a quantization job by ID."""
        return self.store.get(job_id)

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[QuantizationJob]:
        """List quantization jobs."""
        return self.store.list(status=status, limit=limit, offset=offset)

    def update_status(
        self,
        job_id: str,
        status: QuantizationStatus,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[QuantizationJob]:
        """Update job status and progress."""
        job = self.store.get(job_id)
        if not job:
            logger.warning("Job not found for status update", extra={"job_id": job_id})
            return None

        job.status = status
        if progress is not None:
            job.progress = progress
        if current_step is not None:
            job.current_step = current_step
        if error is not None:
            job.error = error

        # Update timestamps
        if status == QuantizationStatus.QUANTIZING and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status in [QuantizationStatus.COMPLETED, QuantizationStatus.FAILED, QuantizationStatus.CANCELLED]:
            job.completed_at = datetime.utcnow()

        self.store.upsert(job)
        logger.info(
            "Updated job status",
            extra={"job_id": job_id, "status": status.value, "progress": progress}
        )
        return job

    def update_output(
        self,
        job_id: str,
        output_id: str,
        file_path: str = None,
        file_size: int = None,
        status: str = None,
        error: str = None,
    ) -> Optional[QuantizationJob]:
        """Update a specific output within a job."""
        job = self.store.get(job_id)
        if not job:
            return None

        for output in job.outputs:
            if output.id == output_id:
                if file_path:
                    output.file_path = file_path
                if file_size is not None:
                    output.file_size = file_size
                if status:
                    output.status = status
                    if status == "completed":
                        job.completed_outputs += 1
                        # Register with model registry if available
                        self._register_output_with_registry(job, output)
                if error:
                    output.error = error
                break

        # Update overall progress
        if job.total_outputs > 0:
            job.progress = (job.completed_outputs / job.total_outputs) * 100

        self.store.upsert(job)
        return job

    def _register_output_with_registry(
        self,
        job: QuantizationJob,
        output: QuantizedOutput,
    ) -> None:
        """Register a completed quantized output with the model registry."""
        if not MODEL_REGISTRY_AVAILABLE:
            return

        try:
            # Cache the model if not already cached
            repo_id = job.source_model
            cached_model = model_registry.get_cached_model(repo_id)
            
            if not cached_model:
                # Cache basic model info
                model_registry.cache_model(
                    repo_id=repo_id,
                    name=job.name,
                    description=f"Quantized from {repo_id}",
                    model_type="quantized",
                    tags=["quantized", output.format.value],
                )
                logger.info("Cached model in registry", extra={"repo_id": repo_id})

            # Add variant for the quantized output
            model_registry.add_variant(
                repo_id=repo_id,
                filename=output.file_path.split("/")[-1],  # Get filename from path
                quantization=output.quant_type,
                size_bytes=output.file_size,
                vram_required_mb=self._estimate_vram_for_quant(job.source_model, output.quant_type),
            )
            logger.info("Registered quantized variant", extra={
                "repo_id": repo_id,
                "quant_type": output.quant_type,
                "output_id": output.id,
            })

        except Exception as e:
            logger.error("Failed to register output with model registry", extra={
                "error": str(e),
                "output_id": output.id,
            })

    def _estimate_vram_for_quant(self, model_name: str, quant_type: str) -> int:
        """Estimate VRAM requirements for a quantized model in MB."""
        # Get base model size
        base_size_gb = self._estimate_model_size(model_name)
        
        # Get quantization factor
        quant_factors = {
            "Q2_K": 0.14,
            "Q3_K_S": 0.19,
            "Q3_K_M": 0.21,
            "Q3_K_L": 0.23,
            "Q4_0": 0.28,
            "Q4_K_S": 0.29,
            "Q4_K_M": 0.31,
            "Q5_0": 0.35,
            "Q5_K_S": 0.36,
            "Q5_K_M": 0.38,
            "Q6_K": 0.50,
            "Q8_0": 0.56,
            "F16": 1.0,
            "4bit": 0.25,  # GPTQ/AWQ
            "8bit": 0.50,
        }
        
        factor = quant_factors.get(quant_type, 0.31)
        vram_gb = base_size_gb * factor * 1.2  # Add 20% overhead
        return int(vram_gb * 1024)  # Convert to MB

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running quantization job."""
        job = self.store.get(job_id)
        if not job:
            return False

        if job.status in [QuantizationStatus.COMPLETED, QuantizationStatus.FAILED, QuantizationStatus.CANCELLED]:
            return False

        self.update_status(job_id, QuantizationStatus.CANCELLED)
        logger.info("Cancelled quantization job", extra={"job_id": job_id})
        return True

    def delete_job(self, job_id: str) -> bool:
        """Delete a quantization job."""
        return self.store.delete(job_id)

    def estimate_resources(
        self,
        source_model: str,
        output_formats: List[QuantizationFormat],
        gguf_quant_types: List[GGUFQuantType] = None,
        gptq_bits: List[int] = None,
        awq_bits: List[int] = None,
    ) -> QuantizationEstimate:
        """Estimate disk space and time requirements for quantization."""
        # This is a rough estimation based on typical model sizes
        # TODO: Improve with actual model size detection
        
        # Estimate base model size from name
        model_size_gb = self._estimate_model_size(source_model)
        
        total_disk = model_size_gb  # Base model in FP16
        outputs_count = 0
        breakdown = {}
        
        if QuantizationFormat.GGUF in output_formats and gguf_quant_types:
            gguf_sizes = {}
            for quant_type in gguf_quant_types:
                size_factor = self._get_gguf_size_factor(quant_type)
                size_gb = model_size_gb * size_factor
                total_disk += size_gb
                outputs_count += 1
                gguf_sizes[quant_type.value] = size_gb
            breakdown["gguf"] = gguf_sizes
        
        if QuantizationFormat.GPTQ in output_formats and gptq_bits:
            gptq_sizes = {}
            for bits in gptq_bits:
                size_gb = model_size_gb * (bits / 16)
                total_disk += size_gb
                outputs_count += 1
                gptq_sizes[f"{bits}bit"] = size_gb
            breakdown["gptq"] = gptq_sizes
        
        if QuantizationFormat.AWQ in output_formats and awq_bits:
            awq_sizes = {}
            for bits in awq_bits:
                size_gb = model_size_gb * (bits / 16)
                total_disk += size_gb
                outputs_count += 1
                awq_sizes[f"{bits}bit"] = size_gb
            breakdown["awq"] = awq_sizes
        
        # Estimate time (very rough - depends on hardware)
        estimated_time = outputs_count * 10  # ~10 minutes per output on average
        
        warnings = []
        if total_disk > 100:
            warnings.append(f"Large disk space required: {total_disk:.1f}GB")
        if outputs_count > 10:
            warnings.append(f"Many outputs requested ({outputs_count}), this will take time")
        
        return QuantizationEstimate(
            disk_space_gb=round(total_disk, 2),
            estimated_time_minutes=estimated_time,
            outputs_count=outputs_count,
            warnings=warnings,
            breakdown=breakdown,
        )

    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in GB from model name."""
        model_lower = model_name.lower()
        
        # Extract parameter count
        if "70b" in model_lower or "72b" in model_lower:
            return 140.0
        elif "34b" in model_lower:
            return 68.0
        elif "13b" in model_lower:
            return 26.0
        elif "8b" in model_lower:
            return 16.0
        elif "7b" in model_lower:
            return 14.0
        elif "3b" in model_lower:
            return 6.0
        elif "1b" in model_lower or "1.5b" in model_lower:
            return 3.0
        else:
            # Default to 7B
            return 14.0

    def _get_gguf_size_factor(self, quant_type: GGUFQuantType) -> float:
        """Get size factor relative to FP16 for GGUF quantization types."""
        size_factors = {
            GGUFQuantType.Q2_K: 0.14,
            GGUFQuantType.Q3_K_S: 0.19,
            GGUFQuantType.Q3_K_M: 0.21,
            GGUFQuantType.Q3_K_L: 0.23,
            GGUFQuantType.Q4_0: 0.28,
            GGUFQuantType.Q4_1: 0.30,
            GGUFQuantType.Q4_K_S: 0.29,
            GGUFQuantType.Q4_K_M: 0.31,
            GGUFQuantType.Q5_0: 0.35,
            GGUFQuantType.Q5_1: 0.37,
            GGUFQuantType.Q5_K_S: 0.36,
            GGUFQuantType.Q5_K_M: 0.38,
            GGUFQuantType.Q6_K: 0.50,
            GGUFQuantType.Q8_0: 0.56,
            GGUFQuantType.F16: 1.0,
            GGUFQuantType.F32: 2.0,
        }
        return size_factors.get(quant_type, 0.31)  # Default to Q4_K_M

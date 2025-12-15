"""
Quantization API routes.

Provides endpoints for creating and managing model quantization jobs.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from enhanced_logger import enhanced_logger as logger

try:
    from modules.quantization import (
        QuantizationJob,
        QuantizationManager,
        QuantizationFormat,
        GGUFQuantType,
        QuantizationEstimate,
    )
    from modules.quantization.executor import QuantizationExecutor
except ImportError:
    from quantization import (
        QuantizationJob,
        QuantizationManager,
        QuantizationFormat,
        GGUFQuantType,
        QuantizationEstimate,
    )
    from quantization.executor import QuantizationExecutor


router = APIRouter(prefix="/api/v1/quantize", tags=["quantization"])

# Initialize manager and executor
quantization_manager = QuantizationManager()
quantization_executor = QuantizationExecutor(quantization_manager)


class CreateQuantizationJobRequest(BaseModel):
    """Request model for creating a quantization job."""
    name: str
    description: Optional[str] = None
    source_model: str
    source_type: str = "huggingface"
    output_formats: List[QuantizationFormat]
    gguf_quant_types: List[GGUFQuantType] = []
    gptq_bits: List[int] = []
    awq_bits: List[int] = []


class EstimateRequest(BaseModel):
    """Request model for estimating quantization resources."""
    source_model: str
    output_formats: List[QuantizationFormat]
    gguf_quant_types: List[GGUFQuantType] = []
    gptq_bits: List[int] = []
    awq_bits: List[int] = []


class FormatInfo(BaseModel):
    """Information about a quantization format."""
    format: QuantizationFormat
    name: str
    description: str
    supported_types: List[str]


@router.post("/jobs", status_code=201)
async def create_quantization_job(request: CreateQuantizationJobRequest):
    """Create a new quantization job."""
    try:
        # Validate that we have quantization types for selected formats
        if QuantizationFormat.GGUF in request.output_formats and not request.gguf_quant_types:
            raise HTTPException(
                status_code=400,
                detail="GGUF format requires at least one quantization type"
            )

        # Create the job
        job = QuantizationJob(
            id="",  # Will be generated
            name=request.name,
            description=request.description,
            source_model=request.source_model,
            source_type=request.source_type,
            output_formats=request.output_formats,
            gguf_quant_types=request.gguf_quant_types,
            gptq_bits=request.gptq_bits or [4],
            awq_bits=request.awq_bits or [4],
        )

        created_job = quantization_manager.create_job(job)

        # Start the job execution
        quantization_executor.start_job(created_job.id)

        logger.info("Created quantization job", extra={
            "job_id": created_job.id,
            "source_model": request.source_model,
        })

        return created_job

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create quantization job", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@router.get("/jobs")
async def list_quantization_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List quantization jobs with optional filtering."""
    try:
        jobs = quantization_manager.list_jobs(status=status, limit=limit, offset=offset)
        return {
            "jobs": jobs,
            "total": len(jobs),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error("Failed to list quantization jobs", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_quantization_job(job_id: str):
    """Get a specific quantization job by ID."""
    job = quantization_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Quantization job not found")
    return job


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_quantization_job(job_id: str):
    """Delete a quantization job."""
    success = quantization_manager.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Quantization job not found")
    logger.info("Deleted quantization job", extra={"job_id": job_id})


@router.post("/jobs/{job_id}/cancel")
async def cancel_quantization_job(job_id: str):
    """Cancel a running quantization job."""
    success = quantization_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job cannot be cancelled (not found or already completed)"
        )
    return {"status": "cancelled", "job_id": job_id}


@router.get("/jobs/{job_id}/outputs")
async def get_job_outputs(job_id: str):
    """Get the outputs of a quantization job."""
    job = quantization_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Quantization job not found")
    
    return {
        "job_id": job_id,
        "outputs": job.outputs,
        "total": len(job.outputs),
        "completed": job.completed_outputs,
    }


@router.post("/estimate")
async def estimate_quantization(request: EstimateRequest) -> QuantizationEstimate:
    """Estimate disk space and time requirements for quantization."""
    try:
        estimate = quantization_manager.estimate_resources(
            source_model=request.source_model,
            output_formats=request.output_formats,
            gguf_quant_types=request.gguf_quant_types,
            gptq_bits=request.gptq_bits,
            awq_bits=request.awq_bits,
        )
        return estimate
    except Exception as e:
        logger.error("Failed to estimate quantization", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to estimate: {str(e)}")


@router.get("/formats")
async def list_supported_formats() -> List[FormatInfo]:
    """List all supported quantization formats and their options."""
    return [
        FormatInfo(
            format=QuantizationFormat.GGUF,
            name="GGUF (llama.cpp)",
            description="Quantized GGUF format for llama.cpp deployment",
            supported_types=[q.value for q in GGUFQuantType],
        ),
        FormatInfo(
            format=QuantizationFormat.GPTQ,
            name="GPTQ",
            description="GPTQ quantization for transformers and vLLM",
            supported_types=["4bit", "8bit"],
        ),
        FormatInfo(
            format=QuantizationFormat.AWQ,
            name="AWQ",
            description="AWQ quantization for transformers and vLLM",
            supported_types=["4bit"],
        ),
        FormatInfo(
            format=QuantizationFormat.ONNX,
            name="ONNX",
            description="ONNX format for ONNX Runtime",
            supported_types=["fp16", "int8"],
        ),
    ]


@router.get("/formats/gguf/types")
async def list_gguf_types():
    """List all GGUF quantization types with descriptions."""
    return {
        "types": [
            {
                "value": GGUFQuantType.Q2_K.value,
                "name": "Q2_K",
                "description": "2-bit quantization, smallest size, lowest quality",
                "size_factor": 0.14,
            },
            {
                "value": GGUFQuantType.Q3_K_S.value,
                "name": "Q3_K_S",
                "description": "3-bit quantization (small)",
                "size_factor": 0.19,
            },
            {
                "value": GGUFQuantType.Q3_K_M.value,
                "name": "Q3_K_M",
                "description": "3-bit quantization (medium)",
                "size_factor": 0.21,
            },
            {
                "value": GGUFQuantType.Q3_K_L.value,
                "name": "Q3_K_L",
                "description": "3-bit quantization (large)",
                "size_factor": 0.23,
            },
            {
                "value": GGUFQuantType.Q4_0.value,
                "name": "Q4_0",
                "description": "4-bit quantization, legacy format",
                "size_factor": 0.28,
            },
            {
                "value": GGUFQuantType.Q4_K_S.value,
                "name": "Q4_K_S",
                "description": "4-bit quantization (small), good balance",
                "size_factor": 0.29,
            },
            {
                "value": GGUFQuantType.Q4_K_M.value,
                "name": "Q4_K_M",
                "description": "4-bit quantization (medium), recommended default",
                "size_factor": 0.31,
            },
            {
                "value": GGUFQuantType.Q5_0.value,
                "name": "Q5_0",
                "description": "5-bit quantization, legacy format",
                "size_factor": 0.35,
            },
            {
                "value": GGUFQuantType.Q5_K_S.value,
                "name": "Q5_K_S",
                "description": "5-bit quantization (small), higher quality",
                "size_factor": 0.36,
            },
            {
                "value": GGUFQuantType.Q5_K_M.value,
                "name": "Q5_K_M",
                "description": "5-bit quantization (medium), good quality/size",
                "size_factor": 0.38,
            },
            {
                "value": GGUFQuantType.Q6_K.value,
                "name": "Q6_K",
                "description": "6-bit quantization, very high quality",
                "size_factor": 0.50,
            },
            {
                "value": GGUFQuantType.Q8_0.value,
                "name": "Q8_0",
                "description": "8-bit quantization, near-lossless",
                "size_factor": 0.56,
            },
            {
                "value": GGUFQuantType.F16.value,
                "name": "F16",
                "description": "16-bit float, no quantization",
                "size_factor": 1.0,
            },
        ]
    }

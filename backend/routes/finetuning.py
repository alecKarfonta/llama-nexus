"""
Fine-tuning API surface.

Current capabilities:
- Upload and store datasets with basic validation/preview
- Inspect arbitrary dataset files
- Register fine-tuning jobs in memory (stub)
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Response
from pathlib import Path

from typing import List, Dict

from pydantic import BaseModel, Field

try:
    from modules.finetuning import (
        DatasetFormat,
        DatasetProcessor,
        DatasetStatus,
        DatasetStore,
        Dataset,
        FineTuningJob,
        LoRAConfig,
        TrainingConfig,
        QLoRAConfig,
        init_dataset_store,
        init_job_store,
        TrainingManager,
        TrainingStatus,
        TrainingEventConsumer,
        PresetName,
        HyperparameterPreset,
        HYPERPARAMETER_PRESETS,
        # Distillation
        TeacherProvider,
        DistillationConfig,
        PromptTemplate,
        DistillationJob,
        DistillationManager,
        DEFAULT_TEMPLATES,
        OutputFormat,
        # VRAM Estimation
        VRAMEstimateRequest,
        VRAMEstimateResponse,
        estimate_training_vram,
        estimate_model_params,
        # Evaluation
        ComparisonType,
        ComparisonSession,
        JudgeConfig,
        EvaluationManager,
        # Benchmarks
        BenchmarkName,
        BenchmarkConfig,
        BenchmarkJob,
        BenchmarkRunner,
        # A/B Testing
        ABTestStatus,
        TrafficSplit,
        ABTest,
        ABTestManager,
        # Adapter Management
        AdapterStatus,
        AdapterMetadata,
        AdapterComparison,
        AdapterManager,
        get_adapter_manager,
    )
except ImportError:
    # Fallback for local execution without package prefix
    from finetuning import (  # type: ignore
        DatasetFormat,
        DatasetProcessor,
        DatasetStatus,
        DatasetStore,
        Dataset,
        FineTuningJob,
        LoRAConfig,
        TrainingConfig,
        QLoRAConfig,
        init_dataset_store,
        TrainingManager,
        init_job_store,
        TrainingStatus,
        TrainingEventConsumer,
        PresetName,
        HyperparameterPreset,
        HYPERPARAMETER_PRESETS,
        TeacherProvider,
        DistillationConfig,
        PromptTemplate,
        DistillationJob,
        DistillationManager,
        DEFAULT_TEMPLATES,
        OutputFormat,
        VRAMEstimateRequest,
        VRAMEstimateResponse,
        estimate_training_vram,
        estimate_model_params,
        ComparisonType,
        ComparisonSession,
        JudgeConfig,
        EvaluationManager,
        BenchmarkName,
        BenchmarkConfig,
        BenchmarkJob,
        BenchmarkRunner,
        ABTestStatus,
        TrafficSplit,
        ABTest,
        ABTestManager,
        AdapterStatus,
        AdapterMetadata,
        AdapterComparison,
        AdapterManager,
        get_adapter_manager,
    )


# Local models directory where merged/trained models are stored
LOCAL_MODELS_DIR = Path("/home/llamacpp/models")


def resolve_local_model_path(model_id: str) -> str:
    """
    Resolve a model identifier to an absolute path if it's a local model.
    
    If the model_id looks like a HuggingFace model ID (org/model format) and
    doesn't exist locally, return it as-is for HuggingFace to download.
    
    If the model_id matches a local model in the models directory, return
    the full absolute path.
    
    Args:
        model_id: Model identifier (e.g., "microsoft/phi-1_5" or "small/test-merged")
    
    Returns:
        The resolved path (absolute local path or HuggingFace ID)
    """
    import re
    
    # If it's already an absolute path, verify it exists
    if model_id.startswith("/"):
        if Path(model_id).exists():
            return model_id
        raise ValueError(f"Model path not found: {model_id}")
    
    # Try to find a matching local model
    if LOCAL_MODELS_DIR.exists():
        # Check for exact match with the model_id as directory name  
        # (e.g., "microsoft_phi-1_5" or "small_test-merged")
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_id)
        direct_match = LOCAL_MODELS_DIR / safe_name
        if direct_match.exists() and (direct_match / "config.json").exists():
            return str(direct_match)
        
        # Check for model_id with suffix replacements (e.g., "small/test-merged")
        # Try matching the last part of the path
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        safe_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
        
        # Look for directories that end with this name
        for item in LOCAL_MODELS_DIR.iterdir():
            if not item.is_dir():
                continue
            # Match by ending (e.g., "small_test-merged" matches "test-merged")
            if item.name.endswith(safe_model_name) or item.name == safe_model_name:
                if (item / "config.json").exists():
                    return str(item)
        
        # Also check with the full safe name (handles "small/test-merged" -> "small_test-merged")
        full_safe = re.sub(r'[/\\]', '_', model_id)
        full_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', full_safe)
        for item in LOCAL_MODELS_DIR.iterdir():
            if item.is_dir() and item.name == full_safe:
                if (item / "config.json").exists():
                    return str(item)
    
    # Return as-is for HuggingFace to handle (will download from hub)
    return model_id


# Simplified request model for job creation from UI
class CreateJobRequest(BaseModel):
    """Simplified request format for creating training jobs from the UI."""
    name: str
    base_model: str
    dataset_ids: List[str]  # Support multiple datasets
    description: Optional[str] = None
    # LoRA settings (flat)
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    # Training settings (flat)
    learning_rate: float = Field(default=2e-4, gt=0)
    batch_size: int = Field(default=4, ge=1)
    num_epochs: int = Field(default=3, ge=1)
    max_seq_length: int = Field(default=2048, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    warmup_steps: int = Field(default=100, ge=0)
    # QLoRA settings
    use_qlora: bool = Field(default=True)
    qlora_bits: int = Field(default=4, ge=2, le=8)


router = APIRouter(prefix="/api/v1/finetune", tags=["finetuning"])
benchmark_runner = BenchmarkRunner()
ab_test_manager = ABTestManager()
adapter_manager = get_adapter_manager()
dataset_processor = DatasetProcessor()
dataset_store: DatasetStore = init_dataset_store()
training_manager = TrainingManager()
training_consumer = TrainingEventConsumer(training_manager)
distillation_manager = DistillationManager(dataset_store=dataset_store)
evaluation_manager = EvaluationManager()
try:
    training_consumer.start()
except Exception:
    # Non-fatal if Redis is unavailable at import time; can be started later.
    pass


@router.post("/datasets/inspect")
def inspect_dataset(file_path: str, format: DatasetFormat | None = None):
    """
    Inspect a dataset file and return detected format plus basic stats.
    """
    detected_format, count, errors = dataset_processor.inspect(file_path, format)
    return {
        "format": detected_format,
        "num_examples": count,
        "validation_errors": errors,
    }


@router.post("/datasets", status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    format: DatasetFormat | None = Form(None),
):
    """
    Upload a dataset file, persist it, and run validation/inspection.
    """
    dataset_id = str(uuid.uuid4())
    target_path = dataset_store.dataset_path(dataset_id, file.filename or "dataset.jsonl")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream file to disk to avoid large memory usage
    with target_path.open("wb") as out_f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)

    detected_format, count, errors = dataset_processor.inspect(str(target_path), format)
    status = DatasetStatus.ERROR if errors else DatasetStatus.READY
    now = datetime.utcnow()
    dataset = Dataset(
        id=dataset_id,
        name=name or Path(file.filename).stem or dataset_id,
        description=description,
        format=detected_format,
        status=status,
        num_examples=count,
        total_tokens=0,
        file_path=str(target_path),
        created_at=now,
        updated_at=now,
        validation_errors=errors or None,
        statistics=None,
    )
    dataset_store.upsert(dataset)
    return {"dataset": dataset, "validation_errors": errors}


@router.get("/datasets")
def list_datasets():
    return dataset_store.list()


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    dataset = dataset_store.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str):
    removed = dataset_store.delete(dataset_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return Response(status_code=204)


@router.post("/datasets/{dataset_id}/validate")
def validate_dataset(dataset_id: str):
    dataset = dataset_store.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    detected_format, count, errors = dataset_processor.inspect(
        dataset.file_path, dataset.format
    )
    status = DatasetStatus.ERROR if errors else DatasetStatus.READY
    dataset.status = status
    dataset.num_examples = count
    dataset.validation_errors = errors or None
    dataset.updated_at = datetime.utcnow()
    dataset_store.upsert(dataset)
    return {"dataset": dataset, "validation_errors": errors}


@router.get("/datasets/{dataset_id}/preview")
def preview_dataset(dataset_id: str, limit: int = 20):
    dataset = dataset_store.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    preview = dataset_processor.preview(dataset.file_path, limit=limit)
    return {"dataset_id": dataset_id, "preview": preview}


@router.get("/datasets/{dataset_id}/stats")
def get_dataset_statistics(dataset_id: str, max_records: int = 5000):
    """
    Get comprehensive statistics for a dataset including:
    - Token distribution histogram
    - Sequence length distribution
    - Field completeness stats
    - Format detection confidence
    """
    dataset = dataset_store.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        stats = dataset_processor.analyze(dataset.file_path, max_records=max_records)
        return {
            "dataset_id": dataset_id,
            "total_records": stats.total_records,
            "detected_format": stats.detected_format,
            "format_confidence": stats.format_confidence,
            "format_indicators": stats.format_indicators,
            "tokens": {
                "total": stats.total_tokens,
                "avg_per_record": round(stats.avg_tokens_per_record, 1),
                "min": stats.min_tokens,
                "max": stats.max_tokens,
                "distribution": stats.token_distribution,
            },
            "sequences": {
                "avg_length": round(stats.avg_sequence_length, 1),
                "min_length": stats.min_sequence_length,
                "max_length": stats.max_sequence_length,
                "distribution": stats.sequence_distribution,
            },
            "fields": [
                {
                    "name": f.name,
                    "present_count": f.present_count,
                    "total_count": f.total_count,
                    "completeness": round(f.completeness * 100, 1),
                    "avg_length": round(f.avg_length, 1),
                    "min_length": f.min_length,
                    "max_length": f.max_length,
                    "empty_count": f.empty_count,
                }
                for f in stats.field_stats
            ],
            "validation": {
                "is_valid": stats.is_valid,
                "errors": stats.validation_errors,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze dataset: {str(e)}")


@router.get("/jobs")
def list_jobs():
    return training_manager.list_jobs()


@router.post("/jobs", status_code=201)
def create_job(request: CreateJobRequest):
    """Create a new fine-tuning job from a simplified request format."""
    # Validate all datasets exist
    if not request.dataset_ids:
        raise HTTPException(status_code=400, detail="At least one dataset is required")
    
    datasets_to_use = []
    for ds_id in request.dataset_ids:
        dataset = dataset_store.get(ds_id)
        if not dataset:
            raise HTTPException(status_code=400, detail=f"Dataset not found: {ds_id}")
        datasets_to_use.append(dataset)
    
    # If multiple datasets, create a combined dataset on the fly
    if len(datasets_to_use) > 1:
        import json
        import tempfile
        from datetime import datetime
        
        combined_data = []
        for ds in datasets_to_use:
            try:
                with open(ds.file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        combined_data.extend(data)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load dataset {ds.id}: {e}")
        
        # Create temporary combined dataset
        combined_id = f"combined_{uuid.uuid4().hex[:8]}"
        combined_path = Path(dataset_store.base_dir) / combined_id / "combined.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(combined_path, 'w') as f:
            json.dump(combined_data, f)
        
        # Register combined dataset
        combined_dataset = Dataset(
            id=combined_id,
            name=f"Combined: {', '.join(ds.name for ds in datasets_to_use)}",
            description=f"Auto-combined from: {', '.join(request.dataset_ids)}",
            format=datasets_to_use[0].format,
            status=DatasetStatus.READY,
            num_examples=len(combined_data),
            total_tokens=sum(ds.total_tokens or 0 for ds in datasets_to_use),
            file_path=str(combined_path),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            validation_errors=None,
            statistics=None,
        )
        dataset_store.upsert(combined_dataset)
        primary_dataset = combined_dataset
        primary_dataset_id = combined_id
    else:
        primary_dataset = datasets_to_use[0]
        primary_dataset_id = request.dataset_ids[0]
    
    # Resolve local model path (handles previously fine-tuned models)
    try:
        resolved_base_model = resolve_local_model_path(request.base_model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Build the full FineTuningJob from the simplified request
    job = FineTuningJob(
        id=str(uuid.uuid4()),
        name=request.name,
        description=request.description,
        base_model=resolved_base_model,
        dataset_id=primary_dataset_id,
        lora_config=LoRAConfig(
            rank=request.lora_rank,
            alpha=request.lora_alpha,
            dropout=request.lora_dropout,
        ),
        training_config=TrainingConfig(
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            num_epochs=request.num_epochs,
            max_seq_length=request.max_seq_length,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            warmup_steps=request.warmup_steps,
        ),
        qlora_config=QLoRAConfig(
            enabled=request.use_qlora,
            bits=request.qlora_bits,
        ),
    )
    
    created = training_manager.create_job(job)
    try:
        training_manager.start_job(created, primary_dataset.dict())
    except Exception as exc:
        training_manager.update_status(created.id, TrainingStatus.FAILED, str(exc))
        raise HTTPException(status_code=500, detail="Failed to start training worker")
    return created


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    """
    Delete a training job and clean up associated files.
    
    If the job is running, it will be stopped first.
    """
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Stop the job if it's still running
    if job.status in (TrainingStatus.TRAINING, TrainingStatus.QUEUED, TrainingStatus.PAUSED):
        training_manager.send_command(job_id, "stop")
    
    # Delete the job and all associated files
    deleted = training_manager.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete job")
    
    return Response(status_code=204)


@router.post("/jobs/{job_id}/pause")
def pause_job(job_id: str):
    """
    Pause a running training job.
    
    The job will save a checkpoint and pause until resumed.
    """
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != TrainingStatus.TRAINING:
        raise HTTPException(status_code=400, detail="Job is not training")
    
    training_manager.send_command(job_id, "pause")
    training_manager.update_status(job_id, TrainingStatus.PAUSED)
    return {"status": "pausing", "job_id": job_id}


@router.post("/jobs/{job_id}/unpause")
def unpause_job(job_id: str):
    """
    Resume a paused training job.
    
    The job will continue from where it left off.
    """
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != TrainingStatus.PAUSED:
        raise HTTPException(status_code=400, detail="Job is not paused")
    
    training_manager.send_command(job_id, "resume")
    training_manager.update_status(job_id, TrainingStatus.TRAINING)
    return {"status": "resuming", "job_id": job_id}


@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: str, checkpoint: Optional[str] = None):
    """
    Resume a failed or cancelled training job from a checkpoint.
    
    Args:
        job_id: The job ID to resume
        checkpoint: Specific checkpoint path to resume from, or "latest" (default).
                   If not specified, automatically uses the latest checkpoint.
    """
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in (TrainingStatus.CANCELLED, TrainingStatus.FAILED):
        raise HTTPException(status_code=400, detail="Job is not in a resumable state")
    dataset = dataset_store.get(job.dataset_id)
    if not dataset:
        raise HTTPException(status_code=400, detail="Dataset not found")
    
    # Determine checkpoint to resume from
    resume_from = checkpoint or "latest"
    
    # Validate checkpoint exists if specific path provided
    if checkpoint and checkpoint != "latest":
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Checkpoint not found: {checkpoint}"
            )
    
    # Check if there are checkpoints available
    available_checkpoints = training_manager.list_checkpoints(job_id)
    if not available_checkpoints and resume_from == "latest":
        # No checkpoints, start fresh
        resume_from = None
    
    updated = training_manager.update_status(job_id, TrainingStatus.QUEUED)
    try:
        training_manager.start_job(
            updated,
            dataset.dict(),
            resume_from_checkpoint=resume_from,
        )
    except Exception as exc:
        training_manager.update_status(job_id, TrainingStatus.FAILED, str(exc))
        raise HTTPException(status_code=500, detail="Failed to start training worker")
    
    return {
        "job": updated,
        "resumed_from": resume_from,
        "available_checkpoints": len(available_checkpoints),
    }


@router.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, limit: int = 200):
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    logs = training_manager.read_logs(job_id, limit=limit)
    return {"job_id": job_id, "logs": logs}


@router.get("/jobs/{job_id}/metrics")
def get_job_metrics(job_id: str):
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "metrics": job.metrics or {}}


@router.get("/jobs/{job_id}/metrics/history")
def get_job_metrics_history(job_id: str):
    """Get metrics history for loss chart visualization."""
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    history = training_manager.read_metrics_history(job_id)
    return {"job_id": job_id, "history": history}


@router.get("/jobs/{job_id}/checkpoints")
def list_checkpoints(job_id: str):
    """
    List available checkpoints for a training job.
    
    Returns checkpoint details including step number, epoch, and loss.
    """
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    checkpoints = training_manager.list_checkpoints(job_id)
    latest = training_manager.get_latest_checkpoint(job_id)
    
    return {
        "job_id": job_id,
        "checkpoints": checkpoints,
        "latest_checkpoint": latest,
        "can_resume": job.status in (TrainingStatus.CANCELLED, TrainingStatus.FAILED) and len(checkpoints) > 0,
    }


@router.get("/adapters")
def list_adapters():
    """
    List all adapters from completed training jobs.
    
    This endpoint returns adapters derived from training jobs.
    For full adapter management with versioning, use /adapters/registry.
    """
    import re
    jobs = training_manager.list_jobs()
    adapters = []
    models_dir = Path("/home/llamacpp/models")
    
    for job in jobs:
        if job.adapter_path:
            # Check if this adapter is registered
            registered = adapter_manager.get_adapter_by_job(job.id)
            
            # Check merge status
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', job.name or job.id[:8])
            merged_dir = models_dir / f"{safe_name}-merged"
            merged_metadata = merged_dir / "training_metadata.json"
            
            if merged_metadata.exists():
                merge_status = "completed"
            elif merged_dir.exists():
                merge_status = "in_progress"
            else:
                merge_status = None
            
            adapter_info = {
                "job_id": job.id,
                "name": job.name,
                "base_model": job.base_model,
                "adapter_path": job.adapter_path,
                "metrics": job.metrics,
                "status": job.status.value if hasattr(job.status, 'value') else str(job.status),
                "registered": registered is not None,
                "registry_id": registered.id if registered else None,
                "merge_status": merge_status,
                "merged_path": str(merged_dir) if merge_status == "completed" else None,
            }
            adapters.append(adapter_info)
    return {"adapters": adapters}


@router.post("/adapters/{job_id}/register")
def register_adapter_from_job(job_id: str, description: Optional[str] = None, tags: Optional[List[str]] = None):
    """
    Register an adapter from a completed training job.
    
    This adds the adapter to the registry for versioning and management.
    """
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.adapter_path:
        raise HTTPException(status_code=400, detail="No adapter available for this job")
    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Training not completed")
    
    # Check if already registered
    existing = adapter_manager.get_adapter_by_job(job_id)
    if existing:
        return {"adapter": existing, "message": "Adapter already registered"}
    
    adapter = adapter_manager.register_adapter(
        name=job.name,
        base_model=job.base_model,
        training_job_id=job_id,
        adapter_path=job.adapter_path,
        lora_config=job.lora_config.dict(),
        training_config=job.training_config.dict(),
        metrics=job.metrics,
        description=description,
        tags=tags or [],
    )
    
    return {"adapter": adapter, "message": "Adapter registered successfully"}


# ============================================================================
# Hyperparameter Presets
# ============================================================================


@router.get("/presets")
def list_presets():
    """List available hyperparameter presets."""
    return {
        "presets": [
            {
                "name": preset.name.value,
                "display_name": preset.display_name,
                "description": preset.description,
                "lora_config": preset.lora_config.dict(),
                "training_config": preset.training_config.dict(),
                "qlora_config": preset.qlora_config.dict(),
            }
            for preset in HYPERPARAMETER_PRESETS.values()
        ]
    }


@router.get("/presets/{preset_name}")
def get_preset(preset_name: str):
    """Get a specific hyperparameter preset by name."""
    try:
        preset_key = PresetName(preset_name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    preset = HYPERPARAMETER_PRESETS.get(preset_key)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    return {
        "name": preset.name.value,
        "display_name": preset.display_name,
        "description": preset.description,
        "lora_config": preset.lora_config.dict(),
        "training_config": preset.training_config.dict(),
        "qlora_config": preset.qlora_config.dict(),
    }


# ============================================================================
# Adapter Merge & Export
# ============================================================================


@router.post("/adapters/{job_id}/merge")
def merge_adapter(job_id: str):
    """
    Merge LoRA adapter weights into the base model to create a standalone model.
    This produces a full model that can be used without the adapter.
    """
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.adapter_path:
        raise HTTPException(status_code=400, detail="No adapter available for this job")
    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Training not completed")

    # Save merged model to models directory so it appears in Model Manager
    # Use a sanitized name based on job name
    import re
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', job.name or job_id[:8])
    models_dir = Path("/home/llamacpp/models")
    merge_output_dir = models_dir / f"{safe_name}-merged"
    merge_output_dir.mkdir(parents=True, exist_ok=True)

    # Publish merge command to Redis
    import redis
    import json
    import os
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    try:
        client = redis.from_url(redis_url, decode_responses=True)
        client.publish(
            f"training:commands:{job_id}",
            json.dumps({
                "command": "merge",
                "adapter_path": job.adapter_path,
                "base_model": job.base_model,
                "output_dir": str(merge_output_dir),
            }),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue merge: {exc}")

    return {
        "job_id": job_id,
        "status": "merge_queued",
        "output_dir": str(merge_output_dir),
        "message": "Merge task queued. Check job status for completion.",
    }


@router.post("/adapters/{job_id}/export")
def export_adapter_to_gguf(job_id: str, quant_type: str = "q4_k_m"):
    """
    Export a merged model or adapter to GGUF format for llama.cpp deployment.
    Uses the existing quantization infrastructure.
    """
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.adapter_path:
        raise HTTPException(status_code=400, detail="No adapter available for this job")
    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Training not completed")

    # Create a quantization job using the existing quantization module
    try:
        from modules.quantization import (
            QuantizationJob,
            QuantizationFormat,
            GGUFQuantType,
        )
        from modules.quantization.manager import QuantizationManager
        from modules.quantization.executor import QuantizationExecutor
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Quantization module not available"
        )

    # Determine source: use merged model if available, otherwise need to merge first
    merged_dir = Path(job.adapter_path).parent / "merged"
    if not merged_dir.exists():
        raise HTTPException(
            status_code=400,
            detail="Model not merged yet. Call /merge first, then export."
        )

    # Map quant_type string to enum
    try:
        gguf_quant = GGUFQuantType(quant_type.upper())
    except ValueError:
        valid_types = [q.value for q in GGUFQuantType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid quant_type. Valid options: {valid_types}"
        )

    # Create quantization job
    quant_manager = QuantizationManager()
    quant_job = QuantizationJob(
        id="",
        name=f"lora-export-{job.name}",
        description=f"GGUF export of fine-tuned adapter from job {job_id}",
        source_model=str(merged_dir),
        source_type="local",
        output_formats=[QuantizationFormat.GGUF],
        gguf_quant_types=[gguf_quant],
    )
    created_quant_job = quant_manager.create_job(quant_job)

    # Start the quantization
    quant_executor = QuantizationExecutor(quant_manager)
    quant_executor.start_job(created_quant_job.id)

    return {
        "job_id": job_id,
        "quantization_job_id": created_quant_job.id,
        "status": "export_queued",
        "quant_type": quant_type,
        "message": "GGUF export queued. Monitor quantization job for progress.",
    }


@router.post("/adapters/{job_id}/deploy")
def register_merged_model_for_deployment(job_id: str, name: Optional[str] = None):
    """
    Register a merged adapter as a deployable model in the models registry.
    
    This creates a model entry that appears in the Models page and can be
    deployed like any other model.
    
    Args:
        job_id: The training job ID
        name: Optional custom name for the model (defaults to job name)
    
    Returns:
        The created model entry with its ID
    """
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.adapter_path:
        raise HTTPException(status_code=400, detail="No adapter available for this job")
    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Training not completed")
    
    # Check if merged model exists
    merged_dir = Path(job.adapter_path).parent / "merged"
    if not merged_dir.exists():
        raise HTTPException(
            status_code=400,
            detail="Model not merged yet. Call /merge first to create the merged model."
        )
    
    # Create model entry in the models registry
    model_name = name or f"{job.name}-trained"
    
    # Check for models service/manager
    try:
        from routes.models import model_store
        
        # Create the model entry
        model_entry = {
            "name": model_name,
            "framework": "transformers",
            "status": "available",
            "path": str(merged_dir),
            "source": "trained",
            "training_job_id": job_id,
            "base_model": job.base_model,
            "parameters": None,  # Could estimate from base model
            "quantization": None,
            "context_length": job.training_config.max_seq_length if job.training_config else 2048,
            "description": f"Fine-tuned from {job.base_model} using training job {job.name}",
            "lora_config": job.lora_config.dict() if job.lora_config else None,
            "training_metrics": job.metrics,
        }
        
        created_model = model_store.create(model_entry)
        
        return {
            "job_id": job_id,
            "model_id": created_model.get("id"),
            "model_name": model_name,
            "model_path": str(merged_dir),
            "status": "registered",
            "message": f"Model '{model_name}' registered successfully. It now appears in the Models page.",
        }
    except ImportError:
        # Fallback: just return the path info if model_store not available
        return {
            "job_id": job_id,
            "model_name": model_name,
            "model_path": str(merged_dir),
            "status": "registered_local",
            "message": f"Merged model available at {merged_dir}. Models registry not available.",
        }


# ============================================================================
# VRAM Estimation
# ============================================================================


@router.post("/estimate")
def estimate_vram(request: VRAMEstimateRequest):
    """Estimate VRAM requirements for training configuration."""
    estimate = estimate_training_vram(
        model_name=request.model_name,
        lora_rank=request.lora_rank,
        batch_size=request.batch_size,
        seq_length=request.seq_length,
        qlora_enabled=request.qlora_enabled,
        gradient_accumulation=request.gradient_accumulation,
    )
    params = estimate_model_params(request.model_name)
    warning = None
    if estimate.total_gb > 24:
        warning = "This configuration may require a high-end GPU (A100/H100)"
    elif estimate.total_gb > 16:
        warning = "This configuration requires at least an RTX 4090 or A10"

    return VRAMEstimateResponse(
        model_name=request.model_name,
        estimated_params_b=params,
        model_memory_gb=estimate.model_memory_gb,
        lora_memory_gb=estimate.lora_memory_gb,
        optimizer_memory_gb=estimate.optimizer_memory_gb,
        activation_memory_gb=estimate.activation_memory_gb,
        total_gb=estimate.total_gb,
        recommended_gpu=estimate.recommended_gpu,
        fits_on=estimate.fits_on,
        warning=warning,
    )


# ============================================================================
# Distillation
# ============================================================================


@router.get("/distillation/templates")
def list_distillation_templates():
    """List available distillation prompt templates."""
    return {"templates": [t.dict() for t in distillation_manager.list_templates()]}


@router.post("/distillation/jobs", status_code=201)
def create_distillation_job(job: DistillationJob):
    """Create a new distillation job to generate training data from a teacher model."""
    created = distillation_manager.create_job(job)
    return created


@router.get("/distillation/jobs")
def list_distillation_jobs():
    """List all distillation jobs."""
    return {"jobs": [j.dict() for j in distillation_manager.list_jobs()]}


@router.get("/distillation/jobs/{job_id}")
def get_distillation_job(job_id: str):
    """Get a specific distillation job."""
    job = distillation_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Distillation job not found")
    return job


@router.post("/distillation/jobs/{job_id}/start")
async def start_distillation_job(job_id: str):
    """Start a distillation job."""
    job = distillation_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Distillation job not found")
    distillation_manager.start_job(job_id)
    return {"job_id": job_id, "status": "started"}


@router.post("/distillation/jobs/{job_id}/cancel")
def cancel_distillation_job(job_id: str):
    """Cancel a running distillation job."""
    success = distillation_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job not running or not found")
    return {"job_id": job_id, "status": "cancelled"}


@router.get("/distillation/jobs/{job_id}/preview")
def preview_distillation_data(job_id: str, limit: int = 20):
    """Preview generated examples from a distillation job."""
    examples = distillation_manager.get_generated_examples(job_id, limit=limit)
    return {"job_id": job_id, "examples": examples, "count": len(examples)}


@router.get("/distillation/jobs/{job_id}/quality-assessment")
def assess_distillation_quality(job_id: str):
    """Analyze the quality of a distillation job and provide recommendations."""
    job = distillation_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Distillation job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # Analyze the job metrics and provide assessment
    assessment = _analyze_distillation_quality(job)
    return assessment


def _analyze_distillation_quality(job):
    """Analyze distillation job quality and provide recommendations."""
    metrics = job.metrics
    config = job.config
    
    # Calculate quality scores
    success_rate = metrics.generated_count / metrics.total_prompts if metrics.total_prompts > 0 else 0
    avg_quality = metrics.avg_quality_score or 0
    refinement_rate = metrics.refined_count / metrics.generated_count if metrics.generated_count > 0 else 0
    
    # Determine overall grade
    if avg_quality >= 0.85 and success_rate >= 0.9:
        grade = "A"
        grade_color = "#059669"
    elif avg_quality >= 0.75 and success_rate >= 0.8:
        grade = "B"
        grade_color = "#1d4ed8"
    elif avg_quality >= 0.65 and success_rate >= 0.7:
        grade = "C"
        grade_color = "#f59e0b"
    else:
        grade = "D"
        grade_color = "#dc2626"
    
    # Generate recommendations
    recommendations = []
    
    if success_rate < 0.8:
        recommendations.append({
            "type": "warning",
            "title": "Low Success Rate",
            "description": f"Only {success_rate:.1%} of prompts generated valid responses. Consider simplifying prompts or adjusting temperature.",
            "action": "Review failed prompts and simplify complex requests"
        })
    
    if avg_quality < 0.7:
        recommendations.append({
            "type": "warning", 
            "title": "Low Quality Scores",
            "description": f"Average quality score is {avg_quality:.2f}. Consider using a stronger teacher model or enabling quality validation.",
            "action": "Try GPT-4o or enable LLM-as-judge quality filtering"
        })
    
    if refinement_rate > 0.3:
        recommendations.append({
            "type": "info",
            "title": "High Refinement Rate", 
            "description": f"{refinement_rate:.1%} of responses were refined. This indicates the teacher model needed multiple attempts.",
            "action": "Consider adjusting prompts or teacher model settings"
        })
    
    if metrics.generated_count < 100:
        recommendations.append({
            "type": "info",
            "title": "Small Dataset Size",
            "description": f"Only {metrics.generated_count} examples generated. Consider generating more data for better fine-tuning results.",
            "action": "Aim for 200-500 examples for most use cases"
        })
    
    if metrics.generated_count > 1000:
        recommendations.append({
            "type": "success",
            "title": "Large Dataset",
            "description": f"Generated {metrics.generated_count} examples. This should provide excellent fine-tuning results.",
            "action": "Consider using lower learning rates and more epochs"
        })
    
    # Training recommendations based on dataset characteristics
    training_recommendations = []
    
    if metrics.generated_count < 200:
        training_recommendations.append("Use higher learning rate (5e-4) and fewer epochs (1-2) to prevent overfitting")
    elif metrics.generated_count > 500:
        training_recommendations.append("Use lower learning rate (1e-4) and can handle more epochs (3-4)")
    
    if avg_quality >= 0.8:
        training_recommendations.append("High quality data - can use standard hyperparameters")
    else:
        training_recommendations.append("Lower quality data - consider data augmentation or additional filtering")
    
    if config.strategy == "chain_of_thought":
        training_recommendations.append("CoT data - focus on reasoning consistency during training")
    
    return {
        "overall_grade": grade,
        "grade_color": grade_color,
        "metrics": {
            "success_rate": round(success_rate * 100, 1),
            "avg_quality_score": round(avg_quality, 2),
            "refinement_rate": round(refinement_rate * 100, 1),
            "total_examples": metrics.generated_count,
            "failed_examples": metrics.failed_count,
            "filtered_examples": metrics.filtered_count
        },
        "recommendations": recommendations,
        "training_recommendations": training_recommendations,
        "ready_for_training": success_rate >= 0.7 and avg_quality >= 0.6,
        "estimated_performance": _estimate_model_performance(avg_quality, metrics.generated_count)
    }


def _estimate_model_performance(avg_quality, dataset_size):
    """Estimate expected model performance based on dataset characteristics."""
    if avg_quality >= 0.85 and dataset_size >= 300:
        return {
            "level": "Excellent",
            "description": "Should produce high-quality, reliable outputs similar to teacher model",
            "confidence": 90
        }
    elif avg_quality >= 0.75 and dataset_size >= 200:
        return {
            "level": "Good", 
            "description": "Should handle most tasks well with occasional inconsistencies",
            "confidence": 75
        }
    elif avg_quality >= 0.65 and dataset_size >= 100:
        return {
            "level": "Fair",
            "description": "Will work for basic tasks but may need prompt engineering",
            "confidence": 60
        }
    else:
        return {
            "level": "Poor",
            "description": "May produce inconsistent results, consider improving dataset",
            "confidence": 40
        }


@router.get("/distillation/jobs/{job_id}/download")
def download_distillation_output(job_id: str):
    """Download the raw output file from a distillation job."""
    from fastapi.responses import FileResponse
    
    job = distillation_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Distillation job not found")
    
    if not job.output_file or not Path(job.output_file).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    filename = f"{job.name.replace(' ', '_')}_distilled_{job.id[:8]}.jsonl"
    return FileResponse(
        path=job.output_file,
        filename=filename,
        media_type="application/jsonl"
    )


@router.post("/distillation/jobs/{job_id}/export")
async def export_distillation_dataset(
    job_id: str, 
    output_format: OutputFormat = OutputFormat.ALPACA,
    include_metadata: bool = False
):
    """Export distillation output in a specific format."""
    from fastapi.responses import FileResponse
    
    job = distillation_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Distillation job not found")
    
    try:
        output_path = await distillation_manager.export_dataset(
            job_id=job_id,
            output_format=output_format,
            include_metadata=include_metadata
        )
        
        filename = f"{job.name.replace(' ', '_')}_exported_{output_format.value}_{job.id[:8]}.jsonl"
        return FileResponse(
            path=output_path,
            filename=filename,
            media_type="application/jsonl"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/distillation/jobs/{job_id}/create-training-job")
async def create_training_job_from_distillation(job_id: str):
    """Create a training job directly from a completed distillation job with smart defaults."""
    job = distillation_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Distillation job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Distillation job not completed")
    
    if not job.output_dataset_id:
        raise HTTPException(status_code=400, detail="No dataset created from distillation job")
    
    # Get dataset info for smart recommendations
    dataset = dataset_store.get(job.output_dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Smart hyperparameter recommendations based on dataset size and quality
    recommendations = _get_smart_hyperparameters(job, dataset)
    
    # Create training job with recommendations
    training_job = FineTuningJob(
        name=f"{job.name} - Fine-tuned",
        dataset_id=job.output_dataset_id,
        base_model="meta-llama/Llama-3.2-8B-Instruct",  # Default model
        preset_name="balanced",
        config=recommendations,
        status=TrainingStatus.PENDING,
        created_at=datetime.utcnow()
    )
    
    # Save the job
    created_job = training_manager.create_job(training_job)
    
    return {
        "training_job": created_job.dict(),
        "recommendations": recommendations,
        "dataset_info": {
            "size": dataset.num_examples,
            "format": dataset.format.value,
            "estimated_training_time": _estimate_training_time(dataset.num_examples)
        }
    }


def _get_smart_hyperparameters(distillation_job, dataset):
    """Generate smart hyperparameter recommendations based on dataset characteristics."""
    size = dataset.num_examples
    
    # Base recommendations on dataset size
    if size < 100:
        # Small dataset - prevent overfitting
        return {
            "num_train_epochs": 2,
            "learning_rate": 5e-4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "warmup_steps": max(10, size // 10),
            "save_steps": max(25, size // 4),
            "eval_steps": max(10, size // 10),
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
        }
    elif size < 500:
        # Medium dataset - balanced approach
        return {
            "num_train_epochs": 3,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_steps": max(20, size // 20),
            "save_steps": max(50, size // 10),
            "eval_steps": max(25, size // 20),
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
        }
    else:
        # Large dataset - can handle more aggressive training
        return {
            "num_train_epochs": 2,
            "learning_rate": 1e-4,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "warmup_steps": max(50, size // 50),
            "save_steps": max(100, size // 20),
            "eval_steps": max(50, size // 50),
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
        }


def _estimate_training_time(num_examples):
    """Estimate training time based on dataset size."""
    # Rough estimates based on typical hardware
    minutes_per_100_examples = 15  # Approximate for 8B model on RTX 4090
    estimated_minutes = (num_examples / 100) * minutes_per_100_examples
    
    if estimated_minutes < 60:
        return f"~{int(estimated_minutes)} minutes"
    else:
        hours = estimated_minutes / 60
        return f"~{hours:.1f} hours"


@router.get("/workflow-templates")
def list_workflow_templates():
    """List all available workflow templates for distillation + fine-tuning."""
    try:
        from modules.finetuning.workflow_templates import list_templates
    except ImportError:
        from finetuning.workflow_templates import list_templates
    return {"templates": [t.dict() for t in list_templates()]}


@router.get("/workflow-templates/{template_id}")
def get_workflow_template(template_id: str):
    """Get a specific workflow template."""
    try:
        from modules.finetuning.workflow_templates import get_template
    except ImportError:
        from finetuning.workflow_templates import get_template
    template = get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template


@router.post("/workflow-templates/{template_id}/start")
async def start_workflow_from_template(template_id: str, customizations: Dict = None):
    """Start a complete workflow (distillation + training) from a template."""
    try:
        from modules.finetuning.workflow_templates import get_template
    except ImportError:
        from finetuning.workflow_templates import get_template
    
    template = get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Apply any customizations
    config = template.distillation_config.copy()
    if customizations:
        for key, value in customizations.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Create distillation job from template
    job = DistillationJob(
        name=f"{template.name} Workflow",
        description=f"Automated workflow: {template.description}",
        config=config,
        prompts=template.sample_prompts,
        target_examples=config.target_examples
    )
    
    # Create and start the distillation job
    created_job = distillation_manager.create_job(job)
    distillation_manager.start_job(created_job.id)
    
    return {
        "distillation_job": created_job.dict(),
        "template": template.dict(),
        "next_steps": [
            "Monitor distillation progress",
            "Review generated examples",
            "Start training job when distillation completes",
            f"Expected completion: {template.estimated_time}"
        ]
    }


# ============================================================================
# Model Evaluation
# ============================================================================


@router.post("/eval/sessions", status_code=201)
def create_evaluation_session(session: ComparisonSession):
    """Create a new model comparison session."""
    created = evaluation_manager.create_session(session)
    return created


@router.get("/eval/sessions")
def list_evaluation_sessions():
    """List all evaluation sessions."""
    return {"sessions": [s.dict() for s in evaluation_manager.list_sessions()]}


@router.get("/eval/sessions/{session_id}")
def get_evaluation_session(session_id: str):
    """Get a specific evaluation session."""
    session = evaluation_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/eval/sessions/{session_id}", status_code=204)
def delete_evaluation_session(session_id: str):
    """Delete an evaluation session."""
    success = evaluation_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return Response(status_code=204)


@router.post("/eval/sessions/{session_id}/compare")
async def run_comparison(session_id: str, prompt: str, system_prompt: Optional[str] = None):
    """Run a single comparison between base and fine-tuned model."""
    try:
        comparison = await evaluation_manager.run_comparison(
            session_id=session_id,
            prompt=prompt,
            system_prompt=system_prompt,
        )
        return comparison
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/eval/sessions/{session_id}/batch")
async def run_batch_comparison(session_id: str):
    """Run comparisons for all prompts in the session."""
    try:
        session = await evaluation_manager.run_batch_comparison(session_id)
        return session
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/eval/sessions/{session_id}/rate")
def submit_rating(
    session_id: str,
    comparison_id: str,
    preferred: str,
    reason: Optional[str] = None,
):
    """Submit human rating for a comparison."""
    try:
        comparison = evaluation_manager.submit_rating(
            session_id=session_id,
            comparison_id=comparison_id,
            preferred=preferred,
            reason=reason,
        )
        return comparison
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/eval/sessions/{session_id}/judge")
async def run_judge_evaluation(
    session_id: str,
    comparison_id: str,
    judge_model: str = "gpt-4",
    api_key: Optional[str] = None,
):
    """Run LLM-as-judge evaluation on a comparison."""
    try:
        config = JudgeConfig(judge_model=judge_model, api_key=api_key)
        evaluation = await evaluation_manager.run_judge_evaluation(
            session_id=session_id,
            comparison_id=comparison_id,
            judge_config=config,
        )
        return evaluation
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/eval/sessions/{session_id}/summary")
def get_session_summary(session_id: str):
    """Get summary statistics for a comparison session."""
    try:
        summary = evaluation_manager.get_session_summary(session_id)
        return summary
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# Benchmarks
# ============================================================================


@router.get("/benchmarks/available")
def list_available_benchmarks():
    """List all available benchmark suites with HuggingFace dataset status."""
    # Static metadata for display
    benchmark_metadata = {
        "mmlu": {
            "display_name": "MMLU",
            "description": "Multi-task language understanding across 57 subjects",
            "metrics": ["accuracy"],
        },
        "hellaswag": {
            "display_name": "HellaSwag",
            "description": "Commonsense reasoning and sentence completion",
            "metrics": ["accuracy"],
        },
        "arc_easy": {
            "display_name": "ARC-Easy",
            "description": "Science question answering (easy subset)",
            "metrics": ["accuracy"],
        },
        "arc_challenge": {
            "display_name": "ARC-Challenge",
            "description": "Science question answering (challenge subset)",
            "metrics": ["accuracy"],
        },
        "truthfulqa": {
            "display_name": "TruthfulQA",
            "description": "Truthfulness evaluation for language models",
            "metrics": ["truthful_rate"],
        },
        "gsm8k": {
            "display_name": "GSM8K",
            "description": "Grade school math word problems",
            "metrics": ["accuracy"],
        },
        "humaneval": {
            "display_name": "HumanEval",
            "description": "Python code generation benchmark",
            "metrics": ["pass_at_1"],
        },
    }
    
    # Get dynamic dataset info from the benchmark runner
    dataset_info = benchmark_runner.list_available_benchmarks()
    
    # Merge static metadata with dynamic dataset info
    benchmarks = []
    for ds_info in dataset_info:
        name = ds_info["name"]
        metadata = benchmark_metadata.get(name, {})
        benchmarks.append({
            "name": name,
            "display_name": metadata.get("display_name", name.upper()),
            "description": metadata.get("description", ""),
            "metrics": metadata.get("metrics", ["accuracy"]),
            "repo_id": ds_info.get("repo_id"),
            "subset": ds_info.get("subset"),
            "split": ds_info.get("split"),
            "loaded": ds_info.get("loaded", False),
            "huggingface_available": ds_info.get("available", False),
            "has_fallback": ds_info.get("has_fallback", False),
            "fallback_sample_count": ds_info.get("fallback_sample_count", 0),
        })
    
    return {"benchmarks": benchmarks}


@router.post("/benchmarks/jobs", status_code=201)
def create_benchmark_job(job: BenchmarkJob):
    """Create a new benchmark evaluation job."""
    # If adapter_id is provided, look up the adapter path
    if job.adapter_id:
        training_job = training_manager.get_job(job.adapter_id)
        if training_job and training_job.adapter_path:
            job.adapter_path = training_job.adapter_path

    created = benchmark_runner.create_job(job)
    return created


@router.get("/benchmarks/jobs")
def list_benchmark_jobs():
    """List all benchmark jobs."""
    jobs = benchmark_runner.list_jobs()
    return {"jobs": [j.dict() for j in jobs]}


@router.get("/benchmarks/jobs/{job_id}")
def get_benchmark_job(job_id: str):
    """Get a specific benchmark job."""
    job = benchmark_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Benchmark job not found")
    return job


@router.post("/benchmarks/jobs/{job_id}/start")
async def start_benchmark_job(job_id: str):
    """Start running a benchmark job."""
    job = benchmark_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Benchmark job not found")

    success = benchmark_runner.start_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job already running or not found")

    return {"job_id": job_id, "status": "started"}


@router.post("/benchmarks/jobs/{job_id}/cancel")
def cancel_benchmark_job(job_id: str):
    """Cancel a running benchmark job."""
    success = benchmark_runner.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job not running or not found")
    return {"job_id": job_id, "status": "cancelled"}


@router.delete("/benchmarks/jobs/{job_id}", status_code=204)
def delete_benchmark_job(job_id: str):
    """Delete a benchmark job."""
    success = benchmark_runner.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return Response(status_code=204)


@router.get("/benchmarks/jobs/{job_id}/summary")
def get_benchmark_summary(job_id: str):
    """Get a summary of benchmark results."""
    summary = benchmark_runner.get_summary(job_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Job not found")
    return summary


# ============================================================================
# Dataset Discovery and Management
# ============================================================================


@router.get("/benchmarks/datasets/search")
def search_benchmark_datasets(
    query: str,
    limit: int = 20,
    task: Optional[str] = None,
):
    """
    Search HuggingFace Hub for datasets.
    
    Args:
        query: Search query string
        limit: Maximum number of results (default 20)
        task: Filter by task type (e.g., 'question-answering', 'text-classification')
    """
    results = benchmark_runner.search_datasets(query, limit, task)
    return {"datasets": results, "query": query, "count": len(results)}


@router.get("/benchmarks/datasets/{repo_id:path}/info")
def get_dataset_info(repo_id: str):
    """
    Get detailed information about a HuggingFace dataset.
    
    Args:
        repo_id: HuggingFace dataset repository ID (e.g., 'cais/mmlu')
    """
    info = benchmark_runner.get_dataset_info(repo_id)
    if "error" in info and not info.get("configs"):
        raise HTTPException(status_code=404, detail=info["error"])
    return info


@router.get("/benchmarks/datasets/{benchmark_name}/preview")
def preview_benchmark_dataset(
    benchmark_name: str,
    num_samples: int = 5,
):
    """
    Preview samples from a benchmark dataset.
    
    Args:
        benchmark_name: Name of the benchmark (mmlu, hellaswag, etc.)
        num_samples: Number of samples to preview (default 5)
    """
    try:
        preview = benchmark_runner.preview_dataset(benchmark_name, num_samples)
        return preview
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/benchmarks/datasets/custom")
def load_custom_benchmark_dataset(
    repo_id: str,
    name: Optional[str] = None,
    split: str = "test",
    num_samples: Optional[int] = None,
):
    """
    Load a custom dataset from HuggingFace Hub for benchmarking.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        name: Dataset configuration/subset name
        split: Dataset split to load (default 'test')
        num_samples: Number of samples to load (None = all)
    """
    result = benchmark_runner.load_custom_benchmark(
        repo_id=repo_id,
        name=name,
        split=split,
        num_samples=num_samples,
    )
    
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.get("/benchmarks/datasets/cache")
def get_dataset_cache_info():
    """Get information about cached benchmark datasets."""
    return benchmark_runner.get_cache_info()


@router.delete("/benchmarks/datasets/cache")
def clear_dataset_cache(benchmark_name: Optional[str] = None):
    """
    Clear cached benchmark datasets.
    
    Args:
        benchmark_name: Specific benchmark to clear (None = clear all)
    """
    benchmark_runner.clear_dataset_cache(benchmark_name)
    return {"status": "cleared", "benchmark": benchmark_name or "all"}


# ============================================================================
# A/B Testing
# ============================================================================


@router.post("/ab-tests", status_code=201)
def create_ab_test(test: ABTest):
    """
    Create a new A/B test for comparing adapters.
    
    Variants can include the base model (adapter_path=None) and/or
    multiple LoRA adapters with different traffic weights.
    """
    created = ab_test_manager.create_test(test)
    return created


@router.get("/ab-tests")
def list_ab_tests():
    """List all A/B tests."""
    tests = ab_test_manager.list_tests()
    return {"tests": [t.dict() for t in tests]}


@router.get("/ab-tests/{test_id}")
def get_ab_test(test_id: str):
    """Get a specific A/B test."""
    test = ab_test_manager.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="A/B test not found")
    return test


@router.delete("/ab-tests/{test_id}", status_code=204)
def delete_ab_test(test_id: str):
    """Delete an A/B test."""
    success = ab_test_manager.delete_test(test_id)
    if not success:
        raise HTTPException(status_code=404, detail="A/B test not found")
    return Response(status_code=204)


@router.post("/ab-tests/{test_id}/start")
def start_ab_test(test_id: str):
    """Start an A/B test."""
    test = ab_test_manager.start_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="A/B test not found")
    return {"test_id": test_id, "status": test.status.value}


@router.post("/ab-tests/{test_id}/pause")
def pause_ab_test(test_id: str):
    """Pause an A/B test."""
    test = ab_test_manager.pause_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="A/B test not found")
    return {"test_id": test_id, "status": test.status.value}


@router.post("/ab-tests/{test_id}/complete")
def complete_ab_test(test_id: str):
    """Complete an A/B test."""
    test = ab_test_manager.complete_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="A/B test not found")
    return {"test_id": test_id, "status": test.status.value}


@router.post("/ab-tests/{test_id}/request")
async def execute_ab_test_request(
    test_id: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
):
    """
    Execute a request through the A/B test.
    Automatically routes to a variant based on traffic weights.
    """
    result = await ab_test_manager.execute_request(
        test_id=test_id,
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not result:
        raise HTTPException(status_code=400, detail="Test not running or not found")
    return result


@router.post("/ab-tests/{test_id}/feedback")
def record_ab_test_feedback(test_id: str, variant_id: str, thumbs_up: bool):
    """Record user feedback for a variant."""
    success = ab_test_manager.record_feedback(test_id, variant_id, thumbs_up)
    if not success:
        raise HTTPException(status_code=404, detail="Test or variant not found")
    return {"status": "recorded"}


@router.get("/ab-tests/{test_id}/results")
def get_ab_test_results(test_id: str, limit: int = 100):
    """Get recent results for an A/B test."""
    test = ab_test_manager.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="A/B test not found")
    results = ab_test_manager.get_results(test_id, limit=limit)
    return {"test_id": test_id, "results": [r.dict() for r in results]}


@router.get("/ab-tests/{test_id}/summary")
def get_ab_test_summary(test_id: str):
    """Get a summary of A/B test results with statistical analysis."""
    summary = ab_test_manager.get_summary(test_id)
    if not summary:
        raise HTTPException(status_code=404, detail="A/B test not found")
    return summary


@router.patch("/ab-tests/{test_id}/weights")
def update_ab_test_weights(test_id: str, weights: dict):
    """Update traffic weights for variants."""
    test = ab_test_manager.update_weights(test_id, weights)
    if not test:
        raise HTTPException(status_code=404, detail="A/B test not found")
    return {"test_id": test_id, "variants": [v.dict() for v in test.variants]}


# ============================================================================
# Adapter Management
# ============================================================================


@router.post("/adapters/register", status_code=201)
def register_adapter(
    name: str,
    base_model: str,
    training_job_id: str,
    adapter_path: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    """
    Register a new adapter in the adapter registry.
    
    This creates a versioned entry for tracking the adapter.
    """
    # Get job for config info
    job = training_manager.get_job(training_job_id)
    lora_config = job.lora_config.dict() if job else {}
    training_config = job.training_config.dict() if job else {}
    metrics = job.metrics if job else {}
    
    adapter = adapter_manager.register_adapter(
        name=name,
        base_model=base_model,
        training_job_id=training_job_id,
        adapter_path=adapter_path,
        lora_config=lora_config,
        training_config=training_config,
        metrics=metrics,
        description=description,
        tags=tags or [],
    )
    return adapter


@router.get("/adapters/registry")
def list_registered_adapters(
    base_model: Optional[str] = None,
    status: Optional[str] = None,
    tags: Optional[str] = None,
):
    """
    List all registered adapters with optional filtering.
    
    Args:
        base_model: Filter by base model name
        status: Filter by status (training, ready, merged, exported, archived)
        tags: Comma-separated list of tags to filter by
    """
    status_enum = None
    if status:
        try:
            status_enum = AdapterStatus(status)
        except ValueError:
            pass
    
    tag_list = tags.split(",") if tags else None
    
    adapters = adapter_manager.list_adapters(
        base_model=base_model,
        status=status_enum,
        tags=tag_list,
    )
    return {"adapters": [a.dict() for a in adapters]}


@router.get("/adapters/registry/{adapter_id}")
def get_registered_adapter(adapter_id: str):
    """Get a specific registered adapter by ID."""
    adapter = adapter_manager.get_adapter(adapter_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return adapter


@router.patch("/adapters/registry/{adapter_id}")
def update_registered_adapter(
    adapter_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    """Update adapter metadata."""
    adapter = adapter_manager.update_adapter(
        adapter_id=adapter_id,
        name=name,
        description=description,
        tags=tags,
    )
    if not adapter:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return adapter


@router.delete("/adapters/registry/{adapter_id}", status_code=204)
def delete_registered_adapter(adapter_id: str):
    """Delete an adapter from the registry."""
    success = adapter_manager.delete_adapter(adapter_id)
    if not success:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return Response(status_code=204)


@router.post("/adapters/registry/{adapter_id}/archive")
def archive_adapter(adapter_id: str):
    """Archive an adapter (soft delete)."""
    adapter = adapter_manager.archive_adapter(adapter_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return {"adapter_id": adapter_id, "status": adapter.status.value}


# ============================================================================
# Adapter Versioning
# ============================================================================


@router.get("/adapters/registry/{adapter_id}/versions")
def list_adapter_versions(adapter_id: str):
    """List all versions of an adapter."""
    versions = adapter_manager.list_versions(adapter_id)
    if not versions:
        adapter = adapter_manager.get_adapter(adapter_id)
        if not adapter:
            raise HTTPException(status_code=404, detail="Adapter not found")
    return {"adapter_id": adapter_id, "versions": [v.dict() for v in versions]}


@router.post("/adapters/registry/{adapter_id}/versions", status_code=201)
def create_adapter_version(
    adapter_id: str,
    version: Optional[str] = None,
    description: Optional[str] = None,
    checkpoint_step: Optional[int] = None,
    tags: Optional[List[str]] = None,
):
    """
    Create a new version of an adapter.
    
    If version is not specified, it will auto-increment the patch version.
    """
    new_version = adapter_manager.create_version(
        adapter_id=adapter_id,
        version=version,
        description=description,
        checkpoint_step=checkpoint_step,
        tags=tags,
    )
    if not new_version:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return new_version


@router.post("/adapters/registry/{adapter_id}/versions/{version}/tag")
def tag_adapter_version(adapter_id: str, version: str, tag: str):
    """Add a tag to a specific adapter version."""
    success = adapter_manager.tag_version(adapter_id, version, tag)
    if not success:
        raise HTTPException(status_code=404, detail="Adapter or version not found")
    return {"adapter_id": adapter_id, "version": version, "tag": tag}


# ============================================================================
# Adapter Export & Download
# ============================================================================


@router.post("/adapters/registry/{adapter_id}/export")
def export_adapter_archive(
    adapter_id: str,
    include_config: bool = True,
    include_metrics: bool = True,
):
    """
    Export an adapter as a ZIP archive.
    
    The archive includes:
    - Adapter weights (safetensors/bin files)
    - adapter_config.json
    - Optional: metadata, metrics, benchmark results
    """
    result = adapter_manager.export_adapter(
        adapter_id=adapter_id,
        include_config=include_config,
        include_metrics=include_metrics,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Adapter not found or export failed")
    
    zip_path, filename = result
    return {
        "adapter_id": adapter_id,
        "filename": filename,
        "path": str(zip_path),
        "size": zip_path.stat().st_size,
    }


@router.get("/adapters/registry/{adapter_id}/exports")
def list_adapter_exports(adapter_id: str):
    """List all exports for an adapter."""
    exports = adapter_manager.list_exports(adapter_id)
    if not exports:
        adapter = adapter_manager.get_adapter(adapter_id)
        if not adapter:
            raise HTTPException(status_code=404, detail="Adapter not found")
    return {"adapter_id": adapter_id, "exports": exports}


@router.get("/adapters/registry/{adapter_id}/download")
def download_adapter(adapter_id: str):
    """
    Get download information for an adapter.
    
    Returns the path to the latest export ZIP file.
    For actual file download, use the returned path with a static file server.
    """
    from fastapi.responses import FileResponse
    
    # First check if there's an existing export
    export_path = adapter_manager.get_export_path(adapter_id)
    
    if not export_path:
        # Create a new export
        result = adapter_manager.export_adapter(adapter_id)
        if not result:
            raise HTTPException(status_code=404, detail="Adapter not found")
        export_path, _ = result
    
    if not export_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    
    return FileResponse(
        path=str(export_path),
        filename=export_path.name,
        media_type="application/zip",
    )


# ============================================================================
# Adapter Comparison
# ============================================================================


@router.post("/adapters/compare")
def compare_adapters(adapter_ids: List[str]):
    """
    Compare multiple adapters.
    
    Returns detailed comparison of:
    - Configuration differences (LoRA config, training config)
    - Metric comparisons
    - Benchmark result comparisons
    - Summary with best performers
    """
    if len(adapter_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 adapter IDs required for comparison"
        )
    
    comparison = adapter_manager.compare_adapters(adapter_ids)
    if not comparison:
        raise HTTPException(status_code=404, detail="One or more adapters not found")
    
    return comparison


@router.get("/adapters/registry/{adapter_id}/benchmarks")
def get_adapter_benchmark_results(adapter_id: str):
    """Get benchmark results for an adapter."""
    adapter = adapter_manager.get_adapter(adapter_id)
    if not adapter:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return {
        "adapter_id": adapter_id,
        "benchmark_results": adapter.benchmark_results,
    }


@router.post("/adapters/registry/{adapter_id}/benchmarks")
def update_adapter_benchmark_results(adapter_id: str, results: dict):
    """Update benchmark results for an adapter."""
    adapter = adapter_manager.update_adapter(
        adapter_id=adapter_id,
        benchmark_results=results,
    )
    if not adapter:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return {
        "adapter_id": adapter_id,
        "benchmark_results": adapter.benchmark_results,
    }

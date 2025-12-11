"""Model management routes."""
from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/models", tags=["models"])

# HuggingFace API - try to import
try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HfApi = None
    HF_AVAILABLE = False


def get_download_manager(request: Request):
    """Get the download manager from app state."""
    return getattr(request.app.state, 'download_manager', None)


def get_manager(request: Request):
    """Get the LlamaCPP manager from app state."""
    return getattr(request.app.state, 'manager', None)


@router.get("")
async def list_models(request: Request):
    """List all available models."""
    download_manager = get_download_manager(request)
    if download_manager is None:
        raise HTTPException(status_code=503, detail="Download manager not available")
    try:
        items = await download_manager.list_local_models()
        return {"success": True, "data": items, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current")
async def get_current_model(request: Request):
    """Get information about the currently configured model."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    try:
        status = manager.get_status()
        model_config = manager.config.get("model", {})
        return {
            "name": model_config.get("name"),
            "variant": model_config.get("variant"),
            "context_size": model_config.get("context_size"),
            "gpu_layers": model_config.get("gpu_layers"),
            "status": "loaded" if status.get("running") else "not_loaded",
            "file_path": f"/home/llamacpp/models/{model_config.get('name')}-{model_config.get('variant')}.gguf",
            "estimated_vram_gb": 19 if "30B" in str(model_config.get("name", "")) else 8
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/local-files")
async def list_local_model_files():
    """List all downloaded model files in the models directory."""
    try:
        models_dir = Path("/home/llamacpp/models")
        if not models_dir.exists():
            return {"success": True, "data": {"files": [], "total_size": 0}, "timestamp": datetime.now().isoformat()}
        
        files = []
        total_size = 0
        
        for item in models_dir.rglob('*'):
            if item.is_file():
                if item.suffix.lower() in ['.gguf', '.safetensors', '.bin', '.pth', '.pt']:
                    stat = item.stat()
                    files.append({
                        "name": item.name,
                        "path": str(item.relative_to(models_dir)),
                        "full_path": str(item),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "extension": item.suffix
                    })
                    total_size += stat.st_size
        
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        logger.info(f"Found {len(files)} local model files, total size: {total_size / (1024**3):.2f} GB")
        return {
            "success": True, 
            "data": {
                "files": files,
                "total_size": total_size,
                "total_count": len(files)
            }, 
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing local model files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/local-files")
async def delete_local_model_file(file_path: str):
    """Delete a model file from the local filesystem."""
    try:
        models_dir = Path("/home/llamacpp/models")
        full_path = (models_dir / file_path).resolve()
        
        if not str(full_path).startswith(str(models_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access denied: path outside models directory")
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not full_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        file_size = full_path.stat().st_size
        file_name = full_path.name
        
        full_path.unlink()
        
        logger.info(f"Deleted model file: {file_name} ({file_size / (1024**3):.2f} GB)")
        return {
            "success": True,
            "data": {
                "deleted_file": file_name,
                "size_freed": file_size
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/repo-files")
async def list_repo_files(repo_id: str, revision: str = "main"):
    """List available model files in a HuggingFace model repository."""
    if not HF_AVAILABLE:
        raise HTTPException(status_code=503, detail="HuggingFace Hub not available")
    
    try:
        if not repo_id or '/' not in repo_id:
            raise HTTPException(status_code=422, detail="Invalid repository ID. Must be in format 'owner/repo'")
            
        api = HfApi(token=os.getenv("HUGGINGFACE_TOKEN"))
        try:
            logger.info(f"Listing repo files for {repo_id}")
            files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")
            model_extensions = ('.gguf', '.safetensors', '.bin', '.pth', '.pt')
            model_files = [f for f in files if f.lower().endswith(model_extensions)]
            logger.info(f"Found {len(model_files)} model files in repo {repo_id}")
            return {"success": True, "data": {"files": model_files}, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error listing repo files for {repo_id}: {str(e)}")
            raise HTTPException(status_code=502, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in list_repo_files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download")
async def download_model(request: Request, payload: Dict[str, Any]):
    """Start downloading a model from HuggingFace."""
    download_manager = get_download_manager(request)
    if download_manager is None:
        raise HTTPException(status_code=503, detail="Download manager not available")
    
    repo_id = payload.get("repositoryId")
    filename = payload.get("filename")
    if not repo_id or not filename:
        raise HTTPException(status_code=400, detail="'repositoryId' and 'filename' are required")
    
    supported_extensions = ('.gguf', '.safetensors', '.bin', '.pth', '.pt')
    if not str(filename).lower().endswith(supported_extensions):
        raise HTTPException(status_code=400, detail=f"Only files with extensions {supported_extensions} are supported")
    
    try:
        rec = await download_manager.start_download(repo_id=repo_id, filename=filename)
        return {"success": True, "data": asdict(rec), "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/downloads")
async def list_downloads(request: Request):
    """List all active and completed downloads."""
    download_manager = get_download_manager(request)
    if download_manager is None:
        raise HTTPException(status_code=503, detail="Download manager not available")
    try:
        downloads = await download_manager.get_downloads()
        return {"success": True, "data": downloads, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/downloads/{model_id}")
async def cancel_download(request: Request, model_id: str):
    """Cancel an active download."""
    download_manager = get_download_manager(request)
    if download_manager is None:
        raise HTTPException(status_code=503, detail="Download manager not available")
    try:
        await download_manager.cancel_download(model_id)
        return {"success": True, "data": {"modelId": model_id, "status": "cancelled"}, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/estimate-vram")
async def estimate_vram(payload: Dict[str, Any]):
    """Estimate VRAM requirements for a model configuration."""
    try:
        params_b = payload.get("parameters_b", 7.0)
        quant_bits = payload.get("quant_bits", 4.0)
        context_size = payload.get("context_size", 4096)
        batch_size = payload.get("batch_size", 1)
        num_layers = payload.get("num_layers", 32)
        head_dim = payload.get("head_dim", 128)
        num_kv_heads = payload.get("num_kv_heads", 8)
        available_vram_gb = payload.get("available_vram_gb", 24.0)
        
        quant_map = {
            "F32": 32.0, "F16": 16.0, "BF16": 16.0,
            "Q8_0": 8.5, "Q6_K": 6.5, "Q5_K_M": 5.5, "Q5_K_S": 5.5,
            "Q4_K_M": 4.5, "Q4_K_S": 4.5, "Q4_0": 4.0,
            "Q3_K_M": 3.5, "Q3_K_S": 3.5, "Q2_K": 2.5,
            "IQ4_XS": 4.25, "IQ3_XXS": 3.0, "IQ2_XXS": 2.0,
            "MXFP4": 4.5
        }
        
        if isinstance(quant_bits, str):
            quant_bits = quant_map.get(quant_bits.upper(), 4.5)
        
        overhead_factor = 1.1
        model_vram_gb = params_b * (quant_bits / 8) * overhead_factor
        
        kv_cache_gb = (2 * num_layers * context_size * batch_size * num_kv_heads * head_dim * 2) / 1e9
        compute_buffer_gb = 0.5 + (params_b / 20)
        cuda_overhead_gb = 0.5
        
        total_vram_gb = model_vram_gb + kv_cache_gb + compute_buffer_gb + cuda_overhead_gb
        fits_in_vram = total_vram_gb <= available_vram_gb
        
        if fits_in_vram:
            headroom = available_vram_gb - total_vram_gb
            if headroom > 4:
                recommendation = f"Model fits comfortably with {headroom:.1f}GB headroom. You could increase context size or use a higher quality quantization."
            elif headroom > 1:
                recommendation = f"Model fits with {headroom:.1f}GB headroom. Should work well for typical workloads."
            else:
                recommendation = f"Model fits with minimal headroom ({headroom:.1f}GB). Consider reducing context size for stability."
        else:
            overage = total_vram_gb - available_vram_gb
            suggestion_bits = quant_bits * (available_vram_gb / total_vram_gb) * 0.9
            suggestions = [q for q, b in quant_map.items() if b <= suggestion_bits]
            suggestion = suggestions[0] if suggestions else "Q2_K"
            recommendation = f"Model exceeds available VRAM by {overage:.1f}GB. Consider using {suggestion} quantization or reducing context size to {int(context_size * 0.5)}."
        
        return {
            "success": True,
            "data": {
                "model_vram_gb": round(model_vram_gb, 2),
                "kv_cache_gb": round(kv_cache_gb, 2),
                "compute_buffer_gb": round(compute_buffer_gb, 2),
                "overhead_gb": round(cuda_overhead_gb, 2),
                "total_vram_gb": round(total_vram_gb, 2),
                "available_vram_gb": available_vram_gb,
                "fits_in_vram": fits_in_vram,
                "utilization_percent": round((total_vram_gb / available_vram_gb) * 100, 1),
                "breakdown": {
                    "weights": round(model_vram_gb, 2),
                    "kv_cache": round(kv_cache_gb, 2),
                    "compute_buffer": round(compute_buffer_gb, 2),
                    "overhead": round(cuda_overhead_gb, 2)
                },
                "recommendation": recommendation,
                "input_parameters": {
                    "parameters_b": params_b,
                    "quant_bits": quant_bits,
                    "context_size": context_size,
                    "batch_size": batch_size,
                    "num_layers": num_layers,
                    "num_kv_heads": num_kv_heads
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error estimating VRAM: {e}")
        raise HTTPException(status_code=500, detail=str(e))

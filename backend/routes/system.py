from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import psutil
import subprocess
from datetime import datetime
from typing import Optional

from enhanced_logger import enhanced_logger as logger
from app_state import manager

try:
    from modules import vram_estimator
except ImportError:
    vram_estimator = None

router = APIRouter(prefix="/api/v1", tags=["system"])

@router.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    status = manager.get_status()
    return {
        "status": "up",
        "timestamp": datetime.now().isoformat(),
        "service": {
            "running": status["running"],
            "backend": "docker" if manager.use_docker else "subprocess",
        }
    }

@router.get("/logs")
async def get_logs(lines: int = 100):
    """Get recent logs from the llamacpp service"""
    return {"logs": manager.get_logs(lines)}

@router.delete("/logs")
async def clear_logs():
    """Clear internal log buffer"""
    manager.log_buffer.clear()
    last_log = f"[{datetime.now().strftime('%H:%M:%S')}] Logs cleared by user\n"
    manager.log_buffer.append(last_log)
    await manager.broadcast_ws_event("log", {"text": last_log})
    return {"status": "cleared"}

@router.get("/container/logs")
async def get_container_logs(lines: int = 100):
    """Fetch logs directly from Docker container"""
    if not manager.use_docker:
        raise HTTPException(status_code=400, detail="Service is not running in Docker mode")
    
    logs = manager.read_docker_logs_cli()
    if logs:
        # Keep only the requested number of lines
        log_lines = logs.split('\n')
        recent_logs = '\n'.join(log_lines[-lines:])
        return {"logs": recent_logs}
    
    return {"logs": "No logs available or container not running"}

@router.get("/container/logs/stream")
async def stream_container_logs():
    """Stream logs directly from Docker container"""
    if not manager.use_docker:
        raise HTTPException(status_code=400, detail="Service is not running in Docker mode")
        
    async def log_generator():
        # First check if the container is running
        check = subprocess.run(
            ["docker", "ps", "-q", "-f", "name=llama-nexus-llamacpp-api"],
            capture_output=True, text=True
        )
        
        if not check.stdout.strip():
            yield "data: Container is not running\n\n"
            return
            
        # Start docker logs process
        process = subprocess.Popen(
            ["docker", "logs", "-f", "--tail", "100", "llama-nexus-llamacpp-api"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    import json
                    # Format as SSE
                    yield f"data: {json.dumps({'text': line})}\n\n"
                else:
                    break
        finally:
            process.terminate()
            
    return StreamingResponse(log_generator(), media_type="text/event-stream")

@router.get("/resources")
async def get_resources():
    """Get system resource usage"""
    try:
        resources = {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total_mb": psutil.virtual_memory().total / (1024 * 1024),
                "used_mb": psutil.virtual_memory().used / (1024 * 1024),
                "available_mb": psutil.virtual_memory().available / (1024 * 1024),
                "percent": psutil.virtual_memory().percent
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Get GPU info using nvidia-smi
        try:
            which_result = subprocess.run(
                ["which", "nvidia-smi"],
                capture_output=True, text=True, timeout=2
            )
            
            if which_result.returncode == 0:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    first_gpu_line = result.stdout.strip().split('\n')[0]
                    values = [v.strip() for v in first_gpu_line.split(',')]
                    if len(values) >= 5:
                        resources["gpu"] = {
                            "vram_used_mb": int(float(values[0])),
                            "vram_total_mb": int(float(values[1])),
                            "usage_percent": float(values[2]) if values[2] != "[N/A]" else 0,
                            "temperature_c": float(values[3]) if values[3] != "[N/A]" else 0,
                            "power_watts": float(values[4]) if values[4] not in ["N/A", "[N/A]"] else None
                        }
                else:
                    resources["gpu"] = {"status": "error", "reason": f"nvidia-smi failed: {result.returncode}"}
            else:
                resources["gpu"] = {"status": "unavailable", "reason": "nvidia-smi not found"}
        except Exception as e:
            resources["gpu"] = {"status": "error", "reason": str(e)}
        
        return resources
        
    except Exception as e:
        logger.error(f"Error getting resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/gpus")
async def list_gpus():
    """List all available GPUs with detailed information"""
    try:
        which_result = subprocess.run(
            ["which", "nvidia-smi"],
            capture_output=True, text=True, timeout=2
        )
        
        if which_result.returncode != 0:
            return {"available": False, "gpus": [], "reason": "nvidia-smi not found"}
        
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            return {"available": False, "gpus": [], "reason": f"nvidia-smi failed code {result.returncode}"}
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip(): continue
            values = [v.strip() for v in line.split(',')]
            if len(values) >= 7:
                gpus.append({
                    "index": int(values[0]),
                    "name": values[1],
                    "vram_total_mb": int(values[2]),
                    "vram_used_mb": int(values[3]),
                    "vram_free_mb": int(values[4]),
                    "utilization_percent": float(values[5]) if values[5] != "N/A" else 0,
                    "temperature_c": float(values[6]) if values[6] != "N/A" else 0,
                    "power_draw_watts": float(values[7]) if len(values) > 7 and values[7] != "N/A" else None,
                    "power_limit_watts": float(values[8]) if len(values) > 8 and values[8] != "N/A" else None,
                })
        
        return {"available": True, "gpus": gpus, "count": len(gpus)}
    except Exception as e:
        return {"available": False, "gpus": [], "reason": str(e)}

@router.get("/models/current")
async def get_current_model():
    """Get information about the currently configured model"""
    status = manager.get_status()
    return {
        "name": manager.config["model"]["name"],
        "variant": manager.config["model"]["variant"],
        "context_size": manager.config["model"]["context_size"],
        "gpu_layers": manager.config["model"]["gpu_layers"],
        "status": "loaded" if status["running"] else "not_loaded",
        "file_path": f"/home/llamacpp/models/{manager.config['model']['name']}-{manager.config['model']['variant']}.gguf",
        "estimated_vram_gb": 19 if "30B" in manager.config["model"]["name"] else 8
    }

@router.get("/config/presets")
async def get_config_presets():
    """Get available configuration presets"""
    return [
        {
            "id": "balanced",
            "name": "Balanced",
            "description": "Good balance between quality and speed",
            "config": {"sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05}}
        },
        {
            "id": "creative",
            "name": "Creative",
            "description": "More creative and varied outputs",
            "config": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.0}}
        },
        {
            "id": "precise",
            "name": "Precise",
            "description": "More deterministic and focused outputs",
            "config": {"sampling": {"temperature": 0.3, "top_p": 0.5, "top_k": 10, "repeat_penalty": 1.1}}
        },
        {
            "id": "fast",
            "name": "Fast Inference",
            "description": "Optimized for speed",
            "config": {"performance": {"batch_size": 512, "ubatch_size": 128, "num_predict": 2048}}
        }
    ]

@router.post("/config/presets/{preset_id}/apply")
async def apply_config_preset(preset_id: str):
    """Apply a configuration preset"""
    presets = await get_config_presets()
    preset = next((p for p in presets if p["id"] == preset_id), None)
    
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset {preset_id} not found")
    
    updated_config = manager.update_config(preset["config"])
    return {"status": "success", "preset": preset_id}

@router.post("/vram/estimate")
async def estimate_vram_api(request: Request):
    """Estimate VRAM requirements for model deployment."""
    if vram_estimator is None:
        raise HTTPException(status_code=503, detail="VRAM estimator not available")
    
    data = await request.json()
    estimate = vram_estimator.estimate_vram(
        model_name=data.get("model_name", ""),
        params_b=data.get("params_b"),
        quantization=data.get("quantization", "Q4_K_M"),
        context_size=data.get("context_size", 4096),
        batch_size=data.get("batch_size", 1),
        gpu_layers=data.get("gpu_layers", -1),
        kv_cache_type=data.get("kv_cache_type", "f16"),
        available_vram_gb=data.get("available_vram_gb", 24.0),
        flash_attention=data.get("flash_attention", True),
    )
    
    return {
        "model_weights_mb": estimate.model_weights_mb,
        "kv_cache_mb": estimate.kv_cache_mb,
        "compute_buffer_mb": estimate.compute_buffer_mb,
        "overhead_mb": estimate.overhead_mb,
        "total_mb": estimate.total_mb,
        "total_gb": estimate.total_gb,
        "fits_in_vram": estimate.fits_in_vram,
        "available_vram_gb": estimate.available_vram_gb,
        "utilization_percent": estimate.utilization_percent,
        "warnings": estimate.warnings,
    }

@router.get("/vram/quantizations")
async def get_quantization_options():
    if vram_estimator is None:
        raise HTTPException(status_code=503, detail="VRAM estimator not available")
    return {"quantizations": vram_estimator.get_quantization_options()}

@router.get("/vram/architectures")
async def get_model_architectures():
    if vram_estimator is None:
        raise HTTPException(status_code=503, detail="VRAM estimator not available")
    return {"architectures": vram_estimator.get_model_architectures()}

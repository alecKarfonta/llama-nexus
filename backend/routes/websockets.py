from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import subprocess
from datetime import datetime
import psutil

from enhanced_logger import enhanced_logger as logger
from app_state import manager

router = APIRouter()

# ============================================================================
# WebSocket Log Streaming
# ============================================================================
@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """
    WebSocket endpoint for real-time log streaming.
    Clients connect to receive new stderr output from llamacpp.
    """
    await websocket.accept()
    
    # Send all recent logs immediately upon connection
    recent_logs = manager.get_logs(100)
    for log_line in recent_logs:
        try:
            await websocket.send_text(log_line)
        except Exception:
            pass # Client might have disconnected
            
    try:
        # Keep connection open and send new logs as they arrive
        manager.ws_clients.append(websocket)
        while True:
            # We just need to keep the connection alive
            # The actual sending happens in add_log_line
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in manager.ws_clients:
            manager.ws_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error in log streaming: {e}")
        if websocket in manager.ws_clients:
            manager.ws_clients.remove(websocket)


# ============================================================================
# General WebSocket endpoint for real-time updates
# ============================================================================
_ws_clients: List[WebSocket] = []

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    General WebSocket endpoint for real-time updates.
    Broadcasts metrics, status changes, download progress, and model events.
    """
    await websocket.accept()
    _ws_clients.append(websocket)
    
    try:
        # Send connection confirmation with current status
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "service_running": manager.running,
                "model_name": manager.current_model_name if hasattr(manager, 'current_model_name') else None
            }
        })
        
        # Keep connection alive and periodically send metrics
        while True:
            await asyncio.sleep(5)  # Send updates every 5 seconds
            
            try:
                # Gather current metrics
                metrics_data = {
                    "cpu": {
                        "percent": psutil.cpu_percent(),
                    },
                    "memory": {
                        "total_mb": psutil.virtual_memory().total / (1024 * 1024),
                        "used_mb": psutil.virtual_memory().used / (1024 * 1024),
                        "percent": psutil.virtual_memory().percent
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                # Try to get GPU info
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        parts = result.stdout.strip().split(",")
                        if len(parts) >= 3:
                            metrics_data["gpu"] = {
                                "vram_used_mb": float(parts[0].strip()),
                                "vram_total_mb": float(parts[1].strip()),
                                "usage_percent": float(parts[2].strip())
                            }
                except:
                    pass  # GPU info not available
                
                await websocket.send_json({
                    "type": "metrics",
                    "timestamp": datetime.now().isoformat(),
                    "payload": metrics_data
                })
                
                # Also send status update
                await websocket.send_json({
                    "type": "status",
                    "timestamp": datetime.now().isoformat(),
                    "payload": {
                        "running": manager.running,
                        "model_name": manager.current_model_name if hasattr(manager, 'current_model_name') else None,
                        "uptime": manager.uptime if hasattr(manager, 'uptime') else 0
                    }
                })
                
            except Exception as e:
                logger.warning(f"Error sending metrics via WebSocket: {e}")
                
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)

async def broadcast_ws_event(event_type: str, data: dict):
    """Broadcast an event to all connected WebSocket clients"""
    message = {
        "type": event_type,
        "timestamp": datetime.now().isoformat(),
        "payload": data
    }
    
    disconnected = []
    for client in _ws_clients:
        try:
            await client.send_json(message)
        except:
            disconnected.append(client)
    
    for client in disconnected:
        _ws_clients.remove(client)

# Wire up the manager to use this broadcast function
manager.broadcast_ws_event = broadcast_ws_event

# ============================================================================
# Training WebSocket Endpoint for Real-time Metrics
# ============================================================================
_training_ws_clients: List[WebSocket] = []

@router.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """
    WebSocket endpoint for real-time training metrics.
    Streams training progress, loss curves, and GPU utilization.
    """
    await websocket.accept()
    _training_ws_clients.append(websocket)
    
    try:
        # Import here to avoid circular imports
        try:
            from modules.finetuning import register_ws_broadcaster, unregister_ws_broadcaster
        except ImportError:
            from finetuning import register_ws_broadcaster, unregister_ws_broadcaster
        
        # Capture the current running event loop for thread-safe scheduling
        main_loop = asyncio.get_running_loop()
        
        # Create async callback for broadcasting
        async def send_training_event(event_type: str, data: dict):
            try:
                await websocket.send_json({
                    "type": event_type,
                    "timestamp": datetime.now().isoformat(),
                    "payload": data
                })
            except Exception as e:
                logger.warning(f"Failed to send training event: {e}")
        
        # Thread-safe wrapper for the async callback - called from Redis consumer thread
        def sync_broadcast(event_type: str, data: dict):
            try:
                # Use run_coroutine_threadsafe to schedule from background thread
                future = asyncio.run_coroutine_threadsafe(
                    send_training_event(event_type, data),
                    main_loop
                )
                # Don't wait for result - fire and forget
            except Exception as e:
                logger.warning(f"Training broadcast error: {e}")
        
        # Register the callback
        register_ws_broadcaster(sync_broadcast)
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.now().isoformat(),
            "data": {"endpoint": "training"}
        })
        
        # Keep connection alive and send GPU metrics periodically
        while True:
            await asyncio.sleep(2)  # Send GPU metrics every 2 seconds
            
            try:
                gpu_metrics = {}
                # Get GPU metrics
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                         "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        parts = [p.strip() for p in result.stdout.strip().split(",")]
                        if len(parts) >= 5:
                            gpu_metrics = {
                                "vram_used_gb": float(parts[0]) / 1024,
                                "vram_total_gb": float(parts[1]) / 1024,
                                "gpu_utilization": float(parts[2]),
                                "temperature_c": float(parts[3]),
                                "power_w": float(parts[4]) if parts[4] != "[N/A]" else None,
                            }
                except Exception:
                    pass  # GPU metrics not available
                
                await websocket.send_json({
                    "type": "gpu_metrics",
                    "timestamp": datetime.now().isoformat(),
                    "payload": gpu_metrics
                })
                
            except Exception as e:
                logger.warning(f"Error sending GPU metrics: {e}")
                
    except WebSocketDisconnect:
        if websocket in _training_ws_clients:
            _training_ws_clients.remove(websocket)
    except Exception as e:
        logger.error(f"Training WebSocket error: {e}")
        if websocket in _training_ws_clients:
            _training_ws_clients.remove(websocket)

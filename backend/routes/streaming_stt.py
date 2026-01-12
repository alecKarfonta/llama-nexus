"""Streaming STT (Speech-to-Text) service management routes.

This module provides endpoints for managing the NVIDIA Nemotron streaming STT
service, including WebSocket relay for real-time audio transcription.
"""
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional
import asyncio
import json
import logging

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streaming-stt"])


def get_streaming_stt_manager(request: Request):
    """Get the Streaming STT manager from app state."""
    return getattr(request.app.state, 'streaming_stt_manager', None)


# =============================================================================
# Streaming STT Service Endpoints
# =============================================================================

@router.post("/api/v1/streaming-stt/start")
async def start_streaming_stt_service(request: Request, config: Optional[Dict[str, Any]] = None):
    """Start the Streaming STT service."""
    manager = get_streaming_stt_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Streaming STT manager not available")
    try:
        if config:
            manager.update_config(config)
        
        logger.info("Streaming STT service start request")
        success = await manager.start()
        status = manager.get_status()
        return {
            "success": success,
            "status": status,
            "message": "Streaming STT service started successfully" if success else "Streaming STT service failed to start"
        }
    except Exception as e:
        logger.error(f"Streaming STT service start error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start Streaming STT service: {str(e)}")


@router.post("/api/v1/streaming-stt/stop")
async def stop_streaming_stt_service(request: Request):
    """Stop the Streaming STT service."""
    manager = get_streaming_stt_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Streaming STT manager not available")
    try:
        success = await manager.stop()
        return {"success": success, "status": manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop Streaming STT service: {str(e)}")


@router.post("/api/v1/streaming-stt/restart")
async def restart_streaming_stt_service(request: Request, config: Optional[Dict[str, Any]] = None):
    """Restart the Streaming STT service."""
    manager = get_streaming_stt_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Streaming STT manager not available")
    try:
        if config:
            manager.update_config(config)
        
        success = await manager.restart()
        return {"success": success, "status": manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart Streaming STT service: {str(e)}")


@router.get("/api/v1/streaming-stt/status")
async def get_streaming_stt_status(request: Request):
    """Get current Streaming STT service status."""
    manager = get_streaming_stt_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Streaming STT manager not available")
    return manager.get_status()


@router.get("/api/v1/streaming-stt/config")
async def get_streaming_stt_config(request: Request):
    """Get current Streaming STT configuration."""
    manager = get_streaming_stt_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Streaming STT manager not available")
    return {
        "config": manager.config,
        "websocket_url": "/api/v1/streaming-stt/ws",
        "protocol": {
            "client_to_server": [
                {"type": "audio_chunk", "data": "<base64 PCM float32>"},
                {"type": "config", "username": "optional"},
                {"type": "end"},
                {"type": "ping"},
            ],
            "server_to_client": [
                {"type": "session_started", "session_id": "uuid"},
                {"type": "partial", "text": "Hello wo..."},
                {"type": "sentence", "text": "Hello world.", "confidence": 0.95},
                {"type": "final", "text": "Full transcription.", "sentences": []},
                {"type": "vad", "is_speech": True},
                {"type": "pong"},
                {"type": "error", "message": "..."},
            ]
        }
    }


@router.put("/api/v1/streaming-stt/config")
async def update_streaming_stt_config(request: Request, config: Dict[str, Any]):
    """Update Streaming STT configuration (requires restart to apply)."""
    manager = get_streaming_stt_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Streaming STT manager not available")
    try:
        updated_config = manager.update_config(config)
        return {
            "success": True,
            "config": updated_config,
            "message": "Configuration updated. Restart service to apply changes."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


# =============================================================================
# WebSocket Relay Endpoint
# =============================================================================

@router.websocket("/api/v1/streaming-stt/ws")
async def streaming_stt_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming speech-to-text.
    
    This acts as a relay between the frontend and the streaming-stt Docker container,
    forwarding audio chunks and returning transcription results.
    
    Protocol:
        Client -> Server:
            {"type": "audio_chunk", "data": "<base64 PCM float32>"}
            {"type": "config", "username": "Streamer"}
            {"type": "end"}
        
        Server -> Client:
            {"type": "session_started", "session_id": "..."}
            {"type": "partial", "text": "Hello wo..."}
            {"type": "sentence", "text": "Hello world.", "confidence": 0.95}
            {"type": "final", "text": "Full transcription.", "sentences": [...]}
            {"type": "vad", "is_speech": true}
            {"type": "error", "message": "..."}
    """
    if not WEBSOCKETS_AVAILABLE:
        await websocket.close(code=4001, reason="WebSocket relay not available (websockets package not installed)")
        return
    
    manager = get_streaming_stt_manager(websocket.app)
    if manager is None or not manager.is_running():
        await websocket.close(code=4002, reason="Streaming STT service is not running")
        return
    
    await websocket.accept()
    
    import uuid
    session_id = str(uuid.uuid4())
    stt_connection = None
    
    logger.info(f"📞 Client connected for streaming STT relay: {session_id}")
    
    try:
        # Connect to the streaming-stt container
        stt_url = manager.get_internal_ws_url()
        logger.info(f"Connecting to STT service at {stt_url}")
        
        stt_connection = await websockets.connect(
            stt_url,
            ping_interval=None,  # Disable ping to match server config
        )
        
        # Wait for session_started confirmation from STT service
        response = await asyncio.wait_for(stt_connection.recv(), timeout=10.0)
        msg = json.loads(response)
        
        if msg.get("type") != "session_started":
            raise Exception(f"Unexpected response from STT service: {msg}")
        
        # Forward session info to client
        await websocket.send_json({
            "type": "session_started",
            "session_id": session_id,
            "stt_session_id": msg.get("session_id"),
        })
        
        # Start background task to receive from STT service
        receive_task = asyncio.create_task(
            _relay_stt_messages(stt_connection, websocket)
        )
        
        # Main loop: receive from client and forward to STT service
        while True:
            try:
                message = await websocket.receive_json()
                msg_type = message.get("type")
                
                if msg_type == "audio_chunk":
                    # Forward audio to STT service
                    await stt_connection.send(json.dumps(message))
                
                elif msg_type == "config":
                    # Forward config updates
                    await stt_connection.send(json.dumps(message))
                
                elif msg_type == "end":
                    # Client ending stream
                    await stt_connection.send(json.dumps({"type": "end_stream"}))
                    break
                
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                
            except WebSocketDisconnect:
                logger.info(f"Client disconnected: {session_id}")
                break
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from client: {e}")
        
        # Clean up
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
    
    except Exception as e:
        logger.error(f"Streaming STT relay error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except:
            pass
    
    finally:
        if stt_connection:
            try:
                await stt_connection.close()
            except:
                pass
        logger.info(f"🧹 Cleaned up streaming session: {session_id}")


async def _relay_stt_messages(stt_connection, client_ws: WebSocket):
    """Background task to relay messages from STT service to client."""
    try:
        while True:
            try:
                raw_message = await asyncio.wait_for(stt_connection.recv(), timeout=30.0)
                message = json.loads(raw_message)
                
                # Forward all message types to client
                await client_ws.send_json(message)
                
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await stt_connection.send(json.dumps({"type": "ping"}))
                except:
                    break
            
            except websockets.exceptions.ConnectionClosed:
                logger.info("STT service connection closed")
                break
    
    except Exception as e:
        logger.error(f"Error relaying STT messages: {e}")

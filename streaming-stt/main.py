"""
Streaming STT Service - NVIDIA Nemotron Speech Streaming

This service provides real-time speech-to-text transcription using
NVIDIA's Nemotron-Speech-Streaming-En-0.6b model with cache-aware
streaming for low-latency voice agent applications.

Protocol:
    Client connects via WebSocket and sends 80ms audio chunks.
    Server returns partial transcriptions, complete sentences, and final results.
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from streaming_engine import NemotronStreamingEngine, StreamingSession

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-speech-streaming-en-0.6b")
CHUNK_SIZE_MS = int(os.getenv("CHUNK_SIZE_MS", "80"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
SENTENCE_END_SILENCE_MS = int(os.getenv("SENTENCE_END_SILENCE_MS", "500"))
MIN_SENTENCE_WORDS = int(os.getenv("MIN_SENTENCE_WORDS", "2"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.01"))
VAD_SILENCE_THRESHOLD = float(os.getenv("VAD_SILENCE_THRESHOLD", "0.005"))
VAD_HYSTERESIS_MS = int(os.getenv("VAD_HYSTERESIS_MS", "100"))
MODEL_BUFFER_MS = int(os.getenv("MODEL_BUFFER_MS", "896"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Set log level
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

# Global engine instance
engine: Optional[NemotronStreamingEngine] = None

# Active sessions
active_sessions: Dict[str, StreamingSession] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load model on startup."""
    global engine
    
    logger.info(f"🚀 Loading Nemotron model: {MODEL_NAME}")
    logger.info(f"   Chunk size: {CHUNK_SIZE_MS}ms")
    logger.info(f"   Sample rate: {SAMPLE_RATE}Hz")
    logger.info(f"   VAD threshold: {VAD_THRESHOLD} (silence: {VAD_SILENCE_THRESHOLD})")
    logger.info(f"   Model buffer: {MODEL_BUFFER_MS}ms")
    
    try:
        engine = NemotronStreamingEngine(
            model_name=MODEL_NAME,
            chunk_size_ms=CHUNK_SIZE_MS,
            sample_rate=SAMPLE_RATE,
            sentence_end_silence_ms=SENTENCE_END_SILENCE_MS,
            min_sentence_words=MIN_SENTENCE_WORDS,
            vad_threshold=VAD_THRESHOLD,
            vad_silence_threshold=VAD_SILENCE_THRESHOLD,
            vad_hysteresis_ms=VAD_HYSTERESIS_MS,
            model_buffer_ms=MODEL_BUFFER_MS,
        )
        await engine.load_model()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down streaming STT service")
    for session_id in list(active_sessions.keys()):
        try:
            await engine.end_session(session_id)
        except Exception as e:
            logger.warning(f"Error cleaning up session {session_id}: {e}")
    active_sessions.clear()


app = FastAPI(
    title="Streaming STT Service",
    description="Real-time speech-to-text using NVIDIA Nemotron",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_loaded": engine is not None and engine.is_loaded,
        "active_sessions": len(active_sessions),
        "config": {
            "chunk_size_ms": CHUNK_SIZE_MS,
            "sample_rate": SAMPLE_RATE,
            "sentence_end_silence_ms": SENTENCE_END_SILENCE_MS,
            "vad_threshold": VAD_THRESHOLD,
        }
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Streaming STT",
        "model": MODEL_NAME,
        "websocket_endpoint": "/ws/stt",
        "docs": "/docs",
    }


@app.websocket("/ws/stt")
async def streaming_stt_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming speech-to-text.
    
    Protocol:
        Client -> Server:
            {"type": "audio_chunk", "data": "<base64 PCM float32>"}
            {"type": "end_stream"}
            {"type": "config", "chunk_size_ms": 80}
        
        Server -> Client:
            {"type": "partial", "text": "Hello wo...", "is_final": false}
            {"type": "sentence", "text": "Hello world.", "confidence": 0.95}
            {"type": "final", "text": "Full transcription.", "sentences": [...]}
            {"type": "vad", "is_speech": true}
            {"type": "error", "message": "..."}
    """
    await websocket.accept()
    
    # Generate unique session ID
    import uuid
    session_id = str(uuid.uuid4())
    
    logger.info(f"📞 New streaming session: {session_id}")
    
    try:
        # Create streaming session
        session = await engine.create_session(session_id)
        active_sessions[session_id] = session
        
        # Send session confirmation
        await websocket.send_json({
            "type": "session_started",
            "session_id": session_id,
            "config": {
                "chunk_size_ms": CHUNK_SIZE_MS,
                "sample_rate": SAMPLE_RATE,
            }
        })
        
        while True:
            try:
                # Receive message from client
                message = await websocket.receive_json()
                msg_type = message.get("type")
                
                if msg_type == "audio_chunk":
                    # Process audio chunk
                    import base64
                    import numpy as np
                    
                    audio_b64 = message.get("data", "")
                    if not audio_b64:
                        continue
                    
                    # Decode base64 to float32 array
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    # Process through streaming engine
                    result = await engine.process_chunk(session_id, audio_data)
                    
                    if not result:
                        logger.warning(f"⚠️ process_chunk returned None for session {session_id}")
                    
                    # Send VAD update if speech state changed
                    if result.vad_changed:
                        await websocket.send_json({
                            "type": "vad",
                            "is_speech": result.is_speech,
                        })
                    
                    # Send partial transcription if available
                    if result.partial_text:
                        await websocket.send_json({
                            "type": "partial",
                            "text": result.partial_text,
                            "is_final": False,
                        })
                    
                    # Send complete sentences
                    for sentence in result.sentences:
                        await websocket.send_json({
                            "type": "sentence",
                            "text": sentence.text,
                            "confidence": sentence.confidence,
                            "start_time": sentence.start_time,
                            "end_time": sentence.end_time,
                        })
                
                elif msg_type == "end_stream":
                    # Client requested end of stream
                    final_result = await engine.end_session(session_id)
                    
                    await websocket.send_json({
                        "type": "final",
                        "text": final_result.full_text,
                        "sentences": [
                            {"text": s.text, "confidence": s.confidence}
                            for s in final_result.sentences
                        ],
                    })
                    break
                
                elif msg_type == "config":
                    # Dynamic configuration update
                    new_chunk_size = message.get("chunk_size_ms")
                    if new_chunk_size:
                        session.update_config(chunk_size_ms=new_chunk_size)
                        await websocket.send_json({
                            "type": "config_updated",
                            "chunk_size_ms": new_chunk_size,
                        })
                
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                
            except WebSocketDisconnect:
                logger.info(f"📴 Client disconnected: {session_id}")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
    
    except Exception as e:
        logger.error(f"Session error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Session error: {str(e)}",
            })
        except:
            pass
    
    finally:
        # Cleanup session
        if session_id in active_sessions:
            del active_sessions[session_id]
        try:
            await engine.cleanup_session(session_id)
        except:
            pass
        logger.info(f"🧹 Cleaned up session: {session_id}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8009,
        ws_ping_interval=None,  # Disable ping for long-running streams
        log_level="info",
    )

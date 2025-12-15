"""
Redis event consumer for training status/log updates.

This runs in a background thread to listen for worker events and update the
TrainingManager / JobStore accordingly. Also broadcasts events to WebSocket clients.
"""

import asyncio
import json
import threading
from datetime import datetime
from typing import Optional, Callable, Any, List

import redis

from enhanced_logger import enhanced_logger as logger

from .training_manager import TrainingManager


# Global list of WebSocket broadcast callbacks
_ws_broadcast_callbacks: List[Callable[[str, dict], Any]] = []


def register_ws_broadcaster(callback: Callable[[str, dict], Any]):
    """Register a callback to broadcast events to WebSocket clients."""
    _ws_broadcast_callbacks.append(callback)


def unregister_ws_broadcaster(callback: Callable[[str, dict], Any]):
    """Unregister a WebSocket broadcast callback."""
    if callback in _ws_broadcast_callbacks:
        _ws_broadcast_callbacks.remove(callback)


def broadcast_training_event(event_type: str, data: dict):
    """Broadcast a training event to all registered WebSocket callbacks."""
    for callback in _ws_broadcast_callbacks:
        try:
            callback(event_type, data)
        except Exception as e:
            logger.warning(f"Failed to broadcast training event: {e}")


class TrainingEventConsumer:
    def __init__(self, manager: TrainingManager, redis_url: str = "redis://redis:6379/0"):
        self.manager = manager
        self.redis_url = redis_url
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("TrainingEventConsumer started", extra={"redis_url": self.redis_url})

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        client = redis.from_url(self.redis_url, decode_responses=True)
        pubsub = client.pubsub()
        pubsub.psubscribe("training:status:*", "training:logs:*")
        for message in pubsub.listen():
            if self._stop.is_set():
                break
            if message["type"] != "pmessage":
                continue
            channel = message.get("channel", "")
            payload_raw = message.get("data")
            try:
                payload = json.loads(payload_raw)
            except Exception:
                logger.warning("Failed to parse training event payload", extra={"channel": channel})
                continue
            job_id = channel.split(":")[-1]
            if "status" in channel:
                self.manager.handle_status_event(job_id, payload)
                # Broadcast status update to WebSocket clients
                broadcast_training_event("training_status", {
                    "job_id": job_id,
                    "step": payload.get("step", 0),
                    "total_steps": payload.get("total_steps", 0),
                    "loss": payload.get("loss"),
                    "status": payload.get("status"),
                    "adapter_path": payload.get("adapter_path"),
                    "metrics": payload.get("metrics", {}),
                    "progress": (payload.get("step", 0) / max(payload.get("total_steps", 1), 1)) * 100,
                    "timestamp": datetime.now().isoformat(),
                })
            elif "logs" in channel:
                msg = payload.get("message", "")
                if msg:
                    self.manager.handle_log_event(job_id, msg)
                    # Broadcast log update to WebSocket clients
                    broadcast_training_event("training_log", {
                        "job_id": job_id,
                        "message": msg,
                        "timestamp": datetime.now().isoformat(),
                    })
        pubsub.close()

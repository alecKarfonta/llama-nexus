"""
Event Bus Module
Provides event-driven architecture using Redis Pub/Sub for real-time updates
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional
import logging

# Configure logging
logger = logging.getLogger("event_bus")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Try to import redis
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis package not installed, event bus will use in-memory fallback")


class EventBus:
    """
    Event bus for publishing and subscribing to real-time events.
    Uses Redis Pub/Sub when available, falls back to in-memory for development.
    """
    
    # Event channels
    CHANNEL_STATUS = "llama-nexus:status"
    CHANNEL_METRICS = "llama-nexus:metrics"
    CHANNEL_DOWNLOAD = "llama-nexus:download"
    CHANNEL_LOGS = "llama-nexus:logs"
    CHANNEL_MODEL = "llama-nexus:model"
    CHANNEL_CONVERSATION = "llama-nexus:conversation"
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the event bus"""
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub: Optional[aioredis.client.PubSub] = None
        self.subscribers: Dict[str, List[Callable]] = {}
        self.is_connected = False
        self._listener_task: Optional[asyncio.Task] = None
        
    async def connect(self) -> bool:
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.info("Redis not available, using in-memory event handling")
            return False
            
        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self.redis.ping()
            self.is_connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis = None
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
            
        if self.pubsub:
            await self.pubsub.close()
            self.pubsub = None
            
        if self.redis:
            await self.redis.close()
            self.redis = None
            
        self.is_connected = False
        logger.info("Disconnected from Redis")
    
    async def publish(self, channel: str, event_type: str, data: Dict[str, Any]) -> bool:
        """
        Publish an event to a channel
        
        Args:
            channel: The channel to publish to
            event_type: Type of event (e.g., 'model_loaded', 'download_progress')
            data: Event data payload
            
        Returns:
            bool: True if published successfully
        """
        message = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Always notify local subscribers
        await self._notify_local_subscribers(channel, message)
        
        # Publish to Redis if connected
        if self.is_connected and self.redis:
            try:
                await self.redis.publish(channel, json.dumps(message))
                logger.debug(f"Published {event_type} to {channel}")
                return True
            except Exception as e:
                logger.error(f"Failed to publish to Redis: {e}")
                return False
        
        return True
    
    async def _notify_local_subscribers(self, channel: str, message: Dict[str, Any]):
        """Notify local in-memory subscribers"""
        handlers = self.subscribers.get(channel, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Error in subscriber handler: {e}")
    
    def subscribe_local(self, channel: str, handler: Callable):
        """
        Subscribe to events with a local handler (in-memory)
        
        Args:
            channel: The channel to subscribe to
            handler: Callback function that receives the message
        """
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        self.subscribers[channel].append(handler)
        logger.debug(f"Added local subscriber to {channel}")
    
    def unsubscribe_local(self, channel: str, handler: Callable):
        """Remove a local subscriber"""
        if channel in self.subscribers:
            self.subscribers[channel] = [h for h in self.subscribers[channel] if h != handler]
    
    async def subscribe_redis(self, channels: List[str], handler: Callable):
        """
        Subscribe to Redis channels
        
        Args:
            channels: List of channels to subscribe to
            handler: Async callback function that receives messages
        """
        if not self.is_connected or not self.redis:
            logger.warning("Not connected to Redis, using local subscriptions only")
            for channel in channels:
                self.subscribe_local(channel, handler)
            return
        
        try:
            self.pubsub = self.redis.pubsub()
            await self.pubsub.subscribe(*channels)
            
            async def listener():
                async for message in self.pubsub.listen():
                    if message["type"] == "message":
                        try:
                            data = json.loads(message["data"])
                            if asyncio.iscoroutinefunction(handler):
                                await handler(message["channel"], data)
                            else:
                                handler(message["channel"], data)
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON in message: {message['data']}")
                        except Exception as e:
                            logger.error(f"Error in Redis subscriber: {e}")
            
            self._listener_task = asyncio.create_task(listener())
            logger.info(f"Subscribed to Redis channels: {channels}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to Redis: {e}")
    
    # Convenience methods for common events
    
    async def emit_status_change(self, status: str, details: Dict[str, Any] = None):
        """Emit a service status change event"""
        await self.publish(
            self.CHANNEL_STATUS,
            "status_change",
            {"status": status, "details": details or {}}
        )
    
    async def emit_model_event(self, event: str, model_name: str, details: Dict[str, Any] = None):
        """Emit a model-related event (loading, loaded, unloaded, error)"""
        await self.publish(
            self.CHANNEL_MODEL,
            event,
            {"model": model_name, "details": details or {}}
        )
    
    async def emit_download_progress(self, download_id: str, progress: float, details: Dict[str, Any] = None):
        """Emit a download progress event"""
        await self.publish(
            self.CHANNEL_DOWNLOAD,
            "download_progress",
            {"download_id": download_id, "progress": progress, "details": details or {}}
        )
    
    async def emit_metrics(self, metrics: Dict[str, Any]):
        """Emit system metrics"""
        await self.publish(
            self.CHANNEL_METRICS,
            "metrics_update",
            metrics
        )
    
    async def emit_log(self, level: str, message: str, source: str = "system"):
        """Emit a log event"""
        await self.publish(
            self.CHANNEL_LOGS,
            "log",
            {"level": level, "message": message, "source": source}
        )
    
    async def emit_conversation_event(self, event: str, conversation_id: str, details: Dict[str, Any] = None):
        """Emit a conversation-related event"""
        await self.publish(
            self.CHANNEL_CONVERSATION,
            event,
            {"conversation_id": conversation_id, "details": details or {}}
        )
    
    # Cache methods (uses Redis as cache when available)
    
    async def cache_set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set a cached value
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (default 1 hour)
        """
        if not self.is_connected or not self.redis:
            return False
            
        try:
            await self.redis.set(f"cache:{key}", json.dumps(value), ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get a cached value"""
        if not self.is_connected or not self.redis:
            return None
            
        try:
            data = await self.redis.get(f"cache:{key}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def cache_delete(self, key: str) -> bool:
        """Delete a cached value"""
        if not self.is_connected or not self.redis:
            return False
            
        try:
            await self.redis.delete(f"cache:{key}")
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False


# Create singleton instance
event_bus = EventBus()

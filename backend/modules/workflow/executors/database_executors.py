"""
Database Node Executors
SQL queries, caching, and data storage operations
"""

from typing import Dict, Any, List, Optional
import os
import json
import hashlib
from datetime import datetime, timedelta
from .base import NodeExecutor, ExecutionContext

# Try to import redis for caching
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Try to import aiosqlite for SQL
try:
    import aiosqlite
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


class SQLQueryExecutor(NodeExecutor):
    """Execute SQL query against a database"""
    
    node_type = "sql_query"
    display_name = "SQL Query"
    category = "database"
    description = "Execute SQL query against a database"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        params = inputs.get("params", {})
        
        connection_name = self.get_config_value("connection", "default")
        query = self.get_config_value("query", "")
        
        if not query:
            raise ValueError("SQL query is required")
        
        context.log(f"Executing SQL query on {connection_name}")
        
        # Get database path from environment or use default
        db_path = os.environ.get(f"DB_{connection_name.upper()}_PATH", f"data/{connection_name}.db")
        
        if not SQLITE_AVAILABLE:
            raise ValueError("aiosqlite is not installed. Run: pip install aiosqlite")
        
        try:
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Handle parameterized queries
                if params:
                    cursor = await db.execute(query, params)
                else:
                    cursor = await db.execute(query)
                
                # Fetch results
                rows = await cursor.fetchall()
                
                # Convert to list of dicts
                results = []
                if rows:
                    columns = [desc[0] for desc in cursor.description]
                    results = [dict(zip(columns, row)) for row in rows]
                
                # Commit if it's a write operation
                if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')):
                    await db.commit()
                    context.log(f"SQL write executed, {cursor.rowcount} rows affected")
                    return {"rows": [], "count": cursor.rowcount}
                
                context.log(f"SQL query returned {len(results)} rows")
                return {"rows": results, "count": len(results)}
                
        except Exception as e:
            context.log(f"SQL query failed: {e}", level="error")
            raise ValueError(f"SQL query failed: {e}")


class CacheGetExecutor(NodeExecutor):
    """Get value from cache (Redis)"""
    
    node_type = "cache_get"
    display_name = "Cache Get"
    category = "database"
    description = "Get a value from the cache"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        key = inputs.get("key", "")
        
        if not key:
            raise ValueError("Cache key is required")
        
        context.log(f"Getting cache key: {key}")
        
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        
        if not REDIS_AVAILABLE:
            # Fallback to in-memory cache in context
            cache = context.variables.get("_cache", {})
            value = cache.get(key)
            return {"value": value, "hit": value is not None}
        
        try:
            redis_client = aioredis.from_url(redis_url)
            value = await redis_client.get(key)
            await redis_client.close()
            
            if value is not None:
                # Try to deserialize JSON
                try:
                    value = json.loads(value)
                except:
                    value = value.decode('utf-8') if isinstance(value, bytes) else value
                
                context.log("Cache hit")
                return {"value": value, "hit": True}
            else:
                context.log("Cache miss")
                return {"value": None, "hit": False}
                
        except Exception as e:
            context.log(f"Cache get failed: {e}", level="warning")
            return {"value": None, "hit": False}


class CacheSetExecutor(NodeExecutor):
    """Set value in cache (Redis)"""
    
    node_type = "cache_set"
    display_name = "Cache Set"
    category = "database"
    description = "Set a value in the cache"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        key = inputs.get("key", "")
        value = inputs.get("value")
        
        if not key:
            raise ValueError("Cache key is required")
        
        ttl = self.get_config_value("ttl")  # TTL in seconds
        
        context.log(f"Setting cache key: {key}")
        
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        
        if not REDIS_AVAILABLE:
            # Fallback to in-memory cache in context
            cache = context.variables.get("_cache", {})
            cache[key] = value
            context.variables["_cache"] = cache
            return {"success": True}
        
        try:
            redis_client = aioredis.from_url(redis_url)
            
            # Serialize value to JSON
            serialized = json.dumps(value, default=str)
            
            if ttl:
                await redis_client.setex(key, ttl, serialized)
            else:
                await redis_client.set(key, serialized)
            
            await redis_client.close()
            
            context.log(f"Cache set successful (TTL: {ttl or 'none'})")
            return {"success": True}
            
        except Exception as e:
            context.log(f"Cache set failed: {e}", level="error")
            raise ValueError(f"Cache set failed: {e}")


class CacheDeleteExecutor(NodeExecutor):
    """Delete value from cache"""
    
    node_type = "cache_delete"
    display_name = "Cache Delete"
    category = "database"
    description = "Delete a value from the cache"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        key = inputs.get("key", "")
        
        if not key:
            raise ValueError("Cache key is required")
        
        context.log(f"Deleting cache key: {key}")
        
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        
        if not REDIS_AVAILABLE:
            cache = context.variables.get("_cache", {})
            deleted = key in cache
            if deleted:
                del cache[key]
            return {"deleted": deleted}
        
        try:
            redis_client = aioredis.from_url(redis_url)
            result = await redis_client.delete(key)
            await redis_client.close()
            
            context.log(f"Cache delete: {result > 0}")
            return {"deleted": result > 0}
            
        except Exception as e:
            context.log(f"Cache delete failed: {e}", level="error")
            raise ValueError(f"Cache delete failed: {e}")


class QdrantSearchExecutor(NodeExecutor):
    """Search Qdrant vector database directly"""
    
    node_type = "qdrant_search"
    display_name = "Qdrant Search"
    category = "database"
    description = "Search Qdrant vector database"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        query_vector = inputs.get("vector", [])
        
        collection = self.get_config_value("collection", "default")
        limit = self.get_config_value("limit", 5)
        score_threshold = self.get_config_value("scoreThreshold")
        
        if not query_vector:
            raise ValueError("Query vector is required")
        
        context.log(f"Searching Qdrant collection: {collection}")
        
        import httpx
        
        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        qdrant_port = os.environ.get("QDRANT_PORT", "6333")
        qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                search_body = {
                    "vector": query_vector,
                    "limit": limit,
                    "with_payload": True,
                }
                
                if score_threshold is not None:
                    search_body["score_threshold"] = score_threshold
                
                response = await client.post(
                    f"{qdrant_url}/collections/{collection}/points/search",
                    json=search_body
                )
                response.raise_for_status()
                result = response.json()
                
                points = result.get("result", [])
                context.log(f"Qdrant search returned {len(points)} results")
                
                return {"points": points}
                
        except Exception as e:
            context.log(f"Qdrant search failed: {e}", level="error")
            raise ValueError(f"Qdrant search failed: {e}")









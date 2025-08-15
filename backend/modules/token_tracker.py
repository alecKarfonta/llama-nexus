"""
Token Usage Tracking Module
Tracks token usage by model and provides endpoints for retrieving statistics
"""

import sqlite3
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import logging

# Configure logging
logger = logging.getLogger("token_tracker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

class TokenTracker:
    """Tracks token usage by model and provides statistics"""
    
    def __init__(self, db_path: str = None):
        """Initialize the token tracker with database path"""
        # Use environment variable for database path if set
        if db_path is None:
            db_path = os.environ.get("TOKEN_DB_PATH", "/tmp/token_usage.db")
        
        self.db_path = db_path
        self.db_lock = threading.Lock()
        self._ensure_db_dir()
        self._init_db()
        logger.info(f"TokenTracker initialized with database at {db_path}")
    
    def _ensure_db_dir(self):
        """Ensure the database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")
    
    def _init_db(self):
        """Initialize the database schema if it doesn't exist"""
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create token_usage table if it doesn't exist
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_id TEXT NOT NULL,
                    model_name TEXT,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    request_id TEXT,
                    user_id TEXT,
                    endpoint TEXT,
                    metadata TEXT
                )
                ''')
                
                # Create index on timestamp for faster queries
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp ON token_usage(timestamp)')
                
                # Create index on model_id for faster queries
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_token_usage_model_id ON token_usage(model_id)')
                
                conn.commit()
                logger.info("Database schema initialized")
            except Exception as e:
                logger.error(f"Error initializing database: {e}")
                raise
            finally:
                if conn:
                    conn.close()
    
    def record_token_usage(self, 
                          model_id: str, 
                          prompt_tokens: int, 
                          completion_tokens: int, 
                          model_name: Optional[str] = None,
                          request_id: Optional[str] = None,
                          user_id: Optional[str] = None,
                          endpoint: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record token usage for a model
        
        Args:
            model_id: Unique identifier for the model
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            model_name: Human-readable name of the model (optional)
            request_id: Unique identifier for the request (optional)
            user_id: Identifier for the user (optional)
            endpoint: API endpoint used (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Convert metadata to JSON if provided
                metadata_json = json.dumps(metadata) if metadata else None
                
                # Insert token usage record
                cursor.execute('''
                INSERT INTO token_usage 
                (model_id, model_name, prompt_tokens, completion_tokens, request_id, user_id, endpoint, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (model_id, model_name, prompt_tokens, completion_tokens, 
                      request_id, user_id, endpoint, metadata_json))
                
                conn.commit()
                logger.debug(f"Recorded token usage: {prompt_tokens} prompt, {completion_tokens} completion for model {model_id}")
                return True
            except Exception as e:
                logger.error(f"Error recording token usage: {e}")
                return False
            finally:
                if conn:
                    conn.close()
    
    def get_token_usage(self, time_range: str = '24h') -> List[Dict[str, Any]]:
        """
        Get token usage statistics grouped by model
        
        Args:
            time_range: Time range to query ('1h', '24h', '7d', '30d')
            
        Returns:
            List of token usage records by model
        """
        # Convert time range to datetime
        now = datetime.now()
        if time_range == '1h':
            start_time = now - timedelta(hours=1)
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
        else:  # Default to 24h
            start_time = now - timedelta(days=1)
        
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row  # Return rows as dictionaries
                cursor = conn.cursor()
                
                # Query token usage grouped by model
                cursor.execute('''
                SELECT 
                    model_id,
                    model_name,
                    SUM(prompt_tokens) as prompt_tokens,
                    SUM(completion_tokens) as completion_tokens,
                    COUNT(*) as requests,
                    MAX(timestamp) as last_used
                FROM token_usage
                WHERE timestamp >= ?
                GROUP BY model_id
                ORDER BY SUM(prompt_tokens + completion_tokens) DESC
                ''', (start_time.strftime('%Y-%m-%d %H:%M:%S'),))
                
                # Convert rows to dictionaries
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'modelId': row['model_id'],
                        'modelName': row['model_name'],
                        'promptTokens': row['prompt_tokens'],
                        'completionTokens': row['completion_tokens'],
                        'requests': row['requests'],
                        'lastUsed': row['last_used']
                    })
                
                return results
            except Exception as e:
                logger.error(f"Error getting token usage: {e}")
                return []
            finally:
                if conn:
                    conn.close()
    
    def get_token_usage_over_time(self, time_range: str = '24h', interval: str = 'hour') -> List[Dict[str, Any]]:
        """
        Get token usage statistics over time
        
        Args:
            time_range: Time range to query ('1h', '24h', '7d', '30d')
            interval: Interval for grouping ('hour', 'day')
            
        Returns:
            List of token usage records by time interval
        """
        # Convert time range to datetime
        now = datetime.now()
        if time_range == '1h':
            start_time = now - timedelta(hours=1)
            interval = 'minute'  # Override to minutes for 1h
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
            interval = 'day'  # Override to days for 7d
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
            interval = 'day'  # Override to days for 30d
        else:  # Default to 24h
            start_time = now - timedelta(days=1)
        
        # Determine the SQLite datetime format based on interval
        if interval == 'minute':
            time_format = '%Y-%m-%d %H:%M'
            group_by = "strftime('%Y-%m-%d %H:%M', timestamp)"
        elif interval == 'hour':
            time_format = '%Y-%m-%d %H'
            group_by = "strftime('%Y-%m-%d %H', timestamp)"
        elif interval == 'day':
            time_format = '%Y-%m-%d'
            group_by = "strftime('%Y-%m-%d', timestamp)"
        else:
            time_format = '%Y-%m-%d %H'
            group_by = "strftime('%Y-%m-%d %H', timestamp)"
        
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Query token usage grouped by time interval
                cursor.execute(f'''
                SELECT 
                    {group_by} as time_interval,
                    SUM(prompt_tokens) as prompt_tokens,
                    SUM(completion_tokens) as completion_tokens,
                    COUNT(*) as requests
                FROM token_usage
                WHERE timestamp >= ?
                GROUP BY time_interval
                ORDER BY time_interval
                ''', (start_time.strftime('%Y-%m-%d %H:%M:%S'),))
                
                # Convert rows to dictionaries
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'timeInterval': row['time_interval'],
                        'promptTokens': row['prompt_tokens'],
                        'completionTokens': row['completion_tokens'],
                        'totalTokens': row['prompt_tokens'] + row['completion_tokens'],
                        'requests': row['requests']
                    })
                
                return results
            except Exception as e:
                logger.error(f"Error getting token usage over time: {e}")
                return []
            finally:
                if conn:
                    conn.close()
    
    def get_total_token_usage(self, time_range: str = 'all') -> Dict[str, Any]:
        """
        Get total token usage statistics
        
        Args:
            time_range: Time range to query ('1h', '24h', '7d', '30d', 'all')
            
        Returns:
            Dictionary with total token usage statistics
        """
        # Convert time range to datetime
        now = datetime.now()
        if time_range == '1h':
            start_time = now - timedelta(hours=1)
        elif time_range == '24h':
            start_time = now - timedelta(days=1)
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
        else:  # Default to all time
            start_time = None
        
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                if start_time:
                    # Query with time filter
                    cursor.execute('''
                    SELECT 
                        SUM(prompt_tokens) as prompt_tokens,
                        SUM(completion_tokens) as completion_tokens,
                        COUNT(*) as requests,
                        COUNT(DISTINCT model_id) as models,
                        MIN(timestamp) as first_request,
                        MAX(timestamp) as last_request
                    FROM token_usage
                    WHERE timestamp >= ?
                    ''', (start_time.strftime('%Y-%m-%d %H:%M:%S'),))
                else:
                    # Query all time
                    cursor.execute('''
                    SELECT 
                        SUM(prompt_tokens) as prompt_tokens,
                        SUM(completion_tokens) as completion_tokens,
                        COUNT(*) as requests,
                        COUNT(DISTINCT model_id) as models,
                        MIN(timestamp) as first_request,
                        MAX(timestamp) as last_request
                    FROM token_usage
                    ''')
                
                row = cursor.fetchone()
                
                # Handle case where no data exists
                if not row or row[0] is None:
                    return {
                        'promptTokens': 0,
                        'completionTokens': 0,
                        'totalTokens': 0,
                        'requests': 0,
                        'models': 0,
                        'timeRange': time_range,
                        'firstRequest': None,
                        'lastRequest': None
                    }
                
                return {
                    'promptTokens': row[0],
                    'completionTokens': row[1],
                    'totalTokens': row[0] + row[1],
                    'requests': row[2],
                    'models': row[3],
                    'timeRange': time_range,
                    'firstRequest': row[4],
                    'lastRequest': row[5]
                }
            except Exception as e:
                logger.error(f"Error getting total token usage: {e}")
                return {
                    'promptTokens': 0,
                    'completionTokens': 0,
                    'totalTokens': 0,
                    'requests': 0,
                    'models': 0,
                    'timeRange': time_range,
                    'firstRequest': None,
                    'lastRequest': None,
                    'error': str(e)
                }
            finally:
                if conn:
                    conn.close()

# Create a singleton instance
token_tracker = TokenTracker()

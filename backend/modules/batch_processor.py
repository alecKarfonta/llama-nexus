"""
Batch Processing Module
Process multiple prompts through the LLM in batch with progress tracking.
"""

import os
import json
import sqlite3
import csv
import io
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """A single item in a batch."""
    id: str
    input_text: str
    output_text: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None
    tokens_used: int = 0
    processing_time_ms: float = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchJob:
    """A batch processing job."""
    id: str
    name: str
    status: str  # pending, running, completed, failed, cancelled
    total_items: int
    completed_items: int
    failed_items: int
    config: Dict[str, Any]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


class BatchProcessor:
    """
    Processes batches of prompts through the LLM API.
    """
    
    def __init__(self, db_path: str = None, api_base_url: str = None):
        """Initialize the batch processor."""
        if db_path is None:
            db_path = os.getenv('BATCH_DB_PATH', '/data/batch_jobs.db')
        
        self.db_path = db_path
        self.api_base_url = api_base_url or os.getenv('LLAMA_API_URL', 'http://llamacpp-api:8080')
        self._ensure_db_directory()
        self._init_database()
        
        # Track active jobs
        self._active_jobs: Dict[str, BatchJob] = {}
        self._cancel_flags: Dict[str, bool] = {}
        
        logger.info(f"Batch processor initialized with database at: {self.db_path}")
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Batch jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_items INTEGER NOT NULL,
                    completed_items INTEGER DEFAULT 0,
                    failed_items INTEGER DEFAULT 0,
                    config TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT
                )
            ''')
            
            # Batch items table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_items (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    item_index INTEGER NOT NULL,
                    input_text TEXT NOT NULL,
                    output_text TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    processing_time_ms REAL DEFAULT 0,
                    metadata TEXT,
                    FOREIGN KEY (job_id) REFERENCES batch_jobs(id)
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_batch_items_job ON batch_items(job_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status)')
            
            conn.commit()
    
    def parse_input_file(self, content: str, file_type: str) -> List[Dict[str, Any]]:
        """Parse input file content (CSV or JSON)."""
        items = []
        
        if file_type == 'json':
            data = json.loads(content)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, str):
                        items.append({'input': item, 'metadata': {}})
                    elif isinstance(item, dict):
                        items.append({
                            'input': item.get('input') or item.get('prompt') or item.get('text', ''),
                            'metadata': {k: v for k, v in item.items() if k not in ['input', 'prompt', 'text']}
                        })
            elif isinstance(data, dict) and 'items' in data:
                return self.parse_input_file(json.dumps(data['items']), 'json')
        
        elif file_type == 'csv':
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                # Look for input column
                input_text = row.get('input') or row.get('prompt') or row.get('text') or ''
                metadata = {k: v for k, v in row.items() if k not in ['input', 'prompt', 'text']}
                items.append({'input': input_text, 'metadata': metadata})
        
        elif file_type == 'txt':
            # One prompt per line
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    items.append({'input': line, 'metadata': {}})
        
        return items
    
    def create_batch_job(
        self,
        name: str,
        items: List[Dict[str, Any]],
        config: Dict[str, Any] = None,
    ) -> BatchJob:
        """Create a new batch job."""
        job_id = str(uuid.uuid4())[:12]
        now = datetime.utcnow().isoformat()
        
        config = config or {}
        default_config = {
            'max_tokens': 512,
            'temperature': 0.7,
            'system_prompt': None,
            'concurrency': 1,  # Process one at a time by default
        }
        config = {**default_config, **config}
        
        job = BatchJob(
            id=job_id,
            name=name,
            status='pending',
            total_items=len(items),
            completed_items=0,
            failed_items=0,
            config=config,
            created_at=now,
            started_at=None,
            completed_at=None,
            error=None,
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert job
            cursor.execute('''
                INSERT INTO batch_jobs (id, name, status, total_items, config, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (job_id, name, 'pending', len(items), json.dumps(config), now))
            
            # Insert items
            for i, item in enumerate(items):
                item_id = f"{job_id}-{i}"
                cursor.execute('''
                    INSERT INTO batch_items (id, job_id, item_index, input_text, status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (item_id, job_id, i, item['input'], 'pending', json.dumps(item.get('metadata', {}))))
            
            conn.commit()
        
        logger.info(f"Created batch job {job_id} with {len(items)} items")
        return job
    
    async def _process_single_item(
        self,
        item_id: str,
        input_text: str,
        config: Dict[str, Any],
        api_key: str = None,
    ) -> Dict[str, Any]:
        """Process a single batch item."""
        import time
        start_time = time.perf_counter()
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        messages = []
        if config.get('system_prompt'):
            messages.append({"role": "system", "content": config['system_prompt']})
        messages.append({"role": "user", "content": input_text})
        
        payload = {
            "model": "default",
            "messages": messages,
            "max_tokens": config.get('max_tokens', 512),
            "temperature": config.get('temperature', 0.7),
            "stream": False,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                    
                    data = await response.json()
                    
                    output_text = ""
                    tokens_used = 0
                    
                    if 'choices' in data and len(data['choices']) > 0:
                        output_text = data['choices'][0].get('message', {}).get('content', '')
                    
                    if 'usage' in data:
                        tokens_used = data['usage'].get('total_tokens', 0)
                    
                    end_time = time.perf_counter()
                    processing_time_ms = (end_time - start_time) * 1000
                    
                    return {
                        'success': True,
                        'output_text': output_text,
                        'tokens_used': tokens_used,
                        'processing_time_ms': processing_time_ms,
                    }
        
        except asyncio.TimeoutError:
            return {'success': False, 'error': 'Request timed out'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def run_batch_job(
        self,
        job_id: str,
        api_key: str = None,
        progress_callback: callable = None,
    ) -> BatchJob:
        """Run a batch job."""
        # Get job
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        
        if job['status'] not in ['pending', 'failed']:
            raise ValueError(f"Job {job_id} cannot be run (status: {job['status']})")
        
        # Update job status
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE batch_jobs SET status = ?, started_at = ? WHERE id = ?
            ''', ('running', now, job_id))
            conn.commit()
        
        self._cancel_flags[job_id] = False
        config = job['config']
        
        # Get pending items
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, input_text, item_index FROM batch_items 
                WHERE job_id = ? AND status IN ('pending', 'failed')
                ORDER BY item_index
            ''', (job_id,))
            items = [dict(row) for row in cursor.fetchall()]
        
        completed = job['completed_items']
        failed = job['failed_items']
        
        try:
            for item in items:
                # Check for cancellation
                if self._cancel_flags.get(job_id, False):
                    break
                
                # Update item status to processing
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE batch_items SET status = ? WHERE id = ?
                    ''', ('processing', item['id']))
                    conn.commit()
                
                # Process item
                result = await self._process_single_item(
                    item['id'],
                    item['input_text'],
                    config,
                    api_key,
                )
                
                # Update item with result
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    if result['success']:
                        cursor.execute('''
                            UPDATE batch_items 
                            SET status = ?, output_text = ?, tokens_used = ?, processing_time_ms = ?
                            WHERE id = ?
                        ''', ('completed', result['output_text'], result['tokens_used'], 
                              result['processing_time_ms'], item['id']))
                        completed += 1
                    else:
                        cursor.execute('''
                            UPDATE batch_items SET status = ?, error = ? WHERE id = ?
                        ''', ('failed', result['error'], item['id']))
                        failed += 1
                    
                    # Update job progress
                    cursor.execute('''
                        UPDATE batch_jobs SET completed_items = ?, failed_items = ? WHERE id = ?
                    ''', (completed, failed, job_id))
                    conn.commit()
                
                # Progress callback
                if progress_callback:
                    await progress_callback({
                        'job_id': job_id,
                        'completed': completed,
                        'failed': failed,
                        'total': job['total_items'],
                        'progress': int((completed + failed) / job['total_items'] * 100),
                    })
            
            # Finalize job
            final_status = 'cancelled' if self._cancel_flags.get(job_id, False) else 'completed'
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE batch_jobs SET status = ?, completed_at = ? WHERE id = ?
                ''', (final_status, datetime.utcnow().isoformat(), job_id))
                conn.commit()
        
        except Exception as e:
            logger.error(f"Batch job {job_id} failed: {e}")
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE batch_jobs SET status = ?, error = ?, completed_at = ? WHERE id = ?
                ''', ('failed', str(e), datetime.utcnow().isoformat(), job_id))
                conn.commit()
        
        finally:
            self._cancel_flags.pop(job_id, None)
        
        return self.get_job(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        self._cancel_flags[job_id] = True
        return True
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM batch_jobs WHERE id = ?', (job_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'status': row['status'],
                    'total_items': row['total_items'],
                    'completed_items': row['completed_items'],
                    'failed_items': row['failed_items'],
                    'config': json.loads(row['config']) if row['config'] else {},
                    'created_at': row['created_at'],
                    'started_at': row['started_at'],
                    'completed_at': row['completed_at'],
                    'error': row['error'],
                    'progress': int((row['completed_items'] + row['failed_items']) / row['total_items'] * 100) if row['total_items'] > 0 else 0,
                }
        
        return None
    
    def get_job_items(
        self,
        job_id: str,
        status: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get items for a job."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM batch_items WHERE job_id = ?'
            params = [job_id]
            
            if status:
                query += ' AND status = ?'
                params.append(status)
            
            # Get total count
            count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]
            
            # Get items
            query += ' ORDER BY item_index LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            cursor.execute(query, params)
            
            items = []
            for row in cursor.fetchall():
                items.append({
                    'id': row['id'],
                    'index': row['item_index'],
                    'input_text': row['input_text'],
                    'output_text': row['output_text'],
                    'status': row['status'],
                    'error': row['error'],
                    'tokens_used': row['tokens_used'],
                    'processing_time_ms': row['processing_time_ms'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                })
            
            return {
                'items': items,
                'total': total,
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total,
            }
    
    def list_jobs(
        self,
        status: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List all batch jobs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM batch_jobs WHERE 1=1'
            params = []
            
            if status:
                query += ' AND status = ?'
                params.append(status)
            
            # Get total count
            count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]
            
            # Get jobs
            query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            cursor.execute(query, params)
            
            jobs = []
            for row in cursor.fetchall():
                jobs.append({
                    'id': row['id'],
                    'name': row['name'],
                    'status': row['status'],
                    'total_items': row['total_items'],
                    'completed_items': row['completed_items'],
                    'failed_items': row['failed_items'],
                    'created_at': row['created_at'],
                    'completed_at': row['completed_at'],
                    'progress': int((row['completed_items'] + row['failed_items']) / row['total_items'] * 100) if row['total_items'] > 0 else 0,
                })
            
            return {
                'jobs': jobs,
                'total': total,
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total,
            }
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its items."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM batch_items WHERE job_id = ?', (job_id,))
            cursor.execute('DELETE FROM batch_jobs WHERE id = ?', (job_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def export_results(self, job_id: str, format: str = 'json') -> str:
        """Export job results."""
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        
        items_data = self.get_job_items(job_id, limit=10000)
        items = items_data['items']
        
        if format == 'json':
            export_data = {
                'job': job,
                'items': items,
                'exported_at': datetime.utcnow().isoformat(),
            }
            return json.dumps(export_data, indent=2)
        
        elif format == 'csv':
            output = io.StringIO()
            if items:
                fieldnames = ['index', 'input_text', 'output_text', 'status', 'tokens_used', 'processing_time_ms', 'error']
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for item in items:
                    writer.writerow({k: item.get(k, '') for k in fieldnames})
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall batch processing statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM batch_jobs')
            total_jobs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM batch_jobs WHERE status = 'completed'")
            completed_jobs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM batch_jobs WHERE status = 'running'")
            running_jobs = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(total_items) FROM batch_jobs')
            total_items = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT SUM(completed_items) FROM batch_jobs")
            processed_items = cursor.fetchone()[0] or 0
            
            return {
                'total_jobs': total_jobs,
                'completed_jobs': completed_jobs,
                'running_jobs': running_jobs,
                'total_items': total_items,
                'processed_items': processed_items,
            }


# Global instance
batch_processor = BatchProcessor()

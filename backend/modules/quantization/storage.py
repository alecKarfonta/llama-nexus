"""
Storage layer for quantization jobs using SQLite.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from enhanced_logger import enhanced_logger as logger

from .config import QuantizationJob, QuantizedOutput


class QuantizationStore:
    """Persistent storage for quantization jobs."""

    def __init__(self, db_path: str = None):
        """Initialize the quantization store."""
        if db_path is None:
            db_path = os.getenv("QUANTIZATION_DB_PATH", "/data/quantization.db")

        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()
        logger.info(f"Quantization store initialized at: {self.db_path}")

    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Quantization jobs table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS quantization_jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    source_model TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    model_architecture TEXT,
                    output_formats TEXT NOT NULL,
                    gguf_quant_types TEXT,
                    gptq_bits TEXT,
                    awq_bits TEXT,
                    config TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    current_step TEXT,
                    total_outputs INTEGER DEFAULT 0,
                    completed_outputs INTEGER DEFAULT 0,
                    outputs TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    estimated_disk_gb REAL,
                    estimated_time_minutes INTEGER
                )
            """
            )

            # Create indices
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON quantization_jobs(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_created ON quantization_jobs(created_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_source ON quantization_jobs(source_model)"
            )

            conn.commit()

    def upsert(self, job: QuantizationJob) -> None:
        """Insert or update a quantization job."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO quantization_jobs 
                (id, name, description, source_model, source_type, model_architecture,
                 output_formats, gguf_quant_types, gptq_bits, awq_bits, config,
                 status, progress, current_step, total_outputs, completed_outputs,
                 outputs, created_at, started_at, completed_at, error,
                 estimated_disk_gb, estimated_time_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    status = excluded.status,
                    progress = excluded.progress,
                    current_step = excluded.current_step,
                    total_outputs = excluded.total_outputs,
                    completed_outputs = excluded.completed_outputs,
                    outputs = excluded.outputs,
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at,
                    error = excluded.error
            """,
                (
                    job.id,
                    job.name,
                    job.description,
                    job.source_model,
                    job.source_type,
                    job.model_architecture,
                    json.dumps([f.value for f in job.output_formats]),
                    json.dumps([q.value for q in job.gguf_quant_types]),
                    json.dumps(job.gptq_bits),
                    json.dumps(job.awq_bits),
                    job.config.model_dump_json(),
                    job.status.value,
                    job.progress,
                    job.current_step,
                    job.total_outputs,
                    job.completed_outputs,
                    json.dumps([o.model_dump(mode='json') for o in job.outputs]),
                    job.created_at.isoformat(),
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    job.error,
                    job.estimated_disk_gb,
                    job.estimated_time_minutes,
                ),
            )
            conn.commit()

    def get(self, job_id: str) -> Optional[QuantizationJob]:
        """Get a quantization job by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM quantization_jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()

            if row:
                return self._row_to_job(dict(row))

        return None

    def list(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[QuantizationJob]:
        """List quantization jobs with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM quantization_jobs"
            params = []

            if status:
                query += " WHERE status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_job(dict(row)) for row in rows]

    def delete(self, job_id: str) -> bool:
        """Delete a quantization job."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM quantization_jobs WHERE id = ?", (job_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_job(self, row: Dict) -> QuantizationJob:
        """Convert a database row to a QuantizationJob object."""
        from .config import (
            QuantizationFormat,
            QuantizationStatus,
            GGUFQuantType,
            QuantizationConfig,
        )
        from datetime import datetime

        # Parse JSON fields
        output_formats = [QuantizationFormat(f) for f in json.loads(row["output_formats"])]
        gguf_quant_types = [GGUFQuantType(q) for q in json.loads(row.get("gguf_quant_types") or "[]")]
        gptq_bits = json.loads(row.get("gptq_bits") or "[]")
        awq_bits = json.loads(row.get("awq_bits") or "[]")
        config = QuantizationConfig.model_validate_json(row["config"])
        outputs_data = json.loads(row.get("outputs") or "[]")
        outputs = [QuantizedOutput(**o) for o in outputs_data]

        return QuantizationJob(
            id=row["id"],
            name=row["name"],
            description=row.get("description"),
            source_model=row["source_model"],
            source_type=row["source_type"],
            model_architecture=row.get("model_architecture"),
            output_formats=output_formats,
            gguf_quant_types=gguf_quant_types,
            gptq_bits=gptq_bits,
            awq_bits=awq_bits,
            config=config,
            status=QuantizationStatus(row["status"]),
            progress=row["progress"],
            current_step=row.get("current_step", ""),
            total_outputs=row["total_outputs"],
            completed_outputs=row["completed_outputs"],
            outputs=outputs,
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row.get("started_at") else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row.get("completed_at") else None,
            error=row.get("error"),
            estimated_disk_gb=row.get("estimated_disk_gb"),
            estimated_time_minutes=row.get("estimated_time_minutes"),
        )


# Global instance
_quantization_store: Optional[QuantizationStore] = None


def init_quantization_store(db_path: str = None) -> QuantizationStore:
    """Initialize and return the global quantization store instance."""
    global _quantization_store
    if _quantization_store is None:
        _quantization_store = QuantizationStore(db_path)
    return _quantization_store


def get_quantization_store() -> QuantizationStore:
    """Get the global quantization store instance."""
    if _quantization_store is None:
        return init_quantization_store()
    return _quantization_store

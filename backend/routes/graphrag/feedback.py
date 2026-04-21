"""
GraphRAG Feedback & Analytics Routes

Stores retrieval quality feedback (thumbs up/down per query), 
provides analytics endpoints for trending quality metrics.
"""

from fastapi import APIRouter, HTTPException, Request
import sqlite3
import os
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager

logger = logging.getLogger(__name__)

router = APIRouter()

# ── SQLite setup ───────────────────────────────────────────────────────

DB_PATH = os.getenv("FEEDBACK_DB_PATH", "/data/retrieval_feedback.db")

_db_initialized = False


def _ensure_schema(conn: sqlite3.Connection):
    """Create tables and indices if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK (rating IN (-1, 0, 1)),
            comment TEXT DEFAULT '',
            total_chunks INTEGER DEFAULT 0,
            total_entities INTEGER DEFAULT 0,
            avg_relevance REAL DEFAULT 0.0,
            cross_ref_count INTEGER DEFAULT 0,
            graph_expansion_count INTEGER DEFAULT 0,
            source TEXT DEFAULT 'unified',
            domain TEXT DEFAULT 'general',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_feedback_created_at 
        ON retrieval_feedback(created_at)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_feedback_domain 
        ON retrieval_feedback(domain)
    """)
    conn.commit()


@contextmanager
def _get_db():
    """Context manager for database connections. Ensures cleanup on error."""
    global _db_initialized
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    
    if not _db_initialized:
        _ensure_schema(conn)
        _db_initialized = True
    
    try:
        yield conn
    finally:
        conn.close()


# ── Submit feedback ────────────────────────────────────────────────────

@router.post("/feedback")
async def submit_feedback(request: Request):
    """
    Submit retrieval quality feedback.
    
    Body:
    - query: the user query
    - rating: -1 (bad), 0 (neutral), 1 (good)
    - comment: optional text
    - quality_signals: optional dict from /chat-context-v2 response
    - domain: optional domain for filtering
    """
    data = await request.json()
    query = data.get("query", "")
    rating = data.get("rating")
    
    if rating is None or rating not in (-1, 0, 1):
        raise HTTPException(status_code=400, detail="rating must be -1, 0, or 1")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    
    quality = data.get("quality_signals", {})
    
    try:
        with _get_db() as conn:
            conn.execute("""
                INSERT INTO retrieval_feedback 
                (query, rating, comment, total_chunks, total_entities, 
                 avg_relevance, cross_ref_count, graph_expansion_count, source, domain)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query,
                rating,
                data.get("comment", ""),
                quality.get("total_chunks", 0),
                quality.get("total_entities", 0),
                quality.get("avg_chunk_relevance", 0.0),
                quality.get("cross_reference_count", 0),
                quality.get("graph_expansion_count", 0),
                data.get("source", "unified"),
                data.get("domain", "general"),
            ))
            conn.commit()
        
        return {"status": "ok", "message": "Feedback recorded"}
    
    except Exception as e:
        logger.error(f"Failed to store feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ── Analytics ──────────────────────────────────────────────────────────

@router.get("/analytics")
async def get_analytics(days: int = 30, domain: str = None):
    """
    Get retrieval quality analytics.
    
    Returns aggregate stats and daily trends.
    """
    try:
        with _get_db() as conn:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # Overall stats
            params: list = [cutoff]
            domain_clause = ""
            if domain:
                domain_clause = " AND domain = ?"
                params.append(domain)
            
            row = conn.execute(f"""
                SELECT 
                    COUNT(*) as total_feedback,
                    SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as negative,
                    SUM(CASE WHEN rating = 0 THEN 1 ELSE 0 END) as neutral,
                    AVG(avg_relevance) as mean_relevance,
                    AVG(total_chunks) as mean_chunks,
                    AVG(total_entities) as mean_entities,
                    AVG(cross_ref_count) as mean_cross_refs,
                    AVG(graph_expansion_count) as mean_graph_expansion
                FROM retrieval_feedback
                WHERE created_at >= ?{domain_clause}
            """, params).fetchone()
            
            total = row["total_feedback"] or 0
            positive = row["positive"] or 0
            
            overall = {
                "total_feedback": total,
                "positive": positive,
                "negative": row["negative"] or 0,
                "neutral": row["neutral"] or 0,
                "satisfaction_rate": round(positive / max(total, 1), 3),
                "mean_relevance": round(row["mean_relevance"] or 0, 3),
                "mean_chunks": round(row["mean_chunks"] or 0, 1),
                "mean_entities": round(row["mean_entities"] or 0, 1),
                "mean_cross_refs": round(row["mean_cross_refs"] or 0, 1),
                "mean_graph_expansion": round(row["mean_graph_expansion"] or 0, 1),
            }
            
            # Daily trends
            trend_params: list = [cutoff]
            if domain:
                trend_params.append(domain)
            
            trends = conn.execute(f"""
                SELECT 
                    date(created_at) as day,
                    COUNT(*) as count,
                    SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as negative,
                    AVG(avg_relevance) as relevance
                FROM retrieval_feedback
                WHERE created_at >= ?{domain_clause}
                GROUP BY date(created_at)
                ORDER BY day DESC
                LIMIT 30
            """, trend_params).fetchall()
            
            daily_trends = [
                {
                    "date": t["day"],
                    "count": t["count"],
                    "positive": t["positive"],
                    "negative": t["negative"],
                    "relevance": round(t["relevance"] or 0, 3),
                }
                for t in trends
            ]
            
            # Domain breakdown
            domains = conn.execute("""
                SELECT 
                    domain,
                    COUNT(*) as count,
                    SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as positive,
                    AVG(avg_relevance) as relevance
                FROM retrieval_feedback
                WHERE created_at >= ?
                GROUP BY domain
                ORDER BY count DESC
            """, [cutoff]).fetchall()
            
            domain_stats = [
                {
                    "domain": d["domain"],
                    "count": d["count"],
                    "satisfaction_rate": round(d["positive"] / max(d["count"], 1), 3),
                    "relevance": round(d["relevance"] or 0, 3),
                }
                for d in domains
            ]
            
            # Recent low-rated queries
            low_rated_params: list = [cutoff]
            if domain:
                low_rated_params.append(domain)
            
            low_rated = conn.execute(f"""
                SELECT query, rating, comment, avg_relevance, created_at
                FROM retrieval_feedback
                WHERE rating = -1 AND created_at >= ?{domain_clause}
                ORDER BY created_at DESC
                LIMIT 10
            """, low_rated_params).fetchall()
            
            problem_queries = [
                {
                    "query": lr["query"],
                    "comment": lr["comment"],
                    "relevance": round(lr["avg_relevance"] or 0, 3),
                    "date": lr["created_at"],
                }
                for lr in low_rated
            ]
        
        return {
            "period_days": days,
            "domain_filter": domain,
            "overall": overall,
            "daily_trends": daily_trends,
            "domain_stats": domain_stats,
            "problem_queries": problem_queries,
        }
    
    except Exception as e:
        logger.error(f"Analytics query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

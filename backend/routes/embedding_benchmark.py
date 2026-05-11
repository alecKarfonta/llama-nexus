"""
Embedding Benchmark Routes

Comprehensive performance measurement for the embedding + vector store pipeline.
Tests end-to-end latency and throughput for embedding generation, Qdrant insertion,
search, update, and deletion across various collection sizes.

Benchmark types:
- embedding_throughput: Raw embedding generation speed (texts/sec, ms/text)
- insertion:         Time to embed + upsert N documents into a collection
- search:           Search latency at various collection sizes (1K, 10K, 50K, 100K)
- update:           Update (re-embed + upsert) latency per document
- deletion:         Delete latency per document and batch
- full_pipeline:    Combined insert -> search -> update -> delete lifecycle
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime
import asyncio
import json
import os
import uuid
import time
import logging
import sqlite3
import statistics
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/embedding-benchmark", tags=["embedding-benchmark"])

DB_PATH = Path("/data/embedding_benchmarks.db") if os.path.exists("/data") else Path("./data/embedding_benchmarks.db")

# Serialize writes to avoid "database is locked" under concurrent access
_db_lock = asyncio.Lock()


# =============================================================================
# Database
# =============================================================================

def _db_connect():
    """Get a SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _db_connect()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS embedding_benchmark_results (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT NOT NULL,
            config TEXT,
            metrics TEXT,
            status TEXT DEFAULT 'completed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_eb_type ON embedding_benchmark_results(type)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_eb_created ON embedding_benchmark_results(created_at DESC)")
    conn.commit()
    conn.close()

init_db()


# =============================================================================
# Enums & Pydantic Models
# =============================================================================

class EmbeddingBenchmarkType(str, Enum):
    EMBEDDING_THROUGHPUT = "embedding_throughput"
    INSERTION = "insertion"
    SEARCH = "search"
    UPDATE = "update"
    DELETION = "deletion"
    FULL_PIPELINE = "full_pipeline"


class EmbeddingBenchmarkRequest(BaseModel):
    type: EmbeddingBenchmarkType = Field(..., description="Type of benchmark to run")
    name: Optional[str] = Field(None, description="Human-readable name for this run")
    # Collection sizing
    collection_sizes: List[int] = Field(
        default=[100, 1000, 5000, 10000, 50000],
        description="Collection sizes to test (number of documents)"
    )
    # Embedding generation settings
    text_lengths: List[int] = Field(
        default=[50, 200, 500],
        description="Approximate word counts for generated test texts"
    )
    batch_sizes: List[int] = Field(
        default=[1, 8, 32, 64, 128],
        description="Batch sizes for embedding throughput tests"
    )
    # Search settings
    search_top_k: List[int] = Field(
        default=[1, 5, 10, 50],
        description="top_k values for search latency tests"
    )
    search_iterations: int = Field(
        default=50,
        description="Number of search queries to run per size/k combination"
    )
    # General
    warmup: bool = Field(default=True, description="Run a warmup embedding before measuring")
    runs: int = Field(default=1, description="Number of times to repeat the benchmark")
    cleanup: bool = Field(default=True, description="Delete benchmark collections after test")


# =============================================================================
# Test Data Generation
# =============================================================================

# Word pools for generating realistic sentences
_SUBJECTS = [
    "the system", "the database", "the application", "the network", "the server",
    "the algorithm", "the model", "the interface", "the framework", "the pipeline",
    "the cache", "the queue", "the scheduler", "the processor", "the container",
    "the module", "the component", "the service", "the endpoint", "the cluster",
]
_VERBS = [
    "processes", "manages", "handles", "optimizes", "validates",
    "transforms", "indexes", "retrieves", "aggregates", "distributes",
    "serializes", "caches", "routes", "monitors", "scales",
    "compiles", "deploys", "encrypts", "compresses", "streams",
]
_OBJECTS = [
    "data streams", "user requests", "configuration files", "log entries", "event messages",
    "API calls", "database queries", "file uploads", "authentication tokens", "cache entries",
    "batch jobs", "task queues", "metric reports", "webhook payloads", "search results",
    "embeddings", "document chunks", "vector records", "schema definitions", "metadata objects",
]
_MODIFIERS = [
    "efficiently", "in parallel", "with low latency", "using batch processing",
    "across multiple nodes", "with high throughput", "asynchronously", "in real-time",
    "with automatic retry", "using persistent storage", "through the load balancer",
    "via the message broker", "with error handling", "during peak hours", "on demand",
]
_CONNECTORS = [
    "Additionally", "Furthermore", "However", "In contrast", "Meanwhile",
    "As a result", "Subsequently", "In practice", "For example", "Specifically",
]


def generate_test_text(word_count: int, seed: int = 0) -> str:
    """Generate a realistic-looking test document of approximately `word_count` words."""
    import random
    rng = random.Random(seed)
    sentences = []
    current_words = 0

    while current_words < word_count:
        subject = rng.choice(_SUBJECTS)
        verb = rng.choice(_VERBS)
        obj = rng.choice(_OBJECTS)
        mod = rng.choice(_MODIFIERS)

        if rng.random() < 0.3 and sentences:
            connector = rng.choice(_CONNECTORS)
            sentence = f"{connector}, {subject} {verb} {obj} {mod}."
        else:
            sentence = f"{subject.capitalize()} {verb} {obj} {mod}."

        sentences.append(sentence)
        current_words += len(sentence.split())

    return " ".join(sentences)[:word_count * 7]  # rough char cap


def generate_test_documents(count: int, word_count: int = 200, seed_start: int = 0) -> List[Dict[str, Any]]:
    """Generate a list of test documents with unique IDs."""
    docs = []
    for i in range(count):
        doc_id = str(uuid.uuid4())
        text = generate_test_text(word_count, seed=seed_start + i)
        docs.append({
            "id": doc_id,
            "text": text,
            "metadata": {
                "benchmark": True,
                "index": i,
                "word_count": word_count,
            }
        })
    return docs


# =============================================================================
# Helpers
# =============================================================================

def _get_embedder(request: Request):
    from app_state import create_embedder, embedding_manager
    # The global EMBEDDING_SERVICE_URL may point to the wrong port (8080 vs actual).
    # Use the embedding manager's configured port which reflects the real container setup.
    embed_port = embedding_manager.config.get("server", {}).get("port", 8080)
    embed_api_key = embedding_manager.config.get("server", {}).get("api_key", "llamacpp-embed")
    from modules.rag.embedders import APIEmbedder
    return APIEmbedder(
        base_url=f"http://llamacpp-embed:{embed_port}/v1",
        model_name="nomic-embed-text-v1.5",
        api_key=embed_api_key,
    )


def _get_vector_store(request: Request):
    vs = getattr(request.app.state, 'vector_store', None)
    if vs is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return vs


async def _ensure_collection(vector_store, name: str, vector_size: int):
    from modules.rag.vector_stores.base import CollectionConfig, DistanceMetric
    if not await vector_store.collection_exists(name):
        await vector_store.create_collection(CollectionConfig(
            name=name,
            vector_size=vector_size,
            distance=DistanceMetric.COSINE,
        ))


async def _cleanup_collection(vector_store, name: str):
    try:
        if await vector_store.collection_exists(name):
            await vector_store.delete_collection(name)
    except Exception:
        pass


def _collection_name(bench_id: str) -> str:
    return f"bench_{bench_id.replace('-', '_')}"


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "mean": 0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    return {
        "p50": sorted_v[int(n * 0.50)],
        "p90": sorted_v[int(n * 0.90)] if n > 1 else sorted_v[0],
        "p95": sorted_v[int(n * 0.95)] if n > 1 else sorted_v[0],
        "p99": sorted_v[int(n * 0.99)] if n > 10 else sorted_v[-1],
        "min": sorted_v[0],
        "max": sorted_v[-1],
        "mean": statistics.mean(sorted_v),
        "stdev": statistics.stdev(sorted_v) if n > 1 else 0,
    }


async def _save_result(bench_id: str, name: Optional[str], bench_type: str, config: dict, metrics: dict):
    async with _db_lock:
        conn = _db_connect()
        conn.execute(
            """
            INSERT INTO embedding_benchmark_results (id, name, type, config, metrics, status, completed_at)
            VALUES (?, ?, ?, ?, ?, 'completed', ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                type = excluded.type,
                config = excluded.config,
                metrics = excluded.metrics,
                status = 'completed',
                completed_at = excluded.completed_at
            """,
            (bench_id, name, bench_type, json.dumps(config), json.dumps(metrics), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()


# =============================================================================
# Benchmark Runners
# =============================================================================

async def run_embedding_throughput(
    embedder, request_config: EmbeddingBenchmarkRequest
) -> Dict[str, Any]:
    """Benchmark raw embedding generation speed across batch sizes and text lengths."""
    results = {}

    for text_len in request_config.text_lengths:
        results[str(text_len)] = {}
        texts = [generate_test_text(text_len, seed=i) for i in range(max(request_config.batch_sizes))]

        for batch_size in request_config.batch_sizes:
            batch = texts[:batch_size]
            latencies_ms = []

            # Warmup
            if request_config.warmup:
                await embedder.embed(batch[:min(4, batch_size)])

            for run in range(request_config.runs):
                # Split into sub-batches of 8 to avoid exceeding server ubatch token limits
                sub_batch_size = 8
                t0 = time.perf_counter()
                total_tokens = 0
                for i in range(0, len(batch), sub_batch_size):
                    sub = batch[i:i + sub_batch_size]
                    result = await embedder.embed(sub)
                    total_tokens = result.token_count
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies_ms.append(elapsed_ms)

            per_text_ms = [l / batch_size for l in latencies_ms]
            results[str(text_len)][str(batch_size)] = {
                "batch_size": batch_size,
                "text_length_words": text_len,
                "total_latency_ms": _percentiles(latencies_ms),
                "per_text_ms": _percentiles(per_text_ms),
                "texts_per_second": round(batch_size / (statistics.mean(latencies_ms) / 1000), 2) if latencies_ms else 0,
                "dimensions": result.dimensions,
                "token_count": total_tokens,
            }

    return {"batch_results": results}


async def run_insertion_benchmark(
    embedder, vector_store, request_config: EmbeddingBenchmarkRequest, bench_id: str
) -> Dict[str, Any]:
    """Benchmark embedding + Qdrant upsert across collection sizes."""
    from modules.rag.vector_stores.base import VectorRecord

    results = {}
    model_info = embedder.get_model_info()
    vector_size = model_info.dimensions

    for size in request_config.collection_sizes:
        coll_name = _collection_name(bench_id) + f"_ins_{size}"
        await _ensure_collection(vector_store, coll_name, vector_size)

        docs = generate_test_documents(size, word_count=200)
        embed_latencies = []
        upsert_latencies = []
        total_start = time.perf_counter()

        # Embed in batches
        embed_batch_size = 64
        all_embeddings = []
        for i in range(0, len(docs), embed_batch_size):
            batch_texts = [d["text"] for d in docs[i:i + embed_batch_size]]
            t0 = time.perf_counter()
            embed_result = await embedder.embed(batch_texts)
            embed_latencies.append((time.perf_counter() - t0) * 1000)
            all_embeddings.extend(embed_result.embeddings)

        # Upsert in batches
        upsert_batch_size = 100
        for i in range(0, len(docs), upsert_batch_size):
            batch = docs[i:i + upsert_batch_size]
            records = [
                VectorRecord(
                    id=batch[j]["id"],
                    vector=all_embeddings[i + j],
                    payload={"text": batch[j]["text"], "metadata": batch[j]["metadata"]}
                )
                for j in range(len(batch))
            ]
            t0 = time.perf_counter()
            await vector_store.upsert(coll_name, records)
            upsert_latencies.append((time.perf_counter() - t0) * 1000)

        total_elapsed = (time.perf_counter() - total_start) * 1000

        results[str(size)] = {
            "collection_size": size,
            "total_time_ms": round(total_elapsed, 2),
            "embedding_time_ms": _percentiles(embed_latencies),
            "upsert_time_ms": _percentiles(upsert_latencies),
            "docs_per_second": round(size / (total_elapsed / 1000), 2),
            "avg_ms_per_doc": round(total_elapsed / size, 3),
        }

        if request_config.cleanup:
            await _cleanup_collection(vector_store, coll_name)

    return {"insertion_results": results}


async def run_search_benchmark(
    embedder, vector_store, request_config: EmbeddingBenchmarkRequest, bench_id: str
) -> Dict[str, Any]:
    """Benchmark search latency at various collection sizes and top_k values."""
    from modules.rag.vector_stores.base import VectorRecord

    results = {}
    model_info = embedder.get_model_info()
    vector_size = model_info.dimensions

    for size in request_config.collection_sizes:
        coll_name = _collection_name(bench_id) + f"_src_{size}"
        await _ensure_collection(vector_store, coll_name, vector_size)

        # Populate collection
        docs = generate_test_documents(size, word_count=200)
        embed_batch_size = 64
        all_embeddings = []
        for i in range(0, len(docs), embed_batch_size):
            batch_texts = [d["text"] for d in docs[i:i + embed_batch_size]]
            embed_result = await embedder.embed(batch_texts)
            all_embeddings.extend(embed_result.embeddings)

        upsert_batch_size = 100
        for i in range(0, len(docs), upsert_batch_size):
            batch = docs[i:i + upsert_batch_size]
            records = [
                VectorRecord(
                    id=batch[j]["id"],
                    vector=all_embeddings[i + j],
                    payload={"text": batch[j]["text"], "metadata": batch[j]["metadata"]}
                )
                for j in range(len(batch))
            ]
            await vector_store.upsert(coll_name, records)

        # Wait for indexing to settle
        await asyncio.sleep(1)

        size_results = {}

        for top_k in request_config.search_top_k:
            latencies_ms = []

            for q in range(request_config.search_iterations):
                query_text = generate_test_text(50, seed=99999 + q)
                t0 = time.perf_counter()
                query_result = await embedder.embed([query_text])
                query_vector = query_result.embeddings[0]

                t_search = time.perf_counter()
                await vector_store.search(coll_name, query_vector, limit=top_k)
                search_ms = (time.perf_counter() - t_search) * 1000
                latencies_ms.append(search_ms)

            size_results[str(top_k)] = {
                "top_k": top_k,
                "search_only_ms": _percentiles(latencies_ms),
                "queries_per_second": round(request_config.search_iterations / (sum(latencies_ms) / 1000), 2) if latencies_ms else 0,
                "iterations": request_config.search_iterations,
            }

        results[str(size)] = {
            "collection_size": size,
            "search_results": size_results,
        }

        if request_config.cleanup:
            await _cleanup_collection(vector_store, coll_name)

    return {"search_results": results}


async def run_update_benchmark(
    embedder, vector_store, request_config: EmbeddingBenchmarkRequest, bench_id: str
) -> Dict[str, Any]:
    """Benchmark update (re-embed + re-upsert) latency at various collection sizes."""
    from modules.rag.vector_stores.base import VectorRecord

    results = {}
    model_info = embedder.get_model_info()
    vector_size = model_info.dimensions
    update_count = min(100, min(request_config.collection_sizes))

    for size in request_config.collection_sizes:
        if size < update_count:
            continue
        coll_name = _collection_name(bench_id) + f"_upd_{size}"
        await _ensure_collection(vector_store, coll_name, vector_size)

        # Populate
        docs = generate_test_documents(size, word_count=200)
        embed_batch_size = 64
        all_embeddings = []
        for i in range(0, len(docs), embed_batch_size):
            batch_texts = [d["text"] for d in docs[i:i + embed_batch_size]]
            embed_result = await embedder.embed(batch_texts)
            all_embeddings.extend(embed_result.embeddings)

        upsert_batch_size = 100
        for i in range(0, len(docs), upsert_batch_size):
            batch = docs[i:i + upsert_batch_size]
            records = [
                VectorRecord(
                    id=batch[j]["id"],
                    vector=all_embeddings[i + j],
                    payload={"text": batch[j]["text"], "metadata": batch[j]["metadata"]}
                )
                for j in range(len(batch))
            ]
            await vector_store.upsert(coll_name, records)

        # Update a subset with new text
        update_docs = docs[:update_count]
        new_texts = [generate_test_text(200, seed=88888 + i) for i in range(update_count)]

        embed_latencies = []
        upsert_latencies = []
        total_start = time.perf_counter()

        # Re-embed
        for i in range(0, update_count, 32):
            batch = new_texts[i:i + 32]
            t0 = time.perf_counter()
            embed_result = await embedder.embed(batch)
            embed_latencies.append((time.perf_counter() - t0) * 1000)
            new_embs = embed_result.embeddings

            records = [
                VectorRecord(
                    id=update_docs[i + j]["id"],
                    vector=new_embs[j],
                    payload={"text": new_texts[i + j], "metadata": update_docs[i + j]["metadata"]}
                )
                for j in range(len(batch))
            ]
            t0 = time.perf_counter()
            await vector_store.upsert(coll_name, records)
            upsert_latencies.append((time.perf_counter() - t0) * 1000)

        total_elapsed = (time.perf_counter() - total_start) * 1000

        results[str(size)] = {
            "collection_size": size,
            "documents_updated": update_count,
            "total_time_ms": round(total_elapsed, 2),
            "re_embed_time_ms": _percentiles(embed_latencies),
            "upsert_time_ms": _percentiles(upsert_latencies),
            "avg_ms_per_update": round(total_elapsed / update_count, 3),
            "updates_per_second": round(update_count / (total_elapsed / 1000), 2),
        }

        if request_config.cleanup:
            await _cleanup_collection(vector_store, coll_name)

    return {"update_results": results}


async def run_deletion_benchmark(
    embedder, vector_store, request_config: EmbeddingBenchmarkRequest, bench_id: str
) -> Dict[str, Any]:
    """Benchmark delete latency (single and batch) at various collection sizes."""
    from modules.rag.vector_stores.base import VectorRecord

    results = {}
    model_info = embedder.get_model_info()
    vector_size = model_info.dimensions
    delete_count = min(200, min(request_config.collection_sizes))

    for size in request_config.collection_sizes:
        if size < delete_count:
            continue
        coll_name = _collection_name(bench_id) + f"_del_{size}"
        await _ensure_collection(vector_store, coll_name, vector_size)

        # Populate
        docs = generate_test_documents(size, word_count=200)
        embed_batch_size = 64
        all_embeddings = []
        for i in range(0, len(docs), embed_batch_size):
            batch_texts = [d["text"] for d in docs[i:i + embed_batch_size]]
            embed_result = await embedder.embed(batch_texts)
            all_embeddings.extend(embed_result.embeddings)

        upsert_batch_size = 100
        for i in range(0, len(docs), upsert_batch_size):
            batch = docs[i:i + upsert_batch_size]
            records = [
                VectorRecord(
                    id=batch[j]["id"],
                    vector=all_embeddings[i + j],
                    payload={"text": batch[j]["text"], "metadata": batch[j]["metadata"]}
                )
                for j in range(len(batch))
            ]
            await vector_store.upsert(coll_name, records)

        ids_to_delete = [d["id"] for d in docs[:delete_count]]

        # Single deletes (first 20)
        single_latencies = []
        single_count = min(20, delete_count)
        for doc_id in ids_to_delete[:single_count]:
            t0 = time.perf_counter()
            await vector_store.delete(coll_name, [doc_id])
            single_latencies.append((time.perf_counter() - t0) * 1000)

        # Batch deletes (remaining)
        batch_latencies = []
        batch_delete_size = 50
        remaining_ids = ids_to_delete[single_count:]
        for i in range(0, len(remaining_ids), batch_delete_size):
            batch = remaining_ids[i:i + batch_delete_size]
            t0 = time.perf_counter()
            await vector_store.delete(coll_name, batch)
            batch_latencies.append((time.perf_counter() - t0) * 1000)

        results[str(size)] = {
            "collection_size": size,
            "single_delete_ms": _percentiles(single_latencies),
            "batch_delete_ms": _percentiles(batch_latencies),
            "batch_size": batch_delete_size,
            "total_deleted": delete_count,
            "deletes_per_second_single": round(single_count / (sum(single_latencies) / 1000), 2) if single_latencies else 0,
            "deletes_per_second_batch": round(len(remaining_ids) / (sum(batch_latencies) / 1000), 2) if batch_latencies else 0,
        }

        if request_config.cleanup:
            await _cleanup_collection(vector_store, coll_name)

    return {"deletion_results": results}


async def run_full_pipeline(
    embedder, vector_store, request_config: EmbeddingBenchmarkRequest, bench_id: str
) -> Dict[str, Any]:
    """Full lifecycle benchmark: insert -> search -> update -> delete at each size."""
    from modules.rag.vector_stores.base import VectorRecord

    results = {}
    model_info = embedder.get_model_info()
    vector_size = model_info.dimensions

    for size in request_config.collection_sizes:
        coll_name = _collection_name(bench_id) + f"_pipe_{size}"
        await _ensure_collection(vector_store, coll_name, vector_size)

        phase_timings = {}

        # Phase 1: Insert
        docs = generate_test_documents(size, word_count=200)
        all_ids = [d["id"] for d in docs]

        t0 = time.perf_counter()
        embed_batch_size = 64
        all_embeddings = []
        for i in range(0, len(docs), embed_batch_size):
            batch_texts = [d["text"] for d in docs[i:i + embed_batch_size]]
            embed_result = await embedder.embed(batch_texts)
            all_embeddings.extend(embed_result.embeddings)

        upsert_batch_size = 100
        for i in range(0, len(docs), upsert_batch_size):
            batch = docs[i:i + upsert_batch_size]
            records = [
                VectorRecord(
                    id=batch[j]["id"],
                    vector=all_embeddings[i + j],
                    payload={"text": batch[j]["text"], "metadata": batch[j]["metadata"]}
                )
                for j in range(len(batch))
            ]
            await vector_store.upsert(coll_name, records)
        phase_timings["insert_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        await asyncio.sleep(1)

        # Phase 2: Search
        search_latencies = []
        t0 = time.perf_counter()
        for q in range(request_config.search_iterations):
            query_text = generate_test_text(50, seed=77777 + q)
            query_result = await embedder.embed([query_text])
            query_vector = query_result.embeddings[0]
            ts = time.perf_counter()
            await vector_store.search(coll_name, query_vector, limit=10)
            search_latencies.append((time.perf_counter() - ts) * 1000)
        phase_timings["search_total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        phase_timings["search_per_query_ms"] = _percentiles(search_latencies)

        # Phase 3: Update (first 100 docs)
        update_count = min(100, size)
        t0 = time.perf_counter()
        update_docs = docs[:update_count]
        for i in range(0, update_count, 32):
            batch = update_docs[i:i + 32]
            new_texts = [generate_test_text(200, seed=66666 + i + j) for j in range(len(batch))]
            embed_result = await embedder.embed(new_texts)
            records = [
                VectorRecord(
                    id=batch[j]["id"],
                    vector=embed_result.embeddings[j],
                    payload={"text": new_texts[j], "metadata": batch[j]["metadata"]}
                )
                for j in range(len(batch))
            ]
            await vector_store.upsert(coll_name, records)
        phase_timings["update_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Phase 4: Delete (first 50 docs)
        delete_count = min(50, size)
        t0 = time.perf_counter()
        await vector_store.delete(coll_name, all_ids[:delete_count])
        phase_timings["delete_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        phase_timings["total_pipeline_ms"] = round(
            phase_timings["insert_ms"] + phase_timings["search_total_ms"] +
            phase_timings["update_ms"] + phase_timings["delete_ms"], 2
        )

        results[str(size)] = {
            "collection_size": size,
            "phases": phase_timings,
        }

        if request_config.cleanup:
            await _cleanup_collection(vector_store, coll_name)

    return {"pipeline_results": results}


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/run")
async def run_benchmark(request: Request, config: EmbeddingBenchmarkRequest, background_tasks: BackgroundTasks):
    """Run an embedding benchmark. Returns immediately with an ID; results stored in DB."""
    bench_id = str(uuid.uuid4())[:8]
    embedder = _get_embedder(request)
    vector_store = _get_vector_store(request)

    # Check availability
    if not await embedder.is_available():
        raise HTTPException(status_code=503, detail="Embedding service is not available")
    if not await vector_store.is_healthy():
        raise HTTPException(status_code=503, detail="Vector store is not healthy")

    async def _run():
        try:
            async with _db_lock:
                conn = _db_connect()
                conn.execute(
                    "INSERT INTO embedding_benchmark_results (id, name, type, config, status) VALUES (?, ?, ?, ?, 'running')",
                    (bench_id, config.name, config.type.value, json.dumps(config.model_dump()))
                )
                conn.commit()
                conn.close()

            bench_config = {
                "type": config.type.value,
                "collection_sizes": config.collection_sizes,
                "text_lengths": config.text_lengths,
                "batch_sizes": config.batch_sizes,
                "search_top_k": config.search_top_k,
                "search_iterations": config.search_iterations,
                "runs": config.runs,
            }

            if config.type == EmbeddingBenchmarkType.EMBEDDING_THROUGHPUT:
                metrics = await run_embedding_throughput(embedder, config)
            elif config.type == EmbeddingBenchmarkType.INSERTION:
                metrics = await run_insertion_benchmark(embedder, vector_store, config, bench_id)
            elif config.type == EmbeddingBenchmarkType.SEARCH:
                metrics = await run_search_benchmark(embedder, vector_store, config, bench_id)
            elif config.type == EmbeddingBenchmarkType.UPDATE:
                metrics = await run_update_benchmark(embedder, vector_store, config, bench_id)
            elif config.type == EmbeddingBenchmarkType.DELETION:
                metrics = await run_deletion_benchmark(embedder, vector_store, config, bench_id)
            elif config.type == EmbeddingBenchmarkType.FULL_PIPELINE:
                metrics = await run_full_pipeline(embedder, vector_store, config, bench_id)
            else:
                metrics = {"error": f"Unknown benchmark type: {config.type}"}

            metrics["model_info"] = {
                "name": embedder.get_model_info().name,
                "dimensions": embedder.get_model_info().dimensions,
                "provider": embedder.get_model_info().provider,
            }

            await _save_result(bench_id, config.name, config.type.value, bench_config, metrics)

        except Exception as e:
            import traceback
            logger.error(f"Benchmark {bench_id} failed: {e}\n{traceback.format_exc()}")
            try:
                async with _db_lock:
                    conn = _db_connect()
                    conn.execute(
                        "UPDATE embedding_benchmark_results SET status = 'failed', metrics = ? WHERE id = ?",
                        (json.dumps({"error": str(e)}), bench_id)
                    )
                    conn.commit()
                    conn.close()
            except Exception as db_err:
                logger.error(f"Also failed to update benchmark status in DB: {db_err}")

    background_tasks.add_task(_run)
    return {"benchmark_id": bench_id, "status": "started", "type": config.type.value}


@router.post("/run-sync")
async def run_benchmark_sync(request: Request, config: EmbeddingBenchmarkRequest):
    """Run a benchmark synchronously and wait for results."""
    bench_id = str(uuid.uuid4())[:8]
    embedder = _get_embedder(request)
    vector_store = _get_vector_store(request)

    if not await embedder.is_available():
        raise HTTPException(status_code=503, detail="Embedding service is not available")
    if not await vector_store.is_healthy():
        raise HTTPException(status_code=503, detail="Vector store is not healthy")

    bench_config = {
        "type": config.type.value,
        "collection_sizes": config.collection_sizes,
        "text_lengths": config.text_lengths,
        "batch_sizes": config.batch_sizes,
        "search_top_k": config.search_top_k,
        "search_iterations": config.search_iterations,
        "runs": config.runs,
    }

    try:
        if config.type == EmbeddingBenchmarkType.EMBEDDING_THROUGHPUT:
            metrics = await run_embedding_throughput(embedder, config)
        elif config.type == EmbeddingBenchmarkType.INSERTION:
            metrics = await run_insertion_benchmark(embedder, vector_store, config, bench_id)
        elif config.type == EmbeddingBenchmarkType.SEARCH:
            metrics = await run_search_benchmark(embedder, vector_store, config, bench_id)
        elif config.type == EmbeddingBenchmarkType.UPDATE:
            metrics = await run_update_benchmark(embedder, vector_store, config, bench_id)
        elif config.type == EmbeddingBenchmarkType.DELETION:
            metrics = await run_deletion_benchmark(embedder, vector_store, config, bench_id)
        elif config.type == EmbeddingBenchmarkType.FULL_PIPELINE:
            metrics = await run_full_pipeline(embedder, vector_store, config, bench_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown benchmark type: {config.type}")

        metrics["model_info"] = {
            "name": embedder.get_model_info().name,
            "dimensions": embedder.get_model_info().dimensions,
            "provider": embedder.get_model_info().provider,
        }

        await _save_result(bench_id, config.name, config.type.value, bench_config, metrics)

        return {
            "benchmark_id": bench_id,
            "status": "completed",
            "type": config.type.value,
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(f"Benchmark {bench_id} failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.get("/results")
async def list_results(
    type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List benchmark results, optionally filtered by type."""
    conn = _db_connect()
    conn.row_factory = sqlite3.Row

    if type:
        rows = conn.execute(
            "SELECT * FROM embedding_benchmark_results WHERE type = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (type, limit, offset)
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM embedding_benchmark_results WHERE type = ?", (type,)
        ).fetchone()[0]
    else:
        rows = conn.execute(
            "SELECT * FROM embedding_benchmark_results ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM embedding_benchmark_results").fetchone()[0]

    results = [dict(r) for r in rows]
    conn.close()
    return {"total": total, "results": results}


@router.get("/results/{bench_id}")
async def get_result(bench_id: str):
    """Get a specific benchmark result by ID."""
    conn = _db_connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM embedding_benchmark_results WHERE id = ?", (bench_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Benchmark result not found")

    return dict(row)


@router.delete("/results/{bench_id}")
async def delete_result(bench_id: str):
    """Delete a benchmark result."""
    conn = _db_connect()
    conn.execute("DELETE FROM embedding_benchmark_results WHERE id = ?", (bench_id,))
    conn.commit()
    conn.close()
    return {"deleted": bench_id}


@router.delete("/results")
async def delete_all_results():
    """Delete all benchmark results."""
    conn = _db_connect()
    conn.execute("DELETE FROM embedding_benchmark_results")
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM embedding_benchmark_results").fetchone()[0]
    conn.close()
    return {"deleted_all": True, "remaining": count}


@router.get("/types")
async def list_benchmark_types():
    """List available benchmark types with descriptions."""
    return {
        "types": [
            {
                "id": "embedding_throughput",
                "name": "Embedding Throughput",
                "description": "Raw embedding generation speed across batch sizes and text lengths",
            },
            {
                "id": "insertion",
                "name": "Insertion",
                "description": "Embed + upsert documents into Qdrant at various collection sizes",
            },
            {
                "id": "search",
                "name": "Search",
                "description": "Vector search latency at various collection sizes and top_k values",
            },
            {
                "id": "update",
                "name": "Update",
                "description": "Re-embed + re-upsert latency for existing documents",
            },
            {
                "id": "deletion",
                "name": "Deletion",
                "description": "Single and batch delete latency at various collection sizes",
            },
            {
                "id": "full_pipeline",
                "name": "Full Pipeline",
                "description": "Complete lifecycle: insert -> search -> update -> delete",
            },
        ]
    }

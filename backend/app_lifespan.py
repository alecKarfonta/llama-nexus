import os
import json
import asyncio
from types import SimpleNamespace
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager

from enhanced_logger import enhanced_logger as logger
from app_state import (
    manager,
    vllm_manager,
    download_manager,
    embedding_manager,
    stt_manager,
    streaming_stt_manager,
    tts_manager,
    create_embedder,
    _merge_and_persist_config,
)

# Optional Imports (Same as original main.py)
try:
    from modules.rag.document_manager import DocumentManager
    from modules.rag.graph_rag import GraphRAG
    from modules.rag.vector_stores.qdrant_store import QdrantStore
    from modules.rag.discovery import DocumentDiscovery
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    logger.warning(f"RAG modules not available: {e}")

try:
    from modules.workflow.storage import WorkflowStorage
    from modules.workflow.engine import WorkflowEngine
    WORKFLOW_AVAILABLE = True
except ImportError as e:
    WORKFLOW_AVAILABLE = False
    logger.warning(f"Workflow modules not available: {e}. Check sqlmodel dependency.")

# Optional module imports for app.state
try:
    from modules.conversation_store import conversation_store
except ImportError:
    try:
        from conversation_store import conversation_store
    except ImportError:
        conversation_store = None

try:
    from modules.model_registry import model_registry
except ImportError:
    try:
        from model_registry import model_registry
    except ImportError:
        model_registry = None

try:
    from modules.prompt_library import prompt_library
except ImportError:
    try:
        from prompt_library import prompt_library
    except ImportError:
        prompt_library = None

try:
    from modules.benchmark import benchmark_runner
except ImportError:
    try:
        from benchmark import benchmark_runner
    except ImportError:
        benchmark_runner = None

try:
    from modules.batch_processor import batch_processor
except ImportError:
    try:
        from batch_processor import batch_processor
    except ImportError:
        batch_processor = None

try:
    from modules.token_tracker import token_tracker
except ImportError:
    try:
        from token_tracker import token_tracker
    except ImportError:
        token_tracker = None

DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "nomic-embed-text-v1.5")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LlamaCPP Management API")

    # Load saved config if exists and merge with defaults
    config_file = Path("/tmp/llamacpp_config.json")
    if config_file.exists():
        with open(config_file) as f:
            saved_config = json.load(f)
            # Merge saved config with default config to ensure ONLY required fields exist
            # Optional fields (like sampling parameters) should remain None/absent if cleared by user
            default_config = manager.load_default_config()
            
            # Define required fields that must always have values
            required_fields = {
                "model": ["name", "variant", "context_size", "gpu_layers", "n_cpu_moe"],
                "server": ["host", "port", "api_key", "metrics"],
                "template": ["directory"],
                "execution": ["mode", "cuda_devices"],
                "performance": ["threads", "batch_size", "ubatch_size"]
            }
            
            for category in default_config:
                if category in saved_config:
                    # Merge category, but only restore required fields if missing/None
                    for key, default_value in default_config[category].items():
                        if key not in saved_config[category] or saved_config[category][key] is None:
                            # Only add back if it's a required field
                            if category in required_fields and key in required_fields[category]:
                                saved_config[category][key] = default_value
                            # Otherwise, leave it as None/absent so llama-server uses its defaults
                else:
                    # Add missing category entirely, but filter to required fields only
                    if category in required_fields:
                        saved_config[category] = {
                            k: v for k, v in default_config[category].items()
                            if k in required_fields[category]
                        }
                    else:
                        # For categories without required fields, add empty dict
                        saved_config[category] = {}
            manager.config = saved_config

    # Initialize RAG system
    app.state.rag_available = RAG_AVAILABLE
    app.state.create_embedder = create_embedder
    app.state.embedding_manager = embedding_manager
    app.state.stt_manager = stt_manager
    app.state.streaming_stt_manager = streaming_stt_manager
    app.state.tts_manager = tts_manager
    
    if RAG_AVAILABLE:
        try:
            rag_db_path = os.getenv("RAG_DB_PATH", "data/rag")
            os.makedirs(rag_db_path, exist_ok=True)

            # Initialize document manager
            app.state.document_manager = DocumentManager(f"{rag_db_path}/documents.db")
            await app.state.document_manager.initialize()

            # Initialize GraphRAG
            app.state.graph_rag = GraphRAG(f"{rag_db_path}/graph.db")
            await app.state.graph_rag.initialize()

            # Initialize Qdrant vector store
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            app.state.vector_store = QdrantStore(host=qdrant_host, port=qdrant_port)
            if await app.state.vector_store.connect():
                logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
            else:
                logger.warning("Failed to connect to Qdrant")
                app.state.vector_store = None

            # Initialize document discovery
            app.state.document_discovery = DocumentDiscovery(f"{rag_db_path}/discovery.db")
            await app.state.document_discovery.initialize()

            # Set up background document processing. FastAPI background tasks run in
            # this API process, so serialize indexing to avoid SQLite writer lock races.
            from routes import rag as rag_routes
            app.state.rag_processing_lock = asyncio.Lock()

            async def process_document_background(*args, **kwargs):
                async with app.state.rag_processing_lock:
                    await rag_routes.process_document_background(*args, **kwargs)

            app.state.process_document_background = process_document_background

            async def drain_pending_documents():
                """Continuously process pending RAG documents in the API process."""
                from modules.rag.document_manager import DocumentStatus

                interval_seconds = int(os.getenv("RAG_PENDING_WORKER_INTERVAL_SECONDS", "5"))
                batch_size = int(os.getenv("RAG_PENDING_WORKER_BATCH_SIZE", "1"))
                request_like = SimpleNamespace(app=app)

                logger.info(
                    "RAG pending document worker started "
                    f"(batch_size={batch_size}, interval={interval_seconds}s)"
                )

                while True:
                    try:
                        if not app.state.document_manager or not app.state.vector_store:
                            await asyncio.sleep(interval_seconds)
                            continue

                        docs, total = await app.state.document_manager.list_documents(
                            status=DocumentStatus.PENDING,
                            limit=batch_size,
                        )

                        if not docs:
                            await asyncio.sleep(interval_seconds)
                            continue

                        logger.info(f"RAG pending worker processing {len(docs)} of {total} pending documents")
                        for doc in docs:
                            await app.state.process_document_background(
                                request_like,
                                doc.id,
                                "semantic",
                                800,
                                80,
                                DEFAULT_EMBEDDING_MODEL,
                            )
                    except asyncio.CancelledError:
                        raise
                    except Exception as worker_error:
                        logger.error(f"RAG pending document worker error: {worker_error}")
                        await asyncio.sleep(interval_seconds)

            if os.getenv("RAG_AUTO_INDEX_PENDING", "true").lower() in ("true", "1", "yes"):
                app.state.rag_pending_worker_task = asyncio.create_task(drain_pending_documents())
            
            # Pre-warm the default embedding model to avoid cold start on first request
            # This loads the model into memory during startup (~4-5s) instead of first query
            try:
                logger.info(f"Pre-warming embedding model: {DEFAULT_EMBEDDING_MODEL}")
                warmup_embedder = create_embedder(model_name=DEFAULT_EMBEDDING_MODEL)
                # Trigger model load by embedding a dummy query
                await warmup_embedder.embed(["warmup query"])
                logger.info(f"Embedding model pre-warmed and cached: {DEFAULT_EMBEDDING_MODEL}")
            except Exception as warmup_error:
                logger.warning(f"Failed to pre-warm embedding model: {warmup_error}")
            
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            app.state.document_manager = None
            app.state.graph_rag = None
            app.state.vector_store = None
            app.state.document_discovery = None

    # Initialize Workflow system
    app.state.workflow_available = WORKFLOW_AVAILABLE
    if WORKFLOW_AVAILABLE:
        try:
            workflow_db_path = os.getenv("WORKFLOW_DB_PATH", "data/workflows.db")
            os.makedirs(os.path.dirname(workflow_db_path), exist_ok=True)

            app.state.workflow_storage = WorkflowStorage(workflow_db_path)
            app.state.workflow_engine = WorkflowEngine(app.state.workflow_storage)

            logger.info("Workflow system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Workflow system: {e}")
            app.state.workflow_storage = None
            app.state.workflow_engine = None

    # Set module instances on app.state for route access
    app.state.conversation_store = conversation_store
    app.state.model_registry = model_registry
    app.state.prompt_library = prompt_library
    app.state.benchmark_runner = benchmark_runner
    app.state.batch_processor = batch_processor
    app.state.manager = manager
    app.state.vllm_manager = vllm_manager
    app.state.download_manager = download_manager
    app.state.merge_and_persist_config = _merge_and_persist_config
    app.state.token_tracker = token_tracker

    # Initialize deploy profiles
    try:
        from modules.deploy_profiles import DeployProfileStore
        profile_db_path = os.getenv("PROFILE_DB_PATH", "data/deploy_profiles.db")
        app.state.profile_store = DeployProfileStore(profile_db_path)
        logger.info("Deploy profile store initialized")
    except Exception as e:
        logger.warning(f"Deploy profiles not available: {e}")
        app.state.profile_store = None

    # Initialize Reddit crawler scheduler
    try:
        from modules.finetuning.reddit_scheduler import get_reddit_scheduler
        app.state.reddit_scheduler = get_reddit_scheduler()
        if app.state.reddit_scheduler.config.enabled:
            await app.state.reddit_scheduler.start()
            logger.info("Reddit crawler scheduler started (auto-enabled)")
        else:
            logger.info("Reddit crawler scheduler initialized (disabled)")
    except Exception as e:
        logger.warning(f"Reddit crawler scheduler not available: {e}")
        app.state.reddit_scheduler = None

    yield

    # Shutdown
    logger.info("Shutting down LlamaCPP Management API")
    rag_pending_worker_task = getattr(app.state, "rag_pending_worker_task", None)
    if rag_pending_worker_task:
        rag_pending_worker_task.cancel()
        try:
            await rag_pending_worker_task
        except asyncio.CancelledError:
            pass

    if not manager.use_docker and manager.process and manager.process.poll() is None:
        await manager.stop()
    
    # Cleanup RAG
    if RAG_AVAILABLE and hasattr(app.state, 'vector_store') and app.state.vector_store:
        await app.state.vector_store.disconnect()

    # Cleanup GraphRAG shared HTTP client
    try:
        from routes.graphrag.helpers import close_http_client
        await close_http_client()
    except Exception:
        pass


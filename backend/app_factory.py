import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from enhanced_logger import enhanced_logger as logger
from app_state import manager

# Import token tracker
try:
    from modules.token_tracker import token_tracker
    from modules.token_middleware import TokenUsageMiddleware
except ImportError:
    token_tracker = None

# Import all route modules
import routes.models as models
import routes.templates as templates
import routes.tokens as tokens
import routes.benchmark as benchmark
import routes.conversations as conversations
import routes.registry as registry
import routes.prompts as prompts
import routes.batch as batch
import routes.service as service
import routes.system as system
import routes.llamacpp as llamacpp
import routes.websockets as websockets

# Optional routes
try:
    import routes.rag as rag
    import routes.graphrag as graphrag
    import routes.workflows as workflows
except ImportError:
    rag = graphrag = workflows = None

# Custom Logging Middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            # Skip logging for health checks to reduce noise
            if request.url.path == "/api/v1/health":
                return await call_next(request)
                
            response = await call_next(request)
            
            # Log non-200 responses automatically
            if response.status_code >= 400:
                logger.warning(
                    f"{request.method} {request.url.path} - "
                    f"Status: {response.status_code}"
                )
            return response
        except Exception as e:
            # Log unhandled exceptions with traceback
            logger.error(
                f"Unhandled exception in {request.method} {request.url.path}: {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

def create_app(lifespan_handler) -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title="LlamaCPP Management API",
        description="API for managing LlamaCPP model deployments",
        version="2.0.0",
        lifespan=lifespan_handler
    )

    # Add middlewares
    app.add_middleware(LoggingMiddleware)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if token_tracker:
        app.add_middleware(TokenUsageMiddleware)

    # Include basic routers
    app.include_router(system.router)
    app.include_router(llamacpp.router)
    app.include_router(websockets.router)
    app.include_router(models.router)
    app.include_router(templates.router)
    app.include_router(tokens.router)
    app.include_router(benchmark.router)
    app.include_router(conversations.router)
    app.include_router(registry.router)
    app.include_router(prompts.router)
    app.include_router(batch.router)
    app.include_router(service.router)

    # Include optional routers
    if rag is not None:
        app.include_router(rag.router)
    if graphrag is not None:
        app.include_router(graphrag.router)
    if workflows is not None:
        app.include_router(workflows.router)

    # Compatibility health check (moved from main.py)
    @app.get("/health")
    async def root_health():
        from datetime import datetime
        status = manager.get_status()
        return {
            "status": "up",
            "timestamp": datetime.now().isoformat(),
            "service": {
                "running": status["running"],
                "backend": "docker" if manager.use_docker else "subprocess",
            }
        }

    # Initialize MCP
    try:
        from mcp_server import create_mcp_endpoints
        app = create_mcp_endpoints(app)
        logger.info("MCP endpoints initialized")
    except ImportError as e:
        logger.warning(f"Failed to initialize MCP endpoints: {e}")

    return app

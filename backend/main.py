"""
Enhanced FastAPI Backend for LlamaCPP Model Management
Provides APIs for managing llamacpp instances with Docker support

Entrypoint for the Llama-Nexus backend. Most logic has been decomposed
into focused modules under `app_state.py`, `app_factory.py`, and `modules/managers/`.
"""

import sys
import uvicorn
from contextlib import asynccontextmanager

from app_factory import create_app
from app_lifespan import lifespan

# Create the application
app = create_app(lifespan)

if __name__ == "__main__":
    # Configure custom uvicorn logging to suppress default logs
    log_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "minimal": {
                "format": "%(message)s"
            }
        },
        "handlers": {
            "null": {
                "class": "logging.NullHandler",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["null"], "level": "WARNING"},
            "uvicorn.error": {"handlers": ["null"], "level": "WARNING"},
            "uvicorn.access": {"handlers": ["null"], "level": "WARNING"},
        },
    }
    
    # Run with custom logging configuration
    uvicorn.run("main:app", host="0.0.0.0", port=8700, log_config=log_config)

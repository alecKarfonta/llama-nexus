"""
Pytest configuration and fixtures for fine-tuning tests.

Sets up the Python path to properly import backend modules.
Mocks modules that require Docker environment (database connections, etc.)

NOTE: Tests should be run inside Docker with:
    docker compose run --rm backend pytest tests/test_finetuning.py -v
    
Or with all dependencies installed locally.
"""

import sys
import os
from pathlib import Path
from types import ModuleType

import pytest

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Set environment variables for testing
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("STORAGE_DIR", "/tmp/test_storage")
os.environ.setdefault("DATA_DIR", "/tmp/test_data")


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    try:
        import pydantic
    except ImportError:
        missing.append("pydantic")
    try:
        import httpx
    except ImportError:
        missing.append("httpx")
    return missing


# Check dependencies at module load time
_missing_deps = check_dependencies()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_deps: mark test as requiring all dependencies"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require dependencies if they're missing."""
    if _missing_deps:
        skip_marker = pytest.mark.skip(
            reason=f"Missing dependencies: {', '.join(_missing_deps)}. "
            f"Run tests in Docker: docker compose run --rm backend pytest tests/"
        )
        for item in items:
            item.add_marker(skip_marker)


# Mock enhanced_logger for cleaner test output
class MockLogger:
    def info(self, msg, *args, **kwargs): pass
    def debug(self, msg, *args, **kwargs): pass
    def warning(self, msg, *args, **kwargs): pass
    def error(self, msg, *args, **kwargs): pass
    def critical(self, msg, *args, **kwargs): pass
    def bind(self, **kwargs): return self


# Only set up mocks if dependencies are available
if not _missing_deps:
    # Create the mock module properly
    enhanced_logger_module = ModuleType('enhanced_logger')
    enhanced_logger_module.enhanced_logger = MockLogger()
    sys.modules['enhanced_logger'] = enhanced_logger_module

    # Create a stub modules package that allows submodule imports
    # but doesn't have side effects from __init__.py
    modules_pkg = ModuleType('modules')
    modules_pkg.__path__ = [str(backend_dir / 'modules')]
    modules_pkg.__file__ = str(backend_dir / 'modules' / '__init__.py')
    sys.modules['modules'] = modules_pkg

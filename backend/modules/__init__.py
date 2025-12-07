"""
LlamaCPP Management API modules
"""

import os
import sys

# Add the current directory to the path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .token_tracker import token_tracker
    from .token_middleware import TokenUsageMiddleware
except ImportError as e:
    print(f"Error importing token modules: {e}")
    print(f"Current sys.path: {sys.path}")
    token_tracker = None
    
    class TokenUsageMiddleware:
        async def dispatch(self, request, call_next):
            return await call_next(request)

try:
    from .conversation_store import conversation_store
except ImportError as e:
    print(f"Error importing conversation_store: {e}")
    conversation_store = None

try:
    from .event_bus import event_bus
except ImportError as e:
    print(f"Error importing event_bus: {e}")
    event_bus = None

try:
    from .model_registry import model_registry
except ImportError as e:
    print(f"Error importing model_registry: {e}")
    model_registry = None

try:
    from .prompt_library import prompt_library
except ImportError as e:
    print(f"Error importing prompt_library: {e}")
    prompt_library = None

try:
    from .benchmark import benchmark_runner
except ImportError as e:
    print(f"Error importing benchmark: {e}")
    benchmark_runner = None

try:
    from .batch_processor import batch_processor
except ImportError as e:
    print(f"Error importing batch_processor: {e}")
    batch_processor = None

__all__ = ['token_tracker', 'TokenUsageMiddleware', 'conversation_store', 'event_bus', 'model_registry', 'prompt_library', 'benchmark_runner', 'batch_processor']

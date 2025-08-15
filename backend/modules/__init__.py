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
    # Create fallback objects to prevent crashes
    token_tracker = None
    
    class TokenUsageMiddleware:
        async def dispatch(self, request, call_next):
            return await call_next(request)

__all__ = ['token_tracker', 'TokenUsageMiddleware']

"""
Token Usage Tracking Middleware
Intercepts API requests and responses to track token usage
"""

import json
import uuid
from typing import Dict, Any, Callable, Awaitable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

try:
    # In Docker image, files are copied to /app root, so absolute import works
    from token_tracker import token_tracker  # type: ignore
except ImportError:
    try:
        # Local dev path (package import)
        from .token_tracker import token_tracker  # type: ignore
    except ImportError:
        # Create a dummy token_tracker if import fails
        import logging
        logging.warning("Failed to import token_tracker, using dummy implementation")
        
        class DummyTokenTracker:
            def record_token_usage(self, *args, **kwargs):
                return True
        
        token_tracker = DummyTokenTracker()

# Configure logging
logger = logging.getLogger("token_middleware")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

class TokenUsageMiddleware(BaseHTTPMiddleware):
    """Middleware to track token usage from API requests"""
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Process the request and track token usage"""
        
        # Skip non-completion endpoints
        if not self._is_completion_endpoint(request.url.path):
            return await call_next(request)
        
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        
        # Store original request body
        request_body = await self._get_request_body(request)
        
        # Process the request
        response = await call_next(request)
        
        # Extract token usage from response
        await self._extract_and_record_token_usage(request, response, request_body, request_id)
        
        return response
    
    def _is_completion_endpoint(self, path: str) -> bool:
        """Check if the path is a completion endpoint"""
        completion_endpoints = [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/generate",
        ]
        return any(path.endswith(endpoint) for endpoint in completion_endpoints)
    
    async def _get_request_body(self, request: Request) -> Dict[str, Any]:
        """Get the request body as a dictionary"""
        try:
            body = await request.body()
            if body:
                return json.loads(body)
            return {}
        except Exception as e:
            logger.warning(f"Error parsing request body: {e}")
            return {}
    
    async def _extract_and_record_token_usage(
        self, 
        request: Request, 
        response: Response,
        request_body: Dict[str, Any],
        request_id: str
    ) -> None:
        """Extract token usage from response and record it"""
        try:
            # Get response body
            response_body = self._get_response_body(response)
            if not response_body:
                return
            
            # Extract token usage
            usage = response_body.get("usage", {})
            if not usage:
                return
            
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            if prompt_tokens == 0 and completion_tokens == 0:
                return  # No tokens to record
            
            # Extract model information
            model_id = response_body.get("model", request_body.get("model", "unknown"))
            
            # Extract endpoint information
            endpoint = request.url.path
            
            # Extract user information
            user_id = request_body.get("user", None)
            
            # Record token usage
            token_tracker.record_token_usage(
                model_id=model_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_name=self._get_model_name(model_id),
                request_id=request_id,
                user_id=user_id,
                endpoint=endpoint,
                metadata={
                    "ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent"),
                    "stream": request_body.get("stream", False),
                }
            )
            
            logger.debug(f"Recorded token usage: {prompt_tokens} prompt, {completion_tokens} completion for model {model_id}")
            
        except Exception as e:
            logger.error(f"Error extracting and recording token usage: {e}")
    
    def _get_response_body(self, response: Response) -> Dict[str, Any]:
        """Get the response body as a dictionary"""
        try:
            # Access the private _body attribute of the response
            # This is a hack but necessary since FastAPI doesn't provide a way to access the response body
            if hasattr(response, "_body"):
                body = response._body
                if isinstance(body, bytes):
                    return json.loads(body.decode("utf-8"))
            return {}
        except Exception as e:
            logger.warning(f"Error parsing response body: {e}")
            return {}
    
    def _get_model_name(self, model_id: str) -> Optional[str]:
        """Get a human-readable model name from the model ID"""
        # Extract model name from model ID
        # This is a simple implementation - enhance as needed
        if "/" in model_id:
            # Format like "owner/repo"
            return model_id.split("/")[-1]
        return model_id

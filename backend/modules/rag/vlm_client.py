"""
VLM (Vision-Language Model) Client for Visual Description

Provides async interface to call a vision-language model endpoint
for describing images extracted from documents.
"""

import os
import base64
import logging
import httpx
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# VLM Configuration from environment
VLM_ENDPOINT_URL = os.environ.get("VLM_ENDPOINT_URL", "http://llamacpp-api:8080/v1")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "placeholder-api-key")
VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "Qwen_Qwen3-VL-4B-Thinking")
VLM_ENABLED = os.environ.get("VLM_ENABLED", "true").lower() in ("true", "1", "yes")

# Default prompt for describing visuals
DEFAULT_VISUAL_PROMPT = """Describe this image in detail. If it's a chart, graph, or diagram:
1. Identify the type of visualization (bar chart, line graph, pie chart, flowchart, etc.)
2. Describe what data or information it presents
3. Note any key trends, patterns, or notable values
4. Mention any labels, legends, or titles visible

Be concise but comprehensive. Format your response as a single paragraph."""


class VLMClient:
    """
    Async client for calling Vision-Language Model endpoints.
    
    Uses OpenAI-compatible chat/completions API with image support.
    """
    
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 120.0
    ):
        self.endpoint_url = endpoint_url or VLM_ENDPOINT_URL
        self.api_key = api_key or VLM_API_KEY
        self.model_name = model_name or VLM_MODEL_NAME
        self.timeout = timeout
        
        # Ensure endpoint URL ends with /v1 for chat completions
        if not self.endpoint_url.endswith('/v1'):
            self.endpoint_url = self.endpoint_url.rstrip('/') + '/v1'
    
    @staticmethod
    def is_enabled() -> bool:
        """Check if VLM processing is enabled."""
        return VLM_ENABLED
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Read image file and encode to base64."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _get_mime_type(self, image_path: str) -> str:
        """Get MIME type from file extension."""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
        }
        return mime_types.get(ext, 'image/png')
    
    async def describe_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_tokens: int = 512
    ) -> Optional[str]:
        """
        Send image to VLM endpoint and get textual description.
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt for description (uses default if None)
            max_tokens: Maximum tokens in response
            
        Returns:
            Description text or None if failed
        """
        if not VLM_ENABLED:
            logger.debug("VLM processing is disabled")
            return None
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        
        try:
            # Encode image
            image_base64 = self._encode_image_to_base64(image_path)
            mime_type = self._get_mime_type(image_path)
            
            # Build request payload (OpenAI vision format)
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt or DEFAULT_VISUAL_PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3  # Lower temperature for more factual descriptions
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.endpoint_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract description from response
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                    if content:
                        logger.info(f"VLM described image: {image_path[:50]}... ({len(content)} chars)")
                        return content.strip()
                
                logger.warning(f"Empty VLM response for image: {image_path}")
                return None
                
        except httpx.HTTPStatusError as e:
            logger.error(f"VLM HTTP error: {e.response.status_code} - {e.response.text[:200]}")
            return None
        except httpx.ConnectError as e:
            logger.warning(f"VLM connection failed (endpoint may be unavailable): {e}")
            return None
        except Exception as e:
            logger.error(f"VLM description failed for {image_path}: {e}")
            return None
    
    async def describe_image_bytes(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        prompt: Optional[str] = None,
        max_tokens: int = 512
    ) -> Optional[str]:
        """
        Describe image from raw bytes (no file save required).
        
        Args:
            image_bytes: Raw image data
            mime_type: MIME type of the image
            prompt: Custom prompt for description
            max_tokens: Maximum tokens in response
            
        Returns:
            Description text or None if failed
        """
        if not VLM_ENABLED:
            return None
        
        try:
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt or DEFAULT_VISUAL_PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3
            }
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.endpoint_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                    return content.strip() if content else None
                return None
                
        except Exception as e:
            logger.error(f"VLM bytes description failed: {e}")
            return None


# Singleton instance for convenience
_default_client: Optional[VLMClient] = None


def get_vlm_client() -> VLMClient:
    """Get the default VLM client instance."""
    global _default_client
    if _default_client is None:
        _default_client = VLMClient()
    return _default_client

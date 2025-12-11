"""
LLM Node Executors - Language model operations
"""

from typing import Dict, Any, List
import httpx
import os
from .base import NodeExecutor, ExecutionContext


class LLMChatExecutor(NodeExecutor):
    """Chat completion with local LLM"""
    
    node_type = "llm_chat"
    display_name = "LLM Chat"
    category = "llm"
    description = "Chat completion with local LLM"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        messages = inputs.get("messages", [])
        system_prompt = inputs.get("system")
        
        # Build messages array
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        if not messages:
            messages = [{"role": "user", "content": "Hello"}]
        
        # Get config
        model = self.get_config_value("model")
        temperature = self.get_config_value("temperature", 0.7)
        max_tokens = self.get_config_value("maxTokens", 2048)
        
        context.log(f"Calling local LLM with {len(messages)} messages")
        
        # Call local LLM endpoint
        llm_url = os.environ.get("LLM_API_URL", "http://llamacpp-api:8080")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{llm_url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            response.raise_for_status()
            data = response.json()
        
        # Extract response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})
        
        context.log(f"LLM response: {usage.get('total_tokens', 0)} tokens used")
        
        return {
            "response": message.get("content", ""),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }


class OpenAIChatExecutor(NodeExecutor):
    """Chat completion via OpenAI API"""
    
    node_type = "openai_chat"
    display_name = "OpenAI Chat"
    category = "llm"
    description = "Chat completion via OpenAI API"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        messages = inputs.get("messages", [])
        system_prompt = inputs.get("system")
        
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        if not messages:
            messages = [{"role": "user", "content": "Hello"}]
        
        # Get config
        model = self.get_config_value("model", "gpt-4")
        api_key = self.get_config_value("apiKey") or os.environ.get("OPENAI_API_KEY")
        temperature = self.get_config_value("temperature", 0.7)
        max_tokens = self.get_config_value("maxTokens")
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        context.log(f"Calling OpenAI {model} with {len(messages)} messages")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    **({"max_tokens": max_tokens} if max_tokens else {}),
                }
            )
            response.raise_for_status()
            data = response.json()
        
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})
        
        context.log(f"OpenAI response: {usage.get('total_tokens', 0)} tokens used")
        
        return {
            "response": message.get("content", ""),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }


class EmbeddingExecutor(NodeExecutor):
    """Generate text embeddings"""
    
    node_type = "embedding"
    display_name = "Generate Embedding"
    category = "llm"
    description = "Generate text embeddings"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        text = inputs.get("text", "")
        
        if not text:
            return {"embedding": []}
        
        model = self.get_config_value("model", "text-embedding-3-small")
        
        context.log(f"Generating embedding for text ({len(text)} chars)")
        
        # Try local embedding service first
        embed_url = os.environ.get("EMBED_API_URL", "http://llamacpp-embed:8080")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{embed_url}/v1/embeddings",
                    json={
                        "input": text,
                        "model": model,
                    }
                )
                response.raise_for_status()
                data = response.json()
            
            embedding = data.get("data", [{}])[0].get("embedding", [])
            context.log(f"Generated embedding with {len(embedding)} dimensions")
            
            return {"embedding": embedding}
            
        except Exception as e:
            context.log(f"Local embedding failed: {e}, trying OpenAI", level="warning")
            
            # Fallback to OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No embedding service available and no OpenAI API key")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": text,
                        "model": model,
                    }
                )
                response.raise_for_status()
                data = response.json()
            
            embedding = data.get("data", [{}])[0].get("embedding", [])
            return {"embedding": embedding}

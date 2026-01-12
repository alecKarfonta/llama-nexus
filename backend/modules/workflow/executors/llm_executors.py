"""
LLM Node Executors - Language model operations
"""

from typing import Dict, Any, List, Optional
import httpx
import os
import json
import asyncio
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
        
        # Validate model is configured
        if not model:
            raise ValueError(
                "No model selected. Please configure the 'model' field in the node properties. "
                "You can select from your deployed models in the property panel."
            )
        
        context.log(f"Calling local LLM '{model}' with {len(messages)} messages")
        
        # Call local LLM endpoint
        llm_url = os.environ.get("LLM_API_URL", "http://llamacpp-api:8080")
        
        try:
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
        except httpx.TimeoutException:
            raise Exception(
                f"Request to model '{model}' timed out after 120 seconds. "
                "The model may be processing a long response or is overloaded. "
                "Try reducing max_tokens or check model availability."
            )
        except httpx.ConnectError:
            raise Exception(
                f"Cannot connect to LLM service at {llm_url}. "
                "Please ensure the LLM service is running and accessible. "
                "Check docker-compose logs for the llamacpp-api service."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise Exception(
                    f"Model '{model}' not found. "
                    "Please check that this model is deployed and available. "
                    "Visit the Models page to see deployed models."
                )
            elif e.response.status_code == 503:
                raise Exception(
                    f"LLM service is unavailable (503). "
                    "The service may be starting up or overloaded. "
                    "Please wait a moment and try again."
                )
            else:
                error_detail = ""
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("error", {}).get("message", "")
                except:
                    pass
                raise Exception(
                    f"LLM API error ({e.response.status_code}): {error_detail or e.response.text[:200]}"
                )
        except Exception as e:
            raise Exception(
                f"Failed to call LLM '{model}': {str(e)}. "
                "Please check node configuration and model availability."
            )
        
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
            raise ValueError(
                "OpenAI API key is required. Please configure the 'apiKey' field in the node properties "
                "or set the OPENAI_API_KEY environment variable."
            )
        
        context.log(f"Calling OpenAI {model} with {len(messages)} messages")
        
        try:
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
        except httpx.TimeoutException:
            raise Exception(
                f"Request to OpenAI model '{model}' timed out after 120 seconds. "
                "The request may be taking too long. Try reducing max_tokens or simplifying your prompt."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise Exception(
                    "Invalid OpenAI API key. Please check your API key in the node configuration."
                )
            elif e.response.status_code == 404:
                raise Exception(
                    f"Model '{model}' not found. "
                    "This model may not exist or you may not have access to it. "
                    "Check your OpenAI account for available models."
                )
            elif e.response.status_code == 429:
                raise Exception(
                    "OpenAI rate limit exceeded. Please wait a moment and try again, "
                    "or check your OpenAI account quota."
                )
            else:
                error_detail = ""
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("error", {}).get("message", "")
                except:
                    pass
                raise Exception(
                    f"OpenAI API error ({e.response.status_code}): {error_detail or e.response.text[:200]}"
                )
        except Exception as e:
            raise Exception(
                f"Failed to call OpenAI model '{model}': {str(e)}"
            )
        
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


class OpenAIAPILLMExecutor(NodeExecutor):
    """Advanced OpenAI-compatible API endpoint with full configuration"""
    
    node_type = "openai_api_llm"
    display_name = "OpenAI API LLM"
    category = "llm"
    description = "Advanced OpenAI-compatible API endpoint with full configuration"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        messages = inputs.get("messages", [])
        system_prompt = inputs.get("system")
        tools = inputs.get("tools", [])
        
        # Build messages array
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        if not messages:
            messages = [{"role": "user", "content": "Hello"}]
        
        # Get connection settings
        endpoint = self.get_config_value("endpoint", "https://api.openai.com/v1/chat/completions")
        api_key = self.get_config_value("apiKey") or os.environ.get("OPENAI_API_KEY")
        organization = self.get_config_value("organization")
        
        if not api_key:
            raise ValueError(
                "API key is required. Please configure the 'apiKey' field in the node properties "
                "or set the OPENAI_API_KEY environment variable."
            )
        
        # Get model settings
        model = self.get_config_value("model", "gpt-4")
        
        # Get generation parameters
        temperature = self.get_config_value("temperature", 0.7)
        top_p = self.get_config_value("topP", 1.0)
        top_k = self.get_config_value("topK")
        max_tokens = self.get_config_value("maxTokens")
        presence_penalty = self.get_config_value("presencePenalty", 0)
        frequency_penalty = self.get_config_value("frequencyPenalty", 0)
        repetition_penalty = self.get_config_value("repetitionPenalty")
        
        # Thinking/reasoning for o1 models
        thinking_level = self.get_config_value("thinkingLevel", 0)
        
        # Advanced settings
        stream = self.get_config_value("stream", False)
        n = self.get_config_value("n", 1)
        stop = self.get_config_value("stop", [])
        logprobs = self.get_config_value("logprobs", False)
        top_logprobs = self.get_config_value("topLogprobs")
        seed = self.get_config_value("seed")
        response_format = self.get_config_value("responseFormat", {})
        
        # Request configuration
        timeout = self.get_config_value("timeout", 120)
        retry_attempts = self.get_config_value("retryAttempts", 2)
        retry_delay = self.get_config_value("retryDelay", 1000)
        custom_headers = self.get_config_value("customHeaders", {})
        
        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "n": n,
        }
        
        # Add optional parameters
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if top_k:
            payload["top_k"] = top_k
        if repetition_penalty:
            payload["repetition_penalty"] = repetition_penalty
        if stop:
            payload["stop"] = stop
        if logprobs:
            payload["logprobs"] = logprobs
        if top_logprobs:
            payload["top_logprobs"] = top_logprobs
        if seed is not None:
            payload["seed"] = seed
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format and response_format.get("type") != "text":
            payload["response_format"] = response_format
            
        # Add thinking level for o1 models
        if thinking_level > 0 and model.startswith("o1"):
            payload["thinking_level"] = thinking_level
        
        # Build headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **custom_headers
        }
        if organization:
            headers["OpenAI-Organization"] = organization
        
        context.log(f"Calling {endpoint} with model '{model}', {len(messages)} messages")
        
        # Execute request with retries
        last_error = None
        for attempt in range(retry_attempts + 1):
            try:
                if stream:
                    # Handle streaming response
                    return await self._handle_streaming(
                        endpoint, headers, payload, timeout, context
                    )
                else:
                    # Regular request
                    async with httpx.AsyncClient(timeout=float(timeout)) as client:
                        response = await client.post(
                            endpoint,
                            headers=headers,
                            json=payload
                        )
                        response.raise_for_status()
                        data = response.json()
                        
                        # Process response
                        return self._process_response(data, context)
                        
            except httpx.TimeoutException:
                last_error = f"Request timed out after {timeout} seconds"
                context.log(f"Attempt {attempt + 1} failed: {last_error}", level="warning")
                
            except httpx.HTTPStatusError as e:
                last_error = self._handle_http_error(e)
                context.log(f"Attempt {attempt + 1} failed: {last_error}", level="warning")
                
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                context.log(f"Attempt {attempt + 1} failed: {last_error}", level="error")
            
            # Wait before retry
            if attempt < retry_attempts:
                await asyncio.sleep(retry_delay / 1000)
                context.log(f"Retrying... (attempt {attempt + 2}/{retry_attempts + 1})")
        
        # All attempts failed
        raise Exception(f"Failed after {retry_attempts + 1} attempts. Last error: {last_error}")
    
    def _process_response(self, data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """Process the API response"""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})
        finish_reason = choice.get("finish_reason", "")
        
        # Log token usage
        total_tokens = usage.get("total_tokens", 0)
        if total_tokens > 0:
            context.log(f"Response received: {total_tokens} tokens used")
        
        # Extract tool calls if present
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            context.log(f"Response includes {len(tool_calls)} tool calls")
        
        return {
            "response": message.get("content", ""),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": total_tokens,
            },
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }
    
    async def _handle_streaming(
        self, 
        endpoint: str, 
        headers: Dict[str, str], 
        payload: Dict[str, Any], 
        timeout: int,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Handle streaming responses"""
        payload["stream"] = True
        content_chunks = []
        tool_calls = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        finish_reason = ""
        
        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            async with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            choice = data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            
                            # Collect content
                            if "content" in delta:
                                content_chunks.append(delta["content"])
                            
                            # Collect tool calls
                            if "tool_calls" in delta:
                                tool_calls.extend(delta["tool_calls"])
                            
                            # Update finish reason
                            if "finish_reason" in choice:
                                finish_reason = choice["finish_reason"]
                            
                            # Update usage if provided
                            if "usage" in data:
                                usage = data["usage"]
                        
                        except json.JSONDecodeError:
                            context.log(f"Failed to parse streaming chunk: {data_str}", level="warning")
        
        content = "".join(content_chunks)
        context.log(f"Streaming response complete: {len(content)} characters")
        
        return {
            "response": content,
            "usage": usage,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }
    
    def _handle_http_error(self, error: httpx.HTTPStatusError) -> str:
        """Handle HTTP errors and return user-friendly messages"""
        status_code = error.response.status_code
        
        # Try to extract error details
        error_detail = ""
        try:
            error_data = error.response.json()
            error_detail = error_data.get("error", {}).get("message", "")
        except:
            error_detail = error.response.text[:200]
        
        # Common error handling
        if status_code == 401:
            return "Invalid API key. Please check your API key configuration."
        elif status_code == 403:
            return "Access forbidden. Check your API permissions or organization settings."
        elif status_code == 404:
            return f"Model or endpoint not found. Check the model name and endpoint URL."
        elif status_code == 429:
            return "Rate limit exceeded. Please wait and try again, or check your API quota."
        elif status_code == 500:
            return "Server error. The API service is experiencing issues."
        elif status_code == 503:
            return "Service unavailable. The API is temporarily down or overloaded."
        else:
            return f"HTTP {status_code} error: {error_detail}"


class EmbeddingExecutor(NodeExecutor):
    """Generate text embeddings"""
    
    node_type = "embedding"
    display_name = "Generate Embedding"
    category = "llm"
    description = "Generate text embeddings"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        text = inputs.get("text", "")
        
        if not text:
            context.log("No text provided, returning empty embedding")
            return {"embedding": []}
        
        model = self.get_config_value("model", "text-embedding-3-small")
        
        context.log(f"Generating embedding for text ({len(text)} chars) using model '{model}'")
        
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
            context.log(f"Generated embedding with {len(embedding)} dimensions using local service")
            
            return {"embedding": embedding}
            
        except httpx.ConnectError:
            context.log(f"Cannot connect to local embedding service at {embed_url}, trying OpenAI fallback", level="warning")
            
            # Fallback to OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise Exception(
                    f"Cannot connect to local embedding service at {embed_url} and no OpenAI API key is configured. "
                    "Please ensure the embedding service is running or configure OPENAI_API_KEY."
                )
            
            try:
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
                context.log(f"Generated embedding with {len(embedding)} dimensions using OpenAI")
                return {"embedding": embedding}
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise Exception("Invalid OpenAI API key. Please check your OPENAI_API_KEY configuration.")
                elif e.response.status_code == 404:
                    raise Exception(f"Embedding model '{model}' not found in OpenAI. Please use a valid model name.")
                else:
                    raise Exception(f"OpenAI embedding API error ({e.response.status_code}): {e.response.text[:200]}")
            except Exception as e:
                raise Exception(f"Failed to generate embedding using OpenAI: {str(e)}")
                
        except httpx.TimeoutException:
            raise Exception(
                f"Embedding request timed out after 60 seconds. "
                "The text may be too long or the service is overloaded. "
                "Try reducing the text length."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise Exception(
                    f"Embedding model '{model}' not found at {embed_url}. "
                    "Please check that the correct model is deployed for embeddings."
                )
            else:
                raise Exception(f"Embedding service error ({e.response.status_code}): {e.response.text[:200]}")
        except Exception as e:
            context.log(f"Local embedding failed with unexpected error: {e}", level="error")
            raise Exception(f"Failed to generate embedding: {str(e)}")

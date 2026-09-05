"""Service management routes for LlamaCPP and Embedding services."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse
from typing import Dict, Any
from datetime import datetime
import copy
import json
import httpx
import logging
import os

from modules.chat_proxy import chat_proxy_body_unchanged, parse_chat_proxy_body
from modules.inference_targets import vllm_proxy_base_url

logger = logging.getLogger(__name__)

router = APIRouter(tags=["service"])


def get_manager(request: Request):
    """Get the LlamaCPP manager from app state."""
    return getattr(request.app.state, 'manager', None)


def get_vllm_manager(request: Request):
    """Get the vLLM manager from app state."""
    return getattr(request.app.state, 'vllm_manager', None)


def get_ninfer_manager(request: Request):
    """Get the NInfer manager from app state."""
    return getattr(request.app.state, 'ninfer_manager', None)


# [2026-08-24 | INFO | service.MODULE | Registry of inference backend names -> app.state attr getters
_BACKEND_MANAGERS = {
    "llamacpp": get_manager,
    "vllm": get_vllm_manager,
    "ninfer": get_ninfer_manager,
}

# Backends whose configs are merged in-memory + persisted by the manager itself
_SELF_PERSISTING_BACKENDS = ("vllm", "ninfer")


def get_backend_manager(request: Request, backend: str = "llamacpp"):
    """Get the appropriate backend manager by type."""
    getter = _BACKEND_MANAGERS.get(backend)
    if getter is not None:
        return getter(request)
    return get_manager(request)


async def _stop_other_backend_if_running(request: Request, backend: str) -> None:
    """Ensure only one inference backend is active at a time."""
    for other_backend, getter in _BACKEND_MANAGERS.items():
        if other_backend == backend:
            continue
        other_mgr = getter(request)
        if other_mgr is None:
            continue
        other_status = other_mgr.get_status() if hasattr(other_mgr, "get_status") else {}
        if not other_status.get("running"):
            continue

        logger.info("Stopping %s before %s to keep single active deployment", other_backend, backend)
        stopped = await other_mgr.stop()
        if stopped:
            continue

        detail = f"Failed to stop {other_backend} before starting {backend}"
        if hasattr(other_mgr, "last_action_error") and other_mgr.last_action_error:
            detail = f"{detail}: {other_mgr.last_action_error}"
        raise HTTPException(status_code=500, detail=detail)


def get_embedding_manager(request: Request):
    """Get the embedding manager from app state."""
    return getattr(request.app.state, 'embedding_manager', None)


def get_merge_config_func(request: Request):
    """Get the config merge function from app state."""
    return getattr(request.app.state, 'merge_and_persist_config', None)


def _set_server_managed_auth(headers: Dict[str, Any], api_key: str) -> None:
    """Forward the active backend key instead of any browser/client key."""
    for key in list(headers):
        if key.lower() == "authorization":
            headers.pop(key, None)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"


def _prepare_proxy_headers(request: Request, api_key: str) -> Dict[str, Any]:
    headers = dict(request.headers)
    for h in ("host", "content-length", "transfer-encoding"):
        headers.pop(h, None)
    _set_server_managed_auth(headers, api_key)
    return headers


def _active_inference_target(request: Request) -> tuple[str, str, str]:
    llamacpp_mgr = get_manager(request)
    vllm_mgr = get_vllm_manager(request)
    ninfer_mgr = get_ninfer_manager(request)

    if vllm_mgr and vllm_mgr.is_running():
        api_key = vllm_mgr.config.get("server", {}).get("api_key", "")
        target = vllm_proxy_base_url(vllm_mgr.internal_http_base())
        return target or vllm_mgr.internal_http_base(), api_key, "vllm"

    if ninfer_mgr and ninfer_mgr.is_running():
        api_key = ninfer_mgr.config.get("server", {}).get("api_key", "")
        return ninfer_mgr.internal_http_base(), api_key, "ninfer"

    if llamacpp_mgr and llamacpp_mgr.is_running():
        target = os.getenv("LLAMACPP_PROXY_BASE_URL") or os.getenv("INFERENCE_BACKEND_URL")
        if target:
            api_key = llamacpp_mgr.config.get("server", {}).get("api_key", "")
            return target.rstrip("/"), api_key, "llamacpp"
        if llamacpp_mgr.use_docker:
            target = "http://llamacpp-api:8080"
        else:
            target = "http://localhost:8080"
        api_key = llamacpp_mgr.config.get("server", {}).get("api_key", "")
        return target, api_key, "llamacpp"

    raise HTTPException(status_code=503, detail="No inference backend is running. Start llama.cpp, vLLM, or NInfer first.")


def _empty_metrics_response(reason: str) -> PlainTextResponse:
    content = (
        "# HELP llama_nexus_metrics_available Whether inference metrics are available.\n"
        "# TYPE llama_nexus_metrics_available gauge\n"
        "llama_nexus_metrics_available 0\n"
        f"# llama_nexus_metrics_unavailable_reason {reason}\n"
    )
    return PlainTextResponse(content, media_type="text/plain; version=0.0.4")


@router.get("/metrics")
async def get_prometheus_metrics(request: Request):
    """Proxy active inference Prometheus metrics without surfacing stopped runtimes as 502s."""
    try:
        target, api_key, _backend = _active_inference_target(request)
    except HTTPException as exc:
        return _empty_metrics_response(str(exc.detail))

    headers: Dict[str, str] = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{target}/metrics", headers=headers)
            if response.status_code >= 400:
                return _empty_metrics_response(f"metrics endpoint returned {response.status_code}")
            return PlainTextResponse(response.text, media_type=response.headers.get("content-type", "text/plain; version=0.0.4"))
    except httpx.HTTPError as exc:
        logger.debug("Metrics endpoint unavailable: %s", exc)
        return _empty_metrics_response(str(exc))


@router.get("/api/metrics")
async def get_prometheus_metrics_legacy(request: Request):
    """Legacy alias for the frontend metrics fallback path."""
    return await get_prometheus_metrics(request)


# =============================================================================
# LlamaCPP Service Endpoints
# =============================================================================

@router.post("/api/v1/service/start")
async def start_service(request: Request):
    """Start the llamacpp service."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    try:
        logger.info("Service start request")
        success = await manager.start()
        status = manager.get_status()
        return {
            "success": success, 
            "status": status,
            "message": "Service started successfully" if success else "Service failed to start"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Service start error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/api/v1/service/stop")
async def stop_service(request: Request):
    """Stop the llamacpp service."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    success = await manager.stop()
    return {"success": success, "status": manager.get_status()}


@router.post("/api/v1/service/restart")
async def restart_service(request: Request):
    """Restart the llamacpp service."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    success = await manager.restart()
    return {"success": success, "status": manager.get_status()}


@router.get("/api/v1/service/status")
async def get_service_status(request: Request):
    """Get current service status including resources."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    status = manager.get_status()
    if status.get("running"):
        health = await manager.get_llamacpp_health()
        status["llamacpp_health"] = health
    
    return status


@router.get("/api/v1/service/backends")
async def get_backends_status(request: Request):
    """Get status of all available inference backends."""
    llamacpp_mgr = get_manager(request)
    vllm_mgr = get_vllm_manager(request)
    ninfer_mgr = get_ninfer_manager(request)

    result = {
        "active": "llamacpp",
        "backends": {},
    }

    if llamacpp_mgr:
        llamacpp_status = llamacpp_mgr.get_status()
        if llamacpp_status.get("running"):
            health = await llamacpp_mgr.get_llamacpp_health()
            llamacpp_status["health"] = health
            result["active"] = "llamacpp"
        result["backends"]["llamacpp"] = llamacpp_status

    if vllm_mgr:
        vllm_status = vllm_mgr.get_status()
        if vllm_status.get("running"):
            health = await vllm_mgr.get_health()
            vllm_status["health"] = health
            result["active"] = "vllm"
        result["backends"]["vllm"] = vllm_status

    if ninfer_mgr:
        ninfer_status = ninfer_mgr.get_status()
        if ninfer_status.get("running"):
            health = await ninfer_mgr.get_health()
            ninfer_status["health"] = health
            result["active"] = "ninfer"
        result["backends"]["ninfer"] = ninfer_status

    return result


@router.get("/api/v1/service/vllm/status")
async def get_vllm_status(request: Request):
    """Get vLLM backend status."""
    vllm_mgr = get_vllm_manager(request)
    if vllm_mgr is None:
        raise HTTPException(status_code=503, detail="vLLM manager not available")
    status = vllm_mgr.get_status()
    if status.get("running"):
        health = await vllm_mgr.get_health()
        status["health"] = health
    return status


@router.get("/api/v1/service/ninfer/status")
async def get_ninfer_status(request: Request):
    """Get NInfer backend status."""
    ninfer_mgr = get_ninfer_manager(request)
    if ninfer_mgr is None:
        raise HTTPException(status_code=503, detail="NInfer manager not available")
    status = ninfer_mgr.get_status()
    if status.get("running"):
        health = await ninfer_mgr.get_health()
        status["health"] = health
    return status


@router.get("/api/v1/service/config")
async def get_config(request: Request, backend: str = "llamacpp"):
    """Get current configuration for the selected backend."""
    mgr = get_backend_manager(request, backend)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Manager not available")

    if backend == "vllm":
        vllm_mgr = mgr
        return {
            "config": vllm_mgr.config,
            "command": " ".join(vllm_mgr.build_command()),
            "backend": "vllm",
            "editable_fields": {
                "model": ["name", "served_name"],
                "sampling": ["temperature", "top_p", "top_k", "repetition_penalty",
                              "frequency_penalty", "presence_penalty"],
                "performance": ["max_model_len", "gpu_memory_utilization", "max_num_seqs",
                                 "max_num_batched_tokens", "kv_cache_dtype"],
                "reasoning": ["reasoning_parser"],
                "server": ["api_key"],
            }
        }

    if backend == "ninfer":
        ninfer_mgr = mgr
        return {
            "config": ninfer_mgr.config,
            "command": " ".join(ninfer_mgr.build_command()),
            "backend": "ninfer",
            "registered_artifacts": ninfer_mgr.registered_artifacts(),
            "artifact_exists": ninfer_mgr.artifact_exists(),
            "editable_fields": {
                "model": ["artifact", "served_name"],
                "performance": ["max_context", "kv_capacity", "kv_dtype", "max_concurrency"],
                "speculative": ["enabled", "draft_tokens", "lm_head_draft"],
                "vision": ["enabled"],
                "sampling": ["temperature", "top_p", "top_k", "presence_penalty", "greedy"],
                "server": ["api_key", "published_port"],
            }
        }

    # llama.cpp backend (default)
    manager = mgr
    return {
        "config": manager.config,
        "command": " ".join(manager.build_command()),
        "backend": "llamacpp",
        "editable_fields": {
            "model": ["context_size", "gpu_layers"],
            "sampling": ["temperature", "top_p", "top_k", "min_p", "repeat_penalty",
                        "frequency_penalty", "presence_penalty", "dry_multiplier"],
            "performance": ["threads", "batch_size", "ubatch_size", "num_predict", "parallel_slots", "ctx_checkpoints"],
            "mtp": ["enabled", "workload_profile", "draft_n_max", "draft_n_min", "draft_p_min"],
            "server": ["api_key", "chat_template_kwargs", "cache_reuse", "reasoning_budget", "jinja"],
            "template": ["directory", "selected"]
        }
    }


def _deep_merge_config(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge incoming into base (mutates base). Explicit null removes a key."""
    for key, value in incoming.items():
        if value is None:
            base.pop(key, None)
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_config(base[key], value)
        else:
            base[key] = value
    return base


@router.put("/api/v1/service/config")
async def update_config(request: Request, config: Dict[str, Any], backend: str = "llamacpp"):
    """Update configuration (requires restart to apply).

    Use query param ``backend=vllm`` or ``backend=ninfer`` to update those
    managers' in-memory config. llama.cpp continues to use persisted merge when
    ``merge_and_persist_config`` is available.
    """
    if backend in _SELF_PERSISTING_BACKENDS:
        mgr = get_backend_manager(request, backend)
        if mgr is None:
            raise HTTPException(status_code=503, detail=f"{backend} manager not available")
        try:
            _deep_merge_config(mgr.config, config)
            mgr.persist_deploy_config()
            return {
                "success": True,
                "config": mgr.config,
                "command": " ".join(mgr.build_command()),
                "backend": backend,
                "message": f"{backend} configuration updated in memory. Restart service to apply.",
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update {backend} config: {str(e)}")

    manager = get_manager(request)
    merge_config = get_merge_config_func(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    if merge_config is None:
        raise HTTPException(status_code=503, detail="Config persistence not available")
    try:
        updated_config = merge_config(config)
        return {
            "success": True,
            "config": updated_config,
            "command": " ".join(manager.build_command()),
            "backend": "llamacpp",
            "message": "Configuration updated. Restart service to apply changes."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@router.post("/api/v1/service/config/preview")
async def preview_config(request: Request, payload: Dict[str, Any]):
    """Preview the command line that would be generated for a configuration without applying it."""
    backend = payload.get("backend", "llamacpp")
    mgr = get_backend_manager(request, backend)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Manager not available")

    config = payload.get("config")
    if not config:
        raise HTTPException(status_code=400, detail="Config is required")

    try:
        original_config = copy.deepcopy(mgr.config)

        temp_config = copy.deepcopy(mgr.config)
        _deep_merge_config(temp_config, config)

        mgr.config = temp_config
        command = " ".join(mgr.build_command())
        mgr.config = original_config

        return {
            "command": command,
            "config": temp_config,
            "backend": backend,
        }
    except Exception as e:
        mgr.config = original_config
        raise HTTPException(status_code=500, detail=f"Failed to preview config: {str(e)}")


@router.get("/api/v1/service/config/fields")
async def get_config_fields(request: Request, backend: str = "llamacpp"):
    """Get field metadata for the selected backend, including scope and cross-framework mapping."""
    mgr = get_backend_manager(request, backend)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Manager not available")

    if hasattr(mgr, "get_field_metadata"):
        return {
            "backend": backend,
            "fields": mgr.get_field_metadata(),
        }
    return {"backend": backend, "fields": {}}


@router.post("/api/v1/service/config/validate")
async def validate_config(request: Request, config: Dict[str, Any]):
    """Validate a configuration without applying it."""
    manager = get_manager(request)
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not available")
    
    errors = []
    warnings = []
    
    # Validate model settings
    if "model" in config:
        model = config["model"]
        if "context_size" in model:
            ctx = model["context_size"]
            if ctx < 512:
                errors.append("context_size must be at least 512")
            elif ctx > 131072:
                warnings.append("context_size > 131072 may cause memory issues")
        if "gpu_layers" in model:
            layers = model["gpu_layers"]
            if layers < -1:
                errors.append("gpu_layers must be >= -1")
    
    # Validate sampling settings
    if "sampling" in config:
        sampling = config["sampling"]
        if "temperature" in sampling:
            temp = sampling["temperature"]
            if temp < 0:
                errors.append("temperature must be >= 0")
            elif temp > 2:
                warnings.append("temperature > 2 may produce incoherent output")

    if hasattr(manager, "validate_mtp_config"):
        mtp_result = manager.validate_mtp_config(config)
        errors.extend(mtp_result.get("errors", []))
        warnings.extend(mtp_result.get("warnings", []))
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


@router.post("/api/v1/service/action")
async def service_action(request: Request, payload: Dict[str, Any]):
    """Perform a service action (start, stop, restart). Optionally specify backend."""
    action = payload.get("action")
    config = payload.get("config")
    backend = payload.get("backend", "llamacpp")

    mgr = get_backend_manager(request, backend)
    if mgr is None:
        raise HTTPException(status_code=503, detail=f"{backend} manager not available")

    # Apply config before start/restart if provided
    if config and action in ("start", "restart"):
        if backend == "llamacpp":
            merge_config = get_merge_config_func(request)
            if merge_config:
                logger.info(f"Applying config before {action}: model={config.get('model', {}).get('name')}/{config.get('model', {}).get('variant')}")
                merge_config(config)
            else:
                logger.info(f"Directly updating manager config before {action}")
                mgr.config = {**mgr.config, **config}
        elif backend in _SELF_PERSISTING_BACKENDS:
            logger.info(f"Applying {backend} config before {action}")
            _deep_merge_config(mgr.config, config)
            mgr.persist_deploy_config()

    if action in ("start", "restart"):
        await _stop_other_backend_if_running(request, backend)

    if action == "start":
        success = await mgr.start()
    elif action == "stop":
        success = await mgr.stop()
    elif action == "restart":
        success = await mgr.restart()
    else:
        raise HTTPException(status_code=400, detail="Unsupported action. Use 'start', 'stop', or 'restart'.")
    if not success:
        parts = [f"{backend} failed to {action}"]
        if hasattr(mgr, "last_action_error") and mgr.last_action_error:
            parts.append(mgr.last_action_error)
        raise HTTPException(status_code=500, detail=" — ".join(parts))
    return {"success": True, "status": mgr.get_status(), "backend": backend}


# Alias endpoints for frontend compatibility
@router.get("/v1/service/config")
async def get_config_alias(request: Request):
    """Get config (frontend compatibility)."""
    return await get_config(request)


@router.put("/v1/service/config")
async def update_config_alias(request: Request, config: Dict[str, Any], backend: str = "llamacpp"):
    """Update config (frontend compatibility)."""
    return await update_config(request, config, backend)


@router.post("/v1/service/config/preview")
async def preview_config_alias(request: Request, payload: Dict[str, Any]):
    """Preview config (frontend compatibility)."""
    return await preview_config(request, payload)


@router.post("/v1/service/config/validate")
async def validate_config_alias(request: Request, config: Dict[str, Any]):
    """Validate config (frontend compatibility)."""
    return await validate_config(request, config)


@router.get("/v1/service/status")
async def get_status_alias(request: Request):
    """Get status (frontend compatibility)."""
    return await get_service_status(request)


@router.post("/v1/service/action")
async def service_action_alias(request: Request, payload: Dict[str, Any]):
    """Service action (frontend compatibility)."""
    return await service_action(request, payload)


# =============================================================================
# LLM Proxy - Routes to whichever inference backend is active
# =============================================================================

@router.api_route("/v1/chat/completions", methods=["POST"])
async def proxy_chat_completions(request: Request):
    """Proxy chat completion requests to the active inference backend.

    Passthrough contract: tools, tool_choice, response_format, and grammar are
    forwarded unchanged in the request body (see modules/chat_proxy.py).
    """
    body = await request.body()
    body = chat_proxy_body_unchanged(body)
    target, api_key, _backend = _active_inference_target(request)
    headers = _prepare_proxy_headers(request, api_key)

    url = f"{target}/v1/chat/completions"

    is_stream = False
    if body:
        try:
            body_json = parse_chat_proxy_body(body)
            is_stream = bool(body_json.get("stream", False))
        except (json.JSONDecodeError, ValueError):
            is_stream = False

    if is_stream:
        return await _proxy_stream(request, url, body, headers)
    else:
        return await _proxy_non_stream(url, body, headers)


@router.api_route("/v1/completions", methods=["POST"])
async def proxy_completions(request: Request):
    """Proxy text completion requests to the active inference backend."""
    body = await request.body()
    target, api_key, _backend = _active_inference_target(request)
    headers = _prepare_proxy_headers(request, api_key)

    return await _proxy_non_stream(f"{target}/v1/completions", body, headers)


@router.api_route("/completion", methods=["POST"])
async def proxy_native_completion(request: Request):
    """Proxy llama.cpp-compatible native completion requests to the deployed backend."""
    body = await request.body()
    target, api_key, _backend = _active_inference_target(request)
    headers = _prepare_proxy_headers(request, api_key)
    return await _proxy_non_stream(f"{target}/completion", body, headers, timeout=600.0)


@router.api_route("/tokenize", methods=["POST"])
async def proxy_tokenize(request: Request):
    """Proxy tokenizer requests to the deployed backend."""
    body = await request.body()
    target, api_key, _backend = _active_inference_target(request)
    headers = _prepare_proxy_headers(request, api_key)
    return await _proxy_non_stream(f"{target}/tokenize", body, headers, timeout=120.0)


@router.api_route("/detokenize", methods=["POST"])
async def proxy_detokenize(request: Request):
    """Proxy detokenizer requests to the deployed backend."""
    body = await request.body()
    target, api_key, _backend = _active_inference_target(request)
    headers = _prepare_proxy_headers(request, api_key)
    return await _proxy_non_stream(f"{target}/detokenize", body, headers, timeout=120.0)


async def _proxy_non_stream(url: str, body: bytes, headers: dict, timeout: float = 120.0):
    """Proxy a non-streaming request."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, content=body, headers=headers)
            return JSONResponse(
                content=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                status_code=response.status_code,
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="Cannot connect to inference backend")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Inference backend timed out")


async def _proxy_stream(request: Request, url: str, body: bytes, headers: dict):
    """Proxy a streaming SSE request.

    Checks the response status before streaming. If the backend returns an
    error (e.g. model-not-found from vLLM), forwards it as a JSON error
    instead of silently streaming a non-SSE body.
    """
    import json as _json

    try:
        async with httpx.AsyncClient(timeout=10.0) as preflight:
            # Read just the status + headers to detect errors early.
            preflight_req = preflight.build_request("POST", url, content=body, headers=headers)
            preflight_resp = await preflight.send(preflight_req, stream=True)
            if preflight_resp.status_code != 200:
                error_body = await preflight_resp.aread()
                await preflight_resp.aclose()
                try:
                    error_json = _json.loads(error_body)
                except Exception:
                    error_json = {"error": error_body.decode(errors="replace")}
                raise HTTPException(
                    status_code=preflight_resp.status_code,
                    detail=error_json,
                )
            await preflight_resp.aclose()
    except HTTPException:
        raise
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Cannot connect to inference backend")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Inference backend timed out")

    async def stream_generator():
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, content=body, headers=headers) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# Embedding Service Endpoints
# =============================================================================

@router.post("/api/v1/embedding/start")
async def start_embedding_service(request: Request):
    """Start the embedding service."""
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    try:
        logger.info("Embedding service start request")
        success = await embedding_manager.start()
        status = embedding_manager.get_status()
        return {
            "success": success,
            "status": status,
            "message": "Embedding service started successfully" if success else "Embedding service failed to start"
        }
    except Exception as e:
        logger.error(f"Embedding service start error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start embedding service: {str(e)}")


@router.post("/api/v1/embedding/stop")
async def stop_embedding_service(request: Request):
    """Stop the embedding service."""
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    try:
        success = await embedding_manager.stop()
        return {"success": success, "status": embedding_manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop embedding service: {str(e)}")


@router.post("/api/v1/embedding/restart")
async def restart_embedding_service(request: Request):
    """Restart the embedding service."""
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    try:
        success = await embedding_manager.restart()
        return {"success": success, "status": embedding_manager.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart embedding service: {str(e)}")


@router.get("/api/v1/embedding/status")
async def get_embedding_status(request: Request):
    """Get current embedding service status."""
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    return embedding_manager.get_status()


@router.get("/api/v1/embedding/config")
async def get_embedding_config(request: Request):
    """Get current embedding configuration."""
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    return {
        "config": embedding_manager.config,
        "command": " ".join(embedding_manager.build_command()),
        "available_models": [
            {
                "name": "nomic-embed-text-v1.5",
                "dimensions": 768,
                "max_tokens": 8192,
                "description": "Nomic AI's long context embedding model (recommended)"
            },
            {
                "name": "e5-mistral-7b",
                "dimensions": 4096,
                "max_tokens": 32768,
                "description": "E5 Mistral 7B instruct model for embeddings"
            },
            {
                "name": "bge-m3",
                "dimensions": 1024,
                "max_tokens": 8192,
                "description": "BAAI BGE-M3 multilingual embedding model"
            },
            {
                "name": "gte-Qwen2-1.5B",
                "dimensions": 1536,
                "max_tokens": 32768,
                "description": "Alibaba GTE Qwen2 1.5B instruct model"
            }
        ]
    }


@router.put("/api/v1/embedding/config")
async def update_embedding_config(request: Request, config: Dict[str, Any]):
    """Update embedding configuration (requires restart to apply)."""
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    try:
        updated_config = embedding_manager.update_config(config)
        return {
            "success": True,
            "config": updated_config,
            "message": "Configuration updated. Restart service to apply changes."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@router.post("/api/v1/embedding/test")
async def test_embedding(request: Request):
    """Test the embedding service by generating embeddings for sample text."""
    import aiohttp
    
    embedding_manager = get_embedding_manager(request)
    if embedding_manager is None:
        raise HTTPException(status_code=503, detail="Embedding manager not available")
    
    status = embedding_manager.get_status()
    if not status.get("running"):
        raise HTTPException(status_code=503, detail="Embedding service is not running")
    
    try:
        data = await request.json()
        text = data.get("text", "The quick brown fox jumps over the lazy dog.")
        
        # Get the service configuration
        config = embedding_manager.config
        api_key = config.get("server", {}).get("api_key", "llamacpp-embed")
        model_name = config.get("model", {}).get("name", "nomic-embed-text-v1.5")
        port = config.get("server", {}).get("port", 8602)
        
        # Connect to the embedding service via Docker network
        # Use the container hostname for inter-container communication
        # IMPORTANT: Always use internal port 8080, not the external mapped port
        service_url = "http://llamacpp-embed:8080/v1/embeddings"
        
        import time
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                service_url,
                json={"input": text, "model": model_name},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Embedding service error: {error_text}")
                
                result = await response.json()
        
        end_time = time.time()
        time_taken = int((end_time - start_time) * 1000)
        
        return {
            "success": True,
            "model": result.get("model"),
            "usage": result.get("usage"),
            "vectorLength": len(result.get("data", [{}])[0].get("embedding", [])),
            "timeTaken": time_taken,
            "data": result.get("data")
        }
        
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=503, detail=f"Failed to connect to embedding service: {str(e)}")
    except Exception as e:
        logger.error(f"Embedding test error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding test failed: {str(e)}")


# =============================================================================
# Worker Service Control Endpoints
# =============================================================================

# Managed worker containers
WORKER_CONTAINERS = {
    "training": "llama-nexus-training",
    "quantization": "quantization-worker",
}


def run_docker_command(args: list) -> tuple[bool, str]:
    """Run a docker CLI command and return success status and output."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker"] + args,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


@router.get("/api/v1/workers/status")
async def get_workers_status():
    """Get status of all managed worker containers."""
    import subprocess
    
    workers = []
    for worker_id, container_name in WORKER_CONTAINERS.items():
        try:
            # Get container status using docker inspect
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                status = result.stdout.strip()
                running = status == "running"
            else:
                status = "not_found"
                running = False
            
            # Check GPU usage if running
            gpu_active = False
            if running:
                try:
                    gpu_result = subprocess.run(
                        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=5
                    )
                    if gpu_result.returncode == 0 and gpu_result.stdout.strip():
                        gpu_active = True
                except Exception:
                    pass
            
            workers.append({
                "id": worker_id,
                "name": container_name,
                "running": running,
                "status": status,
                "gpu_active": gpu_active,
            })
        except Exception as e:
            workers.append({
                "id": worker_id,
                "name": container_name,
                "running": False,
                "status": "error",
                "error": str(e),
            })
    
    return {"workers": workers}


@router.post("/api/v1/workers/{worker_id}/start")
async def start_worker(worker_id: str):
    """Start a worker container."""
    if worker_id not in WORKER_CONTAINERS:
        raise HTTPException(status_code=404, detail=f"Unknown worker: {worker_id}")
    
    container_name = WORKER_CONTAINERS[worker_id]
    success, output = run_docker_command(["start", container_name])
    
    if success:
        return {"success": True, "message": f"{worker_id} started successfully"}
    else:
        logger.error(f"Failed to start {worker_id}: {output}")
        raise HTTPException(status_code=500, detail=f"Failed to start {worker_id}: {output}")


@router.post("/api/v1/workers/{worker_id}/stop")
async def stop_worker(worker_id: str):
    """Stop a worker container."""
    if worker_id not in WORKER_CONTAINERS:
        raise HTTPException(status_code=404, detail=f"Unknown worker: {worker_id}")
    
    container_name = WORKER_CONTAINERS[worker_id]
    success, output = run_docker_command(["stop", "-t", "10", container_name])
    
    if success:
        return {"success": True, "message": f"{worker_id} stopped successfully"}
    else:
        logger.error(f"Failed to stop {worker_id}: {output}")
        raise HTTPException(status_code=500, detail=f"Failed to stop {worker_id}: {output}")

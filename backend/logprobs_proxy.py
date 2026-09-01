#!/usr/bin/env python3
"""
OpenAI-compatible completions proxy for lm-eval.

lm-eval's local-completions model sends prompt=[context+continuation token IDs],
echo=True, max_tokens=1, and expects logprobs for the echoed prompt so it can
slice [ctxlen:-1] to score the continuation. vLLM already returns those prompt
logprobs from /v1/completions, while llama.cpp deployments need the older
incremental /completion scorer below.

The proxy also exposes /tokenizer_info, /tokenize, and /detokenize so lm-eval can
use tokenizer_backend=remote, keeping token IDs aligned with the deployed model.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import requests

UPSTREAM_URL = os.environ.get("LLAMACPP_API_URL", "http://backend-api:8700")
API_KEY = os.environ.get("LM_EVAL_API_KEY", "placeholder-api-key")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8888"))
N_PROBS = int(os.environ.get("PROXY_N_PROBS", "100"))
MAX_CONTINUATION = int(os.environ.get("PROXY_MAX_CONTINUATION", "20"))
POSITION_WORKERS = int(os.environ.get("PROXY_POSITION_WORKERS", "8"))

_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})
if API_KEY:
    _session.headers.update({"Authorization": f"Bearer {API_KEY}"})
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=50,
    pool_maxsize=50,
    max_retries=2,
)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

# Qwen tokenizer special tokens
EOS_TOKEN = "<|endoftext|>"
EOS_TOKEN_ID = 151643
BOS_TOKEN_ID = None  # Qwen doesn't use BOS


def _upstream_tokenize(text):
    """Tokenize text via the deployed backend. Returns list of token IDs."""
    payloads = (
        {"prompt": text, "add_special_tokens": False},  # vLLM
        {"content": text},  # llama.cpp
    )
    for payload in payloads:
        try:
            r = _session.post(
                f"{UPSTREAM_URL}/tokenize",
                json=payload,
                timeout=30,
            )
            if r.status_code == 200:
                return r.json().get("tokens", [])
        except Exception:
            pass
    return None


def _upstream_detokenize(token_ids):
    """Detokenize token IDs via the deployed backend. Returns string."""
    try:
        r = _session.post(
            f"{UPSTREAM_URL}/detokenize",
            json={"tokens": token_ids},
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("content", data.get("prompt", ""))
    except Exception:
        pass
    return None


def _looks_like_model_not_found(resp):
    if resp.status_code != 404:
        return False
    try:
        data = resp.json()
    except Exception:
        return False
    message = ""
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", ""))
        else:
            message = str(data.get("detail", ""))
    return "model" in message.lower() and "does not exist" in message.lower()


def _active_backend():
    """Return the app's active backend name when routed through backend-api."""
    try:
        r = _session.get(
            f"{UPSTREAM_URL.rstrip('/')}/api/v1/service/backends",
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            active = str(data.get("active") or data.get("active_backend") or "").lower()
            if active:
                return active
    except Exception as e:
        print(f"[proxy] Could not determine active backend: {e}", flush=True)
    return os.environ.get("LM_EVAL_BACKEND", "").lower()


def _get_token_logprob_at_position(prefix_tokens, target_token_id):
    """
    Get logprob of target_token_id given prefix_tokens as context.
    Uses llama.cpp's /completion endpoint with cache_prompt.
    Returns (logprob, top_logprobs_dict) or (None, {}) if not found.
    top_logprobs_dict maps token_text -> logprob (uses token text from response).
    """
    try:
        r = _session.post(
            f"{UPSTREAM_URL}/completion",
            json={
                "prompt": prefix_tokens,
                "n_predict": 1,
                "logprobs": True,
                "n_probs": N_PROBS,
                "temperature": 0,
                "seed": 1234,
                "cache_prompt": True,
            },
            timeout=600,
        )
        if r.status_code != 200:
            return None, {}

        data = r.json()
        cp = data.get("completion_probabilities", [])
        if not cp:
            return None, {}

        entry = cp[0]
        top_logprobs = {}
        target_logprob = None
        for te in entry.get("top_logprobs", []):
            token_text = te.get("token", "")
            top_logprobs[token_text] = te["logprob"]
            if te["id"] == target_token_id:
                target_logprob = te["logprob"]

        return target_logprob, top_logprobs

    except Exception:
        return None, {}


class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.rstrip("/")

        # /tokenizer_info - required by lm-eval's RemoteTokenizer
        if path == "/tokenizer_info":
            info = {
                "eos_token": EOS_TOKEN,
                "eos_token_id": EOS_TOKEN_ID,
                "bos_token": None,
                "bos_token_id": BOS_TOKEN_ID,
                "pad_token": None,
                "pad_token_id": None,
                "model_type": "qwen2",
            }
            self._send_json(200, info)
            return

        # Pass through other GETs
        target_url = f"{UPSTREAM_URL.rstrip('/')}{self.path}"
        try:
            resp = _session.get(target_url, timeout=30)
            self.send_response(resp.status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp.content)
        except requests.RequestException as e:
            self.send_error(502, f"Upstream error: {e}")

    def do_POST(self):
        path = self.path.rstrip("/")

        # /tokenize - translate from lm-eval format to llama.cpp format
        if path == "/tokenize":
            self._handle_tokenize()
            return

        # /detokenize - translate response from llama.cpp format
        if path == "/detokenize":
            self._handle_detokenize()
            return

        # /v1/completions - main inference endpoint
        if path == "/v1/completions":
            self._handle_completions()
            return

        # Pass through other POSTs
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        target_url = f"{UPSTREAM_URL.rstrip('/')}{self.path}"
        try:
            resp = _session.post(target_url, data=body, timeout=600)
            self.send_response(resp.status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp.content)
        except requests.RequestException as e:
            self.send_error(502, f"Upstream error: {e}")

    def _handle_tokenize(self):
        """Handle /tokenize request from lm-eval's RemoteTokenizer."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        # lm-eval sends {"prompt": text, "add_special_tokens": false}
        # llama.cpp expects {"content": text}
        text = payload.get("prompt", payload.get("content", ""))
        if isinstance(text, str):
            text = text.encode("utf-8", errors="replace").decode("utf-8")

        tokens = _upstream_tokenize(text)
        if tokens is None:
            self.send_error(502, "Upstream tokenize failed")
            return

        self._send_json(200, {"tokens": tokens})

    def _handle_detokenize(self):
        """Handle /detokenize request from lm-eval's RemoteTokenizer."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        token_ids = payload.get("tokens", [])
        text = _upstream_detokenize(token_ids)
        if text is None:
            self.send_error(502, "Upstream detokenize failed")
            return

        # lm-eval's RemoteTokenizer expects {"prompt": text}
        self._send_json(200, {"prompt": text})

    def _handle_completions(self):
        """Handle /v1/completions request from lm-eval."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        prompt = payload.get("prompt", "")
        echo = payload.get("echo", False)
        logprobs_flag = payload.get("logprobs")
        max_tokens = payload.get("max_tokens", 16)
        ctxlens = payload.get("ctxlens")

        # Handle batch-wrapped prompt: [[token_ids]] -> [token_ids]
        if (isinstance(prompt, list) and len(prompt) == 1
                and isinstance(prompt[0], list)
                and all(isinstance(t, int) for t in prompt[0])):
            prompt = prompt[0]

        is_loglikelihood = (
            echo is True
            and logprobs_flag is not None
            and max_tokens == 1
            and isinstance(prompt, list)
            and all(isinstance(t, int) for t in prompt)
            and len(prompt) > 0
        )

        if is_loglikelihood:
            self._handle_loglikelihood(prompt, logprobs_flag, ctxlens, payload.get("model"))
        else:
            self._handle_normal_completions(payload)

    def _handle_loglikelihood_openai(self, token_ids, logprobs_flag, ctxlens=None, model=None):
        """Handle loglikelihood through vLLM/OpenAI prompt logprobs."""
        self._last_openai_error = None
        try:
            n_logprobs = int(logprobs_flag) if logprobs_flag is not None else 1
        except (TypeError, ValueError):
            n_logprobs = 1
        n_logprobs = max(n_logprobs, 1)

        payload = {
            "model": model or os.environ.get("LM_EVAL_MODEL_NAME", "default"),
            "prompt": token_ids,
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": n_logprobs,
            "seed": 1234,
            "echo": True,
        }
        try:
            resp = _session.post(
                f"{UPSTREAM_URL}/v1/completions",
                json=payload,
                timeout=600,
            )
        except requests.RequestException as e:
            self._last_openai_error = f"OpenAI completions request failed: {e}"
            print(f"[proxy] {self._last_openai_error}", flush=True)
            return False

        if resp.status_code != 200:
            if _looks_like_model_not_found(resp):
                try:
                    self._send_json(resp.status_code, resp.json())
                except Exception:
                    self.send_error(resp.status_code, "Upstream model not found")
                return True
            self._last_openai_error = (
                f"OpenAI completions returned {resp.status_code}: {resp.text[:500]}"
            )
            print(f"[proxy] {self._last_openai_error}", flush=True)
            return False

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            self._last_openai_error = f"OpenAI completions returned invalid JSON: {e}"
            print(f"[proxy] {self._last_openai_error}", flush=True)
            return False

        choices = data.get("choices")
        if not choices:
            self._last_openai_error = "OpenAI completions response had no choices"
            print(f"[proxy] {self._last_openai_error}", flush=True)
            return False

        for choice in choices:
            logprobs = choice.get("logprobs")
            if not isinstance(logprobs, dict):
                self._last_openai_error = "OpenAI completions response had no logprobs object"
                print(f"[proxy] {self._last_openai_error}", flush=True)
                return False
            token_logprobs = logprobs.get("token_logprobs")
            top_logprobs = logprobs.get("top_logprobs")
            if not isinstance(token_logprobs, list) or not isinstance(top_logprobs, list):
                self._last_openai_error = (
                    "OpenAI completions response had malformed prompt logprobs"
                )
                print(f"[proxy] {self._last_openai_error}", flush=True)
                return False
            choice["logprobs"]["top_logprobs"] = [
                top if isinstance(top, dict) else {}
                for top in top_logprobs
            ]

        print(
            f"[proxy] loglikelihood via /v1/completions: {len(token_ids)} tokens, "
            f"ctxlens={ctxlens or []}",
            flush=True,
        )
        self._send_json(200, data)
        return True

    def _handle_loglikelihood(self, token_ids, logprobs_flag, ctxlens=None, model=None):
        """
        Handle loglikelihood evaluation.
        
        lm-eval sends token IDs from context + continuation, with echo=True, max_tokens=1.
        We need to return logprobs for all positions so parse_logprobs can slice [ctxlen:-1].
        
        If ctxlens is provided, we only compute real logprobs for continuation positions
        (ctxlen to N-1), using ignored placeholders for context positions.
        """
        if self._handle_loglikelihood_openai(token_ids, logprobs_flag, ctxlens, model):
            return

        active_backend = _active_backend()
        if active_backend == "vllm":
            message = self._last_openai_error or "OpenAI prompt logprobs failed"
            self._send_json(502, {
                "error": {
                    "message": (
                        "Real deployed vLLM scoring failed; refusing to fall back "
                        f"to the llama.cpp scorer. {message}"
                    ),
                    "type": "upstream_error",
                }
            })
            return

        t0 = time.time()
        n = len(token_ids)
        
        # Determine which positions need real logprobs
        if ctxlens and isinstance(ctxlens, list) and len(ctxlens) > 0:
            # ctxlens from lm-eval tells us where context ends
            # For a single request, ctxlens has one entry
            ctxlen = ctxlens[0]
            compute_positions = list(range(ctxlen, n))
        else:
            # Fallback: compute last MAX_CONTINUATION positions
            compute_start = max(1, n - MAX_CONTINUATION)
            compute_positions = list(range(compute_start, n))
            ctxlen = n  # Unknown, assume all are continuation

        # Build context placeholders, then fill real continuation logprobs.
        results = []
        for i in range(n):
            token_text = _upstream_detokenize([token_ids[i]]) or ""
            results.append({
                "token": token_text,
                "logprob": None if i == 0 else 0.0,
                "top_logprobs": {},
            })

        if compute_positions:
            # Prime the cache by evaluating the full sequence
            try:
                _session.post(
                    f"{UPSTREAM_URL}/completion",
                    json={
                        "prompt": token_ids,
                        "n_predict": 1,
                        "logprobs": True,
                        "n_probs": N_PROBS,
                        "temperature": 0,
                        "seed": 1234,
                        "cache_prompt": True,
                    },
                    timeout=600,
                )
            except Exception:
                pass

            # Compute logprobs in parallel for needed positions
            def _compute_pos(i):
                prefix = token_ids[:i]
                target_id = token_ids[i]
                logprob, top_logprobs = _get_token_logprob_at_position(prefix, target_id)
                return i, logprob, top_logprobs

            with ThreadPoolExecutor(max_workers=POSITION_WORKERS) as executor:
                futures = {executor.submit(_compute_pos, i): i for i in compute_positions}
                for future in as_completed(futures):
                    try:
                        i, logprob, top_logprobs = future.result()
                        results[i] = {
                            "token": results[i]["token"],
                            "logprob": logprob if logprob is not None else -100.0,
                            "top_logprobs": top_logprobs,
                        }
                    except Exception:
                        pass

        # Get the generated token after the full prompt
        gen_token_text = ""
        gen_logprob = -100.0
        gen_top = {}
        try:
            r = _session.post(
                f"{UPSTREAM_URL}/completion",
                json={
                    "prompt": token_ids,
                    "n_predict": 1,
                    "logprobs": True,
                    "n_probs": N_PROBS,
                    "temperature": 0,
                    "seed": 1234,
                    "cache_prompt": True,
                },
                timeout=600,
            )
            data = r.json()
            cp = data.get("completion_probabilities", [])
            if cp:
                gen_token_text = cp[0].get("token", "")
                gen_logprob = cp[0].get("logprob", -100.0)
                gen_top = {}
                for te in cp[0].get("top_logprobs", []):
                    gen_top[te["token"]] = te["logprob"]
        except Exception:
            pass

        # Build response
        tokens = [info["token"] for info in results] + [gen_token_text]
        token_logprobs = [info["logprob"] for info in results] + [gen_logprob]
        top_logprobs = [info["top_logprobs"] for info in results] + [gen_top]

        elapsed = time.time() - t0
        positions = len(compute_positions)
        print(f"[proxy] loglikelihood: {n} tokens, {positions} positions computed, {elapsed:.2f}s", flush=True)

        resp_data = {
            "choices": [{
                "text": gen_token_text,
                "index": 0,
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": token_logprobs,
                    "top_logprobs": top_logprobs,
                },
                "finish_reason": "length",
            }],
            "model": model or os.environ.get("LM_EVAL_MODEL_NAME", "deployed-model"),
            "object": "text_completion",
        }

        self._send_json(200, resp_data)

    def _handle_normal_completions(self, payload):
        """Pass through non-loglikelihood requests."""
        # Sanitize string prompts
        prompt = payload.get("prompt", "")
        if isinstance(prompt, str):
            payload["prompt"] = prompt.encode("utf-8", errors="replace").decode("utf-8")

        target_url = f"{UPSTREAM_URL}/v1/completions"
        try:
            resp = _session.post(target_url, json=payload, timeout=600)
        except requests.RequestException as e:
            self.send_error(502, f"Upstream error: {e}")
            return

        try:
            resp_data = resp.json()
        except json.JSONDecodeError:
            self.send_error(502, "Upstream returned invalid JSON")
            return

        if resp.status_code != 200:
            self._send_json(resp.status_code, resp_data)
            return

        # Translate logprobs format if present
        if "choices" in resp_data:
            for choice in resp_data["choices"]:
                if "logprobs" in choice and choice["logprobs"]:
                    lp = choice["logprobs"]
                    if isinstance(lp, dict) and "content" in lp:
                        content = lp["content"]
                        tokens = []
                        token_logprobs = []
                        top_logprobs_list = []
                        for entry in content:
                            tokens.append(entry.get("token", ""))
                            token_logprobs.append(entry.get("logprob"))
                            top_entries = entry.get("top_logprobs", [])
                            top_dict = {}
                            for te in top_entries:
                                top_dict[te.get("token", "")] = te.get("logprob", 0.0)
                            top_logprobs_list.append(top_dict if top_dict else {})
                        choice["logprobs"] = {
                            "tokens": tokens,
                            "token_logprobs": token_logprobs,
                            "top_logprobs": top_logprobs_list,
                        }
                elif "logprobs" in choice and choice["logprobs"] is None:
                    choice["logprobs"] = {
                        "tokens": [],
                        "token_logprobs": [],
                        "top_logprobs": [],
                    }

        self._send_json(200, resp_data)

    def _send_json(self, status_code, data):
        try:
            body = json.dumps(data).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, format, *args):
        pass


def main():
    print(f"[proxy] Starting logprobs proxy on port {PROXY_PORT}")
    print(f"[proxy] Upstream: {UPSTREAM_URL}")
    print(f"[proxy] n_probs: {N_PROBS}, max_continuation: {MAX_CONTINUATION}")
    print(f"[proxy] Endpoints: /tokenizer_info, /tokenize, /detokenize, /v1/completions")
    print(f"[proxy] Connection pool: 50 connections")

    server = ThreadingHTTPServer(("0.0.0.0", PROXY_PORT), ProxyHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()

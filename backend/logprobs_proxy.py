#!/usr/bin/env python3
"""
LLM Completions Proxy for lm-eval.

Fixes the loglikelihood evaluation pipeline between lm-eval and llama.cpp.

Problem: lm-eval's local-completions sends prompt=[context+continuation token IDs],
echo=True, max_tokens=1, and expects logprobs for ALL tokens so it can slice
[ctxlen:-1] to get continuation logprobs. But llama.cpp only returns logprobs
for generated tokens, never for prompt tokens.

Solution: The proxy intercepts loglikelihood requests and computes per-token
logprobs using llama.cpp's prompt caching. It evaluates incrementally to get
the logprob of each continuation token given its prefix.

Also provides /tokenizer_info, /tokenize, /detokenize endpoints so lm-eval can
use tokenizer_backend=remote, ensuring token IDs match llama.cpp's tokenizer.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import requests

UPSTREAM_URL = os.environ.get("LLAMACPP_API_URL", "http://llamacpp-api:8080")
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
    """Tokenize text via llama.cpp. Returns list of token IDs."""
    try:
        r = _session.post(
            f"{UPSTREAM_URL}/tokenize",
            json={"content": text},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("tokens", [])
    except Exception:
        pass
    return None


def _upstream_detokenize(token_ids):
    """Detokenize token IDs via llama.cpp. Returns string."""
    try:
        r = _session.post(
            f"{UPSTREAM_URL}/detokenize",
            json={"tokens": token_ids},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("content", "")
    except Exception:
        pass
    return None


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


def compute_logprobs_for_sequence(token_ids):
    """
    Compute logprobs for each token position in the sequence.

    lm-eval uses token_logprobs[ctxlen:-1] where ctxlen separates context
    from continuation. We don't know ctxlen, but continuation is always at
    the END of the sequence. So we only compute real logprobs for the last
    MAX_CONTINUATION positions. Earlier positions get dummy values since
    lm-eval ignores them.

    Returns a list of dicts, one per token position, containing:
    - token: the token text
    - logprob: log P(token[i] | tokens[0..i-1]), or None for position 0
    - top_logprobs: dict of {token_text: logprob} for top-N tokens
    """
    n = len(token_ids)
    if n == 0:
        return []

    results = []

    # Determine which positions need real logprobs
    # Only the last MAX_CONTINUATION positions matter (the continuation part)
    compute_start = max(1, n - MAX_CONTINUATION)

    # Fill dummy values for positions 0..compute_start-1
    for i in range(compute_start):
        token_text = _upstream_detokenize([token_ids[i]]) or ""
        results.append({
            "token": token_text,
            "logprob": None if i == 0 else 0.0,
            "top_logprobs": {},
        })

    if n == 1:
        return results

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
        for i in range(compute_start, n):
            token_text = _upstream_detokenize([token_ids[i]]) or ""
            results.append({
                "token": token_text,
                "logprob": -100.0,
                "top_logprobs": {},
            })
        return results

    # Compute real logprobs for positions compute_start..n-1
    for i in range(compute_start, n):
        prefix = token_ids[:i]
        target_id = token_ids[i]
        token_text = _upstream_detokenize([target_id]) or ""

        logprob, top_logprobs = _get_token_logprob_at_position(prefix, target_id)

        results.append({
            "token": token_text,
            "logprob": logprob if logprob is not None else -100.0,
            "top_logprobs": top_logprobs,
        })

    return results


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
            self._handle_loglikelihood(prompt, logprobs_flag, ctxlens)
        else:
            self._handle_normal_completions(payload)

    def _handle_loglikelihood(self, token_ids, logprobs_flag, ctxlens=None):
        """
        Handle loglikelihood evaluation.
        
        lm-eval sends token IDs from context + continuation, with echo=True, max_tokens=1.
        We need to return logprobs for all positions so parse_logprobs can slice [ctxlen:-1].
        
        If ctxlens is provided, we only compute real logprobs for continuation positions
        (ctxlen to N-1), using dummy values for context positions.
        """
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

        # Build results: dummy values for all positions, then fill in computed ones
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
            "model": "Qwen3.6-27B-Q6_K",
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

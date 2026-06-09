#!/usr/bin/env python3
"""Shared helpers for MTP benchmark scripts (llama-server HTTP harness)."""
from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_mtp_log_parser():
    import importlib.util

    candidates = [
        _REPO_ROOT / "backend" / "modules" / "mtp_log_parser.py",
        Path(__file__).resolve().parent / "mtp_log_parser.py",
    ]
    path = next((p for p in candidates if p.is_file()), candidates[0])
    spec = importlib.util.spec_from_file_location("mtp_log_parser", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load mtp_log_parser from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_mtp_stats_from_log_line


parse_mtp_stats_from_log_line = _load_mtp_log_parser()


DEFAULT_DECODE_PROMPT = (
    "Write a Python function `merge_sorted(a, b)` that merges two sorted lists "
    "into one sorted list in O(n) time. Include a short docstring and two assert tests."
)

DEFAULT_PROMPT_PROCESS_PROMPT = (
    "Summarize the following API design in three bullet points:\n\n"
    + ("The service exposes REST endpoints for model deployment, log streaming, "
       "and VRAM estimation. Clients authenticate with bearer tokens. "
       * 40)
)


@dataclass
class MtpServerConfig:
    model_path: str
    host: str = "127.0.0.1"
    port: int = 18080
    n_ctx: int = 8192
    n_gpu_layers: int = 99
    parallel_slots: int = 1
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    flash_attn: str = "on"
    tensor_split: Optional[str] = None
    main_gpu: int = 0
    mtp_enabled: bool = False
    draft_n_max: int = 3
    draft_n_min: int = 0
    draft_p_min: float = 0.75
    extra_args: List[str] = field(default_factory=list)


@dataclass
class InferenceMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_to_first_token_ms: float = 0.0
    decode_tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    total_time_ms: float = 0.0
    error: Optional[str] = None


def find_llama_server() -> str:
    env = os.environ.get("LLAMA_SERVER")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    for candidate in (
        _REPO_ROOT / "llama.cpp" / "build" / "bin" / "llama-server",
        Path("/usr/local/bin/llama-server"),
        Path("/build/llama.cpp/build/bin/llama-server"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which("llama-server")
    if found:
        return found
    raise FileNotFoundError(
        "llama-server not found. Set LLAMA_SERVER or build llama.cpp (b9193+ for MTP)."
    )


def get_gpu_vram_used_mb() -> Optional[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        values = [int(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit()]
        return sum(values) if values else None
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def build_server_command(cfg: MtpServerConfig, server_bin: str) -> List[str]:
    if not Path(cfg.model_path).is_file():
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")
    cmd: List[str] = [
        server_bin,
        "--model",
        cfg.model_path,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "-c",
        str(cfg.n_ctx),
        "-ngl",
        str(cfg.n_gpu_layers),
        "--parallel",
        str(cfg.parallel_slots),
        "--cache-type-k",
        cfg.cache_type_k,
        "--cache-type-v",
        cfg.cache_type_v,
        "--flash-attn",
        cfg.flash_attn,
        "--metrics",
        "--cont-batching",
    ]
    if cfg.tensor_split:
        cmd.extend(["--tensor-split", cfg.tensor_split])
    if cfg.main_gpu is not None:
        cmd.extend(["--main-gpu", str(cfg.main_gpu)])
    if cfg.mtp_enabled:
        cmd.extend(
            [
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(cfg.draft_n_max),
                "--spec-draft-n-min",
                str(cfg.draft_n_min),
                "--spec-draft-p-min",
                str(cfg.draft_p_min),
            ]
        )
    cmd.extend(cfg.extra_args)
    return cmd


def infer_model_family(model_path: str) -> str:
    name = Path(model_path).stem.lower()
    if "qwen3" in name or "qwen3.6" in name or "qwen-3" in name:
        return "qwen3.6"
    if "glm" in name:
        return "glm"
    if "llama" in name:
        return "llama"
    return "default"


def expand_env_path(path: str) -> str:
    path = os.path.expanduser(path)

    def _default_sub(match: re.Match) -> str:
        var = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var, default)

    path = re.sub(r"\$\{([^}:]+):-([^}]*)\}", _default_sub, path)
    return os.path.expandvars(path)


class LlamaServerRunner:
    """Start/stop llama-server and collect MTP stats from stderr."""

    def __init__(self, cfg: MtpServerConfig, server_bin: Optional[str] = None):
        self.cfg = cfg
        self.server_bin = server_bin or find_llama_server()
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_lines: List[str] = []
        self._mtp_stats: List[Dict[str, Any]] = []
        self._reader_thread: Optional[threading.Thread] = None
        self.vram_before_mb: Optional[int] = None
        self.vram_after_mb: Optional[int] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.host}:{self.cfg.port}"

    def start(self, timeout_seconds: float = 300.0) -> None:
        cmd = build_server_command(self.cfg, self.server_bin)
        self.vram_before_mb = get_gpu_vram_used_mb()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader_thread.start()
        deadline = time.time() + timeout_seconds
        health_url = f"{self.base_url}/health"
        while time.time() < deadline:
            if self._proc.poll() is not None:
                err = "".join(self._stderr_lines[-50:])
                raise RuntimeError(f"llama-server exited early (code {self._proc.returncode}): {err}")
            try:
                urllib.request.urlopen(health_url, timeout=2)
                self.vram_after_mb = get_gpu_vram_used_mb()
                return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"llama-server did not become healthy within {timeout_seconds}s")

    def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            line = line.rstrip("\n")
            self._stderr_lines.append(line)
            stats = parse_mtp_stats_from_log_line(line)
            if stats:
                stats["timestamp"] = time.time()
                self._mtp_stats.append(stats)

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        self._proc = None
        if self._reader_thread:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None

    def latest_acceptance_rate(self) -> Optional[float]:
        for entry in reversed(self._mtp_stats):
            rate = entry.get("acceptance_rate")
            if rate is not None:
                return float(rate)
        return None

    def aggregate_acceptance(self) -> Dict[str, Any]:
        rates = [float(e["acceptance_rate"]) for e in self._mtp_stats if e.get("acceptance_rate") is not None]
        accepted = sum(int(e.get("tokens_accepted") or 0) for e in self._mtp_stats)
        drafted = sum(int(e.get("tokens_drafted") or 0) for e in self._mtp_stats)
        return {
            "acceptance_rate_mean": sum(rates) / len(rates) if rates else None,
            "acceptance_rate_last": self.latest_acceptance_rate(),
            "tokens_accepted": accepted or None,
            "tokens_drafted": drafted or None,
            "samples": len(self._mtp_stats),
        }


def _parse_sse_chat_completion(
    raw: str,
    start_time: float,
    first_token_time_ref: List[Optional[float]],
) -> Tuple[int, int, str]:
    prompt_tokens = 0
    completion_tokens = 0
    full_text = ""
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if first_token_time_ref[0] is None:
            delta = data.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                first_token_time_ref[0] = time.perf_counter()
        timings = data.get("timings") or data.get("__verbose", {}).get("timings")
        if timings:
            if timings.get("predicted_per_second"):
                pass  # applied after loop if usage missing
        usage = data.get("usage") or {}
        if usage.get("prompt_tokens"):
            prompt_tokens = int(usage["prompt_tokens"])
        if usage.get("completion_tokens"):
            completion_tokens = int(usage["completion_tokens"])
        delta = data.get("choices", [{}])[0].get("delta", {})
        chunk = delta.get("content") or delta.get("reasoning_content") or ""
        if chunk:
            full_text += chunk
    return prompt_tokens, completion_tokens, full_text


def run_chat_completion(
    base_url: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 42,
    stream: bool = True,
    timeout: float = 180.0,
    api_key: Optional[str] = None,
) -> Tuple[InferenceMetrics, str]:
    from urllib.parse import urlparse

    parsed_url = urlparse(f"{base_url.rstrip('/')}/v1/chat/completions")
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": stream,
        "stream_options": {"include_usage": True},
    }
    payload = json.dumps(body).encode("utf-8")
    auth_headers: Dict[str, str] = {}
    if api_key:
        auth_headers["Authorization"] = f"Bearer {api_key}"
    metrics = InferenceMetrics()
    full_response = ""
    start = time.perf_counter()
    first_token_time: List[Optional[float]] = [None]
    prompt_tokens = 0
    completion_tokens = 0
    try:
        conn = http.client.HTTPConnection(parsed_url.hostname, parsed_url.port or 80, timeout=timeout)
        conn.request(
            "POST",
            parsed_url.path or "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json", **auth_headers},
        )
        resp = conn.getresponse()
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read().decode('utf-8', errors='replace')[:500]}")
        if stream:
            chunks: List[str] = []
            while True:
                line = resp.readline()
                if not line:
                    break
                chunks.append(line.decode("utf-8", errors="replace"))
            raw = "".join(chunks)
            prompt_tokens, completion_tokens, full_response = _parse_sse_chat_completion(
                raw, start, first_token_time
            )
        else:
            raw = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            msg = parsed.get("choices", [{}])[0].get("message", {}) or {}
            full_response = msg.get("content") or msg.get("reasoning_content") or ""
            usage = parsed.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            timings = parsed.get("timings") or {}
            first_token_time[0] = time.perf_counter()
        conn.close()
        end = time.perf_counter()
        if prompt_tokens == 0:
            prompt_tokens = max(1, len(prompt) // 4)
        if completion_tokens == 0 and full_response:
            completion_tokens = max(1, len(full_response) // 4)
        ttft_ms = (
            (first_token_time[0] - start) * 1000 if first_token_time[0] else (end - start) * 1000
        )
        gen_start = first_token_time[0] or start
        gen_s = max(end - gen_start, 1e-6)
        prefill_s = max((first_token_time[0] or end) - start, 1e-6)
        metrics.prompt_tokens = prompt_tokens
        metrics.completion_tokens = completion_tokens
        metrics.time_to_first_token_ms = round(ttft_ms, 2)
        metrics.total_time_ms = round((end - start) * 1000, 2)
        if not stream and timings:
            if timings.get("predicted_per_second") is not None:
                metrics.decode_tokens_per_second = round(float(timings["predicted_per_second"]), 2)
            elif timings.get("predicted_ms") and completion_tokens:
                metrics.decode_tokens_per_second = round(
                    completion_tokens / (float(timings["predicted_ms"]) / 1000.0), 2
                )
            else:
                metrics.decode_tokens_per_second = round(completion_tokens / gen_s, 2)
            if timings.get("prompt_per_second") is not None:
                metrics.prompt_tokens_per_second = round(float(timings["prompt_per_second"]), 2)
            else:
                metrics.prompt_tokens_per_second = round(prompt_tokens / prefill_s, 2)
            if timings.get("prompt_ms") is not None:
                metrics.time_to_first_token_ms = round(float(timings["prompt_ms"]), 2)
        else:
            metrics.decode_tokens_per_second = round(completion_tokens / gen_s, 2)
            metrics.prompt_tokens_per_second = round(prompt_tokens / prefill_s, 2)
        return metrics, full_response
    except Exception as exc:
        metrics.error = str(exc)
        return metrics, ""


def run_benchmark_trial(
    runner: LlamaServerRunner,
    *,
    warmup: bool = True,
    decode_prompt: str = DEFAULT_DECODE_PROMPT,
    prompt_process_prompt: str = DEFAULT_PROMPT_PROCESS_PROMPT,
    max_tokens: int = 256,
    seed: int = 42,
) -> Dict[str, Any]:
    if warmup:
        run_chat_completion(
            runner.base_url,
            "Say OK.",
            max_tokens=8,
            temperature=0.0,
            seed=seed,
        )
        time.sleep(0.5)

    decode_metrics, _ = run_chat_completion(
        runner.base_url,
        decode_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed,
    )
    time.sleep(0.3)
    prompt_metrics, _ = run_chat_completion(
        runner.base_url,
        prompt_process_prompt,
        max_tokens=64,
        temperature=0.0,
        seed=seed + 1,
    )
    acceptance = runner.aggregate_acceptance()
    vram_delta = None
    if runner.vram_before_mb is not None and runner.vram_after_mb is not None:
        vram_delta = runner.vram_after_mb - runner.vram_before_mb

    return {
        "decode": decode_metrics.__dict__,
        "prompt_process": prompt_metrics.__dict__,
        "acceptance": acceptance,
        "vram_used_mb_after_start": runner.vram_after_mb,
        "vram_delta_mb": vram_delta,
    }


def load_yaml_matrix(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML required: pip install pyyaml") from exc
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_matrix_cases(matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand matrix YAML into individual benchmark cases."""
    models = matrix.get("models") or []
    if not models:
        raise ValueError("matrix must define at least one model")

    parallel_slots = matrix.get("parallel_slots") or [1]
    draft_n_max_values = matrix.get("mtp_draft_n_max") or [2, 3, 4, 5]
    draft_p_min_values = matrix.get("mtp_draft_p_min") or [0.5, 0.75, 0.9]
    cache_types = matrix.get("cache_types") or ["q8_0"]
    n_ctx = int(matrix.get("n_ctx", 8192))
    n_gpu_layers = int(matrix.get("n_gpu_layers", 99))
    repetitions = int(matrix.get("repetitions", 1))
    run_baseline = bool(matrix.get("run_baseline", True))
    max_tokens = int(matrix.get("max_tokens", 256))
    port_base = int(matrix.get("port_base", 18080))

    cases: List[Dict[str, Any]] = []
    port_offset = 0
    for model_entry in models:
        model_path = expand_env_path(str(model_entry.get("path", "")))
        model_id = str(model_entry.get("id") or Path(model_path).stem)
        family = str(model_entry.get("family") or infer_model_family(model_path))
        quant = model_entry.get("quant")
        for parallel in parallel_slots:
            for cache in cache_types:
                cache_k = cache
                cache_v = str(matrix.get("cache_type_v") or cache)
                base_kwargs = {
                    "model_id": model_id,
                    "model_path": model_path,
                    "family": family,
                    "quant": quant,
                    "parallel_slots": int(parallel),
                    "cache_type_k": cache_k,
                    "cache_type_v": cache_v,
                    "n_ctx": n_ctx,
                    "n_gpu_layers": n_gpu_layers,
                    "max_tokens": max_tokens,
                }
                if run_baseline:
                    for rep in range(repetitions):
                        cases.append(
                            {
                                **base_kwargs,
                                "case_id": f"{model_id}_np{parallel}_{cache}_baseline_r{rep}",
                                "mtp_enabled": False,
                                "draft_n_max": 3,
                                "draft_p_min": 0.75,
                                "repetition": rep,
                                "port": port_base + port_offset,
                            }
                        )
                        port_offset += 1
                for draft_n_max in draft_n_max_values:
                    for draft_p_min in draft_p_min_values:
                        for rep in range(repetitions):
                            cases.append(
                                {
                                    **base_kwargs,
                                    "case_id": (
                                        f"{model_id}_np{parallel}_{cache}_"
                                        f"mtp{draft_n_max}_p{draft_p_min}_r{rep}"
                                    ),
                                    "mtp_enabled": True,
                                    "draft_n_max": int(draft_n_max),
                                    "draft_n_min": int(matrix.get("mtp_draft_n_min", 0)),
                                    "draft_p_min": float(draft_p_min),
                                    "repetition": rep,
                                    "port": port_base + port_offset,
                                }
                            )
                            port_offset += 1
    return cases

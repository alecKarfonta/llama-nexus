#!/usr/bin/env python3
"""
Run lm-eval with API-based local-completions model.
Sets up proper authentication headers for the llama.cpp server.
"""

import sys
import os
import json
import subprocess


def main():
    api_base_url = os.environ.get("LLAMACPP_API_URL", "http://llamacpp-api:8080")
    api_key = os.environ.get("LM_EVAL_API_KEY", "placeholder-api-key")

    # Strip /v1 if present since lm-eval appends it
    api_base_url = api_base_url.replace("/v1", "").rstrip("/")

    args = sys.argv[1:]

    # Inject model and model_args if not already present
    has_model = "--model" in args or "-M" in args
    has_model_args = "--model_args" in args or "-a" in args

    if not has_model:
        args.extend(["--model", "local-completions"])

    if not has_model_args:
        # Build header dict for auth
        header_dict = json.dumps({"Authorization": f"Bearer {api_key}"})
        args.extend([
            "--model_args",
            f"base_url={api_base_url}/v1,pretrained=Qwen/Qwen2.5-7B,header={header_dict}",
        ])

    cmd = ["lm-eval", "run"] + args
    print(f"[run_lm_eval_api] Running: {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

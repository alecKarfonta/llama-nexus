#!/bin/bash
set -e

API_MODELS=$(python3 -c "import lm_eval.models.api_models; print(lm_eval.models.api_models.__file__)")
if [ -f "$API_MODELS" ]; then
    # Patch 1: Fix UnboundLocalError
    sed -i 's/eval_logger.error(f"Exception:{repr(e)}, {outputs}, retrying.")/eval_logger.error(f"Exception:{repr(e)}, retrying.")/' "$API_MODELS"

    python3 << 'PYEOF'
import lm_eval.models.api_models
path = lm_eval.models.api_models.__file__
with open(path) as f:
    content = f.read()

# Patch 2: Return defaults on exception instead of crashing
old = """        except BaseException as e:
            eval_logger.error(f"Exception:{repr(e)}, retrying.")
            raise e"""
new = """        except BaseException as e:
            eval_logger.error(f"Exception:{repr(e)}, returning default values.")
            if generate:
                return [""] * len(messages)
            else:
                return [(0.0, False)] * len(messages)"""
if old in content:
    content = content.replace(old, new)
    print("Patched amodel_call to return defaults on exception")

# Patch 3: Include ctxlens in API payload so proxy knows where context ends
old_payload = """        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            **kwargs,
        )"""
new_payload = """        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            **kwargs,
        )
        if not generate and ctxlens:
            payload["ctxlens"] = ctxlens"""
if old_payload in content:
    content = content.replace(old_payload, new_payload)
    print("Patched amodel_call to include ctxlens in payload")

with open(path, "w") as f:
    f.write(content)
PYEOF
    echo "[entrypoint] Patched lm-eval api_models.py"
fi

# Start the logprobs proxy in the background
python3 /app/logprobs_proxy.py &
PROXY_PID=$!
echo "[entrypoint] Started logprobs proxy (PID $PROXY_PID) on port ${PROXY_PORT:-8888}"

sleep 1

# Start the worker (foreground)
python3 /app/lm_eval_worker.py &
WORKER_PID=$!

wait -n $PROXY_PID $WORKER_PID 2>/dev/null
EXIT_CODE=$?

kill $PROXY_PID $WORKER_PID 2>/dev/null || true
exit $EXIT_CODE

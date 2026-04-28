#!/bin/bash
set -e

# Start the logprobs proxy in the background
python3 /app/logprobs_proxy.py &
PROXY_PID=$!
echo "[entrypoint] Started logprobs proxy (PID $PROXY_PID) on port ${PROXY_PORT:-8888}"

# Wait briefly for proxy to be ready
sleep 1

# Start the worker (foreground)
python3 /app/lm_eval_worker.py &
WORKER_PID=$!

# Wait for either process to exit
wait -n $PROXY_PID $WORKER_PID 2>/dev/null
EXIT_CODE=$?

# Kill remaining processes
kill $PROXY_PID $WORKER_PID 2>/dev/null || true
exit $EXIT_CODE

#!/usr/bin/env bash
# [TIME] | INFO | SCALE_REPLAY | fan the MTP replay out across every available
# llama-server slot: the local 4x3090 Ti rig, the 1-GPU pod, and a 4-GPU pod.
#
# Topology this builds:
#   local rig   :8600 (existing tee, --parallel 1)        -> 1 worker
#   pod-1x      :8602 (existing tee, --parallel 1)        -> 1 worker
#   pod-4x      :8603..8606 (new tees, one per GPU,       -> 2 workers each
#                --parallel 2 => 98k ctx per slot)
#
# Ten workers total, each taking one of ten equal session shards.
#
# Equal shards cost far more than the ~8% originally estimated here. Measured
# over a 400-session run the per-session cost spread 271s..710s (2.6x) because
# shards are assigned by hash and session depth is not balanced across them.
# The tail is set by the slowest shard, so the run needed 3.2h when eight of ten
# workers were done in 1.9h - the straggler added ~70%, not 8%. Two workers
# pinned to the same GPU made it worse by contending with each other.
#
# Fix on the next run: hand out sessions from a shared work queue instead of
# static shards, or sort sessions by depth and deal them round-robin so every
# shard carries a similar mix.
#
# Every worker replays a disjoint slice of the same --stack natural selection,
# so restarting reshuffles which sessions each machine owns. Captures append and
# dataset_build.py dedupes, so a restart never loses or double-counts samples.
#
# Usage:
#   POD4_ID=xxxx POD4_KEY=... ./scripts/scale_replay.sh start
#   ./scripts/scale_replay.sh status
#   ./scripts/scale_replay.sh watch    # block until done, exit 2 on stall
#   ./scripts/scale_replay.sh stop

set -uo pipefail
cd "$(dirname "$0")/.."

ACTION="${1:-status}"
DATASETS=/mnt/comfyui-models/llama-nexus/datasets
POD4_ID="${POD4_ID:-$(cat /tmp/rp_pod4id 2>/dev/null || echo '')}"
POD4_KEY="${POD4_KEY:-$(cat /tmp/rp_pod4_key 2>/dev/null || echo '')}"
POD4_GPUS="${POD4_GPUS:-4}"
POD4_SLOTS="${POD4_SLOTS:-2}"
SHARDS="${SHARDS:-10}"
REPLAY_ARGS=(--source transcripts --corpus all --stack natural --sessions 400
             --max-turns 25 --deep-frac 0.35 --timeout 900)

say() { echo "[$(date -u +%H:%M:%S)] | INFO | scale_replay | $*"; }

# pgrep -f would match this script and the caller's own shell, and a stray PID
# in the list makes `kill` fail for the whole batch. Match on the interpreter +
# script path and drop our own process tree explicitly.
replay_pids() {
  pgrep -f 'python3 scripts/capture_cursor_synth\.py' 2>/dev/null \
    | grep -vx "$$" || true
}
tee_pids() {
  pgrep -f 'python3 backend/capture_tee\.py' 2>/dev/null | grep -vx "$$" || true
}

case "$ACTION" in
stop)
  for p in $(replay_pids); do kill "$p" 2>/dev/null && say "killed worker $p"; done
  ;;

start)
  [ -z "$POD4_ID" ] && { say "ERROR POD4_ID unset and /tmp/rp_pod4id empty"; exit 1; }
  [ -z "$POD4_KEY" ] && { say "ERROR POD4_KEY unset and /tmp/rp_pod4_key empty"; exit 1; }

  # Refuse to start unless every pod-4x server answers, otherwise workers burn
  # their shard against a dead endpoint and those sessions are silently skipped.
  for i in $(seq 0 $((POD4_GPUS - 1))); do
    url="https://$POD4_ID-$((8080 + i)).proxy.runpod.net"
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 15 \
             -H "Authorization: Bearer $POD4_KEY" \
             -H 'User-Agent: llama-nexus-bench/1.0' "$url/health" 2>/dev/null)
    if [ "$code" != "200" ]; then
      say "ERROR pod-4x gpu$i ($url) health=$code - not starting"; exit 1
    fi
    say "pod-4x gpu$i healthy"
  done

  for p in $(replay_pids); do kill "$p" 2>/dev/null && say "stopped old worker $p"; done
  sleep 3
  mkdir -p "$DATASETS/pod4" logs

  # One tee per pod-4x GPU. They share a capture dir; _write_record holds a
  # threading lock plus flock, so concurrent appends stay line-atomic.
  for i in $(seq 0 $((POD4_GPUS - 1))); do
    port=$((8603 + i))
    if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
      say "tee :$port already listening, reusing"
    else
      TEE_PORT="$port" \
      TEE_UPSTREAM="https://$POD4_ID-$((8080 + i)).proxy.runpod.net" \
      TEE_UPSTREAM_API_KEY="$POD4_KEY" \
      TEE_UPSTREAM_USER_AGENT='llama-nexus-bench/1.0' \
      TEE_CAPTURE_DIR="$DATASETS/pod4" \
        setsid nohup python3 backend/capture_tee.py \
          >"logs/tee-$port.log" 2>&1 &
      say "tee :$port -> pod-4x gpu$i"
      sleep 1
    fi
  done

  # Shard 0 = local rig, shard 1 = pod-1x, shards 2.. = pod-4x slots.
  shard=0
  launch() {  # launch <url> <key> <label>
    setsid nohup python3 scripts/capture_cursor_synth.py "${REPLAY_ARGS[@]}" \
      --shards "$SHARDS" --shard "$shard" \
      ${1:+--url "$1"} ${2:+--key "$2"} \
      >"logs/replay-$3-shard$shard.log" 2>&1 &
    say "worker shard $shard/$SHARDS -> $3"
    shard=$((shard + 1))
    sleep 1
  }

  launch '' '' local
  launch 'http://127.0.0.1:8602/v1/chat/completions' \
         "$(cat /tmp/rp_pod_key 2>/dev/null)" pod1x
  for i in $(seq 0 $((POD4_GPUS - 1))); do
    for _ in $(seq 1 "$POD4_SLOTS"); do
      [ "$shard" -ge "$SHARDS" ] && break
      launch "http://127.0.0.1:$((8603 + i))/v1/chat/completions" \
             "$POD4_KEY" "pod4x-gpu$i"
    done
  done
  say "started $shard workers across $SHARDS shards"
  ;;

watch)
  # Block until every worker exits, then print a terminal marker. Polling for
  # worker count rather than sleeping a fixed ETA means this returns as soon as
  # the run really ends instead of at a guess, and it surfaces a stall instead
  # of sleeping through one.
  #
  # A worker that finishes its shard exits on its own, so zero workers means
  # done. Requiring two consecutive zero reads avoids tripping on the instant
  # between pgrep and a fork.
  INTERVAL="${INTERVAL:-120}"
  MAX_HOURS="${MAX_HOURS:-3}"
  deadline=$(( $(date +%s) + $(python3 -c "print(int($MAX_HOURS*3600))") ))
  zero=0
  last_total=0
  stalled=0
  while :; do
    n=$(replay_pids | wc -l)
    total=$(cat "$DATASETS"/captures-*.jsonl "$DATASETS"/pod/captures-*.jsonl \
                "$DATASETS"/pod4/captures-*.jsonl 2>/dev/null | wc -l)
    say "watch: workers=$n captures=$total (+$((total - last_total)) since last poll)"

    # A live worker set that is not producing captures is the failure mode worth
    # waking up for: dead upstream, exhausted balance, wedged tee.
    if [ "$n" -gt 0 ] && [ "$total" -le "$last_total" ]; then
      stalled=$((stalled + 1))
      if [ "$stalled" -ge 3 ]; then
        say "REPLAY STALLED - $n workers alive, no new captures in $((3*INTERVAL))s"
        exit 2
      fi
    else
      stalled=0
    fi
    last_total=$total

    if [ "$n" -eq 0 ]; then
      zero=$((zero + 1))
      [ "$zero" -ge 2 ] && { say "REPLAY COMPLETE - all workers exited"; exit 0; }
    else
      zero=0
    fi

    if [ "$(date +%s)" -ge "$deadline" ]; then
      say "REPLAY TIMEOUT - ${MAX_HOURS}h cap hit with $n workers still alive"
      exit 3
    fi
    sleep "$INTERVAL"
  done
  ;;

status)
  say "workers: $(replay_pids | wc -l)   tees: $(tee_pids | wc -l)"
  python3 - "$DATASETS" <<'PY'
import glob, json, os, sys, datetime as dt
root = sys.argv[1]
now = dt.datetime.now(dt.timezone.utc)
srcs = [("local", f"{root}/captures-*.jsonl"),
        ("pod1x", f"{root}/pod/captures-*.jsonl"),
        ("pod4x", f"{root}/pod4/captures-*.jsonl")]
tot = 0.0
for label, pat in srcs:
    ts = []
    for f in glob.glob(pat):
        for line in open(f):
            try:
                r = json.loads(line)
                ts.append(dt.datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%S%z"))
            except Exception:
                pass
    if not ts:
        continue
    w = 10
    rate = len([t for t in ts if (now - t).total_seconds() <= w * 60]) / w
    tot += rate
    print(f"  {label:6s} total={len(ts):6,d}  {rate:5.2f} req/min")
print(f"  {'TOTAL':6s} {' ':13s}{tot:5.2f} req/min")
PY
  ;;
*)
  echo "usage: $0 {start|stop|status|watch}"; exit 1;;
esac

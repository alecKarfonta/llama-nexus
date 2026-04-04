#!/bin/bash
set -eo pipefail

LLAMA_BENCH="/home/alec/git/llama-nexus/llama.cpp-spiritbuun/build/bin/llama-bench"
MODEL="/home/alec/git/llama-nexus/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OUTFILE="/home/alec/git/llama-nexus/results/turboquant_speed_spiritbuun_stable.csv"

# Clear or create file
> "$OUTFILE"
HEADER_WRITTEN=false

TYPES=("f16" "q8_0" "q4_0" "turbo4" "turbo3")

for CT in "${TYPES[@]}"; do
    echo "Running benchmark for cache type: $CT..."
    # We do a separate run per cache type to avoid the state leak / segfault
    TMP_CSV=$(mktemp)
    
    # Notice we removed 32768 for the dense types if it's the cause of OOM, 
    # but let's try 32768 first and if it fails, we catch the error and continue.
    set +e
    LD_LIBRARY_PATH=/home/alec/git/llama-nexus/llama.cpp-spiritbuun/build/bin $LLAMA_BENCH \
        -m $MODEL \
        -ngl 99 -r 5 \
        -ctk $CT -ctv $CT \
        -fa 1 \
        -p 512,16384,32768 \
        -n 128 \
        -o csv > "$TMP_CSV" 2> /tmp/llama_bench_err_${CT}.log
    EXIT_CODE=$?
    set -e
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Warning: Cache type $CT crashed. See /tmp/llama_bench_err_${CT}.log"
    fi
    
    if [ "$HEADER_WRITTEN" = false ]; then
        cat "$TMP_CSV" > "$OUTFILE"
        HEADER_WRITTEN=true
    else
        # append avoiding the header (line 1)
        tail -n +2 "$TMP_CSV" >> "$OUTFILE"
    fi
    
    rm "$TMP_CSV"
done

echo "Done! Results in $OUTFILE"

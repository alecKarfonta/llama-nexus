#!/bin/bash
LD_LIBRARY_PATH=/home/alec/git/llama-nexus/llama.cpp-spiritbuun/build/bin /home/alec/git/llama-nexus/llama.cpp-spiritbuun/build/bin/llama-bench \
    -m /home/alec/git/llama-nexus/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -ngl 99 -r 5 \
    -ctk f16,q8_0,q4_0,turbo4,turbo3 \
    -ctv f16,q8_0,q4_0,turbo4,turbo3 \
    -fa 1 \
    -p 512,16384,32768 \
    -n 128 \
    -o csv > /home/alec/git/llama-nexus/results/turboquant_speed_spiritbuun_full.csv

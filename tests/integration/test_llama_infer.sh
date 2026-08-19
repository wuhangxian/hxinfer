#!/bin/bash
# Integration test: LLaMA-2-7B short prompt inference
BUILD_DIR=${HXINFER_BUILD_DIR:-/data/dorianwu/hxinfer/build}
LLAMA_DIR=${HXINFER_LLAMA_DIR:-/data/models/Llama-2-7b-hf}

OUTPUT=$(HXINFER_DATA_DIR=$LLAMA_DIR timeout 20 $BUILD_DIR/demo_gpu_llama7b --greedy 2>&1)
SPEED=$(echo "$OUTPUT" | grep "Speed:" | awk '{print $2}')

echo -n "  LLaMA-2 short prompt: "
if [ -n "$SPEED" ] && (( $(echo "$SPEED > 10" | bc -l) )); then
    echo "PASS (speed=${SPEED} tok/s)"
    exit 0
else
    echo "FAIL (speed=${SPEED})"
    exit 1
fi

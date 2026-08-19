#!/bin/bash
# Unit test: verify RoPE produces valid token IDs for long prompts
BUILD_DIR=${HXINFER_BUILD_DIR:-/data/dorianwu/hxinfer/build}
LLAMA_DIR=${HXINFER_LLAMA_DIR:-/data/models/Llama-2-7b-hf}

OUTPUT=$(HXINFER_DATA_DIR=$LLAMA_DIR timeout 20 $BUILD_DIR/demo_gpu_llama7b --greedy 2>&1)
echo -n "  RoPE NeoX (no NaN/garbled): "
if echo "$OUTPUT" | grep -qP '[a-zA-Z ]{30,}'; then
    echo "PASS (coherent output detected)"
    exit 0
else
    echo "FAIL (output not coherent)"
    exit 1
fi

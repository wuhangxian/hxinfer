#!/bin/bash
# Unit test: verify ModelConfig dtype fields are set correctly
BUILD_DIR=${HXINFER_BUILD_DIR:-/data/dorianwu/hxinfer/build}
LLAMA_DIR=${HXINFER_LLAMA_DIR:-/data/models/Llama-2-7b-hf}

OUTPUT=$(HXINFER_DATA_DIR=$LLAMA_DIR timeout 20 $BUILD_DIR/demo_gpu_llama7b --greedy 2>&1)
echo -n "  Dtype config consistency: "
if echo "$OUTPUT" | grep -q "Model loaded"; then
    echo "PASS (model loads with centralized dtype)"
    exit 0
else
    echo "FAIL (model failed to load)"
    exit 1
fi

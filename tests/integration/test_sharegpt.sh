#!/bin/bash
# Integration test: LLaMA-2 ShareGPT 100-prompt garbled check
BUILD_DIR=${HXINFER_BUILD_DIR:-/data/dorianwu/hxinfer/build}
LLAMA_DIR=${HXINFER_LLAMA_DIR:-/data/models/Llama-2-7b-hf}

if [ ! -f $BUILD_DIR/test_sharegpt_llama ]; then
    echo "  ShareGPT test: SKIP (binary not built)"
    exit 0
fi

OUTPUT=$(HXINFER_DATA_DIR=$LLAMA_DIR timeout 300 $BUILD_DIR/test_sharegpt_llama 2>&1)
GARBLED=$(echo "$OUTPUT" | strings | grep -c "garbled=YES" || true)

echo -n "  ShareGPT (100 prompts): "
if [ "$GARBLED" -le 5 ]; then
    echo "PASS (garbled=$GARBLED/100)"
    exit 0
else
    echo "FAIL (garbled=$GARBLED/100)"
    exit 1
fi

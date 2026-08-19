#!/bin/bash
# Integration test: Qwen2.5-7B short prompt inference
BUILD_DIR=${HXINFER_BUILD_DIR:-/data/dorianwu/hxinfer/build}
QWEN_DIR=${HXINFER_QWEN_DIR:-/data/models/Qwen2.5-7B-Instruct}

OUTPUT=$(HXINFER_DATA_DIR=$QWEN_DIR timeout 30 $BUILD_DIR/demo_qwen 2>&1)
echo -n "  Qwen2.5 short prompt: "
if echo "$OUTPUT" | grep -qP '[a-zA-Z ]{20,}'; then
    SPEED=$(echo "$OUTPUT" | grep "Speed:" | awk '{print $2}')
    echo "PASS (readable English, speed=${SPEED} tok/s)"
    exit 0
else
    echo "FAIL (no readable English)"
    exit 1
fi

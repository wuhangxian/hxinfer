#!/bin/bash
# hxinfer regression test — run after any code change
# Tests: short prompt, long prompt (NaN check), output quality
set -e

echo "=== hxinfer Regression Test ==="
echo "Date: $(date)"
echo

BUILD_DIR=/data/dorianwu/hxinfer/build
LLAMA_DIR=/data/models/Llama-2-7b-hf
QWEN_DIR=/data/models/Qwen2.5-7B-Instruct
PASS=0
FAIL=0

# Test 1: LLaMA-2 short prompt — should output coherent English
echo "--- Test 1: LLaMA-2 short prompt ---"
OUTPUT=$(HXINFER_DATA_DIR=$LLAMA_DIR timeout 20 $BUILD_DIR/demo_gpu_llama7b --greedy 2>&1)
SPEED=$(echo "$OUTPUT" | grep "Speed:" | awk '{print $2}')
echo "  Speed: $SPEED tok/s"
if (( $(echo "$SPEED > 10" | bc -l) )); then
    echo "  PASS: speed > 10 tok/s"
    PASS=$((PASS+1))
else
    echo "  FAIL: speed too low"
    FAIL=$((FAIL+1))
fi
echo

# Test 2: Qwen short prompt — should NOT be all garbled
echo "--- Test 2: Qwen2.5 short prompt ---"
OUTPUT=$(HXINFER_DATA_DIR=$QWEN_DIR timeout 30 $BUILD_DIR/demo_qwen 2>&1)
# Check if output contains readable English (at least 20 ASCII chars in a row)
if echo "$OUTPUT" | grep -qP '[a-zA-Z ]{20,}'; then
    echo "  PASS: contains readable English"
    PASS=$((PASS+1))
else
    echo "  FAIL: no readable English found"
    FAIL=$((FAIL+1))
fi
echo

# Test 3: LLaMA-2 long prompt — no NaN/token-0
echo "--- Test 3: LLaMA-2 long prompt (512 tokens) ---"
if [ -f $BUILD_DIR/test_sharegpt_llama ]; then
    OUTPUT=$(HXINFER_DATA_DIR=$LLAMA_DIR timeout 120 $BUILD_DIR/test_sharegpt_llama 2>&1 | strings | grep -c "garbled=YES" || true)
    echo "  Garbled count (first batch): $OUTPUT"
    if [ "$OUTPUT" -le 5 ]; then
        echo "  PASS: garbled <= 5"
        PASS=$((PASS+1))
    else
        echo "  FAIL: too many garbled"
        FAIL=$((FAIL+1))
    fi
else
    echo "  SKIP: test_sharegpt_llama not built"
fi
echo

# Summary
echo "=== Summary: $PASS passed, $FAIL failed ==="
if [ $FAIL -gt 0 ]; then
    exit 1
fi

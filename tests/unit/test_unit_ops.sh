#!/bin/bash
# Unit test: run all operator unit tests via C++ binary
BUILD_DIR=${HXINFER_BUILD_DIR:-/data/dorianwu/hxinfer/build}

echo -n "  Operator unit tests: "
if [ -f $BUILD_DIR/test_unit_ops ]; then
    OUTPUT=$($BUILD_DIR/test_unit_ops 2>&1)
    PASSED=$(echo "$OUTPUT" | grep "passed" | awk '{print $2}')
    FAILED=$(echo "$OUTPUT" | grep "passed" | awk '{print $4}')
    if [ "$FAILED" -eq 0 ] 2>/dev/null; then
        echo "PASS (${PASSED} assertions, 0 failures)"
        exit 0
    else
        echo "FAIL (${FAILED} failures)"
        exit 1
    fi
else
    echo "SKIP (binary not built)"
    exit 0
fi

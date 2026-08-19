#!/bin/bash
# hxinfer full regression test suite
# Usage: bash tests/run_all.sh
set -e

echo "============================================"
echo "  hxinfer Regression Test Suite"
echo "  $(date)"
echo "============================================"
echo

BUILD_DIR=${HXINFER_BUILD_DIR:-/data/dorianwu/hxinfer/build}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PASS=0
FAIL=0
SKIP=0

echo "--- Unit Tests ---"
for test in $SCRIPT_DIR/unit/*.sh; do
    [ -f "$test" ] || continue
    name=$(basename "$test" .sh)
    if bash "$test" 2>/dev/null; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        echo "    ^ $name FAILED"
    fi
done

echo
echo "--- Integration Tests ---"
for test in $SCRIPT_DIR/integration/*.sh; do
    [ -f "$test" ] || continue
    name=$(basename "$test" .sh)
    if bash "$test" 2>/dev/null; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        echo "    ^ $name FAILED"
    fi
done

echo
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "============================================"
[ $FAIL -eq 0 ]

#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./anemll-profile}"
FIXTURE="${2:-./tests/fixtures/multifunction.mlpackage}"
TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/anemll-profile-test.XXXXXX")"
trap 'rm -rf "$TMPDIR"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_contains() {
    local file="$1"
    local needle="$2"
    grep -Fq "$needle" "$file" || fail "expected '$needle' in $file"
}

assert_not_contains() {
    local file="$1"
    local needle="$2"
    if grep -Fq "$needle" "$file"; then
        fail "did not expect '$needle' in $file"
    fi
}

assert_count() {
    local file="$1"
    local needle="$2"
    local expected="$3"
    local actual
    actual="$(grep -Fc "$needle" "$file")"
    [[ "$actual" == "$expected" ]] || fail "expected '$needle' $expected times in $file, got $actual"
}

run_capture() {
    local name="$1"
    shift
    "$@" >"$TMPDIR/$name.out" 2>&1
}

[[ -x "$BIN" ]] || fail "binary not found: $BIN"
[[ -d "$FIXTURE" ]] || fail "fixture not found: $FIXTURE"

run_capture list "$BIN" --list-functions "$FIXTURE"
assert_contains "$TMPDIR/list.out" "add_one"
assert_contains "$TMPDIR/list.out" "mul_two"

run_capture default "$BIN" "$FIXTURE"
assert_contains "$TMPDIR/default.out" "Function:     add_one"
assert_contains "$TMPDIR/default.out" "ios18.add"
assert_not_contains "$TMPDIR/default.out" "ios18.mul"

run_capture mul_two "$BIN" --function mul_two "$FIXTURE"
assert_contains "$TMPDIR/mul_two.out" "Function:     mul_two"
assert_contains "$TMPDIR/mul_two.out" "ios18.mul"
assert_not_contains "$TMPDIR/mul_two.out" "ios18.add"

run_capture all "$BIN" --all-functions "$FIXTURE"
assert_count "$TMPDIR/all.out" "ANE CostModel Report:" 2
assert_contains "$TMPDIR/all.out" "Function:     add_one"
assert_contains "$TMPDIR/all.out" "Function:     mul_two"

if "$BIN" --function nope "$FIXTURE" >"$TMPDIR/invalid.out" 2>&1; then
    fail "expected invalid function selection to fail"
fi
assert_contains "$TMPDIR/invalid.out" "function 'nope' not found"
assert_contains "$TMPDIR/invalid.out" "Available functions: add_one mul_two"

if "$BIN" --function add_one --all-functions "$FIXTURE" >"$TMPDIR/conflict.out" 2>&1; then
    fail "expected conflicting function selectors to fail"
fi
assert_contains "$TMPDIR/conflict.out" "mutually exclusive selectors"

echo "ok"

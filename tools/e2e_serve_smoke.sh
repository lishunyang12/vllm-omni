#!/usr/bin/env bash
# tools/e2e_serve_smoke.sh — E2E smoke for PR #2383 (config refactor 2/N)
#
# Starts the qwen3_omni vllm-omni server with --deploy-config, waits for
# readiness, sends a single chat completion request, optionally asserts a
# pattern in the server log, and tears down cleanly. Any extra args after
# `--` are forwarded verbatim to `vllm serve`.
#
# Usage:
#
#   # Layer 4 — bare serve (uses deploy YAML defaults)
#   ./tools/e2e_serve_smoke.sh
#
#   # Layer 5 scenario B — explicit CLI override + verify it fired
#   E2E_LOG_GREP='max_num_seqs=8' ./tools/e2e_serve_smoke.sh -- --max-num-seqs 8
#
#   # Layer 5 scenario C — per-stage override via JSON
#   E2E_LOG_GREP='gpu_memory_utilization=0.42' \
#     ./tools/e2e_serve_smoke.sh -- \
#       --stage-overrides '{"0": {"gpu_memory_utilization": 0.42}}'
#
#   # Custom deploy config (e.g. variant)
#   E2E_DEPLOY_CONFIG=vllm_omni/deploy/ci/qwen3_omni_moe.yaml \
#     ./tools/e2e_serve_smoke.sh
#
# Environment variables:
#   E2E_MODEL          (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
#   E2E_PORT           (default: 8091)
#   E2E_DEPLOY_CONFIG  (default: vllm_omni/deploy/qwen3_omni_moe.yaml)
#   E2E_LOG_FILE       (default: /tmp/e2e_serve_smoke_<pid>.log)
#   E2E_READY_TIMEOUT  (default: 900 — 15 min for first-time model load)
#   E2E_LOG_GREP       (default: empty — if set, grep server log for the pattern
#                       after ready and fail if not found)
#   E2E_REQUEST_TIMEOUT (default: 120 — wall-clock for the chat completion call)
#   E2E_KEEP_LOG       (default: 0 — set to 1 to keep the log file after run)
#
# Requirements:
#   - 2x H100-80G (or equivalent) with the model cached
#   - vllm-omni installed in the active Python env
#   - curl, optionally jq for richer response verification
#
# Exits 0 on success, non-zero on first failure.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL="${E2E_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PORT="${E2E_PORT:-8091}"
DEPLOY_CONFIG="${E2E_DEPLOY_CONFIG:-vllm_omni/deploy/qwen3_omni_moe.yaml}"
LOG_FILE="${E2E_LOG_FILE:-/tmp/e2e_serve_smoke_$$.log}"
READY_TIMEOUT="${E2E_READY_TIMEOUT:-900}"
REQUEST_TIMEOUT="${E2E_REQUEST_TIMEOUT:-120}"
LOG_GREP="${E2E_LOG_GREP:-}"
KEEP_LOG="${E2E_KEEP_LOG:-0}"
STREAM_LOG="${E2E_STREAM_LOG:-0}"     # 1 = tail -f the server log to stdout
LOG_SNIPPET_LINES="${E2E_LOG_SNIPPET_LINES:-8}"  # last N lines per polling tick

# Forward anything after `--` to `vllm serve`. If no `--`, EXTRA_ARGS is empty.
EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
    if [[ "$1" == "--" ]]; then
        shift
        EXTRA_ARGS=("$@")
    else
        EXTRA_ARGS=("$@")
    fi
fi

SERVER_PID=""
TAIL_PID=""
RESPONSE_FILE="/tmp/e2e_serve_smoke_response_$$.json"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    if [[ -n "$TAIL_PID" ]] && kill -0 "$TAIL_PID" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null || true
        wait "$TAIL_PID" 2>/dev/null || true
    fi
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "=== Cleaning up server (PID $SERVER_PID) ==="
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 15); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  graceful shutdown timed out, sending SIGKILL"
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$RESPONSE_FILE"
    if [[ -f "$LOG_FILE" ]]; then
        if [[ "$exit_code" -ne 0 ]]; then
            echo ""
            echo "=== FAILURE — last 60 lines of server log ($LOG_FILE) ==="
            tail -n 60 "$LOG_FILE" || true
        fi
        if [[ "$KEEP_LOG" != "1" ]] && [[ "$exit_code" -eq 0 ]]; then
            rm -f "$LOG_FILE"
        else
            echo ""
            echo "Server log preserved at: $LOG_FILE"
        fi
    fi
    exit "$exit_code"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "E2E smoke test for PR #2383 — vllm-omni stage config refactor"
echo "========================================================================"
printf "  %-22s %s\n" "Model:"          "$MODEL"
printf "  %-22s %s\n" "Port:"           "$PORT"
printf "  %-22s %s\n" "Deploy config:"  "$DEPLOY_CONFIG"
printf "  %-22s %s\n" "Extra serve args:" "${EXTRA_ARGS[*]:-(none)}"
printf "  %-22s %s\n" "Ready timeout:"  "${READY_TIMEOUT}s"
printf "  %-22s %s\n" "Request timeout:" "${REQUEST_TIMEOUT}s"
printf "  %-22s %s\n" "Log file:"       "$LOG_FILE"
printf "  %-22s %s\n" "Log grep assert:" "${LOG_GREP:-(disabled)}"
echo "========================================================================"

if [[ ! -f "$DEPLOY_CONFIG" ]]; then
    echo "[FAIL] deploy config not found: $DEPLOY_CONFIG"
    echo "       (run from vllm-omni repo root, or set E2E_DEPLOY_CONFIG=)"
    exit 1
fi

if ! command -v curl >/dev/null; then
    echo "[FAIL] curl not in PATH"
    exit 1
fi

HAS_JQ=0
if command -v jq >/dev/null; then
    HAS_JQ=1
else
    echo "[WARN] jq not in PATH — response parsing will be limited"
fi

if ! command -v vllm >/dev/null; then
    echo "[FAIL] vllm not in PATH (is vllm-omni installed in this env?)"
    exit 1
fi

# Bail early if the port is already in use
if (echo > /dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
    echo "[FAIL] Port $PORT already in use — kill the previous server first"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1 — start server
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: Starting vllm-omni server ==="
echo "+ vllm serve $MODEL --omni --port $PORT --deploy-config $DEPLOY_CONFIG ${EXTRA_ARGS[*]}"

# Use stdbuf to keep stdout line-buffered so the log file fills in real time
stdbuf -oL -eL vllm serve "$MODEL" \
    --omni \
    --port "$PORT" \
    --deploy-config "$DEPLOY_CONFIG" \
    "${EXTRA_ARGS[@]}" \
    >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "Streaming log → $LOG_FILE"

if [[ "$STREAM_LOG" == "1" ]]; then
    echo ""
    echo "=== Live log streaming enabled (E2E_STREAM_LOG=1) ==="
    # Start tail -f after a brief delay so the file exists
    ( sleep 1 && tail -F "$LOG_FILE" 2>/dev/null | sed 's/^/[server] /' ) &
    TAIL_PID=$!
fi

# ---------------------------------------------------------------------------
# Step 2 — wait for ready
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Waiting for server ready (up to ${READY_TIMEOUT}s) ==="
elapsed=0
poll_interval=5
while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[FAIL] Server process died during startup"
        exit 1
    fi
    if curl -fsS --max-time 5 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[OK] Server is ready after ${elapsed}s"
        break
    fi
    if [[ $elapsed -ge $READY_TIMEOUT ]]; then
        echo "[FAIL] Server did not become ready within ${READY_TIMEOUT}s"
        exit 1
    fi
    sleep "$poll_interval"
    elapsed=$((elapsed + poll_interval))
    if (( elapsed % 30 == 0 )); then
        echo "  ... still waiting (${elapsed}s elapsed)"
        # Print last N log lines so the user can see what's happening,
        # unless full streaming is already on (would be redundant).
        if [[ "$STREAM_LOG" != "1" ]] && [[ -s "$LOG_FILE" ]]; then
            echo "  --- last $LOG_SNIPPET_LINES log lines ---"
            tail -n "$LOG_SNIPPET_LINES" "$LOG_FILE" 2>/dev/null \
                | sed 's/^/    | /' \
                || true
            echo "  $(printf '%.0s-' {1..40})"
        fi
    fi
done

# ---------------------------------------------------------------------------
# Step 3 — optional log assertion (Layer 5 precedence verification)
# ---------------------------------------------------------------------------
if [[ -n "$LOG_GREP" ]]; then
    echo ""
    echo "=== Step 3: Asserting pattern in server log: '$LOG_GREP' ==="
    if grep -E "$LOG_GREP" "$LOG_FILE" >/tmp/e2e_grep_$$ 2>/dev/null; then
        match_count=$(wc -l </tmp/e2e_grep_$$)
        echo "[OK] Pattern found ($match_count line(s)):"
        head -n 5 /tmp/e2e_grep_$$ | sed 's/^/    /'
        if [[ $match_count -gt 5 ]]; then
            echo "    ... ($((match_count - 5)) more)"
        fi
        rm -f /tmp/e2e_grep_$$
    else
        rm -f /tmp/e2e_grep_$$
        echo "[FAIL] Pattern '$LOG_GREP' not found in server log"
        echo "       This usually means a CLI override didn't propagate"
        echo "       to the engine layer — check the precedence rule."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 4 — send chat completion
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Sending chat completion request ==="
REQUEST_BODY=$(cat <<JSON
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "Say hi in 5 words."}],
  "stream": false,
  "modalities": ["text"],
  "max_tokens": 32
}
JSON
)

if ! curl -fsS --max-time "$REQUEST_TIMEOUT" \
    "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$REQUEST_BODY" \
    -o "$RESPONSE_FILE"; then
    echo "[FAIL] Chat completion request failed"
    if [[ -f "$RESPONSE_FILE" ]]; then
        echo "Response body (if any):"
        cat "$RESPONSE_FILE"
    fi
    exit 1
fi
echo "[OK] Got HTTP 2xx response"

# ---------------------------------------------------------------------------
# Step 5 — verify response shape
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 5: Verifying response shape ==="
if [[ "$HAS_JQ" -eq 1 ]]; then
    if ! jq -e '.choices[0].message.content' "$RESPONSE_FILE" >/dev/null 2>&1; then
        echo "[FAIL] Response missing choices[0].message.content"
        jq . "$RESPONSE_FILE" || cat "$RESPONSE_FILE"
        exit 1
    fi
    content=$(jq -r '.choices[0].message.content' "$RESPONSE_FILE")
    if [[ -z "$content" ]] || [[ "$content" == "null" ]]; then
        echo "[FAIL] choices[0].message.content is empty/null"
        jq . "$RESPONSE_FILE"
        exit 1
    fi
    echo "[OK] choices[0].message.content:"
    echo "    $content"
    finish_reason=$(jq -r '.choices[0].finish_reason // "?"' "$RESPONSE_FILE")
    prompt_tokens=$(jq -r '.usage.prompt_tokens // "?"' "$RESPONSE_FILE")
    completion_tokens=$(jq -r '.usage.completion_tokens // "?"' "$RESPONSE_FILE")
    echo "[OK] finish_reason=$finish_reason  prompt_tokens=$prompt_tokens  completion_tokens=$completion_tokens"
else
    if ! grep -q '"choices"' "$RESPONSE_FILE"; then
        echo "[FAIL] Response missing 'choices' field"
        cat "$RESPONSE_FILE"
        exit 1
    fi
    echo "[OK] Response contains 'choices' field"
    echo "    (install jq for richer verification)"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "ALL OK — server startup, log assertion (if any), and chat completion all passed"
echo "========================================================================"

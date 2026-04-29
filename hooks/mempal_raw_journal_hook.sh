#!/bin/bash
# MEMPALACE RAW JOURNAL HOOK — cheap append-only transcript mirroring
#
# Goal:
#   Persist raw terminal/session transcripts on every Stop hook with minimal work.
#   This does NOT block the model and does NOT ask the AI to classify anything.
#
# Behavior:
#   - Reads session_id and transcript_path from hook JSON on stdin
#   - Copies only newly appended bytes from the live transcript JSONL
#   - Mirrors them into ~/.mempalace/raw_sessions/<session_id>.jsonl
#   - Tracks the last byte offset in ~/.mempalace/hook_state/
#   - Always returns "{}" so the conversation continues without interruption
#
# This is the crash-safety layer. Mining and judgment extraction should run
# asynchronously on the mirrored files, not in the hot path.

set -u

resolve_palace_path() {
    if [ -n "${MEMPALACE_PALACE_PATH:-}" ]; then
        printf '%s\n' "$MEMPALACE_PALACE_PATH"
        return
    fi

    python3 - <<'PYEOF' 2>/dev/null
import json
import os
from pathlib import Path

config_file = Path(os.path.expanduser("~/.mempalace/config.json"))
default_path = os.path.expanduser("~/.mempalace/palace")

if config_file.exists():
    try:
        data = json.loads(config_file.read_text())
        print(data.get("palace_path", default_path))
    except Exception:
        print(default_path)
else:
    print(default_path)
PYEOF
}

STATE_DIR="${STATE_DIR:-$HOME/.mempalace/hook_state}"
PALACE_PATH="${PALACE_PATH:-$(resolve_palace_path)}"
RAW_ARCHIVE_DIR="${RAW_ARCHIVE_DIR:-$PALACE_PATH/raw_sessions}"

mkdir -p "$STATE_DIR" "$RAW_ARCHIVE_DIR"

INPUT=$(cat)

SESSION_ID=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id','unknown'))" 2>/dev/null)
TRANSCRIPT_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('transcript_path',''))" 2>/dev/null)

SESSION_ID=$(echo "$SESSION_ID" | tr -cd 'a-zA-Z0-9_-')
[ -z "$SESSION_ID" ] && SESSION_ID="unknown"
TRANSCRIPT_PATH="${TRANSCRIPT_PATH/#\~/$HOME}"

OFFSET_FILE="$STATE_DIR/${SESSION_ID}_raw_offset"
TARGET_FILE="$RAW_ARCHIVE_DIR/${SESSION_ID}.jsonl"
LOG_FILE="$STATE_DIR/raw_journal.log"

if [ -f "$TRANSCRIPT_PATH" ]; then
    CURRENT_SIZE=$(wc -c < "$TRANSCRIPT_PATH" 2>/dev/null || echo 0)
    LAST_OFFSET=0

    if [ -f "$OFFSET_FILE" ]; then
        LAST_OFFSET=$(cat "$OFFSET_FILE" 2>/dev/null || echo 0)
    fi

    # Transcript rotated or truncated; reset our cursor and rewrite mirror.
    if [ "$CURRENT_SIZE" -lt "$LAST_OFFSET" ]; then
        LAST_OFFSET=0
        : > "$TARGET_FILE"
        echo "[$(date '+%F %T')] reset session=$SESSION_ID transcript_truncated" >> "$LOG_FILE"
    fi

    if [ "$CURRENT_SIZE" -gt "$LAST_OFFSET" ]; then
        START_BYTE=$((LAST_OFFSET + 1))
        tail -c +"$START_BYTE" "$TRANSCRIPT_PATH" >> "$TARGET_FILE" 2>/dev/null
        echo "$CURRENT_SIZE" > "$OFFSET_FILE"
        echo "[$(date '+%F %T')] append session=$SESSION_ID bytes=$((CURRENT_SIZE - LAST_OFFSET))" >> "$LOG_FILE"
    fi
fi

echo "{}"

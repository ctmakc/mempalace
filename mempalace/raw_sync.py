#!/usr/bin/env python3
"""
raw_sync.py — Incrementally ingest mirrored raw terminal transcripts.

Hot path capture writes append-only JSONL files into <palace>/raw_sessions/.
This module turns those raw transcripts into searchable drawers asynchronously.
"""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from .config import MempalaceConfig
from .convo_miner import detect_convo_room


def _state_path(palace_path: str) -> Path:
    return Path(palace_path) / "raw_sync_state.json"


def _load_state(palace_path: str) -> Dict[str, Dict[str, Any]]:
    path = _state_path(palace_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_state(palace_path: str, state: Dict[str, Dict[str, Any]]) -> None:
    _state_path(palace_path).write_text(json.dumps(state, indent=2, sort_keys=True))


def _get_collection(palace_path: str):
    client = chromadb.PersistentClient(path=palace_path)
    return client.get_or_create_collection("mempalace_drawers")


def _codex_session_id(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(20):
                raw_line = handle.readline()
                if not raw_line:
                    break
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "session_meta":
                    payload = entry.get("payload")
                    if isinstance(payload, dict) and payload.get("id"):
                        return str(payload["id"])
    except OSError:
        pass
    return path.stem


def _mirror_source_sessions(raw_dir: Path) -> Dict[str, int]:
    home = Path.home()
    sources: list[tuple[Path, str]] = [
        (home / ".claude" / "projects", "claude"),
        (home / ".codex" / "sessions", "codex"),
    ]
    seen = 0
    updated = 0

    for root, tool in sources:
        if not root.exists():
            continue
        for source in root.rglob("*.jsonl"):
            if not source.is_file():
                continue
            seen += 1
            session_id = _codex_session_id(source) if tool == "codex" else source.stem
            safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_") or source.stem
            target = raw_dir / f"{safe_id}.jsonl"
            try:
                source_stat = source.stat()
                if target.exists():
                    target_stat = target.stat()
                    if target_stat.st_size == source_stat.st_size and target_stat.st_mtime_ns >= source_stat.st_mtime_ns:
                        continue
                shutil.copy2(source, target)
                updated += 1
            except OSError:
                continue

    return {"source_files_seen": seen, "source_files_updated": updated}


def _extract_message(entry: Dict[str, Any]) -> Optional[Dict[str, str]]:
    msg = entry.get("message")
    if not isinstance(msg, dict):
        return None
    role = msg.get("role")
    content = msg.get("content")
    if not role or not isinstance(content, str) or not content.strip():
        return None
    if role not in {"user", "assistant", "system"}:
        return None
    return {"role": role, "content": content.strip()}


def _segment_messages(messages: List[Dict[str, str]]) -> List[str]:
    segments: List[str] = []
    current: List[Dict[str, str]] = []

    def flush():
        if not current:
            return
        lines: List[str] = []
        for item in current:
            if item["role"] == "user":
                lines.append(f"> {item['content']}")
            else:
                lines.append(item["content"])
        text = "\n".join(lines).strip()
        if text:
            segments.append(text)

    for item in messages:
        if item["role"] == "user" and current:
            flush()
            current = [item]
            continue
        current.append(item)
        if len(current) >= 6:
            flush()
            current = []

    if current:
        flush()

    return segments


def sync_raw_sessions(
    palace_path: Optional[str] = None,
    wing: str = "terminal_sessions",
    limit_files: int = 0,
    mirror_only: bool = False,
) -> Dict[str, Any]:
    cfg = MempalaceConfig()
    palace_path = palace_path or cfg.palace_path
    raw_dir = Path(palace_path) / "raw_sessions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mirror_stats = _mirror_source_sessions(raw_dir)
    if mirror_only:
        return {
            "files_seen": len(list(raw_dir.glob("*.jsonl"))),
            "files_updated": 0,
            "segments_added": 0,
            "raw_dir": str(raw_dir),
            "wing": wing,
            **mirror_stats,
        }

    state = _load_state(palace_path)
    collection = _get_collection(palace_path)
    files = sorted(raw_dir.glob("*.jsonl"))
    if limit_files > 0:
        files = files[:limit_files]

    stats = {
        "files_seen": len(files),
        "files_updated": 0,
        "segments_added": 0,
        "raw_dir": str(raw_dir),
        "wing": wing,
        **mirror_stats,
    }

    for file_path in files:
        session_id = file_path.stem
        current_size = file_path.stat().st_size
        file_state = state.get(session_id, {"offset": 0})
        offset = int(file_state.get("offset", 0))
        if current_size <= offset:
            continue

        with file_path.open("rb") as handle:
            handle.seek(offset)
            delta_bytes = handle.read()

        if not delta_bytes:
            continue

        delta_text = delta_bytes.decode("utf-8", errors="replace")
        messages: List[Dict[str, str]] = []
        for raw_line in delta_text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            message = _extract_message(entry)
            if message:
                messages.append(message)

        segments = _segment_messages(messages)
        added_for_file = 0

        for index, segment in enumerate(segments):
            room = detect_convo_room(segment)
            digest = hashlib.md5(
                f"{session_id}:{offset}:{index}:{segment}".encode(),
                usedforsecurity=False,
            ).hexdigest()[:16]
            drawer_id = f"drawer_{wing}_{room}_{digest}"
            try:
                collection.add(
                    documents=[segment],
                    ids=[drawer_id],
                    metadatas=[
                        {
                            "wing": wing,
                            "room": room,
                            "source_file": str(file_path),
                            "session_id": session_id,
                            "added_by": "raw_sync",
                            "filed_at": file_path.stat().st_mtime_ns,
                            "ingest_mode": "raw_sessions",
                            "raw_offset_start": offset,
                            "raw_offset_end": current_size,
                        }
                    ],
                )
                added_for_file += 1
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise

        state[session_id] = {"offset": current_size}
        stats["files_updated"] += 1
        stats["segments_added"] += added_for_file

    _save_state(palace_path, state)
    return stats

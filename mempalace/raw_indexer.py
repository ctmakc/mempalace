#!/usr/bin/env python3
"""
Incremental background indexer for raw terminal session mirrors.

This consumes <palace_path>/raw_sessions/*.jsonl files written by the cheap
raw journal hook, extracts real user/assistant turns from Codex and Claude
session formats, and upserts them into the main drawer collection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb

from .config import MempalaceConfig
from .convo_miner import MIN_CHUNK_SIZE, detect_convo_room
from .searcher import search_memories


SHARED_HOME = Path("/home/llm")
DEFAULT_CONFIG_DIR = SHARED_HOME / ".mempalace"
DEFAULT_LOG_PATH = SHARED_HOME / ".cache" / "mempalace" / "raw-indexer.log"
DEFAULT_STATE_PATH = DEFAULT_CONFIG_DIR / "raw_indexer_state.json"
DEFAULT_HOOK_LOG_PATH = DEFAULT_CONFIG_DIR / "hook_state" / "raw_journal.log"
DEFAULT_MAX_FILES_PER_RUN = 250
STATE_SAVE_EVERY = 20

TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "http",
    "https",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "project",
    "that",
    "the",
    "this",
    "to",
    "up",
    "we",
    "with",
}

GENERIC_TITLES = {
    "config",
    "continue",
    "go next",
    "help",
    "ok",
    "test",
    "продолжай",
    "продолжить",
    "конфиг",
}

PATH_RE = re.compile(
    r"(?P<path>(?:/[\w.@%+\-~]+)+(?:/[\w.@%+\-~]+)*(?:\.[A-Za-z0-9_-]+)?)"
)
URL_RE = re.compile(r"https?://[^\s<>\"]+")
VALID_PATH_PREFIXES = ("/home/", "/data/", "/etc/", "/var/", "/usr/", "/tmp/")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def read_text_tail(path: Path, max_lines: int = 80, max_bytes: int = 64_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        text = fh.read().decode("utf-8", errors="ignore")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    return text or "general"


def trim_text(value: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def is_meta_user_message(entry: dict[str, Any]) -> bool:
    return bool(entry.get("isMeta") or entry.get("sourceToolUseID"))


def iter_text_candidates(content: Any) -> Iterable[str]:
    if isinstance(content, str):
        text = content.strip()
        if text:
            yield text
        return

    if isinstance(content, dict):
        content_type = content.get("type")
        if content_type in {"text", "input_text", "output_text"}:
            text = str(content.get("text", "")).strip()
            if text:
                yield text
        elif content_type == "tool_result":
            for child in content.get("content", []):
                yield from iter_text_candidates(child)
        return

    if not isinstance(content, list):
        return

    for item in content:
        yield from iter_text_candidates(item)


def format_file_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def extract_text_blocks(content: Any) -> str:
    return "\n".join(iter_text_candidates(content)).strip()


def collapse_messages(messages: list[tuple[str, str]]) -> list[tuple[str, str]]:
    collapsed: list[list[str]] = []
    last_role = None
    last_text = None

    for role, text in messages:
        normalized = text.strip()
        if not normalized:
            continue
        if normalized == last_text and role == last_role:
            continue
        if collapsed and collapsed[-1][0] == role:
            collapsed[-1][1] = f"{collapsed[-1][1]}\n\n{normalized}"
        else:
            collapsed.append([role, normalized])
        last_role = role
        last_text = normalized

    return [(role, text) for role, text in collapsed]


def build_chunks(messages: list[tuple[str, str]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        role, text = messages[i]
        if role == "user":
            content = f"> {text}"
            if i + 1 < len(messages) and messages[i + 1][0] == "assistant":
                content = f"{content}\n{messages[i + 1][1]}"
                i += 2
            else:
                i += 1
        else:
            content = text
            i += 1
        if len(content.strip()) >= MIN_CHUNK_SIZE:
            chunks.append({"chunk_index": len(chunks), "content": content})
    return chunks


@dataclass
class ParsedSession:
    tool: str
    session_id: str
    cwd: str
    messages: list[tuple[str, str]]
    first_user_text: str
    last_message_text: str
    updated_at: str
    raw_lines: int


def derive_project_label(tool: str, cwd: str, wing: str) -> str:
    if cwd:
        if cwd == "/home/llm":
            return "workspace"
        path = Path(cwd)
        parts = [part for part in path.parts if part not in {"", "/"}]
        if "projects" in parts:
            idx = parts.index("projects")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        if len(parts) >= 2 and parts[-1] in {"backend", "frontend", "web", "api", "app"}:
            return parts[-2]
        if parts:
            return parts[-1]
    if wing.endswith("_workspace"):
        return tool.upper()
    return wing


def derive_session_title(first_user_text: str, last_message_text: str, session_id: str) -> str:
    first = normalize_text(first_user_text)
    if first and first.lower() not in GENERIC_TITLES and len(first) >= 8:
        return trim_text(first, 96)
    fallback = normalize_text(last_message_text)
    if fallback:
        return trim_text(fallback, 96)
    return f"Session {session_id[:8]}"


def derive_summary(first_user_text: str, last_message_text: str) -> str:
    first = normalize_text(first_user_text)
    last = normalize_text(last_message_text)
    if first and last and first != last:
        return trim_text(f"{first} -> {last}", 220)
    return trim_text(first or last, 220)


def derive_task_key(project_label: str, first_user_text: str) -> str:
    text = normalize_text(first_user_text).lower()
    words = re.findall(r"[a-zA-Zа-яА-Я0-9_-]{3,}", text)
    words = [word for word in words if word not in TITLE_STOPWORDS]
    if not words:
        return slugify(project_label)
    return slugify(" ".join(words[:6]))


def extract_artifacts(messages: list[tuple[str, str]], limit: int = 24) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for role, text in messages:
        for url in URL_RE.findall(text):
            url = url.rstrip("`.,);]")
            key = ("url", url)
            if key in seen:
                continue
            seen.add(key)
            artifacts.append({"type": "url", "value": url, "source_role": role})
        for match in PATH_RE.finditer(text):
            path = match.group("path")
            if len(path) < 4 or path.endswith(("/.", "/..")):
                continue
            if not path.startswith(VALID_PATH_PREFIXES):
                continue
            key = ("path", path)
            if key in seen:
                continue
            seen.add(key)
            artifacts.append({"type": "path", "value": path, "source_role": role})
        if len(artifacts) >= limit:
            break
    return artifacts[:limit]


def parse_codex_session(lines: list[str], fallback_session_id: str) -> ParsedSession | None:
    session_id = fallback_session_id
    cwd = ""
    updated_at = ""
    messages: list[tuple[str, str]] = []
    fallback_messages: list[tuple[str, str]] = []

    for raw_line in lines:
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue

        entry_type = entry.get("type", "")
        payload = entry.get("payload", {})

        if entry_type == "session_meta" and isinstance(payload, dict):
            session_id = str(payload.get("id") or session_id)
            cwd = str(payload.get("cwd") or cwd)
            updated_at = str(entry.get("timestamp") or payload.get("timestamp") or updated_at)
            continue

        if entry_type == "event_msg" and isinstance(payload, dict):
            payload_type = payload.get("type")
            text = str(payload.get("message") or "").strip()
            if payload_type == "user_message" and text:
                messages.append(("user", text))
            elif payload_type == "agent_message" and text:
                messages.append(("assistant", text))
            continue

        if entry_type == "response_item" and isinstance(payload, dict) and payload.get("type") == "message":
            role = payload.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = extract_text_blocks(payload.get("content"))
            if text:
                fallback_messages.append((role, text))

    normalized = collapse_messages(messages or fallback_messages)
    if len(normalized) < 2:
        return None

    return ParsedSession(
        tool="codex",
        session_id=session_id,
        cwd=cwd,
        messages=normalized,
        first_user_text=next((text for role, text in normalized if role == "user"), ""),
        last_message_text=normalized[-1][1],
        updated_at=updated_at,
        raw_lines=len(lines),
    )


def parse_claude_session(lines: list[str], fallback_session_id: str) -> ParsedSession | None:
    session_id = fallback_session_id
    cwd = ""
    updated_at = ""
    messages: list[tuple[str, str]] = []

    for raw_line in lines:
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue

        session_id = str(entry.get("sessionId") or session_id)
        cwd = str(entry.get("cwd") or cwd)
        updated_at = str(entry.get("timestamp") or updated_at)

        entry_type = entry.get("type", "")
        if entry_type not in {"user", "assistant"}:
            continue

        message = entry.get("message")
        if not isinstance(message, dict):
            continue

        if entry_type == "user" and (entry.get("toolUseResult") or entry.get("sourceToolAssistantUUID")):
            continue
        if entry_type == "user" and is_meta_user_message(entry):
            continue

        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue

        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(item, dict) and item.get("type") == "tool_result" for item in content
        ):
            continue

        text = extract_text_blocks(content)
        if text:
            messages.append((role, text))

    normalized = collapse_messages(messages)
    if len(normalized) < 2:
        return None

    return ParsedSession(
        tool="claude",
        session_id=session_id,
        cwd=cwd,
        messages=normalized,
        first_user_text=next((text for role, text in normalized if role == "user"), ""),
        last_message_text=normalized[-1][1],
        updated_at=updated_at,
        raw_lines=len(lines),
    )


def parse_raw_session(path: Path) -> ParsedSession | None:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return None
    fallback_session_id = path.stem
    parsed = parse_codex_session(lines, fallback_session_id)
    if parsed:
        return parsed
    return parse_claude_session(lines, fallback_session_id)


class RawIndexer:
    def __init__(self, palace_path: str | None = None, config_dir: Path | None = None):
        self.config_dir = config_dir or DEFAULT_CONFIG_DIR
        self.config = MempalaceConfig(config_dir=str(self.config_dir))
        if palace_path:
            self.palace_path = Path(palace_path)
        else:
            self.palace_path = Path(
                os.environ.get("MEMPALACE_PALACE_PATH") or self.config.palace_path
            )
        self.raw_dir = self.palace_path / "raw_sessions"
        self.state_path = DEFAULT_STATE_PATH
        self.log_path = DEFAULT_LOG_PATH
        self.hook_log_path = DEFAULT_HOOK_LOG_PATH
        self.state = read_json(self.state_path, {"sessions": {}, "last_run_at": "", "last_error": ""})

    def _log(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{utc_now()}] {message}\n")

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def _get_collection(self):
        self.palace_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.palace_path))
        return client.get_or_create_collection(self.config.collection_name)

    def _session_signature(self, path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _derive_wing(self, tool: str, cwd: str) -> str:
        if cwd:
            base = Path(cwd).name or "workspace"
            if base not in {"", "llm"} or cwd != "/home/llm":
                return slugify(base)
        return f"{tool}_workspace"

    def _drawer_id(self, session_id: str, chunk_index: int) -> str:
        return f"drawer_raw_{slugify(session_id)}_{chunk_index:05d}"

    def _session_row(self, path: Path, state_row: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": str(path),
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "mtime": format_file_mtime(path),
            "status": state_row.get("status", "new"),
            "tool": state_row.get("tool", ""),
            "session_id": state_row.get("session_id", path.stem),
            "cwd": state_row.get("cwd", ""),
            "wing": state_row.get("wing", ""),
            "project_label": state_row.get("project_label", ""),
            "task_key": state_row.get("task_key", ""),
            "title": state_row.get("title", ""),
            "summary": state_row.get("summary", ""),
            "message_count": state_row.get("message_count", 0),
            "chunk_count": state_row.get("chunk_count", 0),
            "first_user_text": state_row.get("first_user_text", ""),
            "last_message_text": state_row.get("last_message_text", ""),
            "source_updated_at": state_row.get("source_updated_at", ""),
            "raw_lines": state_row.get("raw_lines", 0),
            "artifacts": state_row.get("artifacts", []),
        }

    def _find_session_path(self, selector: str) -> Path | None:
        if not selector:
            return None
        direct = Path(selector)
        if direct.exists():
            return direct

        files = sorted(self.raw_dir.glob("*.jsonl"))
        sessions_state = self.state.get("sessions", {})
        for path in files:
            state_row = sessions_state.get(str(path), {})
            if selector in {str(path), path.name, path.stem, state_row.get("session_id", "")}:
                return path
        return None

    def _pending_files(self) -> list[Path]:
        files = sorted(self.raw_dir.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
        sessions_state = self.state.setdefault("sessions", {})
        pending: list[Path] = []
        for path in files:
            signature = self._session_signature(path)
            previous = sessions_state.get(str(path), {})
            if previous.get("signature") != signature:
                pending.append(path)
        return pending

    def index_once(self, max_files: int | None = None) -> dict[str, Any]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        collection = self._get_collection()
        files = self._pending_files()
        if max_files is None:
            max_files = DEFAULT_MAX_FILES_PER_RUN
        if max_files > 0:
            files = files[:max_files]
        sessions_state = self.state.setdefault("sessions", {})

        processed = 0
        updated = 0
        skipped = 0
        total_chunks = 0
        pending_total = len(files)

        for path in files:
            processed += 1
            signature = self._session_signature(path)
            path_key = str(path)
            previous = sessions_state.get(path_key, {})
            if previous.get("signature") == signature:
                skipped += 1
                continue

            parsed = parse_raw_session(path)
            if not parsed:
                skipped += 1
                sessions_state[path_key] = {
                    **previous,
                    "signature": signature,
                    "status": "unparsed",
                    "updated_at": utc_now(),
                }
                continue

            chunks = build_chunks(parsed.messages)
            wing = self._derive_wing(parsed.tool, parsed.cwd)
            project_label = derive_project_label(parsed.tool, parsed.cwd, wing)
            title = derive_session_title(parsed.first_user_text, parsed.last_message_text, parsed.session_id)
            summary = derive_summary(parsed.first_user_text, parsed.last_message_text)
            task_key = derive_task_key(project_label, parsed.first_user_text)
            artifacts = extract_artifacts(parsed.messages)

            drawer_ids = [self._drawer_id(parsed.session_id, chunk["chunk_index"]) for chunk in chunks]
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [
                {
                    "wing": wing,
                    "project_label": project_label,
                    "task_key": task_key,
                    "session_title": title,
                    "room": detect_convo_room(chunk["content"]),
                    "source_file": str(path),
                    "chunk_index": chunk["chunk_index"],
                    "added_by": parsed.tool,
                    "filed_at": utc_now(),
                    "ingest_mode": "raw_sessions",
                    "extract_mode": "exchange",
                    "source_tool": parsed.tool,
                    "source_session_id": parsed.session_id,
                    "source_cwd": parsed.cwd or "",
                    "source_updated_at": parsed.updated_at or "",
                }
                for chunk in chunks
            ]

            if drawer_ids:
                collection.upsert(ids=drawer_ids, documents=documents, metadatas=metadatas)

            previous_chunk_count = int(previous.get("chunk_count") or 0)
            if previous_chunk_count > len(chunks):
                obsolete_ids = [
                    self._drawer_id(parsed.session_id, index)
                    for index in range(len(chunks), previous_chunk_count)
                ]
                if obsolete_ids:
                    collection.delete(ids=obsolete_ids)

            sessions_state[path_key] = {
                "signature": signature,
                "status": "indexed",
                "tool": parsed.tool,
                "session_id": parsed.session_id,
                "cwd": parsed.cwd,
                "wing": wing,
                "project_label": project_label,
                "task_key": task_key,
                "title": title,
                "summary": summary,
                "message_count": len(parsed.messages),
                "chunk_count": len(chunks),
                "first_user_text": trim_text(parsed.first_user_text),
                "last_message_text": trim_text(parsed.last_message_text),
                "updated_at": utc_now(),
                "source_updated_at": parsed.updated_at,
                "raw_lines": parsed.raw_lines,
                "artifacts": artifacts,
            }

            total_chunks += len(chunks)
            updated += 1
            self._log(
                f"indexed tool={parsed.tool} session={parsed.session_id} chunks={len(chunks)} path={path.name}"
            )
            if updated % STATE_SAVE_EVERY == 0:
                self.state["last_run_at"] = utc_now()
                self.state["last_error"] = ""
                self._save_state()

        self.state["last_run_at"] = utc_now()
        self.state["last_error"] = ""
        self._save_state()

        return {
            "ok": True,
            "processed_files": processed,
            "updated_files": updated,
            "skipped_files": skipped,
            "chunks_seen": total_chunks,
            "remaining_files": max(0, len(self._pending_files())),
            "batch_files": pending_total,
            "raw_dir": str(self.raw_dir),
            "palace_path": str(self.palace_path),
        }

    def status_payload(self) -> dict[str, Any]:
        sessions_state = self.state.get("sessions", {})
        files = sorted(self.raw_dir.glob("*.jsonl")) if self.raw_dir.exists() else []
        collection = None
        total_drawers = 0
        try:
            collection = self._get_collection()
            total_drawers = collection.count()
        except Exception:
            total_drawers = 0

        recent_sessions = []
        for path in sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)[:24]:
            state_row = sessions_state.get(str(path), {})
            recent_sessions.append(self._session_row(path, state_row))

        indexed_sessions = sum(1 for row in sessions_state.values() if row.get("status") == "indexed")
        tool_counts: dict[str, int] = {}
        wing_counts: dict[str, int] = {}
        for row in sessions_state.values():
            if row.get("status") != "indexed":
                continue
            tool = str(row.get("tool") or "unknown")
            wing = str(row.get("wing") or "unknown")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
            wing_counts[wing] = wing_counts.get(wing, 0) + 1

        top_wings = [
            {"wing": wing, "count": count}
            for wing, count in sorted(wing_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
        ]

        return {
            "ok": True,
            "generated_at": utc_now(),
            "palace_path": str(self.palace_path),
            "raw_sessions_dir": str(self.raw_dir),
            "state_path": str(self.state_path),
            "total_raw_sessions": len(files),
            "indexed_sessions": indexed_sessions,
            "total_drawers": total_drawers,
            "last_run_at": self.state.get("last_run_at", ""),
            "last_error": self.state.get("last_error", ""),
            "tool_counts": tool_counts,
            "top_wings": top_wings,
            "recent_sessions": recent_sessions,
            "logs": {
                "raw_journal": read_text_tail(self.hook_log_path, max_lines=60),
                "raw_indexer": read_text_tail(self.log_path, max_lines=60),
            },
        }

    def session_payload(self, selector: str) -> dict[str, Any]:
        path = self._find_session_path(selector)
        if not path:
            return {"ok": False, "error": f"session not found: {selector}"}

        parsed = parse_raw_session(path)
        if not parsed:
            return {"ok": False, "error": f"session could not be parsed: {path.name}", "path": str(path)}

        state_row = self.state.get("sessions", {}).get(str(path), {})
        chunks = build_chunks(parsed.messages)
        wing = state_row.get("wing") or self._derive_wing(parsed.tool, parsed.cwd)
        events = [
            {
                "index": index,
                "role": role,
                "chars": len(text),
                "lines": len(text.splitlines()),
                "preview": trim_text(text, 260),
                "text": text,
            }
            for index, (role, text) in enumerate(parsed.messages)
        ]
        chunk_rows = [
            {
                "chunk_index": chunk["chunk_index"],
                "drawer_id": self._drawer_id(parsed.session_id, chunk["chunk_index"]),
                "room": detect_convo_room(chunk["content"]),
                "chars": len(chunk["content"]),
                "preview": trim_text(chunk["content"], 320),
                "text": chunk["content"],
            }
            for chunk in chunks
        ]
        project_label = derive_project_label(parsed.tool, parsed.cwd, wing)
        title = derive_session_title(
            parsed.first_user_text,
            parsed.last_message_text,
            parsed.session_id,
        )
        task_key = state_row.get("task_key") or derive_task_key(project_label, parsed.first_user_text)
        related = []
        for other_path, other_state in self.state.get("sessions", {}).items():
            if other_state.get("status") != "indexed":
                continue
            if other_state.get("session_id") == parsed.session_id:
                continue
            same_project = other_state.get("project_label") == project_label
            same_task = other_state.get("task_key") == task_key
            if not same_project and not same_task:
                continue
            other_file = Path(other_path)
            related.append(
                {
                    "session_id": other_state.get("session_id", other_file.stem),
                    "title": other_state.get("title", other_file.stem),
                    "summary": other_state.get("summary", ""),
                    "project_label": other_state.get("project_label", ""),
                    "task_key": other_state.get("task_key", ""),
                    "mtime": format_file_mtime(other_file) if other_file.exists() else "",
                    "same_project": same_project,
                    "same_task": same_task,
                }
            )
        related.sort(key=lambda item: item["mtime"], reverse=True)

        return {
            "ok": True,
            "generated_at": utc_now(),
            "path": str(path),
            "name": path.name,
            "tool": parsed.tool,
            "session_id": parsed.session_id,
            "cwd": parsed.cwd,
            "wing": wing,
            "project_label": project_label,
            "task_key": task_key,
            "title": title,
            "summary": derive_summary(parsed.first_user_text, parsed.last_message_text),
            "mtime": format_file_mtime(path),
            "size_bytes": path.stat().st_size,
            "raw_lines": parsed.raw_lines,
            "message_count": len(parsed.messages),
            "chunk_count": len(chunk_rows),
            "source_updated_at": parsed.updated_at or "",
            "first_user_text": trim_text(parsed.first_user_text),
            "last_message_text": trim_text(parsed.last_message_text),
            "artifacts": extract_artifacts(parsed.messages),
            "related_sessions": related[:24],
            "index_status": state_row.get("status", "new"),
            "events": events,
            "chunks": chunk_rows,
        }

    def search_payload(self, query: str, n_results: int = 8) -> dict[str, Any]:
        result = search_memories(
            query=query,
            palace_path=str(self.palace_path),
            wing=None,
            room=None,
            n_results=n_results,
        )
        result["generated_at"] = utc_now()
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index raw MemPalace sessions.")
    parser.add_argument("--once", action="store_true", help="Run one indexing pass.")
    parser.add_argument("--status-json", action="store_true", help="Print dashboard status JSON.")
    parser.add_argument("--search-json", metavar="QUERY", help="Run semantic search and emit JSON.")
    parser.add_argument("--session-json", metavar="SESSION", help="Emit parsed details for one raw session.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES_PER_RUN,
        help="Maximum number of pending files to process in one pass.",
    )
    parser.add_argument("--palace-path", help="Override palace path.")
    args = parser.parse_args(argv)

    indexer = RawIndexer(palace_path=args.palace_path)

    try:
        if args.status_json:
            print(json.dumps(indexer.status_payload(), ensure_ascii=False))
            return 0
        if args.search_json:
            print(json.dumps(indexer.search_payload(args.search_json), ensure_ascii=False))
            return 0
        if args.session_json:
            print(json.dumps(indexer.session_payload(args.session_json), ensure_ascii=False))
            return 0
        result = indexer.index_once(max_files=args.max_files)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        indexer.state["last_error"] = str(exc)
        indexer.state["last_run_at"] = utc_now()
        indexer._save_state()
        indexer._log(f"error {exc}")
        if args.status_json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "generated_at": utc_now(),
                        "error": str(exc),
                        "palace_path": str(indexer.palace_path),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        print(f"raw_indexer_error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

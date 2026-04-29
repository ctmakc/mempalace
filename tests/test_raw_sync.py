import json
from pathlib import Path

import chromadb

from mempalace.raw_sync import sync_raw_sessions


def test_raw_sync_ingests_new_messages_incrementally(palace_path):
    raw_dir = Path(palace_path) / "raw_sessions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session_file = raw_dir / "session-1.jsonl"

    session_file.write_text(
        "\n".join(
            [
                json.dumps({"message": {"role": "user", "content": "Why did we switch to pgvector?"}}),
                json.dumps({"message": {"role": "assistant", "content": "Because it simplified retrieval inside Postgres."}}),
            ]
        )
        + "\n"
    )

    first = sync_raw_sessions(palace_path=palace_path, wing="terminal_sessions")
    assert first["segments_added"] >= 1

    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_collection("mempalace_drawers")
    initial_count = col.count()

    with session_file.open("a") as handle:
        handle.write(json.dumps({"message": {"role": "user", "content": "What was the tradeoff?"}}) + "\n")
        handle.write(
            json.dumps(
                {"message": {"role": "assistant", "content": "Lower infra complexity, weaker cross-project reuse."}}
            )
            + "\n"
        )

    second = sync_raw_sessions(palace_path=palace_path, wing="terminal_sessions")
    assert second["segments_added"] >= 1
    assert col.count() > initial_count


def test_raw_sync_mirrors_codex_source_sessions(palace_path):
    codex_dir = Path.home() / ".codex" / "sessions" / "2026" / "04" / "20"
    codex_dir.mkdir(parents=True, exist_ok=True)
    source = codex_dir / "rollout-2026-04-20T00-00-00-019daaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-04-20T00:00:00Z",
                        "type": "session_meta",
                        "payload": {"id": "019daaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "cwd": "/home/llm"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-20T00:00:01Z",
                        "type": "event_msg",
                        "payload": {"type": "user_message", "message": "Why is MemPalace stale?"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-20T00:00:02Z",
                        "type": "event_msg",
                        "payload": {"type": "agent_message", "message": "The raw mirror stopped updating."},
                    }
                ),
            ]
        )
        + "\n"
    )

    result = sync_raw_sessions(palace_path=palace_path, wing="terminal_sessions")
    mirrored = Path(palace_path) / "raw_sessions" / "019daaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"

    assert result["source_files_updated"] >= 1
    assert mirrored.exists()
    assert mirrored.read_text() == source.read_text()


def test_raw_sync_mirror_only_skips_drawer_indexing(palace_path):
    raw_dir = Path(palace_path) / "raw_sessions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "session-1.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"message": {"role": "user", "content": "Should this index now?"}}),
                json.dumps({"message": {"role": "assistant", "content": "No, mirror-only skips indexing."}}),
            ]
        )
        + "\n"
    )

    result = sync_raw_sessions(palace_path=palace_path, wing="terminal_sessions", mirror_only=True)
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_or_create_collection("mempalace_drawers")

    assert result["files_seen"] >= 1
    assert result["segments_added"] == 0
    assert col.count() == 0

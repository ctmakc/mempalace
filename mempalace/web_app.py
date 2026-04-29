#!/usr/bin/env python3
"""
web_app.py — local conversation-centric dashboard for MemPalace.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .config import MempalaceConfig
from .raw_indexer import RawIndexer, derive_project_label, derive_session_title, derive_summary
from .searcher import search_memories


def _query_db(db_path: Path, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _session_catalog(indexer: RawIndexer) -> List[Dict[str, Any]]:
    sessions_state = indexer.state.get("sessions", {})
    rows: list[Dict[str, Any]] = []
    for path in sorted(indexer.raw_dir.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True):
        state_row = sessions_state.get(str(path), {})
        row = indexer._session_row(path, state_row)
        row["updated_ms"] = path.stat().st_mtime_ns
        row["project_label"] = derive_project_label(row.get("tool", ""), row.get("cwd", ""), row.get("wing", ""))
        row["title"] = derive_session_title(
            row.get("first_user_text", ""),
            row.get("last_message_text", ""),
            row.get("session_id", path.stem),
        )
        row["summary"] = derive_summary(
            row.get("first_user_text", ""),
            row.get("last_message_text", ""),
        )
        rows.append(row)
    return rows


def create_app(palace_path: Optional[str] = None) -> FastAPI:
    cfg = MempalaceConfig()
    palace_path = palace_path or cfg.palace_path
    judgment_db = Path(palace_path) / "judgment_memory.sqlite3"

    app = FastAPI(title="MemPalace Conversations")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MemPalace</title>
  <style>
    :root {
      --bg: #0c1117;
      --bg2: #121924;
      --panel: rgba(18, 25, 36, 0.88);
      --panel-2: rgba(27, 36, 51, 0.9);
      --line: rgba(154, 174, 197, 0.16);
      --text: #e9eef6;
      --muted: #95a3b6;
      --accent: #8de1b7;
      --accent-2: #f2c46d;
      --danger: #ef8d8d;
      --mono: "IBM Plex Mono", "SFMono-Regular", monospace;
      --sans: Inter, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(141,225,183,0.12), transparent 24%),
        radial-gradient(circle at top right, rgba(242,196,109,0.10), transparent 24%),
        linear-gradient(180deg, #0a0f15, #0e141d 32%, #0b1118);
    }
    a { color: inherit; }
    .wrap { max-width: 1540px; margin: 0 auto; padding: 28px 24px 36px; }
    .hero {
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }
    .hero-card, .panel, .session-card {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)), var(--panel);
      backdrop-filter: blur(18px);
      border-radius: 22px;
      box-shadow: 0 18px 60px rgba(0,0,0,0.28);
    }
    .hero-card { padding: 24px; }
    .title {
      margin: 0;
      font-size: 38px;
      line-height: 0.95;
      letter-spacing: -0.05em;
    }
    .sub {
      margin-top: 10px;
      color: var(--muted);
      max-width: 760px;
      line-height: 1.5;
    }
    .path {
      margin-top: 14px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--muted);
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .stat {
      padding: 16px 18px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: var(--panel-2);
    }
    .stat .k {
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .stat .v {
      margin-top: 8px;
      font-size: 30px;
      font-weight: 700;
    }
    .layout {
      display: grid;
      grid-template-columns: 430px minmax(0, 1fr);
      gap: 18px;
      min-height: 72vh;
    }
    .panel {
      padding: 18px;
    }
    .panel h2, .panel h3 {
      margin: 0;
      font-size: 17px;
      letter-spacing: -0.02em;
    }
    .toolbar {
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }
    .toolbar-row {
      display: grid;
      grid-template-columns: 1fr 1fr auto;
      gap: 10px;
    }
    input, select, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(9, 13, 20, 0.78);
      color: var(--text);
      padding: 12px 14px;
      font: inherit;
    }
    button {
      width: auto;
      background: linear-gradient(135deg, var(--accent), #b0f0d1);
      color: #122117;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      background: rgba(255,255,255,0.04);
      color: var(--text);
    }
    .list-meta {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }
    .session-list {
      margin-top: 14px;
      display: grid;
      gap: 10px;
      max-height: calc(72vh - 120px);
      overflow: auto;
      padding-right: 4px;
    }
    .session-card {
      padding: 14px 15px;
      cursor: pointer;
      transition: transform 0.12s ease, border-color 0.12s ease;
    }
    .session-card:hover { transform: translateY(-1px); border-color: rgba(141,225,183,0.32); }
    .session-card.active { border-color: rgba(141,225,183,0.58); box-shadow: inset 0 0 0 1px rgba(141,225,183,0.22); }
    .eyebrow, .micro {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--muted);
    }
    .session-title {
      margin-top: 6px;
      font-size: 16px;
      font-weight: 700;
      line-height: 1.28;
    }
    .session-summary {
      margin-top: 8px;
      font-size: 13px;
      color: #c3cfdd;
      line-height: 1.4;
    }
    .tags, .artifact-list, .related-list, .search-results {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .tag, .artifact, .related-chip {
      border-radius: 999px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      font-size: 12px;
      text-decoration: none;
    }
    .artifact.path { font-family: var(--mono); }
    .artifact.url { color: var(--accent); }
    .detail-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(280px, 0.9fr);
      gap: 16px;
      margin-top: 12px;
    }
    .stack { display: grid; gap: 16px; }
    .detail-card {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.02);
      padding: 16px 16px 18px;
    }
    .detail-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
    }
    .detail-title {
      margin: 8px 0 0;
      font-size: 28px;
      line-height: 1.05;
      letter-spacing: -0.04em;
    }
    .detail-summary {
      margin-top: 12px;
      font-size: 15px;
      line-height: 1.55;
      color: #d9e1ec;
    }
    .kv {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .kv-item {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: rgba(7, 11, 18, 0.35);
    }
    .kv-item .v {
      margin-top: 6px;
      font-family: var(--mono);
      font-size: 12px;
      color: #d3dbe6;
      word-break: break-word;
    }
    .timeline {
      margin-top: 14px;
      display: grid;
      gap: 10px;
      max-height: 56vh;
      overflow: auto;
      padding-right: 4px;
    }
    .turn {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 13px;
      background: rgba(8, 12, 19, 0.46);
    }
    .turn.user { border-color: rgba(242,196,109,0.28); }
    .turn.assistant { border-color: rgba(141,225,183,0.24); }
    .turn-meta {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .role {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 11px;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      display: inline-block;
    }
    .dot.user { background: var(--accent-2); }
    .dot.assistant { background: var(--accent); }
    .turn pre, .search-hit pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--sans);
      font-size: 14px;
      line-height: 1.52;
      color: #e4ebf4;
    }
    .search-hit {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 13px;
      background: rgba(8, 12, 19, 0.46);
    }
    .empty {
      margin-top: 14px;
      border: 1px dashed var(--line);
      border-radius: 16px;
      padding: 18px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 1180px) {
      .hero, .layout, .detail-grid, .toolbar-row, .stats, .kv {
        grid-template-columns: 1fr;
      }
      .session-list, .timeline { max-height: none; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow">Conversation Memory</div>
        <h1 class="title">MemPalace</h1>
        <div class="sub">Open a dialogue, read the actual exchange, see a fast summary, inspect artifacts, and jump through the chain of related sessions for the same project or task.</div>
        <div class="path" id="palacePath"></div>
      </div>
      <div class="hero-card">
        <div class="stats" id="stats"></div>
      </div>
    </section>

    <section class="layout">
      <aside class="panel">
        <h2>Dialogue Catalog</h2>
        <div class="toolbar">
          <div class="toolbar-row">
            <input id="searchInput" placeholder="Find by title, summary, project, cwd..." />
            <select id="projectFilter"></select>
            <button class="secondary" id="refreshBtn" type="button">Refresh</button>
          </div>
          <div class="toolbar-row">
            <select id="toolFilter">
              <option value="">All tools</option>
              <option value="codex">Codex</option>
              <option value="claude">Claude</option>
            </select>
            <select id="statusFilter">
              <option value="">Any status</option>
              <option value="indexed">Indexed</option>
              <option value="unparsed">Unparsed</option>
            </select>
            <button id="reindexBtn" type="button">Catch Up Index</button>
          </div>
        </div>
        <div class="list-meta">
          <span id="catalogMeta">Loading…</span>
          <span id="catalogNote"></span>
        </div>
        <div class="session-list" id="sessionList"></div>
      </aside>

      <main class="panel">
        <div id="detailRoot" class="empty">Choose a dialogue from the left to inspect it.</div>
      </main>
    </section>
  </div>

  <script>
    const state = {
      sessions: [],
      selectedSessionId: "",
      project: "",
      query: "",
      tool: "",
      status: "",
    };

    function esc(text) {
      return (text || "").replace(/[&<>"]/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;" }[ch]));
    }

    function linkifyArtifact(item) {
      const cls = item.type === "path" ? "artifact path" : "artifact url";
      if (item.type === "url") {
        return `<a class="${cls}" href="${esc(item.value)}" target="_blank" rel="noreferrer">${esc(item.value)}</a>`;
      }
      return `<span class="${cls}">${esc(item.value)}</span>`;
    }

    async function loadJson(path, options) {
      const res = await fetch(path, options);
      return await res.json();
    }

    async function hydrateStats() {
      const data = await loadJson("/api/overview");
      document.getElementById("palacePath").textContent = data.palace_path;
      document.getElementById("stats").innerHTML = [
        ["Raw dialogs", data.raw_sessions],
        ["Indexed dialogs", data.indexed_sessions],
        ["Pending index", data.pending_sessions],
        ["Drawers", data.drawers]
      ].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
    }

    async function hydrateFilters() {
      const data = await loadJson("/api/projects");
      const projectFilter = document.getElementById("projectFilter");
      projectFilter.innerHTML = `<option value="">All projects</option>` + data.projects.map(
        item => `<option value="${esc(item.project_label)}">${esc(item.project_label)} (${item.sessions})</option>`
      ).join("");
      projectFilter.value = state.project;
    }

    function buildSessionCard(item) {
      const active = item.session_id === state.selectedSessionId ? " active" : "";
      const meta = [item.project_label || "General", item.tool || "unknown", item.mtime].filter(Boolean).join(" · ");
      const tags = [
        item.status ? `<span class="tag">${esc(item.status)}</span>` : "",
        item.task_key ? `<span class="tag">${esc(item.task_key)}</span>` : "",
        item.chunk_count ? `<span class="tag">${esc(String(item.chunk_count))} chunks</span>` : ""
      ].join("");
      return `
        <article class="session-card${active}" data-session-id="${esc(item.session_id)}">
          <div class="eyebrow">${esc(meta)}</div>
          <div class="session-title">${esc(item.title || item.session_id)}</div>
          <div class="session-summary">${esc(item.summary || item.first_user_text || "No preview yet.")}</div>
          <div class="tags">${tags}</div>
        </article>
      `;
    }

    async function hydrateCatalog() {
      const params = new URLSearchParams();
      if (state.query) params.set("q", state.query);
      if (state.project) params.set("project", state.project);
      if (state.tool) params.set("tool", state.tool);
      if (state.status) params.set("status", state.status);
      params.set("limit", "180");
      const data = await loadJson(`/api/sessions?${params.toString()}`);
      state.sessions = data.sessions || [];
      document.getElementById("catalogMeta").textContent = `${data.count} dialogues`;
      document.getElementById("catalogNote").textContent = data.pending_hint || "";
      const root = document.getElementById("sessionList");
      if (!state.sessions.length) {
        root.innerHTML = `<div class="empty">No dialogues match the current filters.</div>`;
        return;
      }
      root.innerHTML = state.sessions.map(buildSessionCard).join("");
      root.querySelectorAll(".session-card").forEach(card => {
        card.addEventListener("click", () => selectSession(card.dataset.sessionId));
      });
      if (!state.selectedSessionId || !state.sessions.some(item => item.session_id === state.selectedSessionId)) {
        selectSession(state.sessions[0].session_id);
      }
    }

    function renderArtifacts(items) {
      if (!items || !items.length) return `<div class="empty">No file or URL artifacts were extracted from this dialogue yet.</div>`;
      return `<div class="artifact-list">${items.map(linkifyArtifact).join("")}</div>`;
    }

    function renderRelated(items) {
      if (!items || !items.length) return `<div class="empty">No related dialogue chain found yet.</div>`;
      return `<div class="related-list">${
        items.map(item => `<button class="related-chip" data-session-id="${esc(item.session_id)}" type="button">${esc(item.title || item.session_id)}</button>`).join("")
      }</div>`;
    }

    function renderTimeline(items) {
      if (!items || !items.length) return `<div class="empty">Nothing parsed from this dialogue yet.</div>`;
      return `<div class="timeline">${
        items.map(item => `
          <article class="turn ${esc(item.role)}">
            <div class="turn-meta">
              <span class="role"><span class="dot ${esc(item.role)}"></span>${esc(item.role)}</span>
              <span>${esc(String(item.chars))} chars · ${esc(String(item.lines))} lines</span>
            </div>
            <pre>${esc(item.text)}</pre>
          </article>
        `).join("")
      }</div>`;
    }

    function renderSearchHits(items) {
      if (!items || !items.length) return `<div class="empty">No semantic matches yet.</div>`;
      return `<div class="stack">${
        items.map(item => `
          <div class="search-hit">
            <div class="eyebrow">${esc(item.project_label || item.wing || "General")} · ${esc(item.room || "")} · ${esc(String(item.similarity || ""))}</div>
            <pre>${esc(item.text)}</pre>
          </div>
        `).join("")
      }</div>`;
    }

    async function selectSession(sessionId) {
      state.selectedSessionId = sessionId;
      document.querySelectorAll(".session-card").forEach(card => {
        card.classList.toggle("active", card.dataset.sessionId === sessionId);
      });
      const detail = await loadJson(`/api/session/${encodeURIComponent(sessionId)}`);
      const hits = await loadJson(`/api/search?q=${encodeURIComponent(detail.title || detail.first_user_text || sessionId)}&n=5`);
      document.getElementById("detailRoot").innerHTML = `
        <div class="detail-grid">
          <section class="stack">
            <article class="detail-card">
              <div class="detail-header">
                <div>
                  <div class="eyebrow">${esc(detail.project_label || "General")} · ${esc(detail.tool || "unknown")} · ${esc(detail.index_status || "")}</div>
                  <h2 class="detail-title">${esc(detail.title || detail.session_id)}</h2>
                </div>
                <div class="micro">${esc(detail.mtime || "")}</div>
              </div>
              <div class="detail-summary">${esc(detail.summary || detail.first_user_text || "No summary yet.")}</div>
              <div class="kv">
                <div class="kv-item"><div class="eyebrow">Session ID</div><div class="v">${esc(detail.session_id)}</div></div>
                <div class="kv-item"><div class="eyebrow">Task Key</div><div class="v">${esc(detail.task_key || "-")}</div></div>
                <div class="kv-item"><div class="eyebrow">Working Dir</div><div class="v">${esc(detail.cwd || "-")}</div></div>
                <div class="kv-item"><div class="eyebrow">Messages</div><div class="v">${esc(String(detail.message_count || 0))} messages · ${esc(String(detail.chunk_count || 0))} chunks</div></div>
              </div>
            </article>
            <article class="detail-card">
              <h3>Dialogue Timeline</h3>
              ${renderTimeline(detail.events)}
            </article>
          </section>
          <section class="stack">
            <article class="detail-card">
              <h3>Artifacts</h3>
              ${renderArtifacts(detail.artifacts)}
            </article>
            <article class="detail-card">
              <h3>Related Chain</h3>
              ${renderRelated(detail.related_sessions)}
            </article>
            <article class="detail-card">
              <h3>Semantic Neighbors</h3>
              ${renderSearchHits(hits.results)}
            </article>
          </section>
        </div>
      `;
      document.querySelectorAll(".related-chip").forEach(btn => {
        btn.addEventListener("click", () => selectSession(btn.dataset.sessionId));
      });
    }

    async function triggerReindex() {
      const result = await loadJson("/api/reindex?limit=250", { method: "POST" });
      document.getElementById("catalogNote").textContent = `Indexed ${result.updated_files} files, ${result.remaining_files} still pending.`;
      await hydrateStats();
      await hydrateCatalog();
    }

    async function boot() {
      document.getElementById("searchInput").addEventListener("input", event => {
        state.query = event.target.value.trim();
        hydrateCatalog();
      });
      document.getElementById("projectFilter").addEventListener("change", event => {
        state.project = event.target.value;
        hydrateCatalog();
      });
      document.getElementById("toolFilter").addEventListener("change", event => {
        state.tool = event.target.value;
        hydrateCatalog();
      });
      document.getElementById("statusFilter").addEventListener("change", event => {
        state.status = event.target.value;
        hydrateCatalog();
      });
      document.getElementById("refreshBtn").addEventListener("click", async () => {
        await hydrateStats();
        await hydrateFilters();
        await hydrateCatalog();
      });
      document.getElementById("reindexBtn").addEventListener("click", triggerReindex);
      await hydrateStats();
      await hydrateFilters();
      await hydrateCatalog();
    }

    boot();
  </script>
</body>
</html>
        """

    @app.get("/api/overview", response_class=JSONResponse)
    def overview():
        indexer = RawIndexer(palace_path=palace_path)
        status = indexer.status_payload()
        pending = max(0, status["total_raw_sessions"] - status["indexed_sessions"])
        pending_candidates = _query_db(
            judgment_db,
            "SELECT COUNT(*) AS count FROM judgment_candidates WHERE review_status = 'pending'",
        )
        return {
            "palace_path": palace_path,
            "raw_sessions": status["total_raw_sessions"],
            "indexed_sessions": status["indexed_sessions"],
            "pending_sessions": pending,
            "drawers": status["total_drawers"],
            "pending_candidates": pending_candidates[0]["count"] if pending_candidates else 0,
            "last_run_at": status["last_run_at"],
        }

    @app.get("/api/projects", response_class=JSONResponse)
    def projects():
        indexer = RawIndexer(palace_path=palace_path)
        rows = _session_catalog(indexer)
        counts: dict[str, int] = {}
        for row in rows:
            project_label = row.get("project_label") or "General"
            counts[project_label] = counts.get(project_label, 0) + 1
        projects = [
            {"project_label": project_label, "sessions": count}
            for project_label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
        ]
        return {"projects": projects}

    @app.get("/api/sessions", response_class=JSONResponse)
    def sessions(
        q: str = "",
        project: str = "",
        tool: str = "",
        status: str = "",
        limit: int = Query(120, ge=1, le=500),
    ):
        indexer = RawIndexer(palace_path=palace_path)
        rows = _session_catalog(indexer)
        query = q.strip().lower()
        filtered = []
        for row in rows:
            if project and row.get("project_label") != project:
                continue
            if tool and row.get("tool") != tool:
                continue
            if status and row.get("status") != status:
                continue
            if query:
                haystack = " ".join(
                    [
                        str(row.get("title") or ""),
                        str(row.get("summary") or ""),
                        str(row.get("project_label") or ""),
                        str(row.get("task_key") or ""),
                        str(row.get("cwd") or ""),
                    ]
                ).lower()
                if query not in haystack:
                    continue
            filtered.append(row)
        filtered.sort(key=lambda item: item.get("updated_ms", 0), reverse=True)
        pending = max(0, len(rows) - sum(1 for item in rows if item.get("status") == "indexed"))
        return {
            "count": len(filtered[:limit]),
            "pending_hint": f"{pending} dialogues still need indexing." if pending else "",
            "sessions": filtered[:limit],
        }

    @app.get("/api/session/{session_id}", response_class=JSONResponse)
    def session_details(session_id: str):
        indexer = RawIndexer(palace_path=palace_path)
        return indexer.session_payload(session_id)

    @app.post("/api/reindex", response_class=JSONResponse)
    def reindex(limit: int = Query(250, ge=1, le=1000)):
        indexer = RawIndexer(palace_path=palace_path)
        return indexer.index_once(max_files=limit)

    @app.get("/api/search", response_class=JSONResponse)
    def search(q: str = Query(..., min_length=2), n: int = Query(8, ge=1, le=20)):
        result = search_memories(q, palace_path=palace_path, n_results=n)
        for hit in result.get("results", []):
            hit["project_label"] = derive_project_label(
                hit.get("source_tool", ""),
                hit.get("source_cwd", ""),
                hit.get("wing", ""),
            )
        return result

    return app

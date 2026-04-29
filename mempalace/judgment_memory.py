#!/usr/bin/env python3
"""
judgment_memory.py — Judgment Memory Engine for MemPalace
=========================================================

MemPalace is strong at verbatim archival and semantic search. This module adds
an explicit judgment layer on top: operational priors that can be promoted,
reinforced, weakened, and retrieved with context.

The implementation stays local-first and dependency-light:
  - SQLite stores structured judgments, candidates, outcomes, and evidence
  - ChromaDB indexes canonical judgment statements for semantic retrieval
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import chromadb

from .config import MempalaceConfig

JUDGMENT_COLLECTION_NAME = "mempalace_judgments"
VOLATILE_DOMAINS = {"ai", "social", "ads", "ad_platforms", "api_pricing"}
AUTO_CANDIDATE_TYPES = {"decision", "preference", "problem", "milestone"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True)


def _json_loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _normalize_statement(statement: str) -> str:
    return " ".join(statement.strip().lower().split())


def _clip_text(text: str, limit: int = 280) -> str:
    collapsed = " ".join(text.strip().split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _split_sentences(text: str) -> List[str]:
    prose = " ".join(text.strip().split())
    if not prose:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", prose) if part.strip()]


def _infer_tags(memory_type: str, text: str) -> List[str]:
    lowered = text.lower()
    tags = [memory_type]
    for keyword in (
        "seo",
        "outreach",
        "cold email",
        "automation",
        "legal",
        "hvac",
        "smb",
        "enterprise",
        "agent",
        "memory",
        "workflow",
        "sales",
    ):
        if keyword in lowered:
            tags.append(keyword.replace(" ", "_"))
    return sorted(set(tags))


def auto_candidate_payload(
    chunk_text: str,
    memory_type: str,
    domain: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if memory_type not in AUTO_CANDIDATE_TYPES:
        return None

    sentences = _split_sentences(chunk_text)
    if not sentences:
        return None

    first = sentences[0]
    second = sentences[1] if len(sentences) > 1 else ""

    if memory_type == "decision":
        statement = _clip_text(first)
        rationale = _clip_text(second or chunk_text, 220)
        quality_score = 0.78
    elif memory_type == "preference":
        statement = _clip_text(f"Preference: {first}")
        rationale = _clip_text(second or "Derived from repeated user/operator preference language.", 220)
        quality_score = 0.7
    elif memory_type == "problem":
        if second:
            statement = _clip_text(f"When this pattern appears, check: {first} Then: {second}")
        else:
            statement = _clip_text(f"When this pattern appears, check: {first}")
        rationale = _clip_text("Heuristically extracted from a problem/fix trace.", 220)
        quality_score = 0.62
    else:  # milestone
        statement = _clip_text(f"Successful pattern observed: {first}")
        rationale = _clip_text(second or "Derived from a success/breakthrough segment.", 220)
        quality_score = 0.64

    normalized = _normalize_statement(statement)
    if len(normalized) < 24:
        return None

    return {
        "statement_draft": statement,
        "rationale_draft": rationale,
        "domain": domain,
        "tags": _infer_tags(memory_type, chunk_text),
        "quality_score": quality_score,
        "novelty_score": 0.55,
        "redundancy_score": 0.2,
        "review_status": "pending",
        "extracted_by": "heuristic_general_extractor",
    }


def posterior_mean(alpha: float, beta: float) -> float:
    total = alpha + beta
    if total <= 0:
        return 0.0
    return alpha / total


def applicability_match(
    stored_conditions: Optional[Dict[str, Any]], context: Optional[Dict[str, Any]]
) -> float:
    if not stored_conditions:
        return 1.0
    if not context:
        return 0.0
    matched = 0
    comparable = 0
    for key, expected in stored_conditions.items():
        if key not in context:
            continue
        comparable += 1
        if context.get(key) != expected:
            return 0.0
        if context.get(key) == expected:
            matched += 1
    if comparable == 0:
        return 0.0
    return matched / len(stored_conditions)


def freshness_score(timestamp: Optional[str], now: Optional[datetime] = None) -> float:
    if not timestamp:
        return 0.4
    now = now or datetime.now(timezone.utc)
    moment = datetime.fromisoformat(timestamp)
    age_days = max((now - moment).total_seconds() / 86400.0, 0.0)
    return math.exp(-age_days / 90.0)


def decayed_confidence(
    base_confidence: float,
    decay_rate: float,
    last_activity_at: Optional[str],
    domain: Optional[str],
    now: Optional[datetime] = None,
) -> tuple[float, float]:
    if not last_activity_at:
        return base_confidence, 0.0

    now = now or datetime.now(timezone.utc)
    last_activity = datetime.fromisoformat(last_activity_at)
    age_days = max((now - last_activity).total_seconds() / 86400.0, 0.0)
    volatility = 2.0 if (domain or "").lower() in VOLATILE_DOMAINS else 1.0
    raw_decay = math.exp(-decay_rate * age_days * volatility)
    stale_score = 1.0 - raw_decay

    # Keep posterior as the anchor, but let stale priors lose leverage over time.
    confidence = max(0.05, base_confidence * (0.35 + 0.65 * raw_decay))
    return confidence, stale_score


@dataclass
class Event:
    event_id: str
    timestamp: str
    source_system: Optional[str]
    source_entity_type: Optional[str]
    source_entity_id: Optional[str]
    actor_type: str
    event_type: str
    payload: Dict[str, Any]
    domain: Optional[str] = None
    workspace_id: Optional[str] = None
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgmentCandidate:
    candidate_id: str
    generated_at: str
    source_event_ids: List[str]
    source_trace_id: Optional[str]
    statement_draft: str
    rationale_draft: Optional[str]
    domain: Optional[str]
    tags: List[str]
    proposed_applicability: Dict[str, Any]
    extracted_by: str
    quality_score: float
    novelty_score: float
    redundancy_score: float
    review_status: str


@dataclass
class Judgment:
    judgment_id: str
    canonical_statement: str
    normalized_statement: str
    domain: Optional[str]
    subdomain: Optional[str]
    tags: List[str]
    status: str
    created_at: str
    updated_at: str
    created_from_candidate_id: Optional[str]
    applicability_conditions: Dict[str, Any]
    exclusion_conditions: Dict[str, Any]
    evidence_strength: float
    confidence_score: float
    posterior_alpha: float
    posterior_beta: float
    support_count: float
    contradict_count: float
    stale_score: float
    last_confirmed_at: Optional[str]
    last_contradicted_at: Optional[str]
    decay_rate: float
    priority_weight: float
    human_verified: bool
    owner_agent: Optional[str]
    owner_module: Optional[str]


@dataclass
class Outcome:
    outcome_id: str
    timestamp: str
    task_id: Optional[str]
    judgment_id: str
    domain: Optional[str]
    outcome_type: str
    value: Optional[float]
    label: Optional[str]
    confidence: float
    observed_after_interval: Optional[int]
    linked_event_ids: List[str]
    metadata: Dict[str, Any]


@dataclass
class EvidenceLink:
    evidence_id: str
    judgment_id: str
    outcome_id: Optional[str]
    event_id: Optional[str]
    evidence_type: str
    direction: str
    weight: float
    extracted_excerpt: Optional[str]
    created_at: str


@dataclass
class DecisionContextPackage:
    package_id: str
    query_context: Dict[str, Any]
    retrieved_judgments: List[Dict[str, Any]]
    suppressed_judgments: List[Dict[str, Any]]
    ranking_rationale: str
    created_at: str


class JudgmentMemoryEngine:
    """Structured judgment memory stored beside the raw MemPalace drawers."""

    def __init__(
        self,
        palace_path: Optional[str] = None,
        db_path: Optional[str] = None,
        collection_name: str = JUDGMENT_COLLECTION_NAME,
    ):
        cfg = MempalaceConfig()
        self.palace_path = palace_path or cfg.palace_path
        self.db_path = db_path or str(Path(self.palace_path) / "judgment_memory.sqlite3")
        self.collection_name = collection_name
        Path(self.palace_path).mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect_db() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    source_system TEXT,
                    source_entity_type TEXT,
                    source_entity_id TEXT,
                    actor_type TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    domain TEXT,
                    workspace_id TEXT,
                    trace_id TEXT,
                    session_id TEXT,
                    task_id TEXT,
                    metadata TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS judgment_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    generated_at TEXT NOT NULL,
                    source_event_ids TEXT NOT NULL,
                    source_trace_id TEXT,
                    statement_draft TEXT NOT NULL,
                    rationale_draft TEXT,
                    domain TEXT,
                    tags TEXT NOT NULL,
                    proposed_applicability TEXT NOT NULL,
                    extracted_by TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    novelty_score REAL NOT NULL,
                    redundancy_score REAL NOT NULL,
                    review_status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS judgments (
                    judgment_id TEXT PRIMARY KEY,
                    canonical_statement TEXT NOT NULL,
                    normalized_statement TEXT NOT NULL UNIQUE,
                    domain TEXT,
                    subdomain TEXT,
                    tags TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_from_candidate_id TEXT,
                    applicability_conditions TEXT NOT NULL,
                    exclusion_conditions TEXT NOT NULL,
                    evidence_strength REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    posterior_alpha REAL NOT NULL,
                    posterior_beta REAL NOT NULL,
                    support_count REAL NOT NULL,
                    contradict_count REAL NOT NULL,
                    stale_score REAL NOT NULL,
                    last_confirmed_at TEXT,
                    last_contradicted_at TEXT,
                    decay_rate REAL NOT NULL,
                    priority_weight REAL NOT NULL,
                    human_verified INTEGER NOT NULL,
                    owner_agent TEXT,
                    owner_module TEXT
                );

                CREATE TABLE IF NOT EXISTS outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    task_id TEXT,
                    judgment_id TEXT NOT NULL,
                    domain TEXT,
                    outcome_type TEXT NOT NULL,
                    value REAL,
                    label TEXT,
                    confidence REAL NOT NULL,
                    observed_after_interval INTEGER,
                    linked_event_ids TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence_links (
                    evidence_id TEXT PRIMARY KEY,
                    judgment_id TEXT NOT NULL,
                    outcome_id TEXT,
                    event_id TEXT,
                    evidence_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    weight REAL NOT NULL,
                    extracted_excerpt TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def _get_collection(self):
        client = chromadb.PersistentClient(path=self.palace_path)
        return client.get_or_create_collection(self.collection_name)

    def _judgment_from_row(self, row: sqlite3.Row) -> Judgment:
        return Judgment(
            judgment_id=row["judgment_id"],
            canonical_statement=row["canonical_statement"],
            normalized_statement=row["normalized_statement"],
            domain=row["domain"],
            subdomain=row["subdomain"],
            tags=_json_loads(row["tags"], []),
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_from_candidate_id=row["created_from_candidate_id"],
            applicability_conditions=_json_loads(row["applicability_conditions"], {}),
            exclusion_conditions=_json_loads(row["exclusion_conditions"], {}),
            evidence_strength=row["evidence_strength"],
            confidence_score=row["confidence_score"],
            posterior_alpha=row["posterior_alpha"],
            posterior_beta=row["posterior_beta"],
            support_count=row["support_count"],
            contradict_count=row["contradict_count"],
            stale_score=row["stale_score"],
            last_confirmed_at=row["last_confirmed_at"],
            last_contradicted_at=row["last_contradicted_at"],
            decay_rate=row["decay_rate"],
            priority_weight=row["priority_weight"],
            human_verified=bool(row["human_verified"]),
            owner_agent=row["owner_agent"],
            owner_module=row["owner_module"],
        )

    def _upsert_judgment_index(self, judgment: Judgment) -> None:
        col = self._get_collection()
        col.upsert(
            ids=[judgment.judgment_id],
            documents=[judgment.canonical_statement],
            metadatas=[
                {
                    "domain": judgment.domain or "general",
                    "status": judgment.status,
                    "confidence_score": float(judgment.confidence_score),
                    "priority_weight": float(judgment.priority_weight),
                    "human_verified": bool(judgment.human_verified),
                    "updated_at": judgment.updated_at,
                }
            ],
        )

    def ingest_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        actor_type: str = "system",
        source_system: Optional[str] = None,
        source_entity_type: Optional[str] = None,
        source_entity_id: Optional[str] = None,
        domain: Optional[str] = None,
        workspace_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        event = Event(
            event_id=str(uuid.uuid4()),
            timestamp=_now_iso(),
            source_system=source_system,
            source_entity_type=source_entity_type,
            source_entity_id=source_entity_id,
            actor_type=actor_type,
            event_type=event_type,
            payload=payload,
            domain=domain,
            workspace_id=workspace_id,
            trace_id=trace_id,
            session_id=session_id,
            task_id=task_id,
            metadata=metadata or {},
        )
        with self._connect_db() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    event_id, timestamp, source_system, source_entity_type, source_entity_id,
                    actor_type, event_type, payload, domain, workspace_id, trace_id,
                    session_id, task_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.timestamp,
                    event.source_system,
                    event.source_entity_type,
                    event.source_entity_id,
                    event.actor_type,
                    event.event_type,
                    _json_dumps(event.payload),
                    event.domain,
                    event.workspace_id,
                    event.trace_id,
                    event.session_id,
                    event.task_id,
                    _json_dumps(event.metadata),
                ),
            )
        return event.event_id

    def create_candidate(
        self,
        statement_draft: str,
        source_event_ids: Optional[Iterable[str]] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        proposed_applicability: Optional[Dict[str, Any]] = None,
        rationale_draft: Optional[str] = None,
        source_trace_id: Optional[str] = None,
        extracted_by: str = "manual",
        quality_score: float = 0.7,
        novelty_score: float = 0.7,
        redundancy_score: float = 0.0,
        review_status: str = "pending",
    ) -> str:
        candidate = JudgmentCandidate(
            candidate_id=str(uuid.uuid4()),
            generated_at=_now_iso(),
            source_event_ids=list(source_event_ids or []),
            source_trace_id=source_trace_id,
            statement_draft=statement_draft,
            rationale_draft=rationale_draft,
            domain=domain,
            tags=tags or [],
            proposed_applicability=proposed_applicability or {},
            extracted_by=extracted_by,
            quality_score=quality_score,
            novelty_score=novelty_score,
            redundancy_score=redundancy_score,
            review_status=review_status,
        )
        with self._connect_db() as conn:
            conn.execute(
                """
                INSERT INTO judgment_candidates (
                    candidate_id, generated_at, source_event_ids, source_trace_id,
                    statement_draft, rationale_draft, domain, tags,
                    proposed_applicability, extracted_by, quality_score,
                    novelty_score, redundancy_score, review_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.candidate_id,
                    candidate.generated_at,
                    _json_dumps(candidate.source_event_ids),
                    candidate.source_trace_id,
                    candidate.statement_draft,
                    candidate.rationale_draft,
                    candidate.domain,
                    _json_dumps(candidate.tags),
                    _json_dumps(candidate.proposed_applicability),
                    candidate.extracted_by,
                    candidate.quality_score,
                    candidate.novelty_score,
                    candidate.redundancy_score,
                    candidate.review_status,
                ),
            )
        return candidate.candidate_id

    def create_candidate_from_memory_chunk(
        self,
        chunk_text: str,
        memory_type: str,
        source_event_ids: Optional[Iterable[str]] = None,
        domain: Optional[str] = None,
        proposed_applicability: Optional[Dict[str, Any]] = None,
        source_trace_id: Optional[str] = None,
    ) -> Optional[str]:
        payload = auto_candidate_payload(chunk_text, memory_type, domain=domain)
        if not payload:
            return None
        return self.create_candidate(
            statement_draft=payload["statement_draft"],
            source_event_ids=source_event_ids,
            domain=payload["domain"],
            tags=payload["tags"],
            proposed_applicability=proposed_applicability,
            rationale_draft=payload["rationale_draft"],
            source_trace_id=source_trace_id,
            extracted_by=payload["extracted_by"],
            quality_score=payload["quality_score"],
            novelty_score=payload["novelty_score"],
            redundancy_score=payload["redundancy_score"],
            review_status=payload["review_status"],
        )

    def promote_candidate(
        self,
        candidate_id: str,
        canonical_statement: Optional[str] = None,
        status: str = "active",
        human_verified: bool = False,
        decay_rate: float = 0.015,
        priority_weight: float = 1.0,
        owner_agent: Optional[str] = None,
        owner_module: str = "judgment_memory",
    ) -> str:
        with self._connect_db() as conn:
            row = conn.execute(
                "SELECT * FROM judgment_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Unknown candidate_id: {candidate_id}")

            statement = canonical_statement or row["statement_draft"]
            normalized = _normalize_statement(statement)
            existing = conn.execute(
                "SELECT judgment_id FROM judgments WHERE normalized_statement = ?",
                (normalized,),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE judgment_candidates SET review_status = ? WHERE candidate_id = ?",
                    ("merged", candidate_id),
                )
                return existing["judgment_id"]

            now = _now_iso()
            alpha = 1.0
            beta = 1.0
            confidence = posterior_mean(alpha, beta)
            judgment = Judgment(
                judgment_id=str(uuid.uuid4()),
                canonical_statement=statement,
                normalized_statement=normalized,
                domain=row["domain"],
                subdomain=None,
                tags=_json_loads(row["tags"], []),
                status=status,
                created_at=now,
                updated_at=now,
                created_from_candidate_id=candidate_id,
                applicability_conditions=_json_loads(row["proposed_applicability"], {}),
                exclusion_conditions={},
                evidence_strength=0.0,
                confidence_score=confidence,
                posterior_alpha=alpha,
                posterior_beta=beta,
                support_count=0.0,
                contradict_count=0.0,
                stale_score=0.0,
                last_confirmed_at=None,
                last_contradicted_at=None,
                decay_rate=decay_rate,
                priority_weight=priority_weight,
                human_verified=human_verified,
                owner_agent=owner_agent,
                owner_module=owner_module,
            )
            conn.execute(
                """
                INSERT INTO judgments (
                    judgment_id, canonical_statement, normalized_statement, domain, subdomain, tags,
                    status, created_at, updated_at, created_from_candidate_id,
                    applicability_conditions, exclusion_conditions, evidence_strength,
                    confidence_score, posterior_alpha, posterior_beta,
                    support_count, contradict_count, stale_score,
                    last_confirmed_at, last_contradicted_at, decay_rate,
                    priority_weight, human_verified, owner_agent, owner_module
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    judgment.judgment_id,
                    judgment.canonical_statement,
                    judgment.normalized_statement,
                    judgment.domain,
                    judgment.subdomain,
                    _json_dumps(judgment.tags),
                    judgment.status,
                    judgment.created_at,
                    judgment.updated_at,
                    judgment.created_from_candidate_id,
                    _json_dumps(judgment.applicability_conditions),
                    _json_dumps(judgment.exclusion_conditions),
                    judgment.evidence_strength,
                    judgment.confidence_score,
                    judgment.posterior_alpha,
                    judgment.posterior_beta,
                    judgment.support_count,
                    judgment.contradict_count,
                    judgment.stale_score,
                    judgment.last_confirmed_at,
                    judgment.last_contradicted_at,
                    judgment.decay_rate,
                    judgment.priority_weight,
                    int(judgment.human_verified),
                    judgment.owner_agent,
                    judgment.owner_module,
                ),
            )
            conn.execute(
                "UPDATE judgment_candidates SET review_status = ? WHERE candidate_id = ?",
                ("promoted", candidate_id),
            )

        self._upsert_judgment_index(judgment)
        return judgment.judgment_id

    def get_judgment(self, judgment_id: str) -> Judgment:
        with self._connect_db() as conn:
            row = conn.execute(
                "SELECT * FROM judgments WHERE judgment_id = ?",
                (judgment_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Unknown judgment_id: {judgment_id}")
            return self._judgment_from_row(row)

    def list_judgments(
        self,
        limit: int = 10,
        domain: Optional[str] = None,
        statuses: Optional[Iterable[str]] = None,
    ) -> List[Judgment]:
        statuses = list(statuses or ["active", "weakened"])
        placeholders = ", ".join("?" for _ in statuses)
        params: List[Any] = list(statuses)
        query = (
            "SELECT * FROM judgments WHERE status IN ({})".format(placeholders)
            + " ORDER BY confidence_score DESC, priority_weight DESC LIMIT ?"
        )
        params.append(limit)
        if domain:
            query = (
                "SELECT * FROM judgments WHERE domain = ? AND status IN ({})".format(placeholders)
                + " ORDER BY confidence_score DESC, priority_weight DESC LIMIT ?"
            )
            params = [domain] + list(statuses) + [limit]

        with self._connect_db() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._judgment_from_row(row) for row in rows]

    def record_outcome(
        self,
        judgment_id: str,
        outcome_type: str,
        label: Optional[str],
        is_support: bool,
        weight: float = 1.0,
        confidence: float = 1.0,
        value: Optional[float] = None,
        linked_event_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        observed_after_interval: Optional[int] = None,
        task_id: Optional[str] = None,
        extracted_excerpt: Optional[str] = None,
    ) -> str:
        judgment = self.get_judgment(judgment_id)
        weight = max(weight, 0.0)
        effective_weight = weight * max(min(confidence, 1.0), 0.0)
        now = _now_iso()

        outcome = Outcome(
            outcome_id=str(uuid.uuid4()),
            timestamp=now,
            task_id=task_id,
            judgment_id=judgment_id,
            domain=judgment.domain,
            outcome_type=outcome_type,
            value=value,
            label=label,
            confidence=confidence,
            observed_after_interval=observed_after_interval,
            linked_event_ids=list(linked_event_ids or []),
            metadata=metadata or {},
        )
        evidence = EvidenceLink(
            evidence_id=str(uuid.uuid4()),
            judgment_id=judgment_id,
            outcome_id=outcome.outcome_id,
            event_id=outcome.linked_event_ids[0] if outcome.linked_event_ids else None,
            evidence_type="outcome",
            direction="support" if is_support else "contradict",
            weight=effective_weight,
            extracted_excerpt=extracted_excerpt,
            created_at=now,
        )

        alpha = judgment.posterior_alpha + (effective_weight if is_support else 0.0)
        beta = judgment.posterior_beta + (effective_weight if not is_support else 0.0)
        base_confidence = posterior_mean(alpha, beta)

        with self._connect_db() as conn:
            conn.execute(
                """
                INSERT INTO outcomes (
                    outcome_id, timestamp, task_id, judgment_id, domain, outcome_type,
                    value, label, confidence, observed_after_interval, linked_event_ids, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome.outcome_id,
                    outcome.timestamp,
                    outcome.task_id,
                    outcome.judgment_id,
                    outcome.domain,
                    outcome.outcome_type,
                    outcome.value,
                    outcome.label,
                    outcome.confidence,
                    outcome.observed_after_interval,
                    _json_dumps(outcome.linked_event_ids),
                    _json_dumps(outcome.metadata),
                ),
            )
            conn.execute(
                """
                INSERT INTO evidence_links (
                    evidence_id, judgment_id, outcome_id, event_id, evidence_type,
                    direction, weight, extracted_excerpt, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.evidence_id,
                    evidence.judgment_id,
                    evidence.outcome_id,
                    evidence.event_id,
                    evidence.evidence_type,
                    evidence.direction,
                    evidence.weight,
                    evidence.extracted_excerpt,
                    evidence.created_at,
                ),
            )

            support_count = judgment.support_count + (effective_weight if is_support else 0.0)
            contradict_count = judgment.contradict_count + (
                effective_weight if not is_support else 0.0
            )
            evidence_strength = support_count + contradict_count
            status = judgment.status
            if base_confidence < 0.35 and contradict_count >= support_count:
                status = "weakened"
            elif base_confidence >= 0.45 and status in {"candidate", "weakened"}:
                status = "active"

            conn.execute(
                """
                UPDATE judgments
                SET updated_at = ?,
                    evidence_strength = ?,
                    confidence_score = ?,
                    posterior_alpha = ?,
                    posterior_beta = ?,
                    support_count = ?,
                    contradict_count = ?,
                    stale_score = 0.0,
                    last_confirmed_at = ?,
                    last_contradicted_at = ?,
                    status = ?
                WHERE judgment_id = ?
                """,
                (
                    now,
                    evidence_strength,
                    base_confidence,
                    alpha,
                    beta,
                    support_count,
                    contradict_count,
                    now if is_support else judgment.last_confirmed_at,
                    now if not is_support else judgment.last_contradicted_at,
                    status,
                    judgment_id,
                ),
            )

        self._upsert_judgment_index(self.get_judgment(judgment_id))
        return outcome.outcome_id

    def apply_decay(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.now(timezone.utc)
        updated = 0
        with self._connect_db() as conn:
            rows = conn.execute(
                "SELECT * FROM judgments WHERE status NOT IN ('archived', 'rejected')"
            ).fetchall()
            for row in rows:
                judgment = self._judgment_from_row(row)
                base_confidence = posterior_mean(
                    judgment.posterior_alpha,
                    judgment.posterior_beta,
                )
                last_activity = (
                    judgment.last_confirmed_at
                    or judgment.last_contradicted_at
                    or judgment.updated_at
                    or judgment.created_at
                )
                confidence, stale = decayed_confidence(
                    base_confidence,
                    judgment.decay_rate,
                    last_activity,
                    judgment.domain,
                    now=now,
                )
                status = judgment.status
                if confidence < 0.35 and status == "active":
                    status = "weakened"
                conn.execute(
                    """
                    UPDATE judgments
                    SET confidence_score = ?, stale_score = ?, updated_at = ?, status = ?
                    WHERE judgment_id = ?
                    """,
                    (
                        confidence,
                        stale,
                        now.replace(microsecond=0).isoformat(),
                        status,
                        judgment.judgment_id,
                    ),
                )
                updated += 1

        for judgment in self.list_judgments(limit=1000, statuses=["active", "weakened", "candidate"]):
            self._upsert_judgment_index(judgment)
        return updated

    def retrieve_judgments(
        self,
        query: str,
        domain: Optional[str] = None,
        applicability_context: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> DecisionContextPackage:
        col = self._get_collection()
        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": max(limit * 4, limit),
            "include": ["metadatas", "distances", "documents"],
        }
        if domain:
            kwargs["where"] = {"domain": domain}
        results = col.query(**kwargs)

        retrieved: List[Dict[str, Any]] = []
        suppressed: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)

        for judgment_id, doc, meta, dist in zip(
            results.get("ids", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        ):
            judgment = self.get_judgment(judgment_id)
            applicability = applicability_match(
                judgment.applicability_conditions,
                applicability_context,
            )
            if judgment.applicability_conditions and applicability == 0.0:
                suppressed.append(
                    {
                        "judgment_id": judgment_id,
                        "reason": "applicability_mismatch",
                        "canonical_statement": judgment.canonical_statement,
                    }
                )
                continue
            if judgment.status not in {"active", "weakened"}:
                suppressed.append(
                    {
                        "judgment_id": judgment_id,
                        "reason": f"status_{judgment.status}",
                        "canonical_statement": judgment.canonical_statement,
                    }
                )
                continue

            semantic = max(0.0, 1.0 - dist)
            freshness = freshness_score(
                judgment.last_confirmed_at or judgment.updated_at or judgment.created_at,
                now=now,
            )
            contradiction_ratio = judgment.contradict_count / max(
                judgment.support_count + judgment.contradict_count,
                1.0,
            )
            priority_component = min(judgment.priority_weight, 3.0) / 3.0
            human_boost = 0.05 if judgment.human_verified else 0.0
            retrieval_score = (
                semantic * 0.45
                + applicability * 0.20
                + judgment.confidence_score * 0.20
                + freshness * 0.10
                + priority_component * 0.05
                + human_boost
                - contradiction_ratio * 0.15
            )

            retrieved.append(
                {
                    "judgment": asdict(judgment),
                    "semantic_similarity": round(semantic, 4),
                    "applicability_match": round(applicability, 4),
                    "freshness": round(freshness, 4),
                    "retrieval_score": round(retrieval_score, 4),
                    "matched_text": doc,
                    "index_metadata": meta,
                }
            )

        retrieved.sort(key=lambda item: item["retrieval_score"], reverse=True)
        package = DecisionContextPackage(
            package_id=str(uuid.uuid4()),
            query_context={
                "query": query,
                "domain": domain,
                "applicability_context": applicability_context or {},
                "limit": limit,
            },
            retrieved_judgments=retrieved[:limit],
            suppressed_judgments=suppressed,
            ranking_rationale=(
                "semantic similarity + applicability + confidence + freshness "
                "+ human verification - contradiction penalty"
            ),
            created_at=_now_iso(),
        )
        return package

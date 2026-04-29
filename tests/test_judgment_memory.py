from datetime import datetime, timedelta, timezone

from mempalace.judgment_memory import JudgmentMemoryEngine


def test_judgment_lifecycle_from_candidate_to_outcome(palace_path):
    engine = JudgmentMemoryEngine(palace_path=palace_path)

    event_id = engine.ingest_event(
        event_type="recommendation_issued",
        actor_type="agent",
        domain="sales",
        payload={"message": "Start with missed-call text-back automation for HVAC SMBs."},
        trace_id="trace-1",
    )
    candidate_id = engine.create_candidate(
        statement_draft=(
            "For owner-operated HVAC SMBs, missed-call text-back automation is a better "
            "first wedge than full AI orchestration."
        ),
        source_event_ids=[event_id],
        domain="sales",
        tags=["hvac", "automation", "smb"],
        proposed_applicability={
            "company_size": "SMB",
            "owner_operated": True,
            "channel": "phone",
        },
        rationale_draft="Higher adoption and lower implementation friction.",
    )

    judgment_id = engine.promote_candidate(candidate_id, human_verified=True, priority_weight=2.0)
    judgment = engine.get_judgment(judgment_id)
    assert judgment.status == "active"
    assert judgment.human_verified is True
    assert judgment.confidence_score == 0.5

    engine.record_outcome(
        judgment_id=judgment_id,
        outcome_type="workflow_adopted",
        label="adopted_with_positive_feedback",
        is_support=True,
        weight=2.0,
        confidence=1.0,
        linked_event_ids=[event_id],
        extracted_excerpt="Client adopted the text-back wedge and expanded rollout.",
    )
    updated = engine.get_judgment(judgment_id)
    assert updated.posterior_alpha == 3.0
    assert updated.posterior_beta == 1.0
    assert updated.confidence_score == 0.75
    assert updated.support_count == 2.0
    assert updated.last_confirmed_at is not None


def test_decay_weakens_stale_judgments(palace_path):
    engine = JudgmentMemoryEngine(palace_path=palace_path)
    candidate_id = engine.create_candidate(
        statement_draft="Local service pages beat blogs for early SEO traction in local legal markets.",
        domain="seo",
        tags=["seo", "legal"],
    )
    judgment_id = engine.promote_candidate(candidate_id, decay_rate=0.03)
    engine.record_outcome(
        judgment_id=judgment_id,
        outcome_type="page_ctr_change",
        label="ctr_up",
        is_support=True,
        weight=3.0,
        confidence=1.0,
    )

    stale_moment = (datetime.now(timezone.utc) - timedelta(days=180)).replace(microsecond=0)
    with engine._connect_db() as conn:
        conn.execute(
            """
            UPDATE judgments
            SET last_confirmed_at = ?, updated_at = ?
            WHERE judgment_id = ?
            """,
            (
                stale_moment.isoformat(),
                stale_moment.isoformat(),
                judgment_id,
            ),
        )

    before = engine.get_judgment(judgment_id)
    engine.apply_decay(now=datetime.now(timezone.utc))
    after = engine.get_judgment(judgment_id)

    assert after.confidence_score < before.confidence_score
    assert after.stale_score > 0


def test_retrieval_prefers_matching_applicability_and_stronger_evidence(palace_path):
    engine = JudgmentMemoryEngine(palace_path=palace_path)

    first_id = engine.promote_candidate(
        engine.create_candidate(
            statement_draft=(
                "Cold-email outreach for SMB automation sells better with ROI proof and "
                "concrete case metrics than with generic AI promises."
            ),
            domain="sales",
            tags=["outreach", "cold-email"],
            proposed_applicability={
                "company_size": "SMB",
                "channel": "cold_email",
                "vertical": "home_services",
            },
        ),
        human_verified=True,
        priority_weight=2.5,
    )
    second_id = engine.promote_candidate(
        engine.create_candidate(
            statement_draft=(
                "Enterprise outbound motion should lead with compliance posture before ROI."
            ),
            domain="sales",
            tags=["outreach", "enterprise"],
            proposed_applicability={
                "company_size": "enterprise",
                "channel": "cold_email",
            },
        ),
        priority_weight=1.0,
    )

    engine.record_outcome(
        judgment_id=first_id,
        outcome_type="lead_replied",
        label="reply_received",
        is_support=True,
        weight=3.0,
    )
    engine.record_outcome(
        judgment_id=second_id,
        outcome_type="lead_replied",
        label="no_reply",
        is_support=False,
        weight=1.0,
    )

    package = engine.retrieve_judgments(
        query="Best cold email angle for smb automation outreach",
        domain="sales",
        applicability_context={
            "company_size": "SMB",
            "channel": "cold_email",
            "vertical": "home_services",
        },
        limit=3,
    )

    assert package.retrieved_judgments
    top = package.retrieved_judgments[0]
    assert top["judgment"]["judgment_id"] == first_id
    suppressed_ids = {item["judgment_id"] for item in package.suppressed_judgments}
    assert second_id in suppressed_ids

import os
import tempfile
import shutil
import chromadb
from mempalace.convo_miner import mine_convos
from mempalace.judgment_memory import JudgmentMemoryEngine


def test_convo_mining():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "chat.txt"), "w") as f:
        f.write(
            "> What is memory?\nMemory is persistence.\n\n> Why does it matter?\nIt enables continuity.\n\n> How do we build it?\nWith structured storage.\n"
        )

    palace_path = os.path.join(tmpdir, "palace")
    mine_convos(tmpdir, palace_path, wing="test_convos")

    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_collection("mempalace_drawers")
    assert col.count() >= 2

    # Verify search works
    results = col.query(query_texts=["memory persistence"], n_results=1)
    assert len(results["documents"][0]) > 0

    shutil.rmtree(tmpdir)


def test_convo_mining_can_generate_judgment_candidates():
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, "strategy.txt"), "w") as f:
            f.write(
                """
We decided to lead cold outreach with ROI proof and concrete case numbers rather than generic AI claims.
Because vague promises were ignored and the concrete case angle got replies.

I prefer short, direct implementation plans over long essays.

The bug was that the webhook payload was empty because the proxy stripped the body.
The fix was to read the raw request stream before parsing.
""".strip()
            )

        palace_path = os.path.join(tmpdir, "palace")
        mine_convos(
            tmpdir,
            palace_path,
            wing="sales_playbook",
            extract_mode="general",
            enable_judgments=True,
        )

        engine = JudgmentMemoryEngine(palace_path=palace_path)
        judgments = engine.list_judgments(limit=10, statuses=["active", "weakened", "candidate"])
        assert judgments == []

        with engine._connect_db() as conn:
            candidate_count = conn.execute(
                "SELECT COUNT(*) AS count FROM judgment_candidates"
            ).fetchone()["count"]
            event_count = conn.execute("SELECT COUNT(*) AS count FROM events").fetchone()["count"]

        assert candidate_count >= 2
        assert event_count >= candidate_count
    finally:
        shutil.rmtree(tmpdir)

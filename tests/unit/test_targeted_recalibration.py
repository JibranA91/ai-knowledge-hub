"""Unit tests for targeted recalibration feature.

Covers:
- RecalibrateJob.fact_instructions field
- RecalibrateRequest.fact_instructions field
- RecalibrateAgentRunner.run() initial state injection
- triage_node targeted mode (semantic search, no health signals)
- analyze_node targeted directive prepended
- finalize_node [TARGETED] log prefix
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── RecalibrateJob dataclass ──────────────────────────────────────────────

def test_recalibrate_job_fact_instructions_default():
    from app.services.recalibrate_job import RecalibrateJob
    job = RecalibrateJob()
    assert job.fact_instructions == ""


def test_recalibrate_job_fact_instructions_set():
    from app.services.recalibrate_job import RecalibrateJob
    job = RecalibrateJob(fact_instructions="The CEO is John Smith not Jane Doe")
    assert job.fact_instructions == "The CEO is John Smith not Jane Doe"


def test_recalibrate_job_reset_clears_fact_instructions():
    from app.services import recalibrate_job
    org_id = "test-reset-org"
    recalibrate_job.get(org_id).fact_instructions = "some fact"
    recalibrate_job.reset(org_id)
    assert recalibrate_job.get(org_id).fact_instructions == ""


# ── RecalibrateRequest model ──────────────────────────────────────────────

def test_recalibrate_request_fact_instructions_default():
    from app.routes.operations import RecalibrateRequest
    req = RecalibrateRequest()
    assert req.fact_instructions == ""


def test_recalibrate_request_fact_instructions_set():
    from app.routes.operations import RecalibrateRequest
    req = RecalibrateRequest(fact_instructions="Revenue for Q3 is wrong")
    assert req.fact_instructions == "Revenue for Q3 is wrong"


def test_recalibrate_request_deleted_files_unaffected():
    from app.routes.operations import RecalibrateRequest
    req = RecalibrateRequest(deleted_files=["a.pdf"], fact_instructions="fix this")
    assert req.deleted_files == ["a.pdf"]
    assert req.fact_instructions == "fix this"


# ── RecalibrateAgentRunner initial state ──────────────────────────────────

@pytest.mark.asyncio
async def test_runner_injects_fact_instructions_into_state():
    with patch("app.services.recalibrate_agent.build_recalibrate_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock()
        mock_build.return_value = mock_graph

        from app.services.recalibrate_agent import RecalibrateAgentRunner
        runner = RecalibrateAgentRunner()
        await runner.run(fact_instructions="Wrong revenue figure on Q3 page")

        called_state = mock_graph.ainvoke.call_args[0][0]
        assert called_state["fact_instructions"] == "Wrong revenue figure on Q3 page"


@pytest.mark.asyncio
async def test_runner_fact_instructions_defaults_to_empty():
    with patch("app.services.recalibrate_agent.build_recalibrate_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock()
        mock_build.return_value = mock_graph

        from app.services.recalibrate_agent import RecalibrateAgentRunner
        runner = RecalibrateAgentRunner()
        await runner.run()

        called_state = mock_graph.ainvoke.call_args[0][0]
        assert called_state["fact_instructions"] == ""


@pytest.mark.asyncio
async def test_runner_passes_deleted_files_and_fact_instructions_together():
    with patch("app.services.recalibrate_agent.build_recalibrate_graph") as mock_build:
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock()
        mock_build.return_value = mock_graph

        from app.services.recalibrate_agent import RecalibrateAgentRunner
        runner = RecalibrateAgentRunner()
        await runner.run(deleted_files=["old.pdf"], fact_instructions="Fix the org chart")

        called_state = mock_graph.ainvoke.call_args[0][0]
        assert called_state["deleted_files"] == ["old.pdf"]
        assert called_state["fact_instructions"] == "Fix the org chart"


# ── triage_node: targeted mode via semantic search ────────────────────────

@pytest.mark.asyncio
async def test_triage_node_targeted_uses_semantic_search():
    """Targeted mode skips health signals and uses semantic search."""
    mock_pages = {
        "topics/revenue.md": "# Revenue\n\nQ3 revenue was $10M.",
        "topics/org.md": "# Org Chart\n\nCEO is John Smith.",
    }
    semantic_results = [{"path": "topics/revenue.md", "score": 0.95}]

    with (
        patch("app.services.recalibrate_agent.make_chat_llm", return_value=MagicMock()),
        patch("app.services.recalibrate_agent.recalibrate_job") as mock_job_mod,
        patch("app.services.embeddings.embed_text", new_callable=AsyncMock, return_value=[0.1, 0.2]) as mock_embed,
        patch("app.services.wiki_db.semantic_search_wiki", new_callable=AsyncMock, return_value=semantic_results),
    ):
        mock_job = MagicMock()
        mock_job_mod.get.return_value = mock_job

        from app.services.recalibrate_agent import build_recalibrate_graph
        # Build graph to get access to triage_node closure
        graph = build_recalibrate_graph()

        state = {
            "wiki_pages": mock_pages,
            "schema": "",
            "index": "",
            "deleted_files": [],
            "fact_instructions": "Q3 revenue figure is incorrect",
            "triage_candidates": [],
            "improvement_plan": [],
            "new_page_plan": [],
            "delete_plan": [],
            "rename_plan": [],
            "pages_written": [],
            "pages_deleted": [],
            "pages_renamed": [],
            "errors": [],
            "log_entry": "",
            "_t_start": 0.0,
        }

        # Invoke just the triage node
        result = await graph.nodes["triage"].bound.afunc(state)

        candidates = result["triage_candidates"]
        assert len(candidates) == 1
        assert candidates[0]["path"] == "topics/revenue.md"
        assert candidates[0]["reasons"] == ["targeted"]


@pytest.mark.asyncio
async def test_triage_node_targeted_skips_unsafe_paths():
    """Targeted mode must not include protected paths even if semantic search returns them."""
    mock_pages = {
        "topics/revenue.md": "# Revenue content",
        "index.md": "# Index",
    }
    semantic_results = [
        {"path": "topics/revenue.md", "score": 0.9},
        {"path": "index.md", "score": 0.8},  # protected — must be filtered
    ]

    with (
        patch("app.services.recalibrate_agent.make_chat_llm", return_value=MagicMock()),
        patch("app.services.recalibrate_agent.recalibrate_job") as mock_job_mod,
        patch("app.services.embeddings.embed_text", new_callable=AsyncMock, return_value=[0.1]),
        patch("app.services.wiki_db.semantic_search_wiki", new_callable=AsyncMock, return_value=semantic_results),
    ):
        mock_job_mod.get.return_value = MagicMock()

        from app.services.recalibrate_agent import build_recalibrate_graph
        graph = build_recalibrate_graph()

        state = {
            "wiki_pages": mock_pages,
            "schema": "",
            "index": "",
            "deleted_files": [],
            "fact_instructions": "fix revenue",
            "triage_candidates": [],
            "improvement_plan": [],
            "new_page_plan": [],
            "delete_plan": [],
            "rename_plan": [],
            "pages_written": [],
            "pages_deleted": [],
            "pages_renamed": [],
            "errors": [],
            "log_entry": "",
            "_t_start": 0.0,
        }

        result = await graph.nodes["triage"].bound.afunc(state)

        paths = [c["path"] for c in result["triage_candidates"]]
        assert "index.md" not in paths
        assert "topics/revenue.md" in paths


@pytest.mark.asyncio
async def test_triage_node_targeted_returns_empty_when_embed_fails():
    """When embed_text returns None, targeted triage yields no candidates."""
    with (
        patch("app.services.recalibrate_agent.make_chat_llm", return_value=MagicMock()),
        patch("app.services.recalibrate_agent.recalibrate_job") as mock_job_mod,
        patch("app.services.embeddings.embed_text", new_callable=AsyncMock, return_value=None),
        patch("app.services.wiki_db.semantic_search_wiki", new_callable=AsyncMock) as mock_search,
    ):
        mock_job_mod.get.return_value = MagicMock()

        from app.services.recalibrate_agent import build_recalibrate_graph
        graph = build_recalibrate_graph()

        state = {
            "wiki_pages": {"topics/a.md": "content"},
            "schema": "",
            "index": "",
            "deleted_files": [],
            "fact_instructions": "some broken fact",
            "triage_candidates": [],
            "improvement_plan": [],
            "new_page_plan": [],
            "delete_plan": [],
            "rename_plan": [],
            "pages_written": [],
            "pages_deleted": [],
            "pages_renamed": [],
            "errors": [],
            "log_entry": "",
            "_t_start": 0.0,
        }

        result = await graph.nodes["triage"].bound.afunc(state)
        assert result["triage_candidates"] == []
        mock_search.assert_not_called()


# ── finalize_node: [TARGETED] log prefix ─────────────────────────────────

@pytest.mark.asyncio
async def test_finalize_node_targeted_prefix_in_log():
    """When fact_instructions is set, log entry uses (targeted) heading and includes the instruction."""
    captured_log = {}

    async def fake_append_audit_log(operation, raw_text):
        captured_log["entry"] = raw_text

    with (
        patch("app.services.recalibrate_agent.make_chat_llm", return_value=MagicMock()),
        patch("app.services.recalibrate_agent.recalibrate_job") as mock_job_mod,
        patch("app.services.wiki_db.append_audit_log", side_effect=fake_append_audit_log),
        patch("app.services.recalibrate_agent.recalibrate_job.persist_finish", new_callable=AsyncMock),
    ):
        mock_job = MagicMock()
        mock_job.errors = []
        mock_job_mod.get.return_value = mock_job

        from app.services.recalibrate_agent import build_recalibrate_graph
        graph = build_recalibrate_graph()

        import time
        state = {
            "wiki_pages": {"a.md": "content"},
            "schema": "",
            "index": "",
            "deleted_files": [],
            "fact_instructions": "Q3 revenue is wrong — should be $12M",
            "triage_candidates": [{"path": "a.md", "reasons": ["targeted"], "priority": 1}],
            "improvement_plan": [],
            "new_page_plan": [],
            "delete_plan": [],
            "rename_plan": [],
            "pages_written": ["a.md"],
            "pages_deleted": [],
            "pages_renamed": [],
            "errors": [],
            "log_entry": "",
            "_t_start": time.time(),
        }

        await graph.nodes["finalize"].bound.afunc(state)

        entry = captured_log.get("entry", "")
        assert "(targeted)" in entry
        assert "Q3 revenue is wrong" in entry
        assert "Fact instructions:" in entry


@pytest.mark.asyncio
async def test_finalize_node_full_mode_uses_master_recalibrate_heading():
    """Without fact_instructions, log entry uses master-recalibrate heading."""
    captured_log = {}

    async def fake_append_audit_log(operation, raw_text):
        captured_log["entry"] = raw_text

    with (
        patch("app.services.recalibrate_agent.make_chat_llm", return_value=MagicMock()),
        patch("app.services.recalibrate_agent.recalibrate_job") as mock_job_mod,
        patch("app.services.wiki_db.append_audit_log", side_effect=fake_append_audit_log),
        patch("app.services.recalibrate_agent.recalibrate_job.persist_finish", new_callable=AsyncMock),
    ):
        mock_job = MagicMock()
        mock_job.errors = []
        mock_job_mod.get.return_value = mock_job

        from app.services.recalibrate_agent import build_recalibrate_graph
        graph = build_recalibrate_graph()

        import time
        state = {
            "wiki_pages": {},
            "schema": "",
            "index": "",
            "deleted_files": [],
            "fact_instructions": "",
            "triage_candidates": [],
            "improvement_plan": [],
            "new_page_plan": [],
            "delete_plan": [],
            "rename_plan": [],
            "pages_written": [],
            "pages_deleted": [],
            "pages_renamed": [],
            "errors": [],
            "log_entry": "",
            "_t_start": time.time(),
        }

        await graph.nodes["finalize"].bound.afunc(state)

        entry = captured_log.get("entry", "")
        assert "master-recalibrate" in entry
        assert "(targeted)" not in entry

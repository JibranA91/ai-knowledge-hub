"""Integration tests for /api/ops routes.

Requires PostgreSQL via testcontainers.
LLM (Bedrock) calls are mocked throughout.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# ── GET /api/ops/status/{filename} ────────────────────────────────────────

@pytest.mark.asyncio
async def test_status_not_found(client, auth_headers):
    resp = await client.get("/api/ops/status/unknown.pdf", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_status_returns_job(client, auth_headers, user_ctx):
    from app.services.jobs import enqueue
    await enqueue("status_test.txt")

    resp = await client.get("/api/ops/status/status_test.txt", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "status_test.txt"
    assert data["status"] == "queued"


# ── DELETE /api/ops/ingest/{filename} ─────────────────────────────────────

@pytest.mark.asyncio
async def test_cancel_ingest_not_found(client, auth_headers):
    resp = await client.delete("/api/ops/ingest/missing.pdf", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_ingest_success(client, auth_headers, user_ctx):
    from app.services.jobs import enqueue
    await enqueue("cancel_me.txt")

    resp = await client.delete("/api/ops/ingest/cancel_me.txt", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["cancelled"] == "cancel_me.txt"

    # Status should be cancelled
    status = await client.get(
        "/api/ops/status/cancel_me.txt", headers=auth_headers
    )
    assert status.json()["status"] == "cancelled"


# ── POST /api/ops/query ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_returns_answer(client, auth_headers):
    mock_answer = "The answer is 42."
    with patch(
        "app.services.wiki_engine.WikiEngine._find_relevant_pages",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "app.providers.bedrock.BedrockConverseClient.converse",
        new_callable=AsyncMock,
        return_value=mock_answer,
    ):
        resp = await client.post(
            "/api/ops/query",
            json={"question": "What is the answer?"},
            headers=auth_headers,
        )

    assert resp.status_code == 200
    assert resp.json()["answer"] == mock_answer


@pytest.mark.asyncio
async def test_query_empty_question(client, auth_headers):
    resp = await client.post(
        "/api/ops/query", json={"question": "   "}, headers=auth_headers
    )
    assert resp.status_code == 400


# ── POST /api/ops/chat ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_creates_session(client, auth_headers):
    with patch(
        "app.services.wiki_engine.WikiEngine._find_relevant_pages",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "app.providers.bedrock.BedrockConverseClient.converse",
        new_callable=AsyncMock,
        return_value="Hello there!",
    ):
        resp = await client.post(
            "/api/ops/chat",
            json={"message": "Hi!", "session_id": None},
            headers=auth_headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["answer"] == "Hello there!"


@pytest.mark.asyncio
async def test_chat_continues_session(client, auth_headers):
    with patch(
        "app.services.wiki_engine.WikiEngine._find_relevant_pages",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "app.providers.bedrock.BedrockConverseClient.converse",
        new_callable=AsyncMock,
        return_value="Response 1",
    ) as mock_converse:
        r1 = await client.post(
            "/api/ops/chat",
            json={"message": "First message", "session_id": None},
            headers=auth_headers,
        )
        sid = r1.json()["session_id"]

        mock_converse.return_value = "Response 2"
        r2 = await client.post(
            "/api/ops/chat",
            json={"message": "Second message", "session_id": sid},
            headers=auth_headers,
        )

    assert r2.json()["session_id"] == sid


@pytest.mark.asyncio
async def test_chat_empty_message(client, auth_headers):
    resp = await client.post(
        "/api/ops/chat", json={"message": "  ", "session_id": None}, headers=auth_headers
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_clear_chat_session(client, auth_headers):
    # Create session
    with patch(
        "app.services.wiki_engine.WikiEngine._find_relevant_pages",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "app.providers.bedrock.BedrockConverseClient.converse",
        new_callable=AsyncMock,
        return_value="ok",
    ):
        r = await client.post(
            "/api/ops/chat",
            json={"message": "hello", "session_id": None},
            headers=auth_headers,
        )
    sid = r.json()["session_id"]

    resp = await client.delete(f"/api/ops/chat/{sid}", headers=auth_headers)
    assert resp.status_code == 200


# ── POST /api/ops/recalibrate ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recalibrate_returns_202(client, auth_headers, default_user):
    with patch(
        "app.routes.operations.RecalibrateAgentRunner"
    ) as MockRunner:
        MockRunner.return_value.run = AsyncMock()
        resp = await client.post(
            "/api/ops/recalibrate", json={}, headers=auth_headers
        )

    # Reset per-org job state
    from app.services import recalibrate_job
    recalibrate_job.reset(default_user["org_id"])

    assert resp.status_code == 202
    assert resp.json()["status"] == "running"


@pytest.mark.asyncio
async def test_recalibrate_duplicate_rejected(client, auth_headers, default_user):
    from app.services import recalibrate_job as rj
    org_id = default_user["org_id"]
    rj.get(org_id).status = "running"

    resp = await client.post(
        "/api/ops/recalibrate", json={}, headers=auth_headers
    )
    rj.reset(org_id)

    assert resp.status_code == 409


# ── GET /api/ops/recalibrate/status ──────────────────────────────────────

@pytest.mark.asyncio
async def test_recalibrate_status(client, auth_headers):
    resp = await client.get("/api/ops/recalibrate/status", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "progress" in data


# ── Recalibrate 503 middleware ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_middleware_blocks_during_recalibrate(client, auth_headers, default_user):
    from app.services import recalibrate_job as rj
    org_id = default_user["org_id"]
    rj.get(org_id).status = "running"

    resp = await client.post("/api/ops/lint", headers=auth_headers)
    rj.reset(org_id)

    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_middleware_allows_recalibrate_status_during_lock(client, auth_headers, default_user):
    from app.services import recalibrate_job as rj
    org_id = default_user["org_id"]
    rj.get(org_id).status = "running"

    resp = await client.get("/api/ops/recalibrate/status", headers=auth_headers)
    rj.reset(org_id)

    assert resp.status_code == 200


# ── GET/PUT /api/ops/schema ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_schema(client, auth_headers):
    resp = await client.get("/api/ops/schema", headers=auth_headers)
    assert resp.status_code == 200
    assert "content" in resp.json()


@pytest.mark.asyncio
async def test_update_schema(client, auth_headers):
    new_content = "# Schema\n\nNew rules here."
    resp = await client.put(
        "/api/ops/schema",
        json={"content": new_content},
        headers=auth_headers,
    )
    assert resp.status_code == 200

    read = await client.get("/api/ops/schema", headers=auth_headers)
    assert read.json()["content"] == new_content


# ── POST /api/ops/lint ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lint_returns_health_report(client, auth_headers):
    with patch(
        "app.providers.bedrock.BedrockConverseClient.converse",
        new_callable=AsyncMock,
        return_value='{"issues": [], "suggestions": ["All good."], "health_score": 95}',
    ), patch(
        "app.services.wiki_db.list_wiki_pages_with_content",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "app.services.wiki_db.prepend_wiki_file", new_callable=AsyncMock
    ), patch(
        "app.services.wiki_db.append_audit_log", new_callable=AsyncMock
    ):
        resp = await client.post("/api/ops/lint", headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert "health_score" in data
    assert "issues" in data
    assert "suggestions" in data


# ── GET /api/ops/graph ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_graph_data(client, auth_headers):
    resp = await client.get("/api/ops/graph", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data


@pytest.mark.asyncio
async def test_get_graph_html(client, auth_headers):
    resp = await client.get("/api/ops/graph/html", headers=auth_headers)
    assert resp.status_code == 200
    assert "html" in resp.text.lower() or "No wiki pages" in resp.text


@pytest.mark.asyncio
async def test_rebuild_graph(client, auth_headers):
    with patch(
        "app.services.wiki_db.list_wiki_pages_with_content",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "app.services.wiki_db.set_wiki_file", new_callable=AsyncMock
    ):
        resp = await client.post("/api/ops/graph/rebuild", headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data


# ── GET /api/ops/status with pending_review ───────────────────────────────

@pytest.mark.asyncio
async def test_status_pending_review_includes_plan(client, auth_headers, user_ctx):
    from app.services.jobs import enqueue, save, get
    await enqueue("review_test.txt")
    job = await get("review_test.txt")
    job.status = "pending_review"
    job.plan = [{"path": "concepts/foo.md", "action": "create", "brief": "foo"}]
    job.conflicts = [{"path": "concepts/foo.md", "existing_claim": "X", "new_claim": "Y"}]
    await save(job)

    resp = await client.get("/api/ops/status/review_test.txt", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending_review"
    assert "plan" in data
    assert "conflicts" in data


# ── POST /api/ops/ingest/{filename}/plan-chat ────────────────────────────

@pytest.mark.asyncio
async def test_plan_chat_updates_plan(client, auth_headers, user_ctx):
    from app.services.jobs import enqueue, save, get
    await enqueue("chat_plan.txt")
    job = await get("chat_plan.txt")
    job.status = "pending_review"
    job.plan = []
    job.doc_text = "Document content."
    await save(job)

    with patch(
        "app.providers.bedrock.BedrockConverseClient.converse",
        new_callable=AsyncMock,
        return_value='{"reply": "Added foo.", "updated_plan": [{"path": "concepts/foo.md", "action": "create", "brief": "foo"}], "index_additions": null, "log_entry": null}',
    ), patch(
        "app.services.wiki_db.get_wiki_file", new_callable=AsyncMock, return_value=""
    ):
        resp = await client.post(
            "/api/ops/ingest/chat_plan.txt/plan-chat",
            json={"message": "Add a foo page."},
            headers=auth_headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "Added foo."


@pytest.mark.asyncio
async def test_plan_chat_not_pending_returns_400(client, auth_headers):
    resp = await client.post(
        "/api/ops/ingest/nonexistent.txt/plan-chat",
        json={"message": "any"},
        headers=auth_headers,
    )
    assert resp.status_code == 400


# ── POST /api/ops/ingest/{filename}/approve ──────────────────────────────

@pytest.mark.asyncio
async def test_approve_ingest_starts_execution(client, auth_headers, user_ctx):
    from app.services.jobs import enqueue, save, get
    await enqueue("approve_test.txt")
    job = await get("approve_test.txt")
    job.status = "pending_review"
    job.plan = [{"path": "concepts/foo.md", "action": "create", "brief": "foo"}]
    job.doc_text = "content"
    await save(job)

    with patch(
        "app.routes.operations._execute_ingest", new_callable=AsyncMock
    ):
        resp = await client.post(
            "/api/ops/ingest/approve_test.txt/approve",
            json={"notes": ""},
            headers=auth_headers,
        )

    assert resp.status_code == 200
    assert resp.json()["approved"] == "approve_test.txt"


@pytest.mark.asyncio
async def test_approve_ingest_not_pending_returns_400(client, auth_headers):
    resp = await client.post(
        "/api/ops/ingest/missing.txt/approve",
        json={"notes": ""},
        headers=auth_headers,
    )
    assert resp.status_code == 400


# ── GET /api/ops/export ────────────────────────────────────────────────────

def _fake_zip() -> bytes:
    """Return a minimal but valid ZIP for mocking build_wiki_export."""
    import io, zipfile, json
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("wiki/index.md", "# Index")
        zf.writestr("graph.json", "{}")
        zf.writestr("manifest.json", json.dumps({
            "exported_at": "2026-01-01T00:00:00Z",
            "page_count": 1,
            "has_embeddings": False,
            "embedding_dimensions": 1536,
            "embedding_model": "",
        }))
        zf.writestr("retriever.py", "# retriever")
        zf.writestr("README.md", "# README")
    return buf.getvalue()


@pytest_asyncio.fixture
async def supervisor_headers(default_user):
    # Role is resolved per-request from org_memberships, so the token must point
    # at a real supervisor membership (not the admin's id with a faked role).
    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    uid = await create_user(default_user["org_id"], "supervisor@test.com",
                            "password123", role="supervisor")
    token = create_access_token(
        email="supervisor@test.com", user_id=uid,
        org_id=default_user["org_id"], role="supervisor",
    )
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def member_headers(default_user):
    from app.services.orgs import create_user
    from app.services.auth import create_access_token
    uid = await create_user(default_user["org_id"], "member@test.com",
                            "password123", role="member")
    token = create_access_token(
        email="member@test.com", user_id=uid,
        org_id=default_user["org_id"], role="member",
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_export_admin_returns_zip(client, auth_headers):
    with patch(
        "app.services.wiki_engine.WikiEngine.build_wiki_export",
        new_callable=AsyncMock,
        return_value=(_fake_zip(), 1, False),
    ):
        resp = await client.get("/api/ops/export", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "wiki-export-" in resp.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_export_supervisor_returns_zip(client, supervisor_headers):
    with patch(
        "app.services.wiki_engine.WikiEngine.build_wiki_export",
        new_callable=AsyncMock,
        return_value=(_fake_zip(), 1, False),
    ):
        resp = await client.get("/api/ops/export", headers=supervisor_headers)

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"


@pytest.mark.asyncio
async def test_export_member_returns_403(client, member_headers):
    resp = await client.get("/api/ops/export", headers=member_headers)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_export_unauthenticated_returns_401(client):
    resp = await client.get("/api/ops/export")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_export_include_embeddings_false(client, auth_headers):
    with patch(
        "app.services.wiki_engine.WikiEngine.build_wiki_export",
        new_callable=AsyncMock,
        return_value=(_fake_zip(), 1, False),
    ) as mock_build:
        resp = await client.get(
            "/api/ops/export?include_embeddings=false", headers=auth_headers
        )

    assert resp.status_code == 200
    mock_build.assert_awaited_once_with(False)


@pytest.mark.asyncio
async def test_export_zip_contains_required_files(client, auth_headers):
    import io, zipfile

    with patch(
        "app.services.wiki_engine.WikiEngine.build_wiki_export",
        new_callable=AsyncMock,
        return_value=(_fake_zip(), 1, False),
    ):
        resp = await client.get("/api/ops/export", headers=auth_headers)

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    assert "wiki/index.md" in names
    assert "graph.json" in names
    assert "manifest.json" in names
    assert "retriever.py" in names
    assert "README.md" in names


@pytest.mark.asyncio
async def test_export_audit_log_written(client, auth_headers, user_ctx):
    import sqlalchemy as sa
    from app.db import get_db

    with patch(
        "app.services.wiki_engine.WikiEngine.build_wiki_export",
        new_callable=AsyncMock,
        return_value=(_fake_zip(), 2, False),
    ):
        resp = await client.get("/api/ops/export", headers=auth_headers)

    assert resp.status_code == 200
    async with get_db() as db:
        result = await db.execute(
            sa.text("SELECT operation, raw_text FROM audit_log WHERE operation = 'wiki_export'")
        )
        row = result.fetchone()

    assert row is not None
    assert "wiki_export" in row.raw_text


# ── POST /api/ops/recalibrate — targeted mode ─────────────────────────────

@pytest.mark.asyncio
async def test_targeted_recalibrate_passes_fact_instructions_to_runner(client, auth_headers, default_user):
    import asyncio

    run_called = asyncio.Event()
    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        run_called.set()

    with patch("app.routes.operations.RecalibrateAgentRunner") as MockRunner:
        MockRunner.return_value.run = fake_run
        resp = await client.post(
            "/api/ops/recalibrate",
            json={"fact_instructions": "Q3 revenue should be $12M not $10M"},
            headers=auth_headers,
        )
        # runner.run() is called inside an asyncio.create_task — wait for it
        # to execute before leaving the patch context
        await asyncio.wait_for(run_called.wait(), timeout=5.0)

    from app.services import recalibrate_job
    recalibrate_job.reset(default_user["org_id"])

    assert resp.status_code == 202
    assert captured.get("fact_instructions") == "Q3 revenue should be $12M not $10M"


@pytest.mark.asyncio
async def test_targeted_recalibrate_status_returns_fact_instructions(client, auth_headers, default_user):
    from app.services import recalibrate_job as rj
    org_id = default_user["org_id"]
    rj.get(org_id).fact_instructions = "Wrong CEO name on leadership page"

    resp = await client.get("/api/ops/recalibrate/status", headers=auth_headers)
    rj.reset(org_id)

    assert resp.status_code == 200
    data = resp.json()
    assert data["fact_instructions"] == "Wrong CEO name on leadership page"


@pytest.mark.asyncio
async def test_recalibrate_status_fact_instructions_empty_by_default(client, auth_headers, default_user):
    resp = await client.get("/api/ops/recalibrate/status", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["fact_instructions"] == ""


@pytest.mark.asyncio
async def test_targeted_recalibrate_stores_fact_instructions_on_job(client, auth_headers, default_user):
    with patch("app.routes.operations.RecalibrateAgentRunner") as MockRunner:
        MockRunner.return_value.run = AsyncMock()
        await client.post(
            "/api/ops/recalibrate",
            json={"fact_instructions": "Fix the headcount figure"},
            headers=auth_headers,
        )

    from app.services import recalibrate_job as rj
    org_id = default_user["org_id"]
    job = rj.get(org_id)
    stored = job.fact_instructions
    rj.reset(org_id)

    assert stored == "Fix the headcount figure"

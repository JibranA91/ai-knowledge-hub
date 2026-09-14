# AI Knowledge Hub

## Overview

AI Knowledge Hub is an AI-maintained internal wiki. Unlike a traditional document search tool, it does not retrieve raw document snippets at query time. Instead, it permanently converts uploaded documents into structured, interlinked Markdown wiki pages — a living knowledge base that grows richer with every new document added.

---

## Requirements

- **Docker + Docker Compose** (recommended) — or Python 3.12+ with a PostgreSQL 16 instance for local dev.
- **An AWS account with Bedrock access**, with the configured model IDs enabled in your region (see the `BEDROCK_*` variables in [Configuration](#configuration-reference)). All LLM work — planning, page writing, query, recalibration, and optional embeddings — runs through AWS Bedrock.
  > **Note:** Bedrock is currently the only supported LLM provider. Multi-provider support (Anthropic API, OpenAI, local models) is on the roadmap — see the `BEDROCK_*` model IDs in [Configuration](#configuration-reference) for what's wired today.
- **Optional:** an AWS S3 bucket — only if you set `STORAGE_BACKEND=s3`. The default `local` backend stores raw uploads on a Docker volume and needs no S3.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Browser (SPA)                         │
│  Login → Upload → Review → Browse → Chat / Ask Questions    │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTPS  (Bearer token)
┌───────────────────────▼─────────────────────────────────────┐
│                    FastAPI Backend                          │
│  /api/auth   /api/documents   /api/wiki   /api/ops          │
│  /api/orgs   /api/admin                                     │
└──────┬──────────────┬──────────────────────┬────────────────┘
       │              │                      │
  Auth Service   Ingest Agent           Wiki Engine
  (PostgreSQL)   (LangGraph)   (query / chat / chat-stream / lint)
       │              │                      │
       └──────────────┴────────┬─────────────┘
                               │
                  ┌────────────┴──────────────┐
                  │                           │
           AWS Bedrock (LLM)            PostgreSQL
           ─────────────────            ──────────
           Planner  · Writer            State · FTS
           Query    · Recalibrate
```

**PostgreSQL is the single source of truth** for all wiki content. Wiki pages, the knowledge graph cache, and the schema (`AGENTS.md`) all live in PostgreSQL — no filesystem access needed at query or ingest time.

The `STORAGE_BACKEND` variable controls **only raw uploaded files** (PDFs, DOCXs, etc.):

| `STORAGE_BACKEND` | Where raw uploads live |
|---|---|
| `local` *(default)* | `DATA_DIR` on a Docker volume (no S3 bucket required) |
| `s3` | AWS S3 bucket named `DATA_BUCKET` |

PostgreSQL stores auth tokens, ingest jobs, chat sessions, recalibration state, wiki pages (content + FTS index), graph cache, schema, and audit logs.

### The LLM layer

Every model call in the app goes through one module, [`app/model.py`](app/model.py). **No LLM connection is created anywhere else** — `tests/unit/test_no_direct_llm_clients.py` fails the build if one is.

```
call sites  ──►  app/model.py  ──►  app/providers/<vendor>.py  ──►  vendor SDK
(ask for a role)  (role → model name,  (name → the vendor's model ID;
                   usage tracking)      the only place a client is built)
```

Call sites ask for a **role** — what the model is for — never a model ID:

```python
from app import model

llm    = model.get_chat(model.Role.INGEST_PLAN, max_tokens=4096)  # LangChain runnable, supports bind_tools
client = model.get_converse(model.Role.QUERY)                     # converse + streaming
vec    = await model.embed("some text")                           # None when embeddings are disabled
```

Config names a **model**, not a vendor model ID. The provider maps the name to its own ID, so the same config survives a change of backend:

| Role | Used by | Configured with | Default |
|---|---|---|---|
| `INGEST_PLAN` | ingest planner (tool-calling loop) | `MODEL_INGEST_PLAN` | `haiku45` |
| `INGEST_WRITE` | ingest page renderer | `MODEL_INGEST_WRITE` | `haiku45` |
| `QUERY` | chat / Q&A | `MODEL_QUERY` | `haiku45` |
| `RECALIBRATE` | wiki-wide analyze + rewrite | `MODEL_RECALIBRATE` | `sonnet45` |
| `DRAFT_AGENT` | conversational AI Writer | `MODEL_DRAFT_AGENT` | `haiku45` |
| `EDIT` | inline page/section editor | `MODEL_EDIT` | `sonnet45` |
| `EMBEDDING` | semantic search vectors | `MODEL_EMBEDDING` | *(unset — BM25 only)* |

So switching the chat model is one word:

```diff
- MODEL_QUERY=haiku45
+ MODEL_QUERY=sonnet5
```

**Model names for `LLM_PROVIDER=bedrock`** — the catalogue lives in `MODEL_CATALOG` in [`app/providers/bedrock.py`](app/providers/bedrock.py):

| Family | Names |
|---|---|
| Claude | `fable5` `opus5` `opus48` `opus47` `opus46` `opus45` `opus41` `sonnet5` `sonnet46` `sonnet45` `sonnet4` `haiku45` `haiku3` |
| Llama | `llama4maverick` `llama4scout` |
| Embeddings | `titanembedv2` `titanembedv1` `cohereembedv4` `cohereembeden` `cohereembedml` |

Matching ignores case and punctuation — `haiku45`, `haiku-4.5` and `Haiku 4.5` are the same model. Names resolve to Bedrock cross-region inference profile IDs using `BEDROCK_INFERENCE_GEO` (`us` by default) as the prefix:

```
MODEL_QUERY=haiku45  +  BEDROCK_INFERENCE_GEO=us
  → us.anthropic.claude-haiku-4-5-20251001-v1:0
```

Set `BEDROCK_INFERENCE_GEO=global` for the global profile, or leave it blank to call the foundation model directly with no profile. Embedding models are plain foundation models and are never prefixed.

**Escape hatch:** a value containing `.`, `:` or `/` is treated as a raw Bedrock model ID or inference-profile ARN and passed through untouched, so anything the catalogue doesn't cover still works. An unknown *bare* name is rejected at startup with the list of valid names — a typo can't reach Bedrock as an opaque 400. Adding a model to the catalogue is one line; to see what your account exposes:

```bash
aws bedrock list-inference-profiles \
  --query 'inferenceProfileSummaries[].inferenceProfileId' --output table
```

Token usage is recorded centrally in [`app/providers/usage.py`](app/providers/usage.py), so every call is logged to `usage_log` regardless of backend.

> **Upgrading:** the older `BEDROCK_*_MODEL_ID` variables still work — each is folded onto its `MODEL_*` replacement when that one isn't explicitly set — so an existing `.env` needs no changes. They're deprecated and will be removed; see the mapping at the bottom of [`.env.example`](.env.example).

**Adding a provider** (OpenAI, Anthropic direct, Azure, Ollama, …):

1. Implement the `Provider` protocol from [`app/providers/base.py`](app/providers/base.py) in `app/providers/<name>.py`, including its own `resolve_model()` name catalogue.
2. Register it in `app/providers/__init__.py`.
3. Set `LLM_PROVIDER=<name>`.

Model names are provider-neutral, so a role configured as `sonnet45` keeps working — the new provider maps it to whatever that vendor calls the model. A name a provider can't serve is rejected at startup rather than silently substituted.

No call site changes. `app/services/bedrock.py` remains as a deprecated re-export shim and will be removed.

---

## Running Locally

```bash
# 1. Create your env file (docker-compose reads .env — this step is required)
cp .env.example .env
# then edit .env: set AUTH_PASSWORD, JWT_SECRET, and your AWS/Bedrock settings

# 2. Docker (recommended — starts PostgreSQL and the wiki app)
docker compose up --build

# Tail logs
docker compose logs -f wiki
```

The app runs on `http://localhost:8000`. Raw uploads are stored in a Docker named volume by default — no S3 bucket needed. (An AWS account with Bedrock access is still required for the LLM — see [Requirements](#requirements).)

### Local dev without Docker

Requires Python 3.12+ and a PostgreSQL 16 instance.

```bash
pip install -r requirements.txt

export DATABASE_URL="postgresql+asyncpg://wiki:wiki@localhost:5432/wiki"
export STORAGE_BACKEND=local
export DATA_DIR=./data

uvicorn app.main:app --reload
```

To use S3 instead:

```bash
export STORAGE_BACKEND=s3
export DATA_BUCKET=my-wiki-bucket
export AWS_REGION=us-east-1
# AWS credentials via env vars or IAM role
```

Alembic migrations run automatically on startup — no manual migration step needed.

---

## Multi-Tenancy

The system is fully multi-tenant — multiple organisations share one deployment with strict data isolation at every layer.

### How it works

| Layer | Isolation mechanism |
|---|---|
| **Database** | Every table (`wiki_pages`, `ingest_jobs`, `chat_sessions`, `audit_log`, `recalibrate_jobs`, `wiki_files`) has an `org_id UUID NOT NULL` column. All queries are automatically scoped to the current request's org via a Python `ContextVar`. |
| **File storage** | Raw uploads are stored under `{org_id}/raw/{filename}`. Org A's uploads are physically separate from Org B's. |
| **JWT tokens** | Access tokens carry `org_id`, `user_id`, and `role` claims. The auth middleware sets these in the request context before any route handler runs. |
| **Background tasks** | `asyncio.create_task()` copies the calling coroutine's context, so org_id propagates automatically into ingest and recalibrate workers. |
| **Recalibration lock** | The 503 middleware checks the running status for the *current org only* — orgs can recalibrate simultaneously without blocking each other. |

### Default organisation

On first startup the app creates a **Default Organization** and seeds an admin user from `AUTH_USERNAME` / `AUTH_PASSWORD`. All existing data (if upgrading from a single-tenant deployment) is migrated to this org via the Alembic migration `004`.

### User management

A user is a single identity that can belong to **one or more organizations** — each membership (stored in `org_memberships`) carries its own role, permission template, workspace, suspension flag, and token/chat limits. Which org a request acts on is chosen by the `X-Org-Context` header and validated against the caller's memberships, so the same person can be a `supervisor` in one org and a `member` in another.

Roles: `admin` (global — not tied to an org, manages the whole deployment), `supervisor` (manages users within an org they belong to), and `member`. Admins manage memberships from the admin dashboard via the `/api/admin/users/{id}/memberships` endpoints.

```bash
# Create a user in your org (must be admin)
curl -X POST http://localhost:8000/api/orgs/me/users \
  -H "Authorization: Bearer <token>" \
  -d '{"email": "alice@example.com", "password": "secret", "role": "member"}'
```

### Org management endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/orgs/me` | Current org info |
| `GET` | `/api/orgs/me/users` | List users in org (admin only) |
| `POST` | `/api/orgs/me/users` | Create user in org (admin only) |
| `DELETE` | `/api/orgs/me/users/{id}` | Remove user (admin only) |
| `GET` | `/api/orgs/me/usage` | Bedrock usage summary for current org |
| `GET` | `/api/orgs` | List all orgs (admin only) |
| `POST` | `/api/orgs` | Create a new org + admin user (admin only) |

> Broader organization administration — clone/delete orgs, and manage a user's memberships across orgs (add/update/remove, transfer) — lives under `/api/admin/organizations/*` and `/api/admin/users/*` (see [Admin Dashboard](#admin-dashboard)).

### Bedrock usage tracking

Every LLM call is logged to the `usage_log` table with `org_id`, `model_id`, `tokens_in`, `tokens_out`, and `operation`. Aggregate usage is available via `GET /api/orgs/me/usage?days=30`.

---

## Authentication

Every API call requires a valid JWT access token. Login returns two tokens — a short-lived access token and a long-lived refresh token.

```
Browser                               Server
  │                                     │
  │── POST /api/auth/login ────────────▶│  verify email + password (users table)
  │◀─ { access_token, refresh_token,    │  access_token: JWT (15 min, stateless)
  │      org_id, role } ────────────────│    claims: sub=user_id, org_id, role, email
  │   (stores tokens in localStorage)   │  refresh_token: opaque hex (7 days, DB-hashed)
  │── GET /api/wiki ───────────────────▶│
  │   Authorization: Bearer <JWT>        validate JWT → extract org_id → scope all queries
  │◀─ 200 OK ──────────────────────────│
  │
  │  (access token expires after 15 min)
  │
  │── POST /api/auth/refresh ──────────▶│  { refresh_token }
  │◀─ { access_token } ────────────────│  new JWT issued, no re-login needed
  │   (new token stored in localStorage)
```

- **Access tokens** are short-lived JWTs signed with `JWT_SECRET` carrying `org_id`, `user_id`, and `role`. Validated by signature — no DB lookup per request.
- **Refresh tokens** are opaque 32-byte hex values stored hashed in PostgreSQL with a 7-day TTL. Used only to issue new access tokens via `POST /api/auth/refresh`.
- Signing out (`POST /api/auth/logout`) revokes the refresh token immediately. The access token expires naturally within 15 minutes.
- `GET /api/auth/config` returns public app settings — no authentication required.

### Rate Limiting

Write-heavy endpoints enforce per-user per-minute limits tracked in a `rate_limit_counters` Postgres table (no Redis required):

| Endpoint | Default limit |
|---|---|
| `POST /api/documents/upload` | 10 requests/min |
| `POST /api/ops/query` | 30 requests/min |
| `POST /api/ops/chat` and `/chat/stream` | 60 requests/min |
| `POST /api/ops/recalibrate` | 2 requests/min |

Limits are configurable via env vars (`RATE_LIMIT_UPLOAD_PER_MINUTE`, etc.). Exceeding a limit returns **429 Too Many Requests**.

---

## Document Ingestion — Two-Phase Pipeline

Uploading a document does not write pages immediately. The process has two phases: **Plan** then **Execute**.

### Phase 1 — Planning

```
Upload PDF / DOCX / TXT / MD  →  stored in file storage (raw/)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  IngestAgent  (LangGraph — Plan Graph)                  │
│                                                         │
│  START → planner ⇄ read_existing_page tool              │
│              │  (reads pages from file storage)         │
│              ▼  (up to 3 reasoning rounds)              │
│         parse_plan → END                                │
│                                                         │
│  The planner reads:                                     │
│    • document text  (in 28 000-char chunks)             │
│    • compact wiki index  (derived live from DB)         │
│    • semantically similar pages (cosine pre-fetch)      │
│    • existing page content  (via tool, from DB)         │
│                                                         │
│  The planner outputs:                                   │
│    • a JSON page plan  [{path, action, brief}, ...]     │
│    • detected conflicts (contradictions vs. wiki)       │
└─────────────────────────────────────────────────────────┘
        │
        ▼
  Job status → "pending_review"  (persisted to PostgreSQL)
```

### Phase 2 — Execution (after approval)

```
User clicks Approve
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  IngestAgent  (LangGraph — Write Graph)                 │
│                                                         │
│  START → write_pages (all pages in parallel)            │
│               │  per-path advisory locks via Postgres   │
│           finalize → END                                │
│                                                         │
│  Finalize:                                              │
│    • Upserts wiki page content to PostgreSQL            │
│    • Appends entry to audit_log table                   │
│    • Rebuilds the knowledge graph (cached in DB)        │
└─────────────────────────────────────────────────────────┘
```

### Durable write queue

Approving a plan does not write inline — it enqueues the job. The queue **is** the `ingest_jobs` table: an approved job sits at `status='queued_write'` with a `queued_at` timestamp, and a per-org *drain* claims jobs FIFO under a Postgres session-level advisory lock, so only one replica writes for a given org at a time. This makes the write phase:

- **Durable** — jobs and their FIFO order survive a restart; nothing lives in an in-memory queue.
- **Replica-safe** — `pg_try_advisory_lock` guarantees a single writer per org across the whole cluster; a background poller (every 15 s) lets any replica pick up work submitted on a since-dead replica.
- **Visible** — queue position is a DB query, so the admin **Jobs** tab shows the true cross-replica position, not one replica's local view.
- **Self-healing** — a job orphaned mid-write by a dead replica is reclaimed and marked errored when the org's advisory lock is next acquired.

**Conflict handling:** When the new document contradicts existing wiki content the planner surfaces each conflict to the user for resolution via plan chat.

**Page budget:** The planner targets 4–6 pages per ingest, preferring broader pages over many narrow stubs.

### Document Deletion

1. Removes the raw file from file storage (`raw/`)
2. Scans `wiki_pages` in PostgreSQL for any `sources/` page stamped with `uploaded_file: <filename>` in its content and deletes those rows
3. Returns the list of removed wiki pages to the caller

---

## Search — BM25 and Hybrid (Phase 5)

Wiki pages are mirrored into a PostgreSQL `wiki_pages` table with a `tsvector GENERATED ALWAYS AS` column (GIN-indexed). Searches use `plainto_tsquery` for ranked BM25 full-text search, falling back to `LIKE` on path and title.

### Semantic embeddings

**Hybrid search is on by default.** `BEDROCK_EMBEDDING_MODEL_ID` defaults to `amazon.titan-embed-text-v1` (1536-dim), so retrieval combines **BM25 (weight 0.4) + cosine similarity (weight 0.6)** out of the box. `GET /api/auth/config` reports `embedding_enabled: true`.

While enabled (the default):
- Every page write generates and stores a `vector(1536)` embedding in the `wiki_pages` table (via an `IVFFlat` index for approximate nearest-neighbour search).
- `_find_relevant_pages()` in the query/chat pipeline uses the hybrid SQL query instead of an LLM index scan — eliminating one Bedrock call per user question.
- A query with no keyword overlap with the answer (e.g. "distributed coordination" matching a page titled "pg_advisory_xact_lock") is still found through vector similarity.

**To disable vector search** — BM25-only, no Bedrock embedding calls, no pgvector queries — set the variable to empty:

```env
BEDROCK_EMBEDDING_MODEL_ID=
```

With embeddings off, the system uses BM25 and falls back to the LLM-based index scan when BM25 returns no results. To switch models instead, set both the model id and its dimension count:

```env
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
EMBEDDING_DIMENSIONS=1536
```

Supported models:
| Model ID | Dimensions |
|---|---|
| `amazon.titan-embed-text-v1` | 1536 (default) |
| `amazon.titan-embed-text-v2:0` | 1536 |
| `cohere.embed-english-v3` | 1024 — set `EMBEDDING_DIMENSIONS=1024` |
| `cohere.embed-multilingual-v3` | 1024 — set `EMBEDDING_DIMENSIONS=1024` |

### Paginated wiki listing

`GET /api/wiki` now supports pagination:

| Parameter | Default | Description |
|---|---|---|
| `page` | `0` | Page number (1-indexed). `0` returns the legacy nested tree for the sidebar. |
| `limit` | `50` | Items per page (max 200). |
| `tag` | *(none)* | Filter by a single tag. |

---

## Knowledge Graph

After every ingest the graph is rebuilt by scanning all wiki pages in PostgreSQL and extracting three types of links:

| Link type | Example in Markdown |
|---|---|
| Wiki-style double-bracket | `[[evaluator]]` |
| Standard Markdown link | `[Evaluator](../concepts/evaluator.md)` |
| YAML frontmatter field | `related: [../concepts/evaluator.md]` |

The graph is stored in two places:
1. **`wiki_files` (key `wiki/.graph.json`)** — JSON cache of the full adjacency list, used for in-process traversal (graph neighbours, `neighbors(hops=N)`).
2. **`wiki_links` table** — structured `(org_id, from_path, to_path)` rows for SQL graph queries (e.g. recursive CTEs, link counts).

Page links can be fetched directly: `GET /api/wiki/{path}/links` returns `{outgoing: [...], incoming: [...]}`.

### Cluster visualisation (Phase 5)

`GET /api/ops/graph?clusters=true` returns **Louvain community clusters** — a lightweight summary of the graph structure:

```json
{
  "clusters": [
    {"id": 0, "size": 12, "hub_pages": ["concepts/evaluator.md", ...]},
    {"id": 1, "size": 8,  "hub_pages": ["concepts/routing.md", ...]}
  ],
  "nodes": [
    {"id": "concepts/evaluator.md", "title": "Evaluator", "cluster": 0, "degree": 7},
    ...
  ]
}
```

The default `GET /api/ops/graph` (no `clusters` param) continues to return the full `{nodes, edges}` payload for the pyvis force-directed visualisation.

---

## Wiki Export

Supervisors and admins can download the entire wiki as a portable, self-contained ZIP — no LLM or network connection required to use it.

### Triggering an export

**UI:** Click the **Export** button in the header (visible to supervisors and admins only).

**API:**

```bash
curl -H "Authorization: Bearer <token>" \
     "http://localhost:8000/api/ops/export?include_embeddings=true" \
     -o wiki-export.zip
```

| Parameter | Default | Description |
|---|---|---|
| `include_embeddings` | `true` | Include pre-computed page embeddings in the ZIP. Only has effect when `BEDROCK_EMBEDDING_MODEL_ID` is set. |

### ZIP contents

```
wiki-export-{date}.zip
├── wiki/              ← all wiki markdown pages
│   ├── index.md
│   ├── concepts/foo.md
│   └── sources/doc1.md
├── graph.json         ← knowledge graph adjacency list
├── embeddings.json    ← [{path, embedding:[float...]}, ...] (omitted if unavailable)
├── manifest.json      ← {exported_at, page_count, has_embeddings, embedding_model, ...}
├── retriever.py       ← self-contained retrieval script (no LLM)
└── README.md          ← usage guide for the export
```

### Using retriever.py

The bundled `retriever.py` finds relevant pages without an LLM. It uses pure-Python **BM25** by default and upgrades to **hybrid BM25 + cosine similarity** when a pre-computed query vector is supplied.

```bash
# BM25 keyword retrieval — no dependencies
python retriever.py "What is our data retention policy?"

# Hybrid retrieval — requires numpy + a pre-computed embedding
python retriever.py "What is our data retention policy?" --query-vector vec.json
```

| Flag | Default | Description |
|---|---|---|
| `--top-k N` | `5` | Number of pages to return |
| `--graph-hops N` | `1` | Expand results with N-hop graph neighbours |
| `--bm25-weight F` | `0.4` | Weight for BM25 score |
| `--cosine-weight F` | `0.6` | Weight for cosine score (requires `--query-vector`) |
| `--export-dir PATH` | `.` | Path to the unzipped export directory |

**Python API:**

```python
from retriever import retrieve

results = retrieve("What is our vacation policy?", top_k=5)
for r in results:
    print(r["path"], r["score"])
    # feed r["content"] to your own LLM
```

A full README is bundled inside every ZIP.

### Access control

Export is restricted to `supervisor` and `admin` roles. Members receive **403**. Every download is recorded in the audit log.

---

## Wiki Import

The counterpart to export: `POST /api/ops/import` restores an export bundle (the same ZIP produced above) into an organization. Supervisor or admin of the active org only.

- **Empty-org only.** Import refuses with **409** if the target org already has any wiki pages — create or select an empty org to import into. This keeps import a clean restore/seed, never a silent merge.
- The bundle's `AGENTS.md` (if present) overwrites the org's schema; its pages are inserted and the knowledge graph is rebuilt.
- **Embeddings are reused** when the bundle's `embedding_model` and `embedding_dimensions` match the current config — otherwise pages are re-embedded on write. The response reports `{pages_imported, schema_imported, embeddings_reused}`.
- The UI surfaces this as an **Import Wiki** button on the welcome screen, shown only while the current org's wiki is empty.
- Every import is recorded in the audit log (`wiki_import`).

---

## Chat & Querying

### One-shot Query (`POST /api/ops/query`)

```
User asks a question
  Step 1 — LLM reads index.md, identifies relevant page paths
  Step 2 — LLM reads those pages + graph neighbours, synthesises answer
```

### Multi-turn Chat

| `CHAT_STREAM` | Endpoint | Behaviour |
|---|---|---|
| `false` | `POST /api/ops/chat` | Full response returned at once |
| `true` *(default)* | `POST /api/ops/chat/stream` | SSE streaming |

Chat sessions are persisted in PostgreSQL. After 10 messages the history is auto-summarised (keeping the 4 most recent turns). `DELETE /api/ops/chat/{session_id}` clears a session.

---

## AI Writer Mode

A conversational page-authoring agent for admins, supervisors, and any member granted the **Writer** permission (`can_use_writer`). Instead of uploading a finished document, the user chats with an LLM that asks clarifying questions, drafts a markdown page in a live side preview, and revises it section-by-section. When the user is satisfied they click *Save & Ingest* and the draft enters the normal plan-review pipeline.

```
User opens AI Writer (nav bar)
  → Picker modal lists existing drafts (per-user, within TTL)
  → New Draft  OR  Resume <existing>

Writer view (main content area):
  ┌── Live draft preview ──┐    ┌── Chat panel ──┐
  │ # Title                │ ←→ │ Agent: "What    │
  │ ...                    │    │  is the goal?"  │
  └────────────────────────┘    │ User: "..."     │
                                └─────────────────┘

Agent decides autonomously when to write:
  [DRAFT_START] ... [DRAFT_END]              ← full rewrite
  [SECTION_START:Heading] ... [SECTION_END]  ← section-only patch

Save & Ingest
  → In-app confirmation modal previews filename + behavior
  → File written to raw/<filename>.md
  → Standard ingest planner runs
  → User reviews plan, approves, pages appear in wiki
  → Draft is deleted automatically once ingest finishes successfully
```

### Agent behavior

The agent runs a fixed three-phase prompt — implemented in `_writer_system_prompt()` in [`app/services/wiki_engine.py`](app/services/wiki_engine.py):

1. **GATHER** — ask focused questions one at a time (purpose, audience, scope, key facts, contradictions with what already exists). No markdown markers emitted.
2. **WRITE** — once enough context is gathered, emit a short chat reply plus the draft wrapped in `[DRAFT_START]…[DRAFT_END]` (full rewrite) or `[SECTION_START:Heading]…[SECTION_END]` (single-section patch).
3. **REVIEW** — check the draft against the wiki, ask follow-ups, tell the user when it's ready to save.

### Wiki grounding (every turn)

The writer agent is **not** an isolated drafting tool — it is grounded in the existing wiki on every user message. Inside `writer_chat_stream()`:

1. The user's message is fed to `_find_relevant_pages()` — the same hybrid BM25 + vector retriever used by Q&A (with 1-hop graph-neighbour expansion). Up to ~8 pages plus ~3 neighbours per turn.
2. The **full markdown** of each matched page is read from `wiki_pages` and injected into the system prompt under "Relevant existing wiki pages (for grounding and to avoid duplication)", alongside the wiki schema (`AGENTS.md`) and the current draft.
3. The selected page paths are surfaced to the client in the `meta` SSE event as `sources` so the UI can show what was consulted.

This lets the agent flag contradictions with existing content, reuse terminology, and avoid creating duplicate pages. Note: retrieval is keyed off the latest user message only (not the running draft), and the agent has no read-existing-page tool — it gets one retrieval shot per turn.

### Key properties

- **Stateful drafts** live in `chat_sessions` rows with `mode='writer'`; the picker scopes to the current user and current org, ordered by last activity.
- **TTL pruning** — inactive writer drafts are deleted after `WRITER_DRAFT_TTL_DAYS` (default **30 days**) by the daily background prune loop in `app/main.py`.
- **Auto-cleanup on success** — once the spawned ingest job reaches `status='done'`, the frontend deletes the writer draft via `DELETE /api/ops/writer/{id}`. Drafts persist on error/cancelled so the user can retry.
- **Section-patch ambiguity is rejected** — if the patch heading appears more than once in the draft, the server emits an `error` SSE event and the agent must retry with a more specific heading (or fall back to a full rewrite).
- **Filename collisions** return `409` — pick a different name, no silent overwrites. Filenames are validated server-side: must end in `.md`, no path separators, max 200 characters.
- **Recalibrate-lock aware** — `/api/ops/writer/chat/stream` is explicit passthrough during recalibration/revert, so chatting and refining keep working; the final `Save & Ingest`, draft-filename update, and draft delete are blocked with `503` until the lock clears.
- **Audit-logged** — `writer_draft_written`, `writer_section_patched`, `writer_ingest`, `writer_draft_deleted` events are written via `append_audit_log` so revert and admin views can see who did what.

**Endpoints** (require the Writer permission `can_use_writer`; admins and supervisors always have it. *Save & Ingest* additionally needs `can_upload_writer_draft`)**:**

| Method | Path | Description |
|---|---|---|
| `GET`    | `/api/ops/writer/sessions` | List current user's active drafts |
| `POST`   | `/api/ops/writer/chat/stream` | SSE chat; emits `chunk`, `draft_chunk`, `section_chunk`, `error`, `done` events |
| `GET`    | `/api/ops/writer/{id}/draft` | Fetch full draft content + filename |
| `PUT`    | `/api/ops/writer/{id}/draft/filename` | Set/update target filename (must end in `.md`) |
| `POST`   | `/api/ops/writer/{id}/ingest` | Save draft to `raw/` and enqueue the standard ingest pipeline |
| `DELETE` | `/api/ops/writer/{id}` | Delete a draft |

---

## Inline AI Editing

Every wiki page has an **AI Edit** action (anyone with `can_edit_wiki`). Instead of hand-editing markdown, the user picks a scope and an action; the agent streams a proposed rewrite into a side-by-side diff, and **nothing is written until the user clicks Apply**.

```
Open a page → AI Edit
  ├─ Scope:  whole page  │  one section (picked from the page's headings)
  └─ Action: improve · expand · summarize · reconcile · custom (free-text instruction)

  POST /api/ops/wiki/edit/stream   (SSE, read-only)
    meta  {sources}        ← related pages used for grounding
    chunk {text}           ← streamed proposed markdown
    done  {full_content}   ← full page after splicing a section patch back in

  Apply   → PUT /api/wiki/{path}   (the normal tracked, revertible write)
  Discard → aborts the in-flight stream (stops server-side generation)
```

**Read-only endpoint.** `/api/ops/wiki/edit/stream` never writes the page — it only proposes. The client diffs the proposal against the live page and, on accept, applies it through the standard `PUT /api/wiki/{path}`, so every AI edit is change-tracked and revertible exactly like a manual edit.

**Grounded + cross-linking.** Each edit is grounded in related pages (the same hybrid retriever Q&A uses, plus the `reconcile_with` target when `action=reconcile`). The agent is also handed a catalog of linkable pages as `[[page-name]]` tokens, so facts it adds get cross-linked to existing pages; the knowledge graph is refreshed when the edit is applied (`_refresh_graph_after_write`).

| Action | What it does |
|---|---|
| `improve` | Tighten wording and structure, keep the meaning |
| `expand` | Add depth/detail the page is missing |
| `summarize` | Make it more concise |
| `reconcile` | Resolve contradictions against another page (`reconcile_with`) |
| `custom` | Follow a free-text `instruction` |

**Model:** `BEDROCK_EDIT_MODEL_ID` (default Claude Sonnet 4.5). Rate-limited with the chat limiter and counted against the token quota.

---

## Recalibration

```
START
  ├─ load_content    — read all wiki pages from PostgreSQL;
  │                    load compact index (path—title, no blobs)
  │
  ├─ triage          — programmatic scoring (no LLM, no cost)
  │                    flags pages by: stale · orphan · stub
  │                    broken_links · duplicate_candidate
  │                    produces a prioritised shortlist
  │
  ├─ analyze         — strong LLM sees ONLY the shortlist
  │                    + 1-hop graph neighbours for context.
  │                    Shortlist is split into batches (≤20 pages),
  │                    one LLM call each, run concurrently; a final
  │                    cross-batch reconciliation pass catches
  │                    contradictions/duplicates spanning batches.
  │                    Produces a merged, targeted CRUD plan.
  │
  ├─ apply_deletions — execute deletes and renames in PostgreSQL
  │
  ├─ improve_pages   — rewrite / create pages in parallel
  │                    semaphore-capped (max 20 concurrent Bedrock calls)
  │
  ├─ rebuild_index   — no-op: index is derived live from wiki_pages table
  │
  └─ finalize        — append to audit_log, mark job complete in PostgreSQL
                       rebuild knowledge graph
```

**Triage signals**

| Signal | Condition | Flag |
|---|---|---|
| Stale | `date_ingested` frontmatter > 90 days ago | `stale` |
| Orphan | 0 inlinks and 0 outlinks in knowledge graph | `orphan` |
| Stub | fewer than 150 words | `stub` |
| Broken links | `[[wikilinks]]` or `[text](path.md)` pointing to non-existent pages | `broken_links` |
| Duplicate | another page has the same normalised title | `duplicate_candidate` |

Only flagged pages (plus their immediate graph neighbours for context) are sent to the LLM. On a 10,000-page wiki where 150 pages are flagged, the LLM context is the same size as running recalibration on a 150-page wiki today.

**Batched analysis (bounded output at any scale).** The analyze plan grows with the number of flagged pages, so a single LLM call over a large shortlist eventually exceeds the model's output-token limit and truncates mid-JSON. Instead, the shortlist is split into batches of ≤20 pages — one LLM call per batch, run concurrently under the same semaphore as `improve_pages`. Output scales by adding calls rather than growing one response, and a truncated or failed batch is isolated (it contributes nothing instead of sinking the whole run). The per-batch plans are then merged: buckets concatenated, improve/create entries de-duplicated by path, and the delete/rename-vs-improve overlap guard applied once.

**Cross-batch consistency.** Because each batch sees only a slice of the wiki, contradictions and duplicates that span two batches could be missed. Two measures keep related pages together: batches are assigned by **connected link-graph component** (linked pages are ordered into the same batch), and **title-duplicate groups are kept atomic** (never split across batches). Whatever still slips through is caught by a final **reconciliation pass** — one bounded LLM call over compact cards (path + short excerpt) of the whole shortlist that emits only the offending *pairs*, which fold back into the improve plan. The reconciliation pass runs only when there is more than one batch.

While recalibration runs, FastAPI middleware queries PostgreSQL and blocks **write operations** with **503** — ingest, wiki edits, lint, and graph rebuild. Read operations (`GET` requests), query, and chat remain available during recalibration. This lock is replica-safe — all instances check the same DB row.

### Targeted Recalibration

When a user knows a specific fact is wrong (e.g. "the Q3 revenue figure is incorrect"), they can describe it in the recalibrate modal instead of running a full structural audit.

**How it works:**

1. The user types a plain-language description of the error in the **"Describe the fact(s) to fix"** textarea in the recalibrate confirmation dialog.
2. The API receives `fact_instructions` in the `POST /api/ops/recalibrate` body.
3. The `triage` node skips all five health signals (staleness, orphans, stubs, broken links, duplicates) and instead embeds the fact description and runs a **semantic vector search** (`top_k=15`) to find the most relevant pages.
4. The `analyze` node receives a **user-directive block** at the top of its system prompt: *"USER-REPORTED FACTUAL ERROR (PRIMARY TASK): … Do not delete or rename pages unless directly required by the fix."*
5. The `finalize` node logs the run to `audit_log` with a `(targeted)` heading and the exact fact instructions quoted verbatim.

**UI behaviour:**
- The confirm button label live-toggles between *"Start Recalibration"* (textarea empty) and *"Start Targeted Recalibration"* (textarea has content).
- The recalibration overlay title changes to *"Targeted Recalibration"* and shows a badge with the reported fact description throughout the run.
- `GET /api/ops/recalibrate/status` includes a `fact_instructions` field — an empty string for full runs, the user's text for targeted runs.

**Leave the textarea blank** to run the normal full-wiki structural recalibration; targeted mode is entirely opt-in and the existing behavior is unchanged.

---

## Wiki Health Score

`POST /api/ops/lint` returns a **deterministic** structural health score (0–100) plus a qualitative LLM audit. The two parts are independent:

- **The score is computed, not guessed.** It comes from `app/services/wiki_health.py`, which detects the same five structural signals recalibrate's triage uses — stale, orphan, stub, broken-links, duplicate-title — and applies a fixed formula. No LLM, no sampling, so the same wiki always yields the same number.
- **The LLM is used only for the qualitative audit** (contradictions, missing pages, weak cross-referencing) and runs at **temperature 0**. It no longer decides the score.

### Scoring formula

The score starts at 100 and subtracts a severity-weighted penalty proportional to the **fraction of pages** exhibiting each signal:

```
score = 100 − Σ ( weight(signal) × pages_with_signal / total_pages )
```

| Signal | Weight | Detection |
|---|---|---|
| Broken links | 30 | `[[wikilinks]]` / `[text](path.md)` pointing at non-existent pages |
| Duplicate title | 25 | another page shares the same normalised title |
| Orphan | 20 | 0 inlinks **and** 0 outlinks in the link graph |
| Stub | 15 | fewer than 150 words |
| Stale | 10 | `date_ingested` older than 90 days |

Weights sum to 100, so the score is bounded to `[0, 100]`. The formula is **monotonic** — fixing any flagged page can only raise (or hold) the score. An empty wiki scores 100. The health-check response also includes a per-signal `breakdown`, surfaced in the UI so you can see exactly what is dragging the score down.

### Why recalibration improves it

`wiki_health` is the **single source of truth** for the structural signals: recalibrate's `triage` node and the health score both compute from it, so they can never disagree about what's wrong. Because the score is derived from precisely the issues recalibration repairs (broken links, stubs, orphans, duplicates, stale pages), running a recalibration **provably raises the score**. Two practical notes:

- Recalibrate processes a **capped shortlist** (200 pages/run), so a large backlog improves incrementally over several runs.
- Orphan and broken-link counts depend on the link graph, which is rebuilt **after** a recalibration completes — so re-run the health check once the run has finished to see the updated score.

---

## Change Tracking & Revert

Every state-changing action — manual edits, document uploads + ingests, recalibration, schema edits, saved Q&A pages, document deletes — is captured as a **revertable action**. Admins and supervisors can roll the wiki back to the state it was in right after any prior action, with git-`reset` semantics.

### How it's tracked

Five choke-point functions own every wiki mutation: `upsert_wiki_page`, `delete_wiki_page`, `set_wiki_file`, `s3.write_bytes`, `s3.delete`. Each one is wrapped so that whenever an "action" is open in the request context, the write also emits a row in `wiki_revisions` capturing the **before** and **after** content. The action itself lives in `wiki_actions`. Future write paths get tracking automatically — there's nothing to remember to add.

```
Request                                                State
─────────                                              ─────
upload_document  ──▶  begin_action("document_upload")    │
                          │                              │
                          ├─ tracked_write_bytes ────────┼──▶ wiki_revisions
                          │                              │     (op=create | update | delete,
                          ▼                              │      content_before, content_after)
                       async with closes
                          │                              ▼
                  status='done' on wiki_actions
```

### Action types

| Action | Captured at | Reverts |
|---|---|---|
| `manual_edit` | `PUT /api/wiki/{path}` | Page content back to what it was |
| `manual_delete` | `DELETE /api/wiki/{path}` | Recreates the page |
| `document_upload` | `POST /api/documents/upload` | Removes the S3 raw file |
| `document_delete` | `DELETE /api/documents/{filename}` | Restores the source file + all derived `sources/` pages |
| `upload_ingest` | `POST /api/ops/ingest/{f}/approve` | Undoes every page the writer created or updated |
| `saved_query` | `query`/`chat` with `save_to_wiki=true` | Removes the saved Q&A page |
| `recalibrate` / `targeted_recalibrate` | `POST /api/ops/recalibrate` | Undoes every page rewrite, rename, and deletion in one atomic operation |
| `schema_update` | `PUT /api/ops/schema` | Restores the previous `AGENTS.md` |
| `revert` | `POST /api/admin/history/{id}/revert` | Self-referential — reverts can themselves be reverted |

### Storage model

| Kind | Where the snapshot lives |
|---|---|
| Wiki page | `content_before` / `content_after` columns on `wiki_revisions` (inline TEXT — usually < 50 KB) |
| Schema file | Same as above |
| Raw S3 upload | Object is copied to `archive/{action_id}/{key}` before being overwritten or deleted; the archive key is stored in the revision row |
| `wiki_links` / `.graph.json` | **Not tracked** — derived state, rebuilt automatically after each revert |
| `audit_log` | **Not tracked** — the reflog; the revert itself appends its own entry |

### Revert semantics

Git-`reset --hard`: reverting action *N* discards *N* and every action that happened after it. The wiki ends up in the state it was in right before *N* ran.

1. The revert opens its own `wiki_actions` row (`action_type='revert'`, `revert_of_id={N}`) so it's itself revertable.
2. All actions with `started_at >= N.started_at` and status `done`/`error` are walked **newest first**.
3. For each, its revisions are replayed in **reverse id order** with the inverse op (`create` → delete, `update` → restore `content_before`, `delete` → recreate).
4. Each reverted action is marked `status='reverted'`.
5. The knowledge graph is rebuilt for every page touched by the revert.

While a revert runs, the same per-org lock that recalibration uses returns 503 on writes for that org. Reads, query, and chat keep working.

### UI

The **History** tab in the admin dashboard (admin + supervisor only) lists every action with type, summary, user, status, and change count. Click **View** for the per-revision breakdown; click **Revert** for a two-step confirmation modal that previews every action that will be discarded and requires typing the action's summary to enable the destructive button.

### Retention

Each org has a `revision_retention_days` setting (default **21**, configurable on the **Org Limits** page or via `PUT /api/admin/org-limits`). A background task running every 24 hours deletes actions older than the retention window and sweeps the matching S3 archive objects. Pruned actions become unrevertable — that's the contract.

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/admin/history?page=&limit=&action_type=&user_id=` | Paginated list of recent actions for the current org |
| `GET` | `/api/admin/history/{id}` | One action + its revision list (sizes only, no content) |
| `GET` | `/api/admin/history/{id}/diff?target_kind=&target_key=` | `content_before` / `content_after` for one revision |
| `GET` | `/api/admin/history/{id}/preview-revert` | Lists every action that *would* be discarded if you reverted now |
| `POST` | `/api/admin/history/{id}/revert` | Performs the revert; returns `{revert_action_id, reverted_action_ids, pages_touched}` |

All five require role `admin` or `supervisor`. Members get 403.

---

## Notifications

A lightweight in-app notification system keeps users informed about long-running work without polling. Notifications are stored per-user in a `notifications` table and pushed in real time.

| Type | Raised when | Goes to |
|---|---|---|
| `pending_review` | An uploaded document's plan finishes and is ready for review | The uploader |
| `ingest_done` | An approved ingest finishes writing pages | The uploader |
| `ingest_error` | Planning or writing fails | The uploader |
| `recalib_done` | A recalibration run completes | The user who started it |

**Real-time delivery.** `GET /api/notifications/stream` is an SSE stream: it sends an `init` event with the current unread backlog, then a `new` event whenever a notification arrives. Delivery is driven by Postgres `LISTEN/NOTIFY` (fanned out in-process by `notif_stream`), so there is no polling; a 25-second keepalive doubles as a missed-`NOTIFY` backstop and a client-disconnect detector. The client uses `fetch` with the normal Bearer header (not `EventSource`).

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/notifications` | List the user's unread notifications |
| `GET` | `/api/notifications/stream` | SSE stream (`init` backlog + live `new` events) |
| `POST` | `/api/notifications/read-all` | Mark all read |
| `POST` | `/api/notifications/{id}/read` | Mark one read |
| `POST` | `/api/notifications/read-by-link` | Mark read by target link (e.g. when the user opens the linked page) |

---

## Wiki Activity & Page History

The change-tracking data behind revert is also surfaced **read-only to anyone who can view the wiki** — not just the admins/supervisors who use the History tab. This powers an org "recent changes" feed and per-page provenance (last editor, contributors, timeline).

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/activity/recent?limit=` | Org-wide feed of recent page-affecting changes (newest first, with a sample of touched pages) |
| `GET` | `/api/activity/page?path=` | Per-page provenance: last editor, when, and the contributor list |
| `GET` | `/api/activity/page/history?path=&limit=` | Full per-page change timeline |
| `GET` | `/api/activity/revision/{id}` | Before/after content of one page revision (for the diff view) |

All four require `can_view_wiki` and are strictly org-scoped to `target_kind='page'`, so this member-facing route can never expose raw-upload or schema revision blobs.

---

## LLM Models

| Role | Env var | Default | Responsibility |
|---|---|---|---|
| **Ingest Planner** | `BEDROCK_INGEST_MODEL_ID` | Claude Haiku 4.5 | Reads documents, builds page plans, plan chat |
| **Ingest Writer** | `BEDROCK_INGEST_WRITER_MODEL_ID` | Llama 4 Maverick | Renders individual wiki pages from a planned spec (single-shot) |
| **Query / Chat** | `BEDROCK_QUERY_MODEL_ID` | Claude Haiku 4.5 | Answers questions, streaming chat |
| **Recalibrate** | `BEDROCK_RECALIBRATE_MODEL_ID` | Claude Sonnet 4.5 | Deep analysis and full-wiki rewriting |
| **Draft Agent** | `BEDROCK_DRAFT_AGENT_MODEL_ID` | Claude Haiku 4.5 | Conversational AI Writer — multi-turn drafting, Q&A, section revisions |
| **Inline Editor** | `BEDROCK_EDIT_MODEL_ID` | Claude Sonnet 4.5 | Inline AI page/section rewrites (single-shot) |
| **Embeddings** | `BEDROCK_EMBEDDING_MODEL_ID` | Titan Embed Text v1 | Hybrid-search vectors — on by default; set empty to disable |

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://wiki:wiki@localhost:5432/wiki` | PostgreSQL connection string |
| `STORAGE_BACKEND` | `local` | Controls raw upload storage only: `local` = Docker volume / disk; `s3` = AWS S3 |
| `DATA_DIR` | `./data` | Root directory for local raw-file storage (used when `STORAGE_BACKEND=local`) |
| `DATA_BUCKET` | *(empty)* | S3 bucket name for raw uploads (used when `STORAGE_BACKEND=s3`) |
| `AWS_ENDPOINT_URL` | *(empty)* | Override S3 endpoint (e.g. MinIO); leave blank for real AWS |
| `AWS_REGION` | `us-east-1` | AWS / Bedrock region |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | *(empty)* | Static credentials; leave blank for IAM role |
| `ASSUMED_ROLE_ARN` | *(empty)* | ARN to assume for cross-account Bedrock access |
| `ASSUMED_ROLE_SESSION_NAME` | `WikiAgentSession` | STS session name when assuming a role |
| `ASSUMED_ROLE_DURATION` | `3600` | STS assumed-role credential lifetime (seconds) |
| `BEDROCK_INGEST_MODEL_ID` | Claude Haiku 4.5 | Ingest planner model |
| `BEDROCK_INGEST_WRITER_MODEL_ID` | Llama 4 Maverick | Ingest page-renderer model (was `BEDROCK_WRITER_MODEL_ID`; old name still accepted) |
| `BEDROCK_QUERY_MODEL_ID` | Claude Haiku 4.5 | Query/chat model |
| `BEDROCK_RECALIBRATE_MODEL_ID` | Claude Sonnet 4.5 | Recalibration model |
| `BEDROCK_DRAFT_AGENT_MODEL_ID` | Claude Haiku 4.5 | Conversational AI Writer agent model |
| `BEDROCK_EDIT_MODEL_ID` | Claude Sonnet 4.5 | Inline AI editor model (page/section rewrites) |
| `BEDROCK_EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v1` | Embedding model for hybrid search; **empty disables** vector search |
| `EMBEDDING_DIMENSIONS` | `1536` | Vector dimensions for the embedding model |
| `BEDROCK_SSL_VERIFY` | `true` | Verify TLS on Bedrock clients; set `false` behind a TLS-intercepting proxy |
| `APP_TITLE` | `AI Knowledge Hub` | Browser tab title |
| `COMPANY_NAME` | `My Company` | Brand name shown in the UI |
| `APP_PORT` | `8000` | Listening port |
| `APP_TIMEZONE` | `UTC` | IANA timezone for log timestamps |
| `LOG_LEVEL` | `INFO` | Log verbosity (`DEBUG` / `INFO` / `WARNING` / `ERROR`) |
| `INITIAL_ORG_NAME` | `Default Organization` | Name of the org seeded on first startup |
| `MAX_UPLOAD_SIZE_MB` | `2` | Maximum file upload size |
| `AUTH_USERNAME` / `AUTH_PASSWORD` | `admin` / `admin` | Login credentials |
| `CHAT_STREAM` | `true` | Enable SSE streaming for chat responses |
| `JWT_SECRET` | `change-me-in-production` | HMAC signing secret for JWT access tokens — **change this in production** |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `JWT_ACCESS_TTL_MINUTES` | `15` | Access token lifetime (minutes) |
| `JWT_REFRESH_TTL_DAYS` | `7` | Refresh token lifetime (days) |
| `BEDROCK_CONCURRENCY` | `20` | Max concurrent Bedrock calls per model |
| `INGEST_CONCURRENCY` | `10` | Max simultaneous ingest jobs across the process |
| `WRITE_PAGE_CONCURRENCY` | `10` | Max simultaneous page writes within one ingest job |
| `MAX_INGEST_CHARS` | `300000` | Reject uploads larger than this (chars) before any LLM call |
| `MAX_HTML_TABLE_CELLS` | `100` | Reject HTML uploads with more `<td>`/`<th>` cells than this |
| `RATE_LIMIT_UPLOAD_PER_MINUTE` | `10` | Upload rate limit per user |
| `RATE_LIMIT_QUERY_PER_MINUTE` | `30` | Query rate limit per user |
| `RATE_LIMIT_CHAT_PER_MINUTE` | `60` | Chat rate limit per user |
| `RATE_LIMIT_RECALIBRATE_PER_MINUTE` | `2` | Recalibrate rate limit per user |
| `WRITER_DRAFT_TTL_DAYS` | `30` | Days an inactive Writer Mode draft stays in the picker before being pruned |

---

## Admin Dashboard

The admin dashboard is available at `http://ip:port/admin` for users with `role=admin`.

### Access Control — Permission Templates

Every member user can be assigned a **Permission Template** — a named set of boolean flags and numeric quotas that governs exactly what they can do. Admins always have full access regardless of their template.

**Built-in templates** (seeded automatically per org, cannot be deleted):

| Template | Purpose |
|---|---|
| `read_only` | View wiki and graph only — no uploads, no LLM calls |
| `contributor` | Upload documents, query, and chat; daily upload and chat limits |
| `power_user` | Full access except recalibration |
| `admin` | Everything enabled, no quotas |

**Custom templates** can be created with any combination of permissions and limits. Templates are cloned from existing ones with one click.

### Permission Flags

| Flag | Controls |
|---|---|
| `can_upload` | Upload new documents |
| `can_upload_writer_draft` | Save & ingest an AI Writer draft |
| `can_delete_files` | Delete uploaded source files |
| `can_download_files` | Download source files |
| `can_view_wiki` | View/search wiki pages |
| `can_edit_wiki` | Manually edit wiki page content |
| `can_delete_wiki_pages` | Delete wiki pages |
| `can_query` | Run one-shot wiki queries |
| `can_chat` | Use the chat interface |
| `can_use_writer` | Use AI Writer mode (members; admins/supervisors always have it) |
| `can_recalibrate` | Trigger wiki recalibration |
| `can_run_lint` | Run the wiki health check |
| `can_manage_schema` | View/edit `AGENTS.md` |
| `can_view_audit_log` | Access audit log |
| `can_view_graph` | View the knowledge graph |
| `can_rebuild_graph` | Force graph rebuild |
| `can_manage_workspace` | Create/copy/delete workspaces |
| `can_approve_ingest` | Approve ingest plans |
| `can_cancel_ingest` | Cancel in-progress ingests |

### Quota Fields

Numeric limits can be set on any template (blank = unlimited):

| Field | Scope |
|---|---|
| `max_upload_size_mb` | Per-file size override (trumps global `MAX_UPLOAD_SIZE_MB` when stricter) |
| `max_uploads_per_day` | User's daily upload count |
| `max_uploads_per_week` | User's weekly upload count |
| `max_queries_per_day` | User's daily query count |
| `max_chat_messages_per_day` | User's daily chat message count |
| `max_tokens_per_day` | User's daily token budget (input + output) |
| `max_tokens_per_week` | User's weekly token budget |

### Org-Level Caps

In addition to per-user template limits, the org can have its own caps (managed via **Org Limits** tab):

| Setting | Effect |
|---|---|
| `max_uploads_per_day_org` | Total daily upload cap across all users in the org |
| `max_tokens_per_day_org` | Total daily token budget for the entire org |
| `max_members` | Maximum number of users in the org |
| `revision_retention_days` | How long to keep revert-history for actions (default 21). Older actions are auto-pruned and become non-revertable. |

Org-level caps apply on top of per-user template limits — whichever is stricter wins.

### Workspaces

Workspaces are isolated S3 namespaces within an org. Each workspace has its own S3 prefix so raw documents are physically separate. Workspaces can be **copied** (server-side S3 copy) for sandboxing experiments.

### Admin API Endpoints

Most endpoints require `role=admin` (`403` otherwise); the membership and org-scoped management endpoints also allow a `supervisor` acting within their own org.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/admin/templates` | List all permission templates |
| `POST` | `/api/admin/templates` | Create custom template |
| `GET` | `/api/admin/templates/{id}` | Get template details |
| `PUT` | `/api/admin/templates/{id}` | Update custom template (built-ins protected) |
| `DELETE` | `/api/admin/templates/{id}` | Delete custom template (built-ins protected) |
| `POST` | `/api/admin/templates/{id}/clone` | Clone a template |
| `GET` | `/api/admin/users` | List users with template, stats, last login |
| `PUT` | `/api/admin/users/{id}` | Update user role / template / workspace / suspension |
| `POST` | `/api/admin/users/{id}/reset-password` | Reset user password |
| `GET` | `/api/admin/workspaces` | List workspaces |
| `POST` | `/api/admin/workspaces` | Create workspace |
| `POST` | `/api/admin/workspaces/{id}/copy` | Copy workspace |
| `DELETE` | `/api/admin/workspaces/{id}` | Delete workspace |
| `GET` | `/api/admin/usage?days=30` | Token + upload usage by user |
| `GET` | `/api/admin/org-limits` | Get org-level limits |
| `PUT` | `/api/admin/org-limits` | Update org-level limits |
| `GET` | `/api/admin/audit-log` | Paginated, searchable audit log |
| `GET` | `/api/admin/audit-log/operations` | Distinct operation names (for the audit-log filter) |
| `GET` | `/api/admin/identities` | List user identities across orgs |
| `GET` | `/api/admin/users/{id}/memberships` | List a user's org memberships |
| `POST` | `/api/admin/users/{id}/memberships` | Add a user to an org |
| `PUT` | `/api/admin/users/{id}/memberships/{org_id}` | Update a membership (role, template, limits, suspension) |
| `DELETE` | `/api/admin/users/{id}/memberships/{org_id}` | Remove a membership |
| `PUT` | `/api/admin/users/{id}/transfer-org` | Move a user to another org |
| `GET` | `/api/admin/organizations` | List organizations |
| `POST` | `/api/admin/organizations` | Create an organization (+ first supervisor) |
| `POST` | `/api/admin/organizations/{id}/clone` | Clone an organization |
| `DELETE` | `/api/admin/organizations/{id}` | Delete an organization |
| `GET` | `/api/admin/jobs` | Cross-replica ingest + recalibration job queue |
| `POST` | `/api/admin/jobs/ingest/{filename}/cancel` | Cancel a queued/running ingest |
| `POST` | `/api/admin/jobs/recalibrate/cancel` | Cancel a running recalibration |

> History/revert endpoints are listed under [Change Tracking & Revert](#api-endpoints).

---

## Wiki Schema

`schema/AGENTS.md` (stored in PostgreSQL `wiki_files` table, key: `schema/AGENTS.md`) is the "constitution" that governs how the LLM structures wiki pages — directory layout, frontmatter format, cross-linking rules, and writing style. It is read on every ingest and recalibration. Changes take effect immediately without a restart and can be edited from the UI (`GET /PUT /api/ops/schema`).

---

## Key API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Load-balancer health check — returns `{status, db}` |
| `GET` | `/api/auth/config` | Public config (company name, chat_stream flag) |
| `POST` | `/api/auth/login` | Obtain JWT access token + refresh token |
| `POST` | `/api/auth/refresh` | Exchange refresh token for new access token |
| `POST` | `/api/auth/logout` | Revoke refresh token |
| `POST` | `/api/documents/upload` | Upload file; auto-starts background plan phase |
| `GET` | `/api/documents` | List uploaded source files |
| `DELETE` | `/api/documents/{filename}` | Delete file + linked source wiki pages |
| `GET` | `/api/ops/status/{filename}` | Poll ingest job status |
| `POST` | `/api/ops/ingest/{filename}/plan-chat` | Chat to refine ingest plan |
| `POST` | `/api/ops/ingest/{filename}/approve` | Approve plan, start write phase |
| `DELETE` | `/api/ops/ingest/{filename}` | Cancel in-progress ingest |
| `POST` | `/api/ops/query` | One-shot question |
| `POST` | `/api/ops/chat` | Non-streaming multi-turn chat |
| `POST` | `/api/ops/chat/stream` | Streaming multi-turn chat (SSE) |
| `DELETE` | `/api/ops/chat/{session_id}` | Clear chat session |
| `POST` | `/api/ops/lint` | Wiki health check |
| `POST` | `/api/ops/recalibrate` | Start recalibration — accepts `{deleted_files:[...], fact_instructions:"..."}`. Non-empty `fact_instructions` activates targeted mode (semantic search instead of full triage). |
| `GET` | `/api/ops/recalibrate/status` | Poll recalibration progress — includes `fact_instructions` field (empty for full runs) |
| `GET` | `/api/ops/schema` | Read `AGENTS.md` |
| `GET` | `/api/ops/schema/default` | Read bundled default `AGENTS.md` |
| `POST` | `/api/ops/schema/validate` | Validate candidate `AGENTS.md` (returns errors + warnings) |
| `PUT` | `/api/ops/schema` | Update `AGENTS.md` |
| `GET` | `/api/ops/log` | Recent audit log entries rendered as markdown (most recent first) |
| `GET` | `/api/ops/export` | Download portable ZIP export (supervisor + admin only) |
| `GET` | `/api/wiki` | Wiki page tree (paginated; `page=0` returns the nested sidebar tree) |
| `GET` | `/api/wiki/search?q=` | Full-text search |
| `GET` | `/api/wiki/{path}` | Read a wiki page |
| `PUT` | `/api/wiki/{path}` | Edit a wiki page (tracked + revertible) |
| `DELETE` | `/api/wiki/{path}` | Delete a wiki page |
| `GET` | `/api/wiki/{path}/links` | Outgoing + incoming links for a page |
| `GET` | `/api/ops/permissions` | Effective permissions for the current user |
| `POST` | `/api/ops/wiki/edit/stream` | Inline AI edit — stream a proposed rewrite (SSE) |
| `POST` | `/api/ops/writer/chat/stream` | AI Writer chat (SSE) |
| `GET` | `/api/ops/graph` | Knowledge-graph nodes + edges (or `?clusters=true`) |
| `POST` | `/api/ops/graph/rebuild` | Force a knowledge-graph rebuild |
| `POST` | `/api/ops/import` | Import an export bundle (empty org only) |
| `GET` | `/api/notifications/stream` | Real-time notification stream (SSE) |
| `GET` | `/api/activity/recent` | Recent wiki-changes feed |

Admin (`/api/admin/*`), AI Writer (`/api/ops/writer/*`), notification, and activity endpoints are documented in full in their respective sections above.

---

## Testing

The project ships with a three-tier test suite (unit + integration + frontend). **All async tests share a single session-scoped event loop** — this is required for testcontainers + SQLAlchemy asyncpg to work correctly and is already configured in `pytest.ini`.

### Install test dependencies

```bash
# via uv (recommended — matches the project's venv)
uv pip install -r requirements-test.txt

# or plain pip
pip install -r requirements-test.txt
```

### Unit tests — no external services needed

```bash
pytest tests/unit -v
```

**538 tests** covering:

| Module | What is tested |
|---|---|
| `test_auth.py` | `verify_credentials`, `create_access_token`, `validate_access_token`, `create_refresh_token`, `validate_refresh_token`, `revoke_token`, `purge_expired` |
| `test_rate_limit.py` | `check_rate_limit` (under/at/over limit), per-endpoint dependency functions, anonymous fallback |
| `test_jobs.py` | `enqueue`, `get`, `save`, `cancel`, task lifecycle, status transitions |
| `test_chat_sessions.py` | `get_or_create`, `save`, `clear`, session lifecycle |
| `test_graph.py` | Link parsing (`[[wikilinks]]`, markdown, frontmatter), neighbor traversal, `as_dict`, `rebuild`, `_save`/`_load`, `generate_html`, `update_pages` |
| `test_wiki_engine.py` | `_parse_json`, `_find_relevant_pages`, `_maybe_summarize`, `lint`, `query`, `chat`, `chat_stream`, `plan_chat`, helper methods |
| `test_wiki_db.py` | `get_compact_index`, `semantic_search_wiki`, `get_rendered_log` |
| `test_semantic_search.py` | Embeddings service, hybrid/BM25 search, paginated listing, wiki links, Louvain clusters |
| `test_provider_bedrock.py` | `converse`, `converse_stream`, credential rotation, embedding payload shaping |
| `test_model.py` | role → name → ID resolution, startup validation, chat/converse factories, embedding fallback, provider registry |
| `test_config_model_vars.py` | model-name defaults and the legacy `BEDROCK_*_MODEL_ID` promotion rules |
| `test_no_direct_llm_clients.py` | architecture guard — no LLM client constructed outside `app/providers/` |
| `test_aws_auth.py` | Static creds, STS role assumption, expiry buffer, cache, refresh loop, start/stop task |
| `test_s3.py` | Local filesystem backend (read, write, delete, exists, list, size, `ensure_bucket`) and S3 backend with mocked boto3 |
| `test_ingest_agent.py` | `_parse_json`, `_split_chunks`, `_extract_text` (txt/md/fallback), `_base_state` |
| `test_permissions.py` | `require_permission`, `get_user_permissions`, built-in template seeding, quota checks |
| `test_orgs.py` | `verify_user`, `create_user`, password hashing, org lookup |
| `test_auth_phase6.py` | JWT org_id/user_id/role claims, token round-trip |
| `test_utils_frontmatter.py` | Frontmatter `type:` detection, path-prefix fallback, `stamp_frontmatter_field` idempotency, CRLF handling |
| `test_wiki_state.py` | `begin_action` lifecycle (running → done / error), `tracked_action` decorator, ContextVar propagation through `asyncio.create_task`, nested-action no-op |
| `test_wiki_state_wrappers.py` | Each of the 5 choke-point wrappers — page create/update/delete revisions, file writes, the `.graph.json` skip case, S3 archive logic |
| `test_revert_algorithm.py` | Revert correctness over a stub storage — guard clauses, all three op kinds, multi-action reverse-chronological replay, S3 archive restore, revision id ordering |
| `test_revision_pruning.py` | Retention-driven pruning of `wiki_actions`/`wiki_revisions` + S3 archive cleanup, non-archive paths skipped, S3-failure resilience |
| `test_writer_stream.py` | Writer-mode SSE parser — full-draft markers, section markers, multi-chunk marker splits, false-trigger resilience (lone end markers, bracketed prose) |
| `test_chat_sessions_writer.py` | Writer-mode helpers — section-bounds finder (single/ambiguous/not-found/nested), `patch_draft_section`, `get_draft`, `prune_expired_writer_sessions`, `_excerpt`, `ChatSession` defaults |
| `test_writer_route_gating.py` | Writer endpoint gating — admin/supervisor bypass, `can_use_writer`, suspension |
| `test_ingest_queue.py` | Durable write-queue submit/claim SQL, drain bookkeeping, dedup guard |
| `test_md_sections.py` | Section extract/splice by heading (single, ambiguous, nested, not-found) |
| `test_wiki_health.py` | Deterministic health score — per-signal detection, weighting, monotonicity, empty = 100 |
| `test_wiki_import.py` | Export-bundle parsing, schema/page import, embedding reuse vs re-embed |
| `test_notif_stream.py` | Notification hub subscribe/unsubscribe + `NOTIFY` fan-out |
| `test_retriever.py` | Bundled `retriever.py` — BM25, hybrid scoring, graph-hop expansion |
| `test_recalibrate_analyze_batching.py` | Shortlist batching, per-batch plan merge, reconciliation trigger |
| `test_recalibrate_robustness.py` | Truncated/failed-batch isolation, plan de-dup, guard clauses |
| `test_recalibrate_job.py` | Recalibration job status lifecycle + fact-instruction plumbing |
| `test_targeted_recalibration.py` | Targeted mode — triage skip, semantic search, user-directive block |

All LLM, AWS, and database calls are mocked — no credentials or running services required.

### Integration tests — require Docker

```bash
pytest tests/integration -v
```

Uses **testcontainers** to spin up a throwaway PostgreSQL 16 container with pgvector (`pgvector/pgvector:pg16`) and applies Alembic migrations automatically. Docker must be running. CI uses the same image for its integration database. Hybrid-search tests require the embedding column and fail if it is missing rather than skipping. If `DATABASE_URL` is explicitly set, it must point to a disposable test database with pgvector available: the fixtures clear its tables. The suite exercises the full HTTP API surface:

- `test_routes_auth.py` — login, logout, JWT refresh, 401 flows, `/health`
- `test_routes_wiki.py` — wiki tree, full-text search, page CRUD
- `test_routes_documents.py` — upload, size/type enforcement, download, delete, cleanup
- `test_routes_operations.py` — ingest status/cancel, plan-chat, approve, query, chat, lint, graph, recalibrate, schema, 503 middleware
- `test_routes_orgs.py` — org info, user management, usage endpoint
- `test_admin.py` / `test_admin_workflows.py` — permission templates, user admin, workspaces, usage, org-limits, audit log; end-to-end admin flows
- `test_admin_jobs.py` — cross-replica Jobs tab, ingest/recalibrate cancel
- `test_memberships.py` / `test_org_clone.py` — multi-org membership management, org cloning
- `test_multi_tenancy.py` / `test_graph_isolation.py` — cross-org data + graph isolation
- `test_wiki_edit.py` — inline AI edit stream (read-only proposal, section scope, error paths)
- `test_wiki_import.py` — import a bundle into an empty org (409 on non-empty)
- `test_notifications.py` / `test_notif_stream.py` — notification CRUD + SSE stream
- `test_activity.py` — recent-changes feed + per-page provenance
- `test_ingest_queue.py` — durable queue drain, FIFO order, failover/recovery
- `test_revert_jobs.py` — async revert job + progress polling
- `test_hybrid_search.py` — BM25 + vector hybrid ranking end-to-end
- `test_lint_gaps.py` — health-check edge cases

> **Note:** Integration tests use `asyncio_default_test_loop_scope = session` (set in `pytest.ini`). This ensures the session-scoped SQLAlchemy asyncpg engine and function-scoped async fixtures (auth tokens, cleanup) all share the same event loop. Do not change this setting without understanding the asyncpg event-loop binding constraints.

### Frontend tests — require Docker + Playwright

```bash
playwright install chromium   # one-time setup
pytest tests/frontend -v
```

Starts the FastAPI app on a random port against a fresh Postgres container. **40 tests** across 10 modules drive a real Chromium browser via Playwright:

- `test_auth.py` — login/logout flows, token in `localStorage`
- `test_wiki.py` — sidebar navigation, search
- `test_chat.py` — send message, new session (API stubbed via route interception)
- `test_upload.py` — upload flow, error toast for bad extensions
- `test_ai_edit.py` — inline AI-edit modal: diff, apply, 401-refresh retry, XSS sanitization, fenced-markdown preview
- `test_admin_ui.py` — admin dashboard interactions
- `test_dialogs.py` — shared modal / confirm-dialog behavior
- `test_org_context.py` — org switching via `X-Org-Context`
- `test_import_ui.py` / `test_create_org_import.py` — Import Wiki button + empty-org import flow

### Coverage report

```bash
# Unit + integration combined (recommended)
pytest tests/unit tests/integration --cov=app/services --cov=app/routes --cov-report=term-missing

# Unit tests only (fast, no Docker needed)
pytest tests/unit --cov=app/services --cov=app/routes --cov-report=term-missing
```

**Current coverage summary (unit + integration, 5,978 statements):**

| Module | Coverage |
|---|---|
| `app/routes/auth.py` | 100% |
| `app/routes/wiki.py` | 92% |
| `app/routes/activity.py` | 92% |
| `app/routes/org.py` | 88% |
| `app/routes/operations.py` | 79% |
| `app/routes/admin.py` | 79% |
| `app/routes/documents.py` | 74% |
| `app/routes/notifications.py` | 66% |
| `app/services/auth.py` · `rate_limit.py` · `md_sections.py` | 100% |
| `app/services/chat_sessions.py` · `aws_auth.py` | 99% |
| `app/services/permissions.py` | 96% |
| `app/services/graph.py` | 94% |
| `app/services/wiki_state.py` · `jobs.py` | 93% |
| `app/services/wiki_health.py` · `notif_stream.py` · `writer_stream.py` | 91–92% |
| `app/services/wiki_import.py` | 87% |
| `app/providers/bedrock.py` · `wiki_db.py` | 83% |
| `app/services/wiki_engine.py` | 78% |
| `app/services/ingest_queue.py` | 76% |
| `app/services/s3.py` | 68% |
| `app/services/recalibrate_agent.py` | 67%¹ |
| `app/services/ingest_agent.py` | 53%¹ |
| **Overall** | **81%** |

¹ The two LangGraph agents carry the heaviest LLM-mocking burden — their node closures and graph execution need a full `ChatBedrockConverse` mock — so they sit lowest. Excluding them, coverage across the other services and routes is **~85%**.

### CI

GitHub Actions runs unit tests on every push and integration + frontend tests on every PR to `main`. See [`.github/workflows/test.yml`](.github/workflows/test.yml).

---

## Deployment

The application runs as two Docker services defined in `docker-compose.yml`:

```
  Host machine
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  ┌───────────────────────────────────────────────────────────────┐  │
  │  │  wiki  (llm-wiki-app)                                         │  │
  │  │  image : llm-wiki-app:latest  (built from Dockerfile)         │  │
  │  │  port  : ${APP_PORT:-8000} → 8000                             │  │
  │  │                                                               │  │
  │  │  env   : DATABASE_URL=postgresql+asyncpg://wiki:wiki@         │  │
  │  │                        postgres:5432/wiki                     │  │
  │  │          STORAGE_BACKEND=local  (default)                     │  │
  │  │          DATA_DIR=/data                                       │  │
  │  │                                                               │  │
  │  │  volume: data  ──────────────────────────────► /data         │  │
  │  │          (raw uploads, graph cache)                           │  │
  │  │                                                               │  │
  │  │  limits: pids=512  nofile=65536                               │  │
  │  └───────────────────────┬───────────────────────────────────────┘  │
  │                          │ TCP 5432  (depends_on: postgres healthy)  │
  │  ┌───────────────────────▼───────────────────────────────────────┐  │
  │  │  postgres  (llm-wiki-data)                                    │  │
  │  │  image : pgvector/pgvector:pg16                               │  │
  │  │  port  : ${POSTGRES_PORT:-<none>} → 5432  (optional expose)   │  │
  │  │                                                               │  │
  │  │  env   : POSTGRES_USER=wiki  POSTGRES_PASSWORD=wiki           │  │
  │  │          POSTGRES_DB=wiki                                     │  │
  │  │                                                               │  │
  │  │  volume: postgres_data ──────────────────► /var/lib/          │  │
  │  │          (wiki pages, jobs, sessions,        postgresql/data  │  │
  │  │           auth tokens, audit log)                             │  │
  │  │                                                               │  │
  │  │  health : pg_isready -U wiki  (every 5 s, 10 retries)        │  │
  │  └───────────────────────────────────────────────────────────────┘  │
  │                                                                      │
  │  Named volumes: postgres_data   data                                 │
  └──────────────────────────────────────────────────────────────────────┘

  External (STORAGE_BACKEND=s3)
  ┌──────────────────────┐
  │  AWS S3  (DATA_BUCKET│◄── wiki container uploads raw files here
  │  + Bedrock LLM API)  │    instead of the data volume
  └──────────────────────┘
```

By default `STORAGE_BACKEND=local` — all files are written to the `data` named volume mounted at `/data`. To use S3 instead, set `STORAGE_BACKEND=s3` and `DATA_BUCKET` in `.env` (remove `DATA_DIR`).

```bash
# Build and run all services
docker compose up --build

# Run detached
docker compose up -d

# Tail app logs
docker compose logs -f wiki

# Rebuild after dependency changes
docker compose build --no-cache wiki && docker compose up -d wiki
```

State is stored in named Docker volumes (`postgres_data`, `data`) and persists across container restarts and updates.

### First-time setup

```bash
git clone <your-fork-url> && cd ai-knowledge-hub
cp .env.example .env          # required — docker-compose reads .env
# edit .env (see the production checklist below), then:
docker compose up -d --build
docker compose logs -f wiki   # watch for "startup | default org seeded"
```

On first boot the app runs its migrations and seeds the default organization plus an admin user from `AUTH_USERNAME` / `AUTH_PASSWORD`. Log in at `http://<host>:8000` with those credentials, then create real users from the Admin dashboard.

### Production checklist

Before exposing the app to anything but localhost:

- [ ] **Set `ENVIRONMENT=production`** in `.env`. The app refuses to boot if any security-sensitive setting below is left at its default.
- [ ] **Set a strong `JWT_SECRET`** — e.g. `openssl rand -hex 32`. This signs all auth tokens; the default is public knowledge.
- [ ] **Set a strong `AUTH_USERNAME` / `AUTH_PASSWORD`** for the seeded admin (don't ship `admin`/`changeme`).
- [ ] **Change the PostgreSQL password.** It's `wiki`/`wiki` by default — edit `POSTGRES_PASSWORD` in `docker-compose.yml` **and** the matching `DATABASE_URL` in `.env` so they agree.
- [ ] **Configure AWS Bedrock access** — `AWS_REGION` plus credentials (env vars or an IAM role), with the configured `BEDROCK_*` model IDs enabled in that region.
- [ ] **Terminate TLS in front of the app.** The container serves plain HTTP on port 8000; put a reverse proxy (nginx, Caddy, Traefik, an ALB, etc.) in front to handle HTTPS.
- [ ] **Back up the `postgres_data` volume** — it holds all wiki content, users, and audit history.

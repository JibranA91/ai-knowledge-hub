from urllib.parse import urlsplit

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings

# Legacy raw-model-ID var → the MODEL_* name var that replaced it. Applied in
# Settings.model_post_init so a .env written before the model-name catalogue
# keeps working untouched.
LEGACY_MODEL_VARS = {
    "BEDROCK_INGEST_MODEL_ID":        "MODEL_INGEST_PLAN",
    "BEDROCK_INGEST_WRITER_MODEL_ID": "MODEL_INGEST_WRITE",
    "BEDROCK_QUERY_MODEL_ID":         "MODEL_QUERY",
    "BEDROCK_RECALIBRATE_MODEL_ID":   "MODEL_RECALIBRATE",
    "BEDROCK_DRAFT_AGENT_MODEL_ID":   "MODEL_DRAFT_AGENT",
    "BEDROCK_EDIT_MODEL_ID":          "MODEL_EDIT",
    "BEDROCK_EMBEDDING_MODEL_ID":     "MODEL_EMBEDDING",
}


class Settings(BaseSettings):
    # ── LLM provider ───────────────────────────────────────────────────────
    # Which backend serves every model call. All connections are created in
    # app/providers/<name>.py and reached through app/model.py — nothing else
    # in the codebase talks to an LLM vendor directly.
    # Registered providers: see app/providers/__init__.py.
    LLM_PROVIDER: str = "bedrock"
    # Direct API-key/endpoint adapters; Bedrock continues to use AWS auth.
    LLM_API_KEY: SecretStr = SecretStr("")
    LLM_BASE_URL: str = ""
    MODEL_DEFAULT: str = ""

    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    ASSUMED_ROLE_ARN: str = ""
    ASSUMED_ROLE_SESSION_NAME: str = "WikiAgentSession"
    ASSUMED_ROLE_DURATION: int = 3600
    # ── Models, one per logical role (app.model.Role) ──────────────────────
    # These hold a *model name*, not a vendor model ID: "haiku45", "sonnet45",
    # "opus5". The active provider maps the name to its own concrete ID (for
    # bedrock, see MODEL_CATALOG in app/providers/bedrock.py), so the same
    # config works against a different vendor. Names are matched loosely —
    # "haiku45", "haiku-4.5" and "Haiku 4.5" are the same model.
    # A value containing '.', ':' or '/' is treated as a raw vendor model ID
    # and passed through untouched, as an escape hatch for anything the
    # catalogue doesn't cover. Unknown bare names fail at startup.

    # Ingest planner — a tool-calling reasoning loop.
    MODEL_INGEST_PLAN: str = "haiku45"
    # Page renderer inside the ingest pipeline — writes the markdown body of an
    # individual planned page. Single-shot, no reasoning loop.
    MODEL_INGEST_WRITE: str = "llama4maverick"
    # Chat / Q&A over the wiki.
    MODEL_QUERY: str = "haiku45"
    # Wiki-wide analysis and rewrite.
    MODEL_RECALIBRATE: str = "sonnet45"
    # Conversational AI Writer agent (the chat-driven document authoring
    # flow). Multi-turn reasoner — asks clarifying questions, drafts the page,
    # accepts revisions.
    MODEL_DRAFT_AGENT: str = "haiku45"
    # Inline AI editor — rewrites a whole page or one section against an
    # instruction. Single-shot, no reasoning loop.
    MODEL_EDIT: str = "sonnet45"
    # Optional semantic embeddings. Leave empty to disable vector search.
    # Storage currently requires 1536 dimensions (e.g. titanembedv1).
    MODEL_EMBEDDING: str = "titanembedv1"

    # Cross-region inference profile geography for Bedrock text models: "us",
    # "eu", "global", … It becomes the model ID's prefix (us.anthropic.…).
    # Leave empty to call foundation models directly with no profile.
    BEDROCK_INFERENCE_GEO: str = "us"

    # ── Back-compat: the pre-name-catalogue variables ──────────────────────
    # These held raw Bedrock model IDs. Still honoured — a role falls back to
    # its BEDROCK_*_MODEL_ID when the MODEL_* above is unset — so existing .env
    # files keep working. Prefer the MODEL_* names for new deployments.
    BEDROCK_INGEST_MODEL_ID: str = ""
    BEDROCK_INGEST_WRITER_MODEL_ID: str = ""
    BEDROCK_QUERY_MODEL_ID: str = ""
    BEDROCK_RECALIBRATE_MODEL_ID: str = ""
    BEDROCK_DRAFT_AGENT_MODEL_ID: str = ""
    BEDROCK_EDIT_MODEL_ID: str = ""
    BEDROCK_EMBEDDING_MODEL_ID: str = ""
    # Older still: BEDROCK_WRITER_MODEL_ID, promoted onto
    # BEDROCK_INGEST_WRITER_MODEL_ID in model_post_init below.
    BEDROCK_WRITER_MODEL_ID: str = ""

    EMBEDDING_DIMENSIONS: int = 1536

    # Deployment environment. When set to "production", the app refuses to boot
    # if any security-sensitive setting is still at its insecure default (see
    # model_post_init). Leave "development" for local/test runs.
    ENVIRONMENT: str = "development"

    APP_TITLE: str = "AI Knowledge Hub"
    COMPANY_NAME: str = "My Company"
    APP_PORT: int = 8000
    APP_TIMEZONE: str = "UTC"
    MAX_UPLOAD_SIZE_MB: int = 2

    # Seed credentials — used only to bootstrap the default org's admin user on first run.
    # log in via the users table; these can be rotated freely.
    AUTH_USERNAME: str = "admin"
    AUTH_PASSWORD: str = "admin"
    CHAT_STREAM: bool = True

    # Multi-tenancy
    INITIAL_ORG_NAME: str = "Default Organization"

    # JWT settings
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TTL_MINUTES: int = 15
    JWT_REFRESH_TTL_DAYS: int = 7

    # Set to false to disable TLS certificate verification on all Bedrock
    # model clients (chat, converse, embeddings). Useful behind a TLS-
    # intercepting proxy with a self-signed CA. Leave true in production.
    BEDROCK_SSL_VERIFY: bool = True

    # Bedrock concurrency cap per model
    BEDROCK_CONCURRENCY: int = 20
    # Max simultaneous ingest jobs across the whole process
    INGEST_CONCURRENCY: int = 10
    # Max simultaneous page writes within a single ingest job
    WRITE_PAGE_CONCURRENCY: int = 10
    # Complexity guards — files exceeding either limit are rejected before any LLM call
    MAX_INGEST_CHARS: int = 300_000        # ~10 LLM planning passes at default chunk size
    MAX_HTML_TABLE_CELLS: int = 100        # total <td>/<th> tags across the whole file

    # Per-user rate limits (requests/minute) for key endpoints
    RATE_LIMIT_UPLOAD_PER_MINUTE: int = 10
    RATE_LIMIT_QUERY_PER_MINUTE: int = 30
    RATE_LIMIT_CHAT_PER_MINUTE: int = 60
    RATE_LIMIT_RECALIBRATE_PER_MINUTE: int = 2

    # Writer mode: TTL (in days) for inactive writer-mode chat sessions. Expired
    # sessions are hidden from the draft picker and actively pruned by the daily loop.
    WRITER_DRAFT_TTL_DAYS: int = 30

    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR

    DATABASE_URL: str = "postgresql+asyncpg://wiki:wiki@localhost:5432/wiki"

    # Storage backend: "local" = data/ folder on disk; "s3" = AWS S3
    STORAGE_BACKEND: str = "local"
    DATA_DIR: str = "./data"        # Used when STORAGE_BACKEND=local
    DATA_BUCKET: str = ""           # S3 bucket name; used when STORAGE_BACKEND=s3
    AWS_ENDPOINT_URL: str = ""      # Override S3 endpoint (advanced); usually empty

    model_config = {"env_file": ".env", "extra": "ignore"}

    @field_validator("LLM_BASE_URL")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip()
        if value:
            url = urlsplit(value)
            if (url.scheme not in {"http", "https"} or not url.hostname
                    or url.username is not None or url.password is not None
                    or url.query or url.fragment):
                raise ValueError("LLM_BASE_URL must be an HTTP(S) endpoint without credentials, query or fragment")
        return value

    def model_post_init(self, __context) -> None:
        # BEDROCK_WRITER_MODEL_ID is older still — fold it into the var that
        # replaced it before the legacy promotion below runs.
        if self.BEDROCK_WRITER_MODEL_ID and not self.BEDROCK_INGEST_WRITER_MODEL_ID:
            object.__setattr__(self, "BEDROCK_INGEST_WRITER_MODEL_ID", self.BEDROCK_WRITER_MODEL_ID)

        # Promote each legacy raw-model-ID var onto its MODEL_* replacement, so
        # a .env written before the name catalogue keeps working untouched.
        # An explicitly-set MODEL_* always wins — that's the newer intent —
        # which is why this tests model_fields_set rather than truthiness: the
        # MODEL_* vars have non-empty defaults, so "is it set" and "is it
        # non-empty" are different questions.
        for legacy, current in LEGACY_MODEL_VARS.items():
            value = getattr(self, legacy, "")
            explicitly_disabled = legacy == "BEDROCK_EMBEDDING_MODEL_ID" and legacy in self.model_fields_set
            if (value or explicitly_disabled) and current not in self.model_fields_set:
                object.__setattr__(self, current, value)

        # Explicit role settings (including legacy ones) outrank the shared default.
        if self.MODEL_DEFAULT.strip():
            for legacy, current in LEGACY_MODEL_VARS.items():
                if (current != "MODEL_EMBEDDING" and current not in self.model_fields_set
                        and not getattr(self, legacy, "")):
                    object.__setattr__(self, current, self.MODEL_DEFAULT.strip())

        # Refuse to boot in production while security-sensitive settings are
        # left at their shipped defaults. These defaults are convenient for
        # local development but are public knowledge, so they must be changed
        # before the app is exposed. Set ENVIRONMENT=production to enforce.
        if self.ENVIRONMENT.strip().lower() == "production":
            insecure_defaults = {
                "JWT_SECRET": {"change-me-in-production"},
                "AUTH_PASSWORD": {"admin", "changeme"},
            }
            insecure = [
                name
                for name, defaults in insecure_defaults.items()
                if getattr(self, name) in defaults
            ]
            if insecure:
                raise ValueError(
                    "Refusing to start with ENVIRONMENT=production while these "
                    f"settings are left at insecure defaults: {', '.join(insecure)}. "
                    "Set strong, unique values in your environment or .env file "
                    "before deploying."
                )


settings = Settings()

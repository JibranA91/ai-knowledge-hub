from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    ASSUMED_ROLE_ARN: str = ""
    ASSUMED_ROLE_SESSION_NAME: str = "WikiAgentSession"
    ASSUMED_ROLE_DURATION: int = 3600
    BEDROCK_INGEST_MODEL_ID: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    # Page renderer used inside the ingest pipeline — writes the markdown body
    # of an individual planned page. Single-shot, no reasoning loop.
    BEDROCK_INGEST_WRITER_MODEL_ID: str = "us.meta.llama4-maverick-17b-instruct-v1:0"
    BEDROCK_QUERY_MODEL_ID: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    BEDROCK_RECALIBRATE_MODEL_ID: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    # Conversational AI Writer agent (the chat-driven document authoring
    # flow). Multi-turn reasoner — asks clarifying questions, drafts the page,
    # accepts revisions. Defaults to the query model.
    BEDROCK_DRAFT_AGENT_MODEL_ID: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    # Inline AI editor — rewrites a whole page or one section against an
    # instruction. Single-shot (no reasoning loop); defaults to the draft model.
    BEDROCK_EDIT_MODEL_ID: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    # ── Back-compat for the old name. If BEDROCK_WRITER_MODEL_ID is set in
    # the environment, pydantic-settings will populate this; we copy it onto
    # BEDROCK_INGEST_WRITER_MODEL_ID in model_post_init below.
    BEDROCK_WRITER_MODEL_ID: str = ""
    # optional semantic embeddings. Leave empty to disable vector search.
    # Recommended: amazon.titan-embed-text-v2:0 (1536-dim) or cohere.embed-english-v3 (1024-dim).
    BEDROCK_EMBEDDING_MODEL_ID: str = "amazon.titan-embed-text-v1"
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

    def model_post_init(self, __context) -> None:
        # If the legacy var BEDROCK_WRITER_MODEL_ID is set in the environment,
        # promote it onto the new name so existing deployments keep working.
        if self.BEDROCK_WRITER_MODEL_ID:
            object.__setattr__(self, "BEDROCK_INGEST_WRITER_MODEL_ID", self.BEDROCK_WRITER_MODEL_ID)

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

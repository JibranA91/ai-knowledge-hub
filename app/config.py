from urllib.parse import urlsplit

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings

from app.providers.base import UnknownModelError

TEXT_MODEL_SETTINGS = (
    "MODEL_INGEST_PLAN", "MODEL_INGEST_WRITE", "MODEL_QUERY",
    "MODEL_RECALIBRATE", "MODEL_DRAFT_AGENT", "MODEL_EDIT",
)


def _reject_obsolete_models(values):
    obsolete = sorted(str(key).upper() for key in values
                      if str(key).upper().startswith("BEDROCK_") and str(key).upper().endswith("_MODEL_ID"))
    if obsolete:
        raise UnknownModelError(
            f"Obsolete model settings: {', '.join(obsolete)}. Remove these keys from the environment and .env; "
            "select friendly names with MODEL_DEFAULT and MODEL_* using app/model_catalog.yaml.")


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
    MODEL_DEFAULT: str = "haiku45"

    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    ASSUMED_ROLE_ARN: str = ""
    ASSUMED_ROLE_SESSION_NAME: str = "WikiAgentSession"
    ASSUMED_ROLE_DURATION: int = 3600
    # Friendly names only; provider IDs and capabilities live in model_catalog.yaml.
    # Omitted text roles inherit MODEL_DEFAULT. Explicit blanks are invalid.
    MODEL_INGEST_PLAN: str = ""
    MODEL_INGEST_WRITE: str = ""
    MODEL_QUERY: str = ""
    MODEL_RECALIBRATE: str = ""
    MODEL_DRAFT_AGENT: str = ""
    MODEL_EDIT: str = ""
    # Embeddings are opt-in and never inherit the text default.
    MODEL_EMBEDDING: str = ""

    # Cross-region inference profile geography for Bedrock text models: "us",
    # "eu", "global", … It becomes the model ID's prefix (us.anthropic.…).
    # Leave empty to call foundation models directly with no profile.
    BEDROCK_INFERENCE_GEO: str = "us"

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

    model_config = {"env_file": ".env", "extra": "ignore", "hide_input_in_errors": True}

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                   dotenv_settings, file_secret_settings):
        # Environment sources discard unknown fields. Check their original keys
        # first, including obsolete keys shadowed by newer settings or left blank.
        _reject_obsolete_models(init_settings.init_kwargs)
        _reject_obsolete_models(env_settings.env_vars)
        _reject_obsolete_models(dotenv_settings.env_vars)
        return init_settings, env_settings, dotenv_settings, file_secret_settings

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
        for name in TEXT_MODEL_SETTINGS:
            if name not in self.model_fields_set:
                object.__setattr__(self, name, self.MODEL_DEFAULT.strip())

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

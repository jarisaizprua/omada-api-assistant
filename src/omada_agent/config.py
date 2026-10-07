"""Central configuration.

Every tunable value lives in the ``.env`` file (see ``.env_template``) and is
validated once, at start-up, by Pydantic. Nothing else in the code base reads
environment variables directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]

# Optional values that are allowed to be present-but-empty in the .env file
# (for example ``OPENAI_TEMPERATURE=``). Empty strings are converted to ``None``.
_NULLABLE_FIELDS = (
    "openai_api_key",
    "openai_base_url",
    "openai_embedding_dimensions",
    "openai_reasoning_effort",
    "openai_temperature",
    "openai_max_output_tokens",
    "portal_access_token",
    "openai_judge_model",
    "eval_price_input_per_mtok",
    "eval_price_output_per_mtok",
    "price_input_per_mtok",
    "price_output_per_mtok",
    "portal_user_header",
    "log_file",
)


class ConfigError(RuntimeError):
    """Raised when a required setting is missing. The message is user-facing."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- OpenAI ---------------------------------------------------------------
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-5.5"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int | None = None
    openai_reasoning_effort: str | None = None
    openai_temperature: float | None = None
    openai_max_output_tokens: int | None = None
    openai_timeout_seconds: float = 120.0
    openai_max_retries: int = 3

    # --- Knowledge base (RAG) ---------------------------------------------------
    spec_path: str = "data/spec"
    guides_dir: str = "data/guides"
    index_dir: str = "data/index"
    rag_top_k: int = 8
    rag_candidates: int = 40
    rag_rrf_k: int = 60
    rag_query_expansion: bool = True

    # --- Agent ----------------------------------------------------------------
    agent_max_tool_rounds: int = 6
    agent_guide_prefetch: bool = True
    chat_max_history_messages: int = 20
    chat_max_message_chars: int = 8000

    # --- Evaluations ----------------------------------------------------------
    openai_judge_model: str | None = None  # model that grades answers (default: OPENAI_MODEL)
    eval_concurrency: int = 3
    eval_retrieval_samples: int = 300
    eval_price_input_per_mtok: float | None = None
    eval_price_output_per_mtok: float | None = None
    evals_dir: str = "evals"

    # --- Web server -----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = ""
    portal_access_token: SecretStr | None = None
    rate_limit_per_minute: int = 20
    # Who is asking? Behind a login proxy this header carries the user's e-mail (e.g. X-Forwarded-Email).
    # Only turn it on when the proxy sets it and strips any value sent by the client.
    portal_user_header: str | None = None
    # True when a reverse proxy sits in front: the client IP is then read from X-Forwarded-For.
    portal_behind_proxy: bool = False
    # Comma-separated identities allowed to see everybody's usage (empty = every signed-in user can).
    portal_admins: str = ""

    # --- Usage and spending limits (0 = limit off) ------------------------------
    usage_db_path: str = "data/usage/usage.db"
    usage_timezone: str = "UTC"
    budget_daily_usd: float = 5.0
    budget_monthly_usd: float = 50.0
    budget_user_daily_usd: float = 1.0
    user_daily_requests: int = 200
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None

    # --- Logging --------------------------------------------------------------
    log_format: str = "console"  # console | json
    log_file: str | None = None
    log_questions: bool = False  # never write the text of questions to the logs unless asked

    @field_validator(*_NULLABLE_FIELDS, mode="before")
    @classmethod
    def _empty_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # --- Helpers --------------------------------------------------------------
    @staticmethod
    def resolve(path: str | Path) -> Path:
        """Resolve a configured path relative to the project root."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else (ROOT_DIR / p).resolve()

    @property
    def spec_location(self) -> Path:
        return self.resolve(self.spec_path)

    @property
    def guides_location(self) -> Path:
        return self.resolve(self.guides_dir)

    @property
    def evals_location(self) -> Path:
        return self.resolve(self.evals_dir)

    @property
    def index_location(self) -> Path:
        return self.resolve(self.index_dir)

    @property
    def usage_db_location(self) -> Path:
        return self.resolve(self.usage_db_path)

    @property
    def admin_list(self) -> set[str]:
        return {a.strip().lower() for a in self.portal_admins.split(",") if a.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())

    @property
    def access_token(self) -> str | None:
        if self.portal_access_token and self.portal_access_token.get_secret_value().strip():
            return self.portal_access_token.get_secret_value().strip()
        return None

    def require_openai_key(self) -> str:
        if not self.has_openai_key:
            raise ConfigError(
                "OPENAI_API_KEY is not set. Copy .env_template to .env and add your key."
            )
        return self.openai_api_key.get_secret_value().strip()  # type: ignore[union-attr]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

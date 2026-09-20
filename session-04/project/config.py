"""Configuration for the ShopWise assistant — project version v1.0.

Same idea as v0.2 (a Pydantic model guarding the environment boundary), plus what
v1 needs: the vendor is now a field, `PROVIDER`, and `model_id` joins it with the
model into LangChain's one string:

    init_chat_model("google_genai:gemini-3.1-flash-lite")
                     └─ provider ─┘ └────── model ──────┘

Moving ShopWise to another vendor is two lines in labs/.env and no code change.
Two more things arrive for tracing: `PROMPT_VERSION` (an override for the label
stamped on every run) and the three `LANGSMITH_*` fields.

Every field is read from `SHOPWISE_<FIELD>`. The exceptions keep their plain
names because another library reads them straight from os.environ: Google's SDK
looks for GEMINI_API_KEY, the LangSmith SDK for LANGSMITH_TRACING / _API_KEY /
_PROJECT. Other vendors' keys (GROQ_API_KEY, OPENAI_API_KEY, ...) just need to be
in .env; their clients read them themselves.

    uv run python config.py      # prints the resolved configuration
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# labs/session-04/project/config.py -> labs/
LABS = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Everything the assistant can be told that is not code. Read top to bottom."""

    model_config = SettingsConfigDict(
        env_file=LABS / ".env",
        env_file_encoding="utf-8",
        env_prefix="SHOPWISE_",
        case_sensitive=True,
        extra="ignore",     # the shared .env also holds keys for later sessions
    )

    # Identity
    APP_NAME: str = "ShopWise Support Assistant"
    VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # Credentials. SecretStr prints as '**********'; read it with .get_secret_value().
    GEMINI_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="GEMINI_API_KEY",   # no SHOPWISE_ prefix — the SDK's own name
        description="Google AI Studio key. Free: https://aistudio.google.com/apikey",
    )

    # Provider: who SERVES the model (not who made it) — the left half of the provider
    # string. LangChain's ids use underscores: google_genai, groq, ollama, openai, anthropic.
    #   SHOPWISE_PROVIDER=groq  +  SHOPWISE_MODEL=openai/gpt-oss-20b   -> OpenAI's open
    #   weights, served by Groq, and no code changes.
    PROVIDER: str = "google_genai"

    # Models. Names are fixed for the whole course; code says settings.MODEL and a
    # model ID is never spelled out at a call site.
    MODEL: str = "gemini-3.1-flash-lite"        # the course default: fast, cheap, free tier
    MODEL_STRONG: str = "gemini-3.6-flash"      # one tier up, for tasks the default fails
    MODEL_SEARCH: str = "gemini-3.5-flash-lite" # search grounding — billed key only (session 12)
    MODEL_PRO: str = "gemini-2.5-pro"           # strongest and slowest (session 9 evals)
    EMBED_MODEL: str = "gemini-embedding-2"     # embeddings (session 8)

    # Generation
    MAX_OUTPUT_TOKENS: int = Field(default=1024, ge=64, le=8192,
                                   description="Too low truncates JSON mid-object.")
    REQUEST_TIMEOUT_S: float = Field(default=60.0, gt=0,
                                     description="Give up on one call after this many seconds.")
    MAX_RETRIES: int = Field(default=3, ge=1, le=6,
                             description="Attempts for transient failures only.")

    # Prompt version OVERRIDE. Empty means "whatever prompts.yml says" — the file is the
    # source of truth (ADR-0005), because the version belongs with the wording and moves in
    # the same commit. Set SHOPWISE_PROMPT_VERSION=v2-x for one run to label an A/B arm.
    PROMPT_VERSION: str = ""

    # Few-shot examples in the triage prompt. Measured with ../score_triage.py:
    #   rules only .......... 18/20   <- shipped
    #   rules + examples .... 18/20   <- bought nothing, costs tokens every call
    USE_FEW_SHOT: bool = False

    # Tracing. Plain names on purpose: the LangSmith SDK reads these from os.environ
    # itself. Mirrored here so `python config.py` can say whether runs are recorded.
    LANGSMITH_TRACING: bool = Field(
        default=False, validation_alias="LANGSMITH_TRACING",
        description="Record every LLM call to LangSmith. Leave off until a key is in .env.",
    )
    LANGSMITH_PROJECT: str = Field(
        default="shopwise", validation_alias="LANGSMITH_PROJECT",
        description="Which LangSmith project traces land in (your own workspace).",
    )
    LANGSMITH_API_KEY: SecretStr = Field(
        default=SecretStr(""), validation_alias="LANGSMITH_API_KEY",
        description="Free at https://smith.langchain.com. Optional.",
    )

    # Paths. One copy of the fixtures for the whole course.
    DATA_DIR: Path = LABS / "data"

    @field_validator("MODEL", "MODEL_STRONG", "MODEL_SEARCH", "MODEL_PRO", "EMBED_MODEL")
    @classmethod
    def no_blank_model_ids(cls, value: str) -> str:
        """`SHOPWISE_MODEL=` (empty) should fail here, not as a 404 from the API."""
        value = value.strip()
        if not value:
            raise ValueError("must not be empty — remove the line or give it a model ID")
        return value

    @field_validator("PROVIDER")
    @classmethod
    def provider_ids_use_underscores(cls, value: str) -> str:
        """Catch `google-genai` (the pip package name) — LangChain's own error for it
        points at the model half of the string and costs people an evening."""
        value = value.strip()
        if not value:
            raise ValueError("must not be empty — e.g. google_genai, groq, ollama")
        if "-" in value:
            raise ValueError(f"{value!r} is a package name, not a provider id — "
                             f"did you mean {value.replace('-', '_')!r}?")
        return value

    @model_validator(mode="after")
    def production_needs_a_key(self) -> Settings:
        """Development may run keyless (outputs are committed); production may not.
        Scoped to Gemini, the course default; widen it when you deploy on another vendor."""
        if (self.ENVIRONMENT == "production"
                and self.PROVIDER == "google_genai"
                and not self.GEMINI_API_KEY.get_secret_value()):
            raise ValueError("GEMINI_API_KEY is required when ENVIRONMENT=production")
        return self

    @property
    def model_id(self) -> str:
        """`<provider>:<model>` — the one string init_chat_model takes. LangChain
        splits on the FIRST colon, so `ollama:llama3.2:3b` works."""
        return self.model_id_for(self.MODEL)

    def model_id_for(self, model: str) -> str:
        """The same join for the other tiers, e.g. settings.model_id_for(settings.MODEL_STRONG)."""
        return f"{self.PROVIDER}:{model}"

    @property
    def has_key(self) -> bool:
        return bool(self.GEMINI_API_KEY.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    """Build the settings once; every caller gets the same object.

    load_dotenv is here because the Gemini client and the LangSmith tracer read their
    variables straight from os.environ — importing config is what switches tracing on.
    """
    load_dotenv(LABS / ".env", override=True)
    return Settings()


settings = get_settings()


if __name__ == "__main__":
    print(f"{settings.APP_NAME} v{settings.VERSION}  [{settings.ENVIRONMENT}]")
    for name, value in settings.model_dump().items():
        print(f"  {name:20} {value}")
    print(f"  {'model_id':20} {settings.model_id}  (computed)")
    print(f"\nAPI key present: {settings.has_key}")
    print(f"Tracing: {settings.LANGSMITH_TRACING}"
          f"  ·  LangSmith key present: {bool(settings.LANGSMITH_API_KEY.get_secret_value())}"
          f"  ·  project: {settings.LANGSMITH_PROJECT}")

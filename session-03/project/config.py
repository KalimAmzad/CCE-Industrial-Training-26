"""Configuration for the ShopWise assistant — project version v0.2.

Session 2 read settings with `os.getenv(...)`. That breaks the moment a setting
is not a string: `USE_FEW_SHOT=false` in .env arrives as the string "false",
which is truthy. So from v0.2 configuration is a Pydantic model — the same
library as schemas.py, guarding the other boundary:

    schemas.py   validates what comes IN from the model
    config.py    validates what comes IN from the environment

Every field is read from `SHOPWISE_<FIELD>` (SHOPWISE_MODEL, SHOPWISE_USE_FEW_SHOT).
The prefix keeps our settings apart from every other program that also wants a
variable called MODEL. `GEMINI_API_KEY` keeps its plain name because Google's SDK
looks for it.

    uv run python config.py      # prints the resolved configuration
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# labs/session-03/project/config.py -> labs/
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
    VERSION: str = "0.2.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # Credentials. SecretStr prints as '**********' so the key never leaks into a
    # log or a screen share; read it deliberately with .get_secret_value().
    GEMINI_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="GEMINI_API_KEY",   # no SHOPWISE_ prefix — the SDK's own name
        description="Google AI Studio key. Free: https://aistudio.google.com/apikey",
    )

    # Models. Names are fixed for the whole course; code says settings.MODEL and a
    # model ID is never spelled out at a call site. Retiring one = one line in .env.
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

    # Few-shot examples in the triage prompt. Measured with ../score_triage.py:
    #   rules only .......... 18/20   <- shipped
    #   rules + examples .... 18/20   <- bought nothing, costs tokens every call
    # Flip it for one run with SHOPWISE_USE_FEW_SHOT=true.
    USE_FEW_SHOT: bool = False

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

    @model_validator(mode="after")
    def production_needs_a_key(self) -> Settings:
        """Development may run keyless (outputs are committed); production may not."""
        if self.ENVIRONMENT == "production" and not self.GEMINI_API_KEY.get_secret_value():
            raise ValueError("GEMINI_API_KEY is required when ENVIRONMENT=production")
        return self

    @property
    def has_key(self) -> bool:
        return bool(self.GEMINI_API_KEY.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    """Build the settings once; every caller gets the same object.

    load_dotenv is here because the Google SDK reads GEMINI_API_KEY straight from
    os.environ — the variable must exist there, not only inside this object.
    """
    load_dotenv(LABS / ".env", override=True)
    return Settings()


settings = get_settings()


if __name__ == "__main__":
    print(f"{settings.APP_NAME} v{settings.VERSION}  [{settings.ENVIRONMENT}]")
    for name, value in settings.model_dump().items():
        print(f"  {name:20} {value}")
    print(f"\nAPI key present: {settings.has_key}")

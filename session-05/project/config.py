"""Configuration for the ShopWise assistant — project version v2.0.

WHAT CHANGED FROM v1 — NOTHING, AND THAT IS WORTH A SENTENCE
    v2 gives the assistant hands: tools.py and agent.py are new, and the
    assistant now answers from looked-up facts rather than from whatever the
    model remembers. None of that needed a new setting.

    The one field it leans on, DATA_DIR, has been here since v0.2 holding the
    path to data/ — where tickets.yml and prices.yml live, and now
    backend.yml with it (ADR-0006). shopwise_data.py reads through this object
    like every other module, so no file in the snapshot does its own
    `.parent.parent` arithmetic and no path is spelled out twice.

    A version bump with no new configuration is a good sign, not a boring one.
    It means the capability arrived behind an interface that already existed.

WHAT CHANGED FROM v0.2, AND WHY IT IS NOT COSMETIC
    v0.2 talked to Google, and only to Google. The choice of vendor was made by
    an import at the top of triage.py — `from google import genai` — so there
    was nothing here to configure about it, because there was nothing you could
    change without rewriting the file.

    v1 runs on LangChain, and LangChain names a model with a single string:

        init_chat_model("google_genai:gemini-3.1-flash-lite")
                         └─ provider ─┘ └────── model ──────┘

    That string is the whole vendor decision, and a string is data. So what used
    to be an import statement becomes a field — PROVIDER — and `model_id` joins
    the two halves back together. Moving the entire assistant from Gemini to
    Groq is now one line in .env and no code change anywhere. That is the
    claim session 4 makes, and this file is what makes it a fact rather than a
    slogan.

    Two smaller fields arrive with it, for one shared reason: from v1 every call
    is traced, and a trace you cannot attribute is just a screenshot.
    PROMPT_VERSION records which wording produced an answer; the LANGSMITH_*
    fields say where traces go and whether they are collected at all.

WHAT CHANGED FROM v0.1 — WHY CONFIGURATION IS A PYDANTIC MODEL AT ALL
    Session 2's config.py was a list of module-level constants read with
    `os.getenv("MODEL", "gemini-3.1-flash-lite")`. That is the right amount of
    machinery for a program with three settings and one developer. It stops
    being right the moment any of these are true, and by v0.2 two of them are:

      - a setting is not a string. `USE_FEW_SHOT=false` in a .env file arrives
        as the *string* "false", which is truthy. `os.getenv` will hand you a
        switch that is stuck in the on position and never say a word about it.
      - a setting is required. A missing key should stop the program at startup
        with a sentence a human can read, not two hundred lines later inside an
        SDK, on the one ticket that mattered.
      - more than one person edits the .env. Then "what settings exist?" needs
        an answer that is a file, not a grep for `getenv` across the repo.

    So configuration becomes a Pydantic model. This is the same library, on the
    same day, as schemas.py — and that symmetry is the lesson:

        schemas.py   validates what comes IN from the model
        config.py    validates what comes IN from the environment

    Both are boundaries where data you did not write enters a program you are
    responsible for. Pydantic is how you stand at a boundary and check.

THE PAYOFF, CONCRETELY
    Every value below is typed, defaulted, documented and range-checked in one
    place. `settings.MODEL` autocompletes. `settings.USE_FEW_SHOT` is a real
    bool. A typo'd ENVIRONMENT fails on import with the field name and the
    allowed values. Nothing downstream ever calls `os.getenv` again.

ENV VAR NAMES
    Every field is read from `SHOPWISE_<FIELD>` — `SHOPWISE_MODEL`,
    `SHOPWISE_PROVIDER`, `SHOPWISE_USE_FEW_SHOT`. The prefix is not decoration:
    `MODEL` and `DEBUG` are names half the software on your machine also wants,
    and an unprefixed setting is a variable you will one day inherit from a
    shell you forgot you exported in. Namespace your environment.

    `GEMINI_API_KEY` and the three `LANGSMITH_*` fields are the exceptions, and
    both exceptions have the same single cause: another library reads those
    variables straight out of `os.environ`, under a name it chose, without ever
    being handed this object. Google's SDK looks for GEMINI_API_KEY; the
    LangSmith SDK looks for LANGSMITH_TRACING, LANGSMITH_API_KEY and
    LANGSMITH_PROJECT. Prefixing them would mean two variables per setting that
    have to agree, and one afternoon they would not. `validation_alias` is how a
    field opts out of the prefix and keeps the name its library already hunts
    for, so there is exactly one line in your .env per thing.

    Read the LANGSMITH_* fields as *mirrors*, then. This file does not configure
    LangSmith; it reports what LangSmith is about to do — which is precisely
    what you want `uv run python config.py` to be able to tell you at 11pm.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# session-05/project/config.py -> repo root
LABS = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Everything the assistant needs to know that is not code.

    Read the field list top to bottom and you know what this program can be
    told to do. That is the point of having one.
    """

    model_config = SettingsConfigDict(
        env_file=LABS / ".env",
        env_file_encoding="utf-8",
        env_prefix="SHOPWISE_",
        case_sensitive=True,
        # A .env shared by sixteen sessions holds keys this snapshot has never
        # heard of (QDRANT_*, OPENAI_API_KEY for the authoring tools). "ignore"
        # lets them coexist. Use "forbid" in a service you own end to end — then
        # a typo'd variable name is a startup error instead of a setting that
        # silently never applied.
        extra="ignore",
    )

    # ── Identity ────────────────────────────────────────────────────────────
    APP_NAME: str = "ShopWise Support Assistant"
    VERSION: str = "2.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # ── Credentials ─────────────────────────────────────────────────────────
    # SecretStr prints as `SecretStr('**********')`. It will not leak into a log
    # line, a traceback, or a `print(settings)` during a screen share. Read it
    # deliberately with .get_secret_value() — which is the point: the only way
    # to see the key is to write code that says you meant to.
    GEMINI_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="GEMINI_API_KEY",   # no SHOPWISE_ prefix — the SDK's own name
        description="Google AI Studio key. Free: https://aistudio.google.com/apikey",
    )

    # There is deliberately no GROQ_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
    # field here, and the omission is the same lesson as the line above. Each
    # vendor's client reads its own key from os.environ under its own name, so
    # putting the key in .env is the whole of the work. A field would only
    # earn its place if this program had to *reason* about the key — as it does
    # for Gemini, in `has_key` and in the production check below.

    # ── Provider ────────────────────────────────────────────────────────────
    PROVIDER: str = "google_genai"
    """Who serves the model — the left half of LangChain's provider string.

    The five this course teaches, and what each costs you to run:

        google_genai   Gemini, via AI Studio      free tier — the course default
        groq           open weights, hosted       free tier, and very fast
        ollama         models on your own laptop  no key, no network, no bill
        openai         GPT                        billed from the first token
        anthropic      Claude                     billed from the first token

    Legal values are LangChain's own provider ids, not the vendors' brand names
    and not their pip package names. All 27 of them use underscores; ask for the
    list any time by passing a nonsense string to `init_chat_model` — the
    ValueError prints every supported provider, which is a better source than
    any comment that can go stale.

    THE TRAP, AND IT HAS COST PEOPLE AN EVENING: the separator inside a provider
    id is an UNDERSCORE. `google_genai` is right. `google-genai` — which is the
    name of the pip package, and the spelling every JavaScript example uses —
    raises `ValueError: Unable to infer model provider`, an error that points at
    the wrong half of the string and sends you off checking your model id. The
    package is hyphenated; the provider id is not. The validator below turns
    that mistake into a sentence at startup rather than a puzzle at call time.

    Changing this line changes the vendor for the entire program — that is the
    claim of session 4. Changing it also means changing MODEL to match, because
    a Gemini model id means nothing to Groq, and putting that vendor's key in
    .env. Two lines, still no code:

        SHOPWISE_PROVIDER=groq
        SHOPWISE_MODEL=openai/gpt-oss-20b     # yes: OpenAI's open weights, served by Groq

    That last example is worth a second look. The provider is who *serves* the
    model, not who *made* it — which is exactly the separation a provider string
    exists to express.
    """

    # ── Models ──────────────────────────────────────────────────────────────
    # Names are fixed for the whole course. Code says settings.MODEL; a model ID
    # is never spelled out at a call site. When Google retires one, the fix is a
    # line in .env and nothing is redeployed.
    MODEL: str = "gemini-3.1-flash-lite"
    """The course default: fast, cheap, on the free tier. Start here always."""

    MODEL_STRONG: str = "gemini-3.6-flash"
    """One tier up. Reach for it only when you can name a task the default fails."""

    MODEL_SEARCH: str = "gemini-3.5-flash-lite"
    """Search-grounded answers. Billed key only — not used before session 12."""

    MODEL_PRO: str = "gemini-2.5-pro"
    """Strongest and slowest. Used sparingly, from session 9's evaluation work."""

    EMBED_MODEL: str = "gemini-embedding-2"
    """Embeddings. Not used until session 8 — the name is fixed now so every
    snapshot's config.py has the same shape."""

    # ── Generation ──────────────────────────────────────────────────────────
    MAX_OUTPUT_TOKENS: int = Field(
        default=1024, ge=64, le=8192,
        description="Ceiling on one reply. Too low truncates JSON mid-object — see the notebook.",
    )
    REQUEST_TIMEOUT_S: float = Field(
        default=60.0, gt=0,
        description="Give up on one call. Without this, a hung connection hangs the whole batch.",
    )
    MAX_RETRIES: int = Field(
        default=3, ge=1, le=6,
        description="Attempts for a TRANSIENT failure only. Retrying a bug just runs it again.",
    )

    # ── Prompts ─────────────────────────────────────────────────────────────
    PROMPT_VERSION: str = ""
    """An OVERRIDE for the prompt version. Empty means "whatever prompts.yml says".

    The name of the current wording is stamped onto every traced run from v1, so
    a bad answer in LangSmith can be traced back to the exact wording that
    produced it. Without it a trace tells you what the model said and nothing
    about why — and "why" is almost always the prompt.

    WHAT CHANGED IN v2 — IT NOW HAS TO SAY *WHICH* PROMPT
        v1 had one `prompt_version` key covering the whole of prompts.yml, so an
        override was a bare label. v2 gives every prompt its own
        `<name>_version` key beside the wording it names (ADR-0005, amended
        2026-09-01), so an override has to name its target:

            SHOPWISE_PROMPT_VERSION=triage_prompt:v2-ordered

        `triage.prompt_version(prompt)` raises a ValueError on the bare form
        rather than guessing. An override that renamed every prompt at once
        would recreate exactly the bug the amendment fixes — one wording under
        two names — and would do it only during the run you were trying to
        measure.

    THE PRECEDENCE, AND WHY IT GOES THIS WAY ROUND
        The version is written in `prompts.yml`, directly above the wording it
        names, and that file wins. This field is empty by default; a prompt with
        no version in the file fails at import rather than falling back to
        anything here.

        The reason is ADR-0005's first rule: the prompt file in git is the single
        source of truth. A version is a property of the *wording*, so it has to
        move in the same edit, in the same file, in the same commit — and a
        default here would be a second place the version lives. Two places
        disagree eventually, and a stamp that lags the wording is worse than no
        stamp, because by then you trust it.

        The field survives as an override because an A/B needs to *label an arm*
        without editing the file it is measuring — one run, one shell, one
        prompt renamed. (../score_triage.py takes the other route for its
        candidate arm: it passes `version=` straight through `stamp()`, which
        needs no environment at all. This field is for the times you are not
        editing the harness.)

    It is a label a human chooses, not a hash and not a timestamp, because its
    whole job is to be sayable out loud: "the v2 prompt regressed refunds." It
    does not track VERSION above — the project version moves when capability
    moves, this moves when *wording* moves, and they are rarely the same commit.

    Git remains the real audit trail — `git log -p prompts.yml` is free and you
    already have it — and this is the cheap label that lets a trace point INTO
    that history.
    """

    # ── Behaviour, measured ─────────────────────────────────────────────────
    # Scored over all 20 hand-labelled tickets (../score_triage.py):
    #     bare instruction only .................... 15/20
    #     + the conventions written out as rules ... 18/20   <- the cheap win, shipped
    #     + three few-shot examples ................ 18/20   <- bought nothing
    # The examples fixed one ticket and broke another, for tokens resent on every
    # call forever. So: off. It lives here rather than as a constant in triage.py
    # because a flag with a measurement attached is a decision someone may revisit
    # with different data — and they should be able to revisit it with
    # SHOPWISE_USE_FEW_SHOT=true, not with a pull request.
    USE_FEW_SHOT: bool = False

    # ── Tracing ─────────────────────────────────────────────────────────────
    # All three keep their unprefixed names. See ENV VAR NAMES in the module
    # docstring: the LangSmith SDK reads them from os.environ itself, so a
    # SHOPWISE_ copy would be a second variable that has to agree with the first.
    # They are mirrored here so that `uv run python config.py` can answer "are my
    # runs being recorded, and where?" without anyone opening a browser.
    LANGSMITH_TRACING: bool = Field(
        default=False,
        validation_alias="LANGSMITH_TRACING",
        description="Record every LLM call to LangSmith. Off until you have pasted a key.",
    )
    """Off by default, and deliberately so.

    Turning this on without a key does not fail loudly and does not fail quietly
    either — it warns `LangSmithMissingAPIKeyWarning` and then spends a
    background thread posting traces nobody receives. The default that costs a
    keyless reader nothing is False; session 4 is where you flip it, and feeling
    that one-line edit change the program's behaviour is part of the lesson.
    """

    LANGSMITH_PROJECT: str = Field(
        default="shopwise",
        validation_alias="LANGSMITH_PROJECT",
        description="Which LangSmith project traces land in. Yours alone — no naming scheme needed.",
    )
    """The folder traces land in, inside your own workspace.

    Everyone on the course can use "shopwise" because everyone has their own
    account. On a team, this is the field that stops one person's load test from
    burying another person's bug hunt — split by service, or by environment, and
    never point staging at the same project as production.
    """

    LANGSMITH_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="LANGSMITH_API_KEY",
        description="Free at https://smith.langchain.com. Optional — nothing in the course requires it.",
    )

    # ── Paths ───────────────────────────────────────────────────────────────
    # One copy of the fixtures for the whole course (ADR-0001). Resolved once,
    # here, so no module ever does its own `.parent.parent` arithmetic.
    DATA_DIR: Path = LABS / "data"

    @field_validator("MODEL", "MODEL_STRONG", "MODEL_SEARCH", "MODEL_PRO", "EMBED_MODEL")
    @classmethod
    def _no_blank_model_ids(cls, value: str) -> str:
        """Catch `SHOPWISE_MODEL=` — an empty line in a .env is a real mistake.

        Without this you get a 404 from the API naming the empty string, which
        is a much longer walk to the same conclusion.
        """
        value = value.strip()
        if not value:
            raise ValueError("must not be empty — remove the line or give it a model ID")
        return value

    @field_validator("PROVIDER")
    @classmethod
    def _provider_ids_use_underscores(cls, value: str) -> str:
        """Catch `SHOPWISE_PROVIDER=google-genai` here, where it is still cheap.

        LangChain's own message for the hyphenated form is "Unable to infer
        model provider for model='google-genai:gemini-3.1-flash-lite'", which is
        accurate and unhelpful: it reads as a complaint about the model, so
        people go and check the model. Failing at startup, naming the field and
        printing the corrected value, is the difference between a typo and an
        evening. No LangChain provider id contains a hyphen, so this rule has no
        false positives to worry about.
        """
        value = value.strip()
        if not value:
            raise ValueError("must not be empty — e.g. google_genai, groq, ollama")
        if "-" in value:
            fixed = value.replace("-", "_")
            raise ValueError(
                f"{value!r} is a package name, not a provider id. LangChain "
                f"provider ids use underscores — did you mean {fixed!r}?"
            )
        return value

    @model_validator(mode="after")
    def _production_needs_a_key(self) -> Settings:
        """A rule no single field can express, which is what model_validator is for.

        In development a missing key is fine: the notebooks ship with committed
        outputs and every cell degrades gracefully. In production it is an outage
        that has not happened yet.

        Scoped to Gemini on purpose. This snapshot's default provider is
        google_genai, and a rule that tried to police five vendors' key names
        would be guessing. When a deployment genuinely runs on another provider,
        the honest change is to widen this check with that provider's key — not
        to delete it.
        """
        if (self.ENVIRONMENT == "production"
                and self.PROVIDER == "google_genai"
                and not self.GEMINI_API_KEY.get_secret_value()):
            raise ValueError("GEMINI_API_KEY is required when ENVIRONMENT=production")
        return self

    @property
    def model_id(self) -> str:
        """The single string `init_chat_model` takes: `<provider>:<model>`.

        This is the one place the two halves are joined, which is the whole
        reason the property exists rather than an f-string at each call site:
        change the provider and every call in the program follows, including the
        ones written in a session you have not reached yet.

        The join is a plain colon, and LangChain splits on the FIRST one — so a
        model id that contains its own colon still works. `ollama:llama3.2:3b`
        resolves to provider `ollama`, model `llama3.2:3b`, which is what you
        want and is not obvious.
        """
        return self.model_id_for(self.MODEL)

    def model_id_for(self, model: str) -> str:
        """The same join for the other tiers: `settings.model_id_for(settings.MODEL_STRONG)`.

        Two lines of duplication saved is not the point. The point is that the
        strong and pro tiers move providers with everything else, instead of
        quietly staying on Gemini because someone hardcoded the colon once.
        """
        return f"{self.PROVIDER}:{model}"

    @property
    def has_key(self) -> bool:
        """True when a key is present. Cells and scripts branch on this rather
        than crashing, so a keyless reader still gets a useful message.

        Gemini's key specifically — it is the course default and the one every
        committed notebook output was produced with. If you have switched
        PROVIDER, this answers a question you are no longer asking; the vendor's
        own client will tell you soon enough, and loudly."""
        return bool(self.GEMINI_API_KEY.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    """Build the settings once and hand the same object to every caller.

    `lru_cache` is doing real work here. Without it, every `get_settings()` call
    re-reads the .env from disk and re-validates — cheap, but it also means two
    parts of your program can disagree about configuration if the file changed
    underneath them. One process, one config object.

    `load_dotenv` looks redundant next to `env_file=` above, and for Settings it
    is: pydantic-settings reads the file itself. It is here because other
    libraries read the environment directly and never see this object — the
    Gemini client picks up `GEMINI_API_KEY`, and from v1 the LangSmith tracer
    picks up `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`.
    Those variables have to actually exist in `os.environ`, not only as fields
    here. Importing config is therefore what switches tracing on.
    """
    load_dotenv(LABS / ".env", override=True)
    return Settings()


# The one import every other module wants: `from config import settings`.
settings = get_settings()


if __name__ == "__main__":
    # `uv run python config.py` — prints the resolved configuration. Worth having
    # in anything you deploy: the fastest way to answer "what does it think its
    # settings are?" without adding a print statement to something else.
    print(f"{settings.APP_NAME} v{settings.VERSION}  [{settings.ENVIRONMENT}]")
    for name, value in settings.model_dump().items():
        print(f"  {name:20} {value}")
    # Properties are computed, so model_dump() never sees them. Print this one
    # anyway: "what provider string is it actually going to build?" is the first
    # question a failed run raises, and v1 is the version that made it a question.
    print(f"  {'model_id':20} {settings.model_id}  (computed)")
    print(f"\nAPI key present: {settings.has_key}")
    print(f"Tracing: {settings.LANGSMITH_TRACING}"
          f"  ·  LangSmith key present: {bool(settings.LANGSMITH_API_KEY.get_secret_value())}"
          f"  ·  project: {settings.LANGSMITH_PROJECT}")

"""Helpers shared by every ShopWise script — project version v0.2.

Nothing in here knows about triage. It is the plumbing every LLM program needs:
a configured client, a guard against an empty answer, the fixtures, and a cost
line.
"""

from __future__ import annotations

import yaml
from google import genai
from google.genai import types

from config import settings


class ModelReturnedNothing(RuntimeError):
    """The call came back 200 OK but `.parsed` is None (cut off, blocked, ...)."""


def build_client() -> genai.Client:
    """One client for the whole program, with a timeout and the SDK's own retries."""
    return genai.Client(
        http_options=types.HttpOptions(
            timeout=int(settings.REQUEST_TIMEOUT_S * 1000),          # the SDK wants milliseconds
            retry_options=types.HttpRetryOptions(
                attempts=settings.MAX_RETRIES,
                http_status_codes=[408, 429, 500, 502, 503, 504],   # transient only: no 400, no 403
            ),
        ),
    )


def checked(response, what: str):
    """Return `response.parsed`, or raise with the reason it is missing.

    `.parsed` is None when the JSON was cut off (MAX_TOKENS), when the model
    declined (SAFETY), or when Pydantic rejected the answer. Treating None as
    "no result" silently drops the ticket — so we raise instead.
    """
    if response.parsed is not None:
        return response.parsed
    reason = "unknown"
    if response.candidates:
        reason = getattr(response.candidates[0].finish_reason, "name", "unknown")
    raise ModelReturnedNothing(f"{what}: no parsed result (finish_reason={reason})")


def load_tickets() -> list[dict]:
    return yaml.safe_load((settings.DATA_DIR / "tickets.yml").read_text(encoding="utf-8"))


def load_prices() -> dict[str, dict]:
    """Token prices per model, from data/prices.yml."""
    rows = yaml.safe_load((settings.DATA_DIR / "prices.yml").read_text(encoding="utf-8"))
    return {r["id"]: r for r in rows}


def as_ticket_block(ticket: dict) -> str:
    """Wrap the customer's words in <ticket> tags, so the model reads them as data,
    not as instructions (the first defence against prompt injection — session 7)."""
    return f"<ticket>\nSubject: {ticket['subject']}\n{ticket['body']}\n</ticket>"


def cost(usage, model: str, prices: dict) -> float:
    """USD for one call. Output tokens cost several times more than input tokens."""
    row = prices.get(model)
    if not row:
        return 0.0
    return (usage.prompt_token_count * row["input"]
            + usage.candidates_token_count * row["output"]) / 1_000_000

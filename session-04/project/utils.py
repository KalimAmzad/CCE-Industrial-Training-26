"""Helpers shared by every ShopWise script — project version v1.

Nothing in here knows about triage: the model object, the fixtures, the ticket
wrapper, and the content hash the trace stamp uses.
"""

from __future__ import annotations

import hashlib

import yaml
from langchain.chat_models import init_chat_model

from config import settings


def build_model():
    """One model object for the whole program.

    `settings.model_id` is the provider string ("google_genai:gemini-3.1-flash-lite")
    and the only place the vendor is named. Change SHOPWISE_PROVIDER / SHOPWISE_MODEL
    in .env and every call goes somewhere else. `timeout` is in seconds here;
    `max_retries` covers transport failures only.
    """
    return init_chat_model(
        settings.model_id,
        timeout=settings.REQUEST_TIMEOUT_S,
        max_retries=settings.MAX_RETRIES,
        max_tokens=settings.MAX_OUTPUT_TOKENS,
    )


def load_tickets() -> list[dict]:
    return yaml.safe_load((settings.DATA_DIR / "tickets.yml").read_text(encoding="utf-8"))


def as_ticket_block(ticket: dict) -> str:
    """Wrap the customer's words in <ticket> tags, so the model reads them as data,
    not as instructions (the first defence against prompt injection — session 7)."""
    return f"<ticket>\nSubject: {ticket['subject']}\n{ticket['body']}\n</ticket>"


def prompt_sha(*parts: str) -> str:
    """A 12-character hash of the prompt AS ASSEMBLED — not of the file, not the ticket.

    The version label is typed by a human and sometimes forgotten; the hash cannot
    be. Two runs with one prompt_version and two hashes mean the wording moved.
    """
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]

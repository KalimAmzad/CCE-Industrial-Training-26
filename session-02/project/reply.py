"""ShopWise support replier — project version v0.1.

    uv run python reply.py               # every ticket in the inbox
    uv run python reply.py --limit 3     # just the first three

What v0.1 is: a system prompt, a hand-built message history, and a loop over the
real inbox. That is the whole assistant. Everything after this session adds a
capability to this shape rather than replacing it.

What v0.1 is NOT, named honestly so the next sessions have something to fix:
  - free text only, so nothing downstream can sort or count it        -> S03
  - no knowledge of any ShopWise fact that isn't in the ticket        -> S08
  - guardrails are prompt-only, which is a soft boundary              -> S07
  - one provider, hand-written                                        -> S04

Importing this module does nothing. The client is created inside main(), so a
missing key is an error when you RUN it, not when something imports it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

import config

HERE = Path(__file__).resolve().parent
# labs/session-02/project/ -> labs/ . The fixture and the key live at the tree
# root: snapshots never keep their own copy of the data (ADR-0001).
LABS = HERE.parent.parent
TICKETS_PATH = LABS / "data" / "tickets.yml"
PROMPTS_PATH = HERE / "prompts.yml"


def load_system_prompt() -> str:
    """Read the 8-block system prompt out of prompts.yml."""
    return yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))["system_prompt"]


def load_tickets() -> list[dict]:
    """Read the shared support inbox."""
    return yaml.safe_load(TICKETS_PATH.read_text(encoding="utf-8"))


def as_message(ticket: dict) -> str:
    """Render one ticket the way a customer's email would actually arrive."""
    return f"Subject: {ticket['subject']}\n{ticket['body']}"


def answer(client, system_prompt: str, ticket: dict) -> str:
    """Send one ticket and return the reply text.

    The history list is built by hand and sent in full, because the API is
    stateless — it remembers nothing between calls. For v0.1 each ticket is a
    fresh one-turn conversation, so the list has a single entry; the shape is
    here because from session 5 it stops being a single entry.
    """
    from google.genai import types

    history = [types.Content(role="user", parts=[types.Part(text=as_message(ticket))])]

    response = client.models.generate_content(
        model=config.MODEL,
        contents=history,
        config={"system_instruction": system_prompt},
    )
    return response.text


def main() -> int:
    parser = argparse.ArgumentParser(description="Reply to the ShopWise inbox.")
    parser.add_argument("--limit", type=int, default=None,
                        help="only handle the first N tickets")
    args = parser.parse_args()

    load_dotenv(LABS / ".env")
    if not os.getenv("GEMINI_API_KEY"):
        print("No GEMINI_API_KEY found. Put it in labs/.env and try again.")
        return 1

    from google import genai

    client = genai.Client()
    system_prompt = load_system_prompt()
    tickets = load_tickets()
    if args.limit:
        tickets = tickets[: args.limit]

    print(f"ShopWise assistant v0.1 — {len(tickets)} ticket(s)\n")
    for ticket in tickets:
        print("=" * 72)
        print(f"{ticket['id']}  [{ticket['category']}]  {ticket['subject']}")
        print("-" * 72)
        try:
            print(answer(client, system_prompt, ticket).strip())
        except Exception as err:  # a rate limit shouldn't lose the whole run
            print(f"  ! could not answer {ticket['id']}: {str(err)[:160]}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

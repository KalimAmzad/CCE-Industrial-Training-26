"""ShopWise support assistant — project version v0.2.

    uv run python triage.py --limit 3

The first pipeline: one model call feeds the next.

    ticket -> triage() -> TriageResult -> draft_reply() -> guarded reply

v0.1 could only write prose. v0.2 first produces a typed TriageResult the code can
branch on, then hands it to the reply call — so the reply already knows what kind
of ticket it is answering and whether a human must sign it off.

The plumbing (client, fixtures, cost, the `.parsed is None` guard) is in utils.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from google.genai import errors, types

from config import settings
from schemas import TriageResult
from utils import (ModelReturnedNothing, as_ticket_block, build_client, checked,
                   cost, load_prices, load_tickets)

HERE = Path(__file__).resolve().parent
PROMPTS = yaml.safe_load((HERE / "prompts.yml").read_text(encoding="utf-8"))


def build_triage_prompt(*, with_examples: bool | None = None) -> str:
    """The triage system prompt, with the few-shot examples appended if enabled.

    Examples are a flag (settings.USE_FEW_SHOT) so score_triage.py can measure
    what they buy. Measured: nothing — so they ship off.
    """
    if with_examples is None:
        with_examples = settings.USE_FEW_SHOT
    prompt = PROMPTS["triage_prompt"]
    if with_examples and PROMPTS.get("few_shot_examples"):
        shown = [
            f"<ticket>\nSubject: {ex['subject']}\n{ex['body'].strip()}\n</ticket>\ncategory: {ex['category']}"
            for ex in PROMPTS["few_shot_examples"]
        ]
        prompt += "\n\nWorked examples of how ShopWise labels the awkward cases:\n\n" + "\n\n".join(shown)
    return prompt


def triage(client, ticket: dict, *, with_examples: bool | None = None,
           schema=None) -> tuple[TriageResult, object]:
    """Classify one ticket. Returns (TriageResult, usage_metadata).

    `schema` is only used by score_triage.py to compare schema shapes.
    """
    response = client.models.generate_content(
        model=settings.MODEL,
        contents=as_ticket_block(ticket),
        config=types.GenerateContentConfig(
            system_instruction=build_triage_prompt(with_examples=with_examples),
            response_mime_type="application/json",
            response_schema=schema or TriageResult,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        ),
    )
    return checked(response, f"triage {ticket['id']}"), response.usage_metadata


def draft_reply(client, ticket: dict, result: TriageResult) -> tuple[str, object]:
    """Write the customer's reply, now that triage has decided what the ticket is."""
    steer = (f"\n\nThis ticket has been triaged as {result.category}, "
             f"urgency {result.urgency}, customer sounds {result.sentiment}.")
    if result.needs_human:
        steer += ("\nA human must handle the actual decision. Do not state or imply any "
                  "outcome — say a colleague will follow up, and what happens next.")

    response = client.models.generate_content(
        model=settings.MODEL,
        contents=as_ticket_block(ticket),
        config=types.GenerateContentConfig(
            system_instruction=PROMPTS["system_prompt"] + steer,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        ),
    )
    return response.text, response.usage_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Triage and answer the ShopWise inbox.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not settings.has_key:
        print("No GEMINI_API_KEY found. Put it in .env and try again.")
        return 1

    client = build_client()
    prices = load_prices()
    tickets = load_tickets()[: args.limit] if args.limit else load_tickets()

    running = 0.0
    print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.MODEL} · {len(tickets)} ticket(s)\n")
    for ticket in tickets:
        try:
            result, triage_usage = triage(client, ticket)
            reply, reply_usage = draft_reply(client, ticket, result)
        except errors.ClientError as err:          # 4xx — our request was wrong; do not retry
            print(f"  ! {ticket['id']}: client error {err.code} {err.status}\n")
            continue
        except errors.ServerError as err:          # 5xx — their side, even after the SDK's retries
            print(f"  ! {ticket['id']}: server error {err.code} — requeue this ticket\n")
            continue
        except ModelReturnedNothing as err:        # 200 OK, nothing usable
            print(f"  ! {ticket['id']}: {err}\n")
            continue

        spent = cost(triage_usage, settings.MODEL, prices) + cost(reply_usage, settings.MODEL, prices)
        running += spent

        print("=" * 74)
        print(f"{ticket['id']}  {result.category} | {result.urgency} | "
              f"{result.sentiment} | needs_human={result.needs_human}")
        print(f"  {result.summary}")
        print("-" * 74)
        print(reply.strip())
        print(f"\n  [${spent:.6f} this ticket · ${running:.6f} running]\n")

    print("=" * 74)
    print(f"Total: ${running:.6f} for {len(tickets)} tickets "
          f"(~${running / max(len(tickets), 1) * 200:.2f} per 200-ticket day)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

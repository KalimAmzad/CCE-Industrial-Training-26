"""ShopWise support assistant — project version v1.

    uv run python triage.py --limit 3

Same pipeline as v0.2, rebuilt on LangChain:

    ticket -> triage() -> TriageResult -> draft_reply() -> guarded reply

What changed: the vendor is a string in settings (build_model, in utils.py),
structured output is `with_structured_output`, cost accounting moved to the
trace, and every call carries a stamp saying which prompt wording produced it.
What got worse: one `except Exception` where v0.2 had three typed branches —
the exception classes are still the vendor's (see main()).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from config import settings
from schemas import TriageResult
from utils import as_ticket_block, build_model, load_tickets, prompt_sha

HERE = Path(__file__).resolve().parent
PROMPTS = yaml.safe_load((HERE / "prompts.yml").read_text(encoding="utf-8"))

# The reply prompt gets two clauses appended at call time. They are templates so the
# stamp can hash the WORDING only: `{category}` is data, the sentence is prompt text.
STEER = ("\n\nThis ticket has been triaged as {category}, urgency {urgency}, "
         "customer sounds {sentiment}.")
ESCALATE = ("\nA human must handle the actual decision. Do not state or imply any "
            "outcome — say a colleague will follow up, and what happens next.")


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


def prompt_version() -> str:
    """The label for the wording in use. prompts.yml wins; SHOPWISE_PROMPT_VERSION
    overrides it for one run (how score_triage.py labels the arms of an A/B)."""
    return settings.PROMPT_VERSION or PROMPTS["prompt_version"]


def stamp(*prompt_parts: str, version: str | None = None) -> dict:
    """The `config=` every traced call carries: which wording, and a hash of it.

    `metadata` is what you filter by afterwards; the `prompt:` tag is what you
    click in the run list. Both land on every run in the chain, so count model
    calls with run_type="llm" or your totals triple.
    """
    version = version or prompt_version()
    return {
        "metadata": {"prompt_version": version, "prompt_sha": prompt_sha(*prompt_parts)},
        "tags": [f"prompt:{version}"],
    }


def triage(model, ticket: dict, *, with_examples: bool | None = None,
           schema=None, prompt: str | None = None,
           version: str | None = None) -> TriageResult:
    """Classify one ticket. Returns the typed result.

    `schema`, `prompt` and `version` exist only for score_triage.py, so it can
    run a candidate shape or wording through the real call. Production passes none.
    """
    system = build_triage_prompt(with_examples=with_examples) if prompt is None else prompt
    structured = model.with_structured_output(schema or TriageResult)
    return structured.invoke(
        [("system", system), ("human", as_ticket_block(ticket))],
        config=stamp(system, version=version),
    )


def draft_reply(model, ticket: dict, result: TriageResult) -> str:
    """Write the customer's reply, now that triage has decided what the ticket is."""
    steer = STEER.format(category=result.category, urgency=result.urgency,
                         sentiment=result.sentiment)
    if result.needs_human:
        steer += ESCALATE
    reply = model.invoke(
        [("system", PROMPTS["system_prompt"] + steer), ("human", as_ticket_block(ticket))],
        config=stamp(PROMPTS["system_prompt"], STEER, ESCALATE),   # hash the templates, not the ticket
    )
    return reply.text


def main() -> int:
    parser = argparse.ArgumentParser(description="Triage and answer the ShopWise inbox.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if settings.PROVIDER == "google_genai" and not settings.has_key:
        print("No GEMINI_API_KEY found. Put it in labs/.env and try again.")
        return 1

    model = build_model()
    tickets = load_tickets()[: args.limit] if args.limit else load_tickets()

    print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.model_id} · {len(tickets)} ticket(s)")
    if settings.LANGSMITH_TRACING:
        print(f"  tracing -> project {settings.LANGSMITH_PROJECT!r}")
        print(f"  prompt  -> {prompt_version()} · sha {prompt_sha(build_triage_prompt())}")
    print()

    failures = 0
    for ticket in tickets:
        try:
            result = triage(model, ticket, with_examples=True)
            reply = draft_reply(model, ticket, result)
        except Exception as err:
            # v0.2 had three typed branches. Behind LangChain the exception classes are
            # still the vendor's, and importing them would re-weld us to one vendor —
            # so print the class name and let the error name itself.
            failures += 1
            print(f"  ! {ticket['id']}: {type(err).__name__}: {err}\n")
            continue

        print("=" * 74)
        print(f"{ticket['id']}  {result.category} | {result.urgency} | "
              f"{result.sentiment} | needs_human={result.needs_human}")
        print(f"  {result.summary}")
        print("-" * 74)
        print(reply.strip())
        print()

    print("=" * 74)
    print(f"{len(tickets) - failures}/{len(tickets)} triaged"
          + (f", {failures} failed" if failures else ""))
    if settings.LANGSMITH_TRACING:
        print("Tokens, latency and cost: https://smith.langchain.com")   # cost lives in the trace now
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

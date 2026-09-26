"""ShopWise support assistant — triage, the first entry point. Project version v2.

    uv run python triage.py --limit 3

Same pipeline as v0.2, rebuilt on LangChain:

    ticket -> triage() -> TriageResult -> draft_reply() -> guarded reply

WHAT CHANGED, AND WHAT IT COST
    Read this next to v0.2's triage.py. Three things went away, and it is worth
    knowing which of them the framework earned and which it only moved.

    build_client() collapsed into two keyword arguments. v0.2 built a
        genai.Client with an HttpOptions holding a millisecond timeout and a
        HttpRetryOptions listing the status codes worth retrying. init_chat_model
        takes `timeout=` and `max_retries=` and hands them to whichever vendor
        client it builds. That is the abstraction doing real work: the same two
        arguments now mean the same thing across five providers.

    checked() is gone. v0.2 needed it because `response.parsed` could be None on
        a 200 OK, and silently treating that as "no result" is how a support
        queue quietly drops every ticket that mentions a lawyer.
        with_structured_output raises instead of handing back a None, so the
        guard is absorbed rather than deleted. Section 5 of the notebook shows
        the other form, include_raw=True, which hands you {parsed, raw,
        parsing_error} when you want to see the failure rather than catch it.

    cost() and load_prices() are gone, and this one is a MOVE, not a saving.
        From v1 every call is traced, and the trace carries tokens and cost per
        run. Maintaining a price table inside the program stopped being the
        cheapest way to know what it spends — see the notebook, section 8, and
        note the operational catch: the platform prices a run when it logs it
        and never backfills, so a model it does not know is recorded at zero
        forever.

    One thing ARRIVED, and it is the smallest change in the file. Every call now
        passes `config=stamp(...)`, which puts the prompt's version and a content
        hash of the prompt as assembled onto the run. Two keys and a tag; see
        stamp() below and ADR-0005. It is what makes a trace explainable rather
        than merely readable — a recorded answer whose prompt is unidentifiable
        tells you what the model said and nothing about why.

    And one thing got WORSE, which the honest version of this file has to say.
        v0.2 caught errors.ClientError and errors.ServerError — 4xx and 5xx as
        separate types, so "fix your request" and "requeue this ticket" were
        different branches. Behind the standard interface those types are still
        the vendor's: a bad model id here raises
        langchain_google_genai.chat_models.ChatGoogleGenerativeAIError, and the
        Groq path raises something else entirely. Catching precisely means
        importing a vendor's exception class, which is the vendor-specific
        coupling you just paid an abstraction to remove. See main().

Importing this module does nothing: the model is built inside main().
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml
from langchain.chat_models import init_chat_model

from config import settings
from schemas import TriageResult

HERE = Path(__file__).resolve().parent
PROMPTS = yaml.safe_load((HERE / "prompts.yml").read_text(encoding="utf-8"))

# Every prompt carries its own version, and this is where a missing one is
# caught. It fails at IMPORT, on a file nobody has run yet, rather than at the
# first traced call — which is the whole reason there is no file-level default
# to fall back on (ADR-0005, as amended). A default would make this line
# unnecessary and would make the stamp a lie instead of an error.
_UNVERSIONED = sorted(k for k in PROMPTS
                      if k.endswith("_prompt") and f"{k}_version" not in PROMPTS)
if _UNVERSIONED:
    raise ValueError(
        f"prompts.yml: {_UNVERSIONED} have no version. Every prompt carries its own "
        f"`<name>_version` key beside it — add e.g. `{_UNVERSIONED[0]}_version: v1` "
        f"directly above the wording. There is deliberately no file-level default."
    )

# The reply call splices two clauses onto the system prompt at call time. They
# are templates rather than inline f-strings for one reason: the stamp below has
# to hash the WORDING and only the wording. `{category}` is data; "A human must
# handle the actual decision" is prompt text somebody can edit at 11pm, and an
# edit to it has to move the hash.
STEER = ("\n\nThis ticket has been triaged as {category}, urgency {urgency}, "
         "customer sounds {sentiment}.")
ESCALATE = ("\nA human must handle the actual decision. Do not state or imply any "
            "outcome — say a colleague will follow up, and what happens next.")


# ── The model ────────────────────────────────────────────────────────────────

def build_model():
    """One model object, configured once, for the whole program.

    `settings.model_id` is the provider string — "google_genai:gemini-3.1-flash-lite"
    — and it is the only place in this program that knows which vendor we are on.
    Change SHOPWISE_PROVIDER and SHOPWISE_MODEL in .env and every call below
    goes somewhere else, with no edit here.

    timeout — without one, a connection that hangs hangs your whole batch.
        LangChain takes seconds; v0.2 had to convert to milliseconds itself.

    max_retries — the client retries transient failures for you. As in v0.2,
        what it does NOT cover is anything that is not a transport failure: a
        truncated body, a validation error, an answer you dislike. Only the
        first of those is worth retrying, and none of them are retried here.
    """
    return init_chat_model(
        settings.model_id,
        timeout=settings.REQUEST_TIMEOUT_S,
        max_retries=settings.MAX_RETRIES,
        max_tokens=settings.MAX_OUTPUT_TOKENS,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

def load_tickets() -> list[dict]:
    return yaml.safe_load((settings.DATA_DIR / "tickets.yml").read_text(encoding="utf-8"))


# ── Prompt assembly ──────────────────────────────────────────────────────────
# Unchanged from v0.2. Worth noticing: nothing in this section knows or cares
# which vendor is downstream. Prompt assembly was already portable; it is the
# CALL that was welded to Google, and only the call had to move.

def render_examples(examples: list[dict]) -> str:
    """Render the few-shot block exactly as the model will see it.

    Same <ticket> delimiters as the real input, so the examples and the question
    have identical shape — an example that looks different from the real thing
    teaches the model the wrong pattern.
    """
    out = []
    for ex in examples:
        out.append(
            f"<ticket>\nSubject: {ex['subject']}\n{ex['body'].strip()}\n</ticket>\n"
            f"category: {ex['category']}"
        )
    return "\n\n".join(out)


def build_triage_prompt(*, with_examples: bool | None = None) -> str:
    """Assemble the triage system prompt.

    The examples are a flag rather than a constant so that ../score_triage.py can
    measure exactly what they buy. See settings.USE_FEW_SHOT for the number that
    decided the default.
    """
    if with_examples is None:
        with_examples = settings.USE_FEW_SHOT
    prompt = PROMPTS["triage_prompt"]
    if with_examples and PROMPTS.get("few_shot_examples"):
        prompt += ("\n\nWorked examples — these show how ShopWise labels the "
                   "awkward cases:\n\n"
                   + render_examples(PROMPTS["few_shot_examples"]))
    return prompt


def prompt_version(prompt: str) -> str:
    """Which wording of ONE prompt this process is running — the label on its runs.

    v1 took no argument: there was a single `prompt_version` key covering the
    whole file. v2 adds `agent_prompt`, and a shared key means editing the
    agent's wording silently renames the triage prompt's version — one wording
    under two names, which is the failure a version exists to prevent, and the
    one thing the content hash cannot catch, because there the hash is the half
    telling the truth. So the version moved down a level, next to the wording it
    names (ADR-0005, amended 2026-09-01).

    `prompts.yml` wins. `SHOPWISE_PROMPT_VERSION` overrides ONE prompt for one
    run, and has to say which:

        SHOPWISE_PROMPT_VERSION=triage_prompt:v2-ordered

    The colon is not ceremony. An override that renamed every prompt at once
    would recreate exactly the bug above, and would do it only during the run
    you were trying to measure — the worst possible place to keep one.
    """
    override = settings.PROMPT_VERSION.strip()
    if override:
        named, sep, label = override.partition(":")
        if not sep:
            raise ValueError(
                f"SHOPWISE_PROMPT_VERSION must name the prompt it renames, e.g. "
                f"'triage_prompt:v2-ordered' — got {override!r}. From v2 a version "
                f"belongs to one wording, so an override has to say which."
            )
        if named == prompt:
            return label
    return PROMPTS[f"{prompt}_version"]


def prompt_sha(*parts: str) -> str:
    """A short content hash of the prompt AS ASSEMBLED — not of the file.

    The version above is typed by a human and therefore sometimes not typed at
    all. This is the half nobody can forget, because nobody enters it: two runs
    with one `prompt_version` and two hashes mean the wording moved and nobody
    announced it.

    What it hashes is what the model was *instructed* with, few-shot block
    included if it is switched on. What it deliberately leaves out is the
    customer's ticket — that is data, not wording, and a hash that changed with
    every ticket could never answer the one question the hash exists for.

    Twelve hex characters, because this is read by humans comparing two runs in a
    trace list, not by anything that needs collision resistance.
    """
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def stamp(*prompt_parts: str, prompt_name: str, version: str | None = None) -> dict:
    """The `config=` every traced call carries: who wrote this prompt, and what is in it.

    Three keys and no new dependency. `metadata` is what you filter and group by
    afterwards; the `prompt:` tag is what makes the two arms of an A/B one click
    apart in the run list. LangChain copies both onto every run in the chain, so
    a structured-output call — a sequence of model then parser — carries the
    stamp on all three of its runs. Count with `run_type="llm"` or your totals
    triple.
    """
    version = version or prompt_version(prompt_name)
    return {
        "metadata": {"prompt_version": version, "prompt_sha": prompt_sha(*prompt_parts)},
        "tags": [f"prompt:{version}"],
    }


def as_ticket_block(ticket: dict) -> str:
    """Wrap the customer's words in delimiters. Never interpolate raw.

    Two reasons, and the second matters more than it looks: the model stops
    reading the ticket's own words as instructions, and attacker text becomes
    visibly *data* rather than *instruction* — the cheapest first move against
    prompt injection (session 7).
    """
    return f"<ticket>\nSubject: {ticket['subject']}\n{ticket['body']}\n</ticket>"


# ── The two calls ────────────────────────────────────────────────────────────

def triage(model, ticket: dict, *, with_examples: bool | None = None,
           schema=None, prompt: str | None = None,
           version: str | None = None) -> TriageResult:
    """Classify one ticket. Returns the typed result.

    Compare the v0.2 signature: it returned `(TriageResult, usage_metadata)`,
    because the program had to add the tokens up itself. It no longer does, so
    the tuple is gone and every caller got simpler. That is what it looks like
    when a responsibility genuinely leaves your codebase rather than moving to
    another line of it.

    The system prompt and the ticket are two messages, not one string with a
    field for each: LangChain's interface is messages, and the roles are how a
    model tells your instructions apart from the customer's words.

    `schema` overrides the shipped `TriageResult`. Production never passes it —
    it exists so ../score_triage.py can score a candidate schema shape against
    the real prompt and the real model, rather than against a reimplementation
    of them that has quietly drifted. An eval that does not exercise the shipped
    code path is measuring the wrong program.

    `prompt` and `version` are the same idea one level up: a candidate WORDING,
    with the label to stamp it under. Production passes neither — prompts.yml is
    the source of truth and this function reads it. They exist so the harness can
    run a rival prompt through the real call, with the real schema and the real
    model, and have the two arms come back tagged and separable in the trace.

    `config=` is the whole of the stamping. Everything else about the call is
    unchanged, which is the point: observability that costs one argument.
    """
    system = build_triage_prompt(with_examples=with_examples) if prompt is None else prompt
    structured = model.with_structured_output(schema or TriageResult)
    return structured.invoke(
        [("system", system), ("human", as_ticket_block(ticket))],
        config=stamp(system, prompt_name="triage_prompt", version=version),
    )


def draft_reply(model, ticket: dict, result: TriageResult) -> str:
    """Write the customer's reply, now that we know what kind of ticket it is.

    The triage is handed to the reply call as context. This is the pipeline: the
    second call is better because the first one already decided something.

    `.text` is a property, not a method. Older code and most of the internet
    still writes `.content`, which also works; `.text()` with parentheses was the
    method form and is deprecated. If you see it called, you are reading
    something written before the v1 rewrite.
    """
    steer = STEER.format(category=result.category, urgency=result.urgency,
                         sentiment=result.sentiment)
    if result.needs_human:
        steer += ESCALATE

    # The stamp hashes the two templates, not this ticket's filled-in steer, so
    # every reply run on one build carries one hash. Whether THIS ticket
    # escalated is already in the trace — it is in the recorded input.
    reply = model.invoke(
        [("system", PROMPTS["system_prompt"] + steer), ("human", as_ticket_block(ticket))],
        config=stamp(PROMPTS["system_prompt"], STEER, ESCALATE,
                     prompt_name="system_prompt"),
    )
    return reply.text


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Triage and answer the ShopWise inbox.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if settings.PROVIDER == "google_genai" and not settings.has_key:
        print("No GEMINI_API_KEY found. Put it in .env and try again.")
        return 1

    model = build_model()
    tickets = load_tickets()
    if args.limit:
        tickets = tickets[: args.limit]

    print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.model_id} · "
          f"{len(tickets)} ticket(s)")
    if settings.LANGSMITH_TRACING:
        print(f"  tracing -> project {settings.LANGSMITH_PROJECT!r}")
        # Printed rather than assumed: this is the pair every run is stamped with,
        # so seeing it at startup is how you notice a version you forgot to bump.
        print(f"  triage  -> {prompt_version('triage_prompt')} · sha "
              f"{prompt_sha(build_triage_prompt())}")
        print(f"  reply   -> {prompt_version('system_prompt')} · sha "
              f"{prompt_sha(PROMPTS['system_prompt'], STEER, ESCALATE)}")
    print()

    failures = 0
    for ticket in tickets:
        try:
            result = triage(model, ticket)
            reply = draft_reply(model, ticket, result)
        except Exception as err:
            # v0.2 had three except branches here, one per failure family, and
            # each one told you what to DO about it. This single branch is the
            # abstraction's bill, arriving in the open.
            #
            # The vendor's exception classes still exist and are still precise —
            # they are just no longer in this file's imports, and importing them
            # would re-weld this program to one vendor for the sake of a nicer
            # error message. So we print the class name instead and let it name
            # itself. Printing the type is not a workaround for laziness: it is
            # the only vendor-neutral thing left to say.
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
    # v0.2 printed a running dollar total here. That number now lives in the
    # trace, per call, without this program maintaining a price list that was
    # out of date the week it was written. See notebook section 8.
    if settings.LANGSMITH_TRACING:
        print("Tokens, latency and cost: https://smith.langchain.com")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score the assistant's triage against the hand-labelled tickets.

    uv run python score_triage.py                              # the shipped configuration
    uv run python score_triage.py --examples                   # with the few-shot block
    uv run python score_triage.py --diff                       # run both and compare
    uv run python score_triage.py --schemas                    # compare three schema SHAPES
    uv run python score_triage.py --prompt-version v2-ordered  # A/B a candidate WORDING

This is session 3's second half, ported to v1: **a prompt change you cannot
measure is a guess.** Every line of `triage_prompt` in project/prompts.yml was
kept or dropped because of a number this script printed.

It deliberately does NOT live in project/. It is a teaching artifact and a
regression check you should run, not part of the assistant — and snapshots are
copied forward, so putting it there would drag it through twelve later sessions
for no reason.

WHAT CHANGED IN THE PORT, AND WHY IT IS THE INTERESTING PART
    v0.2's `triage()` returned `(TriageResult, usage_metadata)` and the project
    owned a `cost()` and a `load_prices()`. v1 has none of the three: the tuple
    collapsed to a bare result, and costing moved to the platform (see notebook
    section 8). So this harness had a choice, and only one of the options is
    honest.

    It could have kept its own price table and its own token counting — a second
    implementation, drifting away from production one release at a time, so that
    the eval eventually reports a cost the service does not have. Instead it
    **reads its own tokens and cost out of the trace**, from the same records
    production is priced by. One number, one source, and the eval is now exercising
    the observability the application actually ships with.

    The mechanics are three lines. Every call in a scoring run is wrapped in a
    `tracing_context` carrying a tag unique to this invocation; afterwards the
    runs are queried back by that tag and summed. Two details bite:

      - **scope to `run_type="llm"`.** A structured-output call is a sequence —
        model, then parser — so the stamp lands on three runs per ticket. Count
        them all and every total triples.
      - **poll.** Traces are posted by a background thread and priced when the
        platform ingests them, so a query fired immediately finds the runs with
        their token counts still empty. `bill()` waits.

WHAT IT MEASURES, AND WHAT IT REFUSES TO
    Accuracy on `category`, because that is the only field with ground truth.
    There is no hand-labelled urgency or sentiment, so any claim about those
    would be a vibe with a percentage sign on it.

    It also reports two numbers people forget to measure and then get asked about
    in a review: how many calls came back **unusable**, and what the run **cost**.
    An eval that reports only accuracy quietly hides the model that is 2% better
    and four times the price.

    And read the misses. Several are genuinely arguable — TCK-003 is labelled
    `complaint` and could fairly be `order_status`. That is not a flaw in the
    data, it is a property of the task: **some of your ceiling is the label, not
    the model.**
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client, tracing_context
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "project"))   # reuse the snapshot, don't duplicate it

import triage as project                            # noqa: E402
from config import settings                         # noqa: E402
from schemas import TicketCategory, TriageResult    # noqa: E402

# One tag for this whole invocation, so an A/B's two arms are also one filter in
# the browser: `has(tags, "eval-20260901-...")`. Each arm adds its own suffix.
RUN_TAG = f"eval-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"

TRACED = settings.LANGSMITH_TRACING and bool(settings.LANGSMITH_API_KEY.get_secret_value())


# ── Schema variants, for --schemas ───────────────────────────────────────────
# These live in the EVAL HARNESS, not in project/schemas.py. That separation is
# deliberate and worth copying: the application holds the one shape it ships,
# the harness holds the candidates you are still arguing about. A variant that
# wins moves into the application; one that does not stays here as evidence.

class ThinkFirst(BaseModel):
    """The folklore version: reason before deciding, so the reasoning can inform
    the decision. A model writes fields in declaration order, so this genuinely
    does put the working before the answer."""

    reasoning: str = Field(description="Which two categories could this be, and which "
                                       "ShopWise convention settles it? Two sentences.")
    category: TicketCategory
    urgency: Literal["low", "medium", "high"]
    sentiment: Literal["calm", "frustrated", "angry"]
    needs_human: bool
    summary: str


class ThinkLast(BaseModel):
    """The control. Same field, same description, declared last — so it costs the
    same output tokens but cannot influence the answer. Without this arm you
    cannot tell 'reasoning helped' from 'a longer answer helped'."""

    category: TicketCategory
    urgency: Literal["low", "medium", "high"]
    sentiment: Literal["calm", "frustrated", "angry"]
    needs_human: bool
    summary: str
    reasoning: str = Field(description="Which two categories could this be, and which "
                                       "ShopWise convention settles it? Two sentences.")


# ── Prompt candidates, for --prompt-version ──────────────────────────────────
# The same separation as the schema variants, one level up. project/prompts.yml
# holds the one wording ShopWise ships and stamps on every run; this file holds
# the wordings we are still arguing about. A candidate that wins is moved into
# prompts.yml WITH a bumped `prompt_version`; one that does not stays here, as
# evidence that somebody already tried it and what it scored.
#
# The rule that makes a result readable: **change one thing.** The candidate
# below states exactly the same four ShopWise conventions as the shipped prompt,
# in the same order, in the same words — as a numbered procedure applied in
# order, first match wins, instead of four independent statements. If you edit
# the meaning as well as the structure, you have measured nothing, because you
# can no longer say which of the two changes moved the number.

PROMPT_CANDIDATES: dict[str, str] = {
    "v2-ordered": """You classify incoming ShopWise support tickets. You do not reply to them.

Read the ticket between the <ticket> tags and return the structured result.

Choosing a category when more than one fits — this is ShopWise's convention,
it is not obvious, and it is the reason the examples exist. Apply these rules
in order and stop at the first one that matches:
1. Did ShopWise get something wrong — ignored them, sent the wrong item,
   delivered it damaged? It is a complaint, even when the customer also
   wants a refund or a return.
2. Are they asking where it is? It is order_status, even when the wait has
   made them cross.
3. Do they want money back? It is refund. Do they want to send goods back?
   It is return.
4. Is it about what they were charged? It is billing. If the argument is
   about a shipping fee or a delivery entitlement, it is shipping.
""",
}


# ── Reading the bill back out of the trace ───────────────────────────────────

BILL_FIELDS = ["ID", "RUN_TYPE", "PROMPT_TOKENS", "COMPLETION_TOKENS",
               "TOTAL_TOKENS", "TOTAL_COST"]


def run_async(coro):
    """Run a coroutine from synchronous code, loop or no loop.

    `client.runs.query` is async and there is no sync public path, so something
    has to drive it. From a script there is no event loop and `asyncio.run` is
    the whole answer. From a notebook cell there already IS one — ipykernel runs
    your code inside it, which is how a cell can `await` at the top level — and
    `asyncio.run` raises `RuntimeError: cannot be called from a running event
    loop`. Handing the coroutine to a thread with a loop of its own works in
    both, which is why score() below can be called from a terminal and from
    exercises.ipynb without knowing which it is in.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _billed_runs(tag: str, since, expect: int, tries: int, pause: float) -> list:
    """Every priced LLM run carrying `tag`, once the platform has finished with them.

    `run_type="llm"` is not a tidiness choice. `with_structured_output` builds a
    sequence, so one scored ticket produces three runs — the sequence, the model
    call, the parser — and the tags and metadata are copied onto all three.
    Without this filter every token count and every dollar in the report is
    exactly three times too large, which is the kind of wrong that looks right.

    The polling is the other half. A trace is posted by a background thread and
    costed when the platform ingests it, so a query fired one second after the
    last call finds the runs with `total_tokens` still 0. Runs that arrive
    unpriced are skipped rather than counted as free.
    """
    client = Client()
    project_id = str(client.read_project(project_name=settings.LANGSMITH_PROJECT).id)
    runs: list = []
    for _ in range(tries):
        page = await client.runs.query(
            project_ids=[project_id], run_type="llm", min_start_time=since,
            page_size=1000, selects=BILL_FIELDS, filter=f'has(tags, "{tag}")',
        )
        runs = [r async for r in page if r.total_tokens]
        if len(runs) >= expect:
            return runs
        await asyncio.sleep(pause)
    return runs


@dataclass
class Run:
    """One scoring pass. Accuracy is the headline; the rest is the fine print."""

    label: str
    total: int
    tag: str = ""                                        # how to find this pass in the trace
    hits: int = 0
    failures: int = 0                                    # no usable answer at all
    spent: float = 0.0                                   # USD, read from the trace
    out_tokens: int = 0                                  # what the answers cost to write
    billed: bool = False                                 # did the trace actually answer?
    misses: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def bill(self, since, tries: int = 15, pause: float = 3.0) -> None:
        """Fill in tokens and cost from the platform's records of this pass.

        v0.2 added these up as it went, from a `usage_metadata` the call returned
        and a price table the project maintained. Both are gone, so the eval asks
        the same place production reads its bill from.
        """
        if not TRACED:
            return
        wait_for_all_tracers()          # the tracer posts in a background thread
        expect = self.total - self.failures
        runs = run_async(_billed_runs(self.tag, since, expect, tries, pause))
        self.out_tokens = sum(r.completion_tokens or 0 for r in runs)
        self.spent = sum(r.total_cost or 0.0 for r in runs)
        self.billed = len(runs) >= expect

    def report(self) -> None:
        print(f"\n{self.label}: {self.hits}/{self.total} = {self.accuracy * 100:.0f}%")
        if self.billed:
            print(f"  {self.failures} unusable response(s) · {self.out_tokens} output tokens · "
                  f"${self.spent:.4f} for the run "
                  f"(${self.spent / max(self.total, 1) * 1000:.2f} per 1,000 tickets)")
        else:
            # Say which number is missing and why. An eval that silently reports
            # $0.0000 because it could not read the trace is worse than one that
            # admits it — somebody will quote the zero.
            why = ("tracing is off" if not TRACED
                   else "the trace had not finished pricing this pass")
            print(f"  {self.failures} unusable response(s) · tokens and cost unavailable "
                  f"({why})")
        if self.tag:
            print(f"  trace filter: has(tags, \"{self.tag}\")")
        if self.misses:
            print(f"\n  {'ticket':<10}{'hand label':<19}{'assistant said'}")
            for tid, truth, pred in self.misses:
                print(f"  {tid:<10}{truth:<19}{pred}")
            worst = Counter(f"{t} -> {p}" for _, t, p in self.misses).most_common(3)
            print("\n  commonest confusions: "
                  + ", ".join(f"{pair} (x{n})" for pair, n in worst))


def score(model, tickets, *, with_examples: bool = False, label: str, slug: str,
          schema=None, prompt: str | None = None, version: str | None = None) -> Run:
    """Run one arm: every ticket, one configuration, one tag.

    The whole loop happens inside a `tracing_context` holding this arm's tag, so
    every run it produces — model calls and the parsers above them — is findable
    afterwards by that one string. Nothing in project/ knows this is happening,
    which is the property that lets the eval exercise the shipped code path
    rather than a copy of it.
    """
    tag = f"{RUN_TAG}-{slug}"
    run = Run(label=label, total=len(tickets), tag=tag)
    started = datetime.now(timezone.utc)

    tagged = tracing_context(tags=[RUN_TAG, tag]) if TRACED else contextlib.nullcontext()
    with tagged:
        for ticket in tickets:
            try:
                result = project.triage(model, ticket, with_examples=with_examples,
                                        schema=schema, prompt=prompt, version=version)
            except Exception as err:
                run.failures += 1
                run.misses.append((ticket["id"], ticket["category"],
                                   f"FAILED: {type(err).__name__}"))
                continue
            if result.category == ticket["category"]:
                run.hits += 1
            else:
                run.misses.append((ticket["id"], ticket["category"], result.category))

    run.bill(started)
    return run


def compare(base: Run, other: Run, *, bought: str) -> None:
    """The two-arm verdict: what changed, what it cost, and which tickets moved.

    The last two lines are the ones people skip and should not. A net zero with
    two tickets fixed and two broken is a different finding from a net zero with
    nothing moving at all, and only the second one means "no effect".
    """
    delta = other.hits - base.hits
    print(f"\n{'=' * 62}")
    print(f"{bought} {delta:+d} ticket(s) "
          f"= {delta / base.total * 100:+.0f} percentage points, "
          f"for {(other.spent / max(base.spent, 1e-9) - 1) * 100:+.0f}% cost.")
    fixed = {m[0] for m in base.misses} - {m[0] for m in other.misses}
    broke = {m[0] for m in other.misses} - {m[0] for m in base.misses}
    if fixed:
        print(f"  fixed : {', '.join(sorted(fixed))}")
    if broke:
        print(f"  BROKE : {', '.join(sorted(broke))}")
    if not delta:
        print("  No net change. That is a real result — report it and try something else.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", action="store_true",
                        help="score WITH the few-shot block (off by default — it bought nothing)")
    parser.add_argument("--diff", action="store_true",
                        help="score both ways and show what the examples bought")
    parser.add_argument("--schemas", action="store_true",
                        help="compare schema SHAPES: shipped vs reasoning-first vs reasoning-last")
    parser.add_argument("--prompt-version", metavar="NAME",
                        help="A/B a candidate WORDING against the shipped one. Defined here: "
                             + (", ".join(PROMPT_CANDIDATES) or "(none)"))
    args = parser.parse_args()

    if settings.PROVIDER == "google_genai" and not settings.has_key:
        print("No GEMINI_API_KEY found. Put it in .env and try again.")
        return 1

    model = project.build_model()
    tickets = project.load_tickets()
    # Names the prompt now: from v2 a version belongs to one wording, and the
    # one this harness measures is the triage prompt (ADR-0005, as amended).
    shipped = project.prompt_version("triage_prompt")

    print(f"{settings.model_id} · prompt {shipped} "
          f"· sha {project.prompt_sha(project.build_triage_prompt())}")
    print(f"tag: {RUN_TAG}" if TRACED
          else "not tracing — accuracy only. Set LANGSMITH_TRACING=true for tokens and cost.")

    if args.prompt_version:
        name = args.prompt_version
        if name not in PROMPT_CANDIDATES:
            print(f"\nNo candidate called {name!r}. Defined in this file: "
                  f"{', '.join(PROMPT_CANDIDATES) or '(none)'}")
            print("Candidates live here, not in prompts.yml — the application ships one")
            print("wording, the harness holds the ones you are still arguing about.")
            return 2

        base = score(model, tickets, label=f"shipped ({shipped})", slug=shipped)
        base.report()
        cand = score(model, tickets, label=f"candidate ({name})", slug=name,
                     prompt=PROMPT_CANDIDATES[name], version=name)
        cand.report()
        compare(base, cand, bought=f"{name} bought")
        print(f"\n  Ship it only if it won. If it tied, the shipped wording stays — a change")
        print(f"  that does not move the number is still a change you have to support.")
        print(f"  Both arms are in the trace: has(tags, \"{RUN_TAG}\")")
        return 0

    if args.diff:
        base = score(model, tickets, with_examples=False, label="WITHOUT examples",
                     slug="no-examples")
        base.report()
        few = score(model, tickets, with_examples=True, label="WITH examples",
                    slug="examples")
        few.report()
        compare(base, few, bought="The examples bought")
        print("  (a broken ticket means examples can cost you tickets too)")
        return 0

    if args.schemas:
        runs = [score(model, tickets, label=name, slug=slug, schema=schema)
                for name, slug, schema in (("shipped         ", "shipped", TriageResult),
                                           ("reasoning FIRST ", "think-first", ThinkFirst),
                                           ("reasoning LAST  ", "think-last", ThinkLast))]
        for r in runs:
            r.report()
        base = runs[0]
        print(f"\n{'=' * 62}")
        print(f"  {'variant':18}{'accuracy':>10}{'out tokens':>13}   {'vs shipped':<22}")
        for r in runs:
            delta = r.hits - base.hits
            extra = (r.out_tokens / base.out_tokens - 1) * 100 if base.out_tokens else 0
            verdict = f"{delta:+d} tickets, {extra:+.0f}% tokens"
            print(f"  {r.label:18}{f'{r.hits}/{r.total}':>10}{r.out_tokens:>13}   {verdict:<22}")
        print("\n  A reasoning field costs output tokens on every call, forever.")
        print("  Before you ship one, make it earn them — and note that a 20-ticket set")
        print("  moves by a point or two between identical runs, so a small win is not a win.")
        return 0

    use = args.examples or settings.USE_FEW_SHOT
    run = score(model, tickets, with_examples=use,
                label="WITH examples" if use else "WITHOUT examples",
                slug="examples" if use else "no-examples")
    run.report()
    print("\nRead the misses before you change anything. Several are arguable,")
    print("and an arguable label is a convention to teach, not a bug to fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

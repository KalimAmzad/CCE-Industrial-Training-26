"""Score the assistant's triage against the hand-labelled tickets.

    uv run python score_triage.py                 # the shipped configuration
    uv run python score_triage.py --examples      # with the few-shot block
    uv run python score_triage.py --diff          # run both and compare
    uv run python score_triage.py --schemas       # compare three schema shapes (appendix G)

A prompt change you cannot measure is a guess. This scores `category` accuracy
(the only field with ground truth), plus two numbers accuracy hides: how many
replies were unusable, and what the run cost.

It lives beside the notebook, not in project/: it is a measurement tool, not part
of the assistant. Read the misses before changing anything — several are arguable
labels, and an arguable label is a convention to teach, not a bug to fix.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "project"))   # reuse the snapshot, don't duplicate it

from schemas import TicketCategory, TriageResult   # noqa: E402


# Schema variants for --schemas. They live here, not in project/schemas.py: the
# application ships one shape, the harness holds the candidates.

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


@dataclass
class Run:
    """One scoring pass. Accuracy is the headline; the rest is the fine print."""

    label: str
    total: int
    hits: int = 0
    failures: int = 0                                    # no usable answer at all
    spent: float = 0.0                                   # USD
    out_tokens: int = 0                                  # what the answers cost to write
    misses: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def report(self) -> None:
        print(f"\n{self.label}: {self.hits}/{self.total} = {self.accuracy * 100:.0f}%")
        print(f"  {self.failures} unusable response(s) · {self.out_tokens} output tokens · "
              f"${self.spent:.4f} for the run "
              f"(${self.spent / max(self.total, 1) * 1000:.2f} per 1,000 tickets)")
        if self.misses:
            print(f"\n  {'ticket':<10}{'hand label':<19}{'assistant said'}")
            for tid, truth, pred in self.misses:
                print(f"  {tid:<10}{truth:<19}{pred}")
            worst = Counter(f"{t} -> {p}" for _, t, p in self.misses).most_common(3)
            print("\n  commonest confusions: "
                  + ", ".join(f"{pair} (x{n})" for pair, n in worst))


def score(client, tickets, prices, *, with_examples: bool, label: str, schema=None) -> Run:
    import triage as project
    import utils

    run = Run(label=label, total=len(tickets))
    for ticket in tickets:
        try:
            result, usage = project.triage(client, ticket, with_examples=with_examples,
                                           schema=schema)
        except Exception as err:
            run.failures += 1
            run.misses.append((ticket["id"], ticket["category"], f"FAILED: {type(err).__name__}"))
            continue
        run.spent += utils.cost(usage, project.settings.MODEL, prices)
        run.out_tokens += usage.candidates_token_count
        if result.category == ticket["category"]:
            run.hits += 1
        else:
            run.misses.append((ticket["id"], ticket["category"], result.category))
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", action="store_true",
                        help="score WITH the few-shot block (off by default — it bought nothing)")
    parser.add_argument("--diff", action="store_true",
                        help="score both ways and show what the examples bought")
    parser.add_argument("--schemas", action="store_true",
                        help="compare schema SHAPES: shipped vs reasoning-first vs reasoning-last")
    args = parser.parse_args()

    import utils
    from config import settings

    if not settings.has_key:
        print("No GEMINI_API_KEY found. Put it in .env and try again.")
        return 1

    client = utils.build_client()
    prices = utils.load_prices()
    tickets = utils.load_tickets()

    if args.diff:
        base = score(client, tickets, prices, with_examples=False, label="WITHOUT examples")
        base.report()
        few = score(client, tickets, prices, with_examples=True, label="WITH examples")
        few.report()

        delta = few.hits - base.hits
        print(f"\n{'=' * 62}")
        print(f"The examples bought {delta:+d} ticket(s) "
              f"= {delta / base.total * 100:+.0f} percentage points, "
              f"for {(few.spent / max(base.spent, 1e-9) - 1) * 100:+.0f}% cost.")
        fixed = {m[0] for m in base.misses} - {m[0] for m in few.misses}
        broke = {m[0] for m in few.misses} - {m[0] for m in base.misses}
        if fixed:
            print(f"  fixed : {', '.join(sorted(fixed))}")
        if broke:
            print(f"  BROKE : {', '.join(sorted(broke))}  <- examples can cost you tickets too")
        if not delta:
            print("  No net change. That is a real result — report it and try something else.")
        return 0

    if args.schemas:
        runs = [score(client, tickets, prices, with_examples=False,
                      label=name, schema=schema)
                for name, schema in (("shipped         ", TriageResult),
                                     ("reasoning FIRST ", ThinkFirst),
                                     ("reasoning LAST  ", ThinkLast))]
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
    run = score(client, tickets, prices, with_examples=use,
                label="WITH examples" if use else "WITHOUT examples")
    run.report()
    print("\nRead the misses before you change anything. Several are arguable,")
    print("and an arguable label is a convention to teach, not a bug to fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

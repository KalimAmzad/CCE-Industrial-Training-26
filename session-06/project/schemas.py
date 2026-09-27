"""The shapes the assistant speaks in — project version v0.2.

`TriageResult` is the most important object in this course. It is born here, and
it survives: S10 makes it the graph's triage-node output, S13 makes it the API
response model. **Do not rename its fields.** Later sessions reuse them by name,
and the continuity is the point — a participant should be able to watch the same
object flow through fourteen sessions.

The nine categories are ground truth, hand-labelled in data/tickets.yml, and
the assistant is scored against them (see ../score_triage.py). They are fixed by
docs/adr/0003 — do not add, remove or rename a value without reading it first.

READ THE FIELD ORDER BEFORE YOU CHANGE IT
    A model writes JSON one token at a time, in the order the schema declares. A
    field can therefore only be informed by the fields declared *above* it. That
    makes declaration order a design decision, not a style question: `summary`
    sits last because writing a summary is easy once the decisions are made, and
    the decisions are what we want the model spending its attention on.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# The nine. Exactly the values in data/tickets.yml, in the same order.
#
# Named separately because three later sessions import this alias rather than
# re-typing the list: a category added in S10 must appear everywhere at once, and
# a Literal copy-pasted into four files will disagree with itself inside a month.
TicketCategory = Literal[
    "billing", "complaint", "order_status", "other",
    "product_question", "refund", "return", "shipping", "warranty",
]

# Categories where money moves. The assistant is never allowed to sound like it
# has settled one of these on its own — see the model_validator below.
MONEY_CATEGORIES: frozenset[str] = frozenset({"billing", "refund", "return", "warranty"})


class TriageResult(BaseModel):
    """What the assistant decides about a ticket before it answers it.

    Every field earns its place by changing what the code does next: `category`
    routes, `urgency` orders the queue, `needs_human` stops the assistant
    promising money, and `summary` is what a human reads in a list. A field no
    code reads is a field you are paying output tokens for on every call.
    """

    category: TicketCategory = Field(
        description="What the ticket is about. Pick the single best fit."
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="high = the customer is blocked, out of pocket, or threatening to leave."
    )
    sentiment: Literal["calm", "frustrated", "angry"] = Field(
        description="How the customer sounds, not how serious the problem is."
    )
    needs_human: bool = Field(
        description="True if money moves, or the customer explicitly asks for a person."
    )
    summary: str = Field(
        description="One sentence, under 20 words, for a dashboard row. No greeting, no advice."
    )

    # ── Cleaning: fix what is merely untidy ──────────────────────────────────
    @field_validator("summary")
    @classmethod
    def _tidy_summary(cls, value: str) -> str:
        """Collapse whitespace and drop a trailing newline.

        Models are fond of leading spaces and stray line breaks inside JSON
        strings. This is invisible until the day the summary is a table cell in
        a dashboard, or a key in a log aggregator, and then it is very visible.
        Normalise at the boundary, once, rather than in nine display templates.
        """
        return re.sub(r"\s+", " ", value).strip()

    # ── Policy: fix what is wrong ────────────────────────────────────────────
    @model_validator(mode="after")
    def _money_always_needs_a_human(self) -> TriageResult:
        """ShopWise's rule, enforced where it cannot be argued with.

        A `model_validator` sees the whole object, so it can express rules no
        single field can: *if the category involves money, a human decides.*

        The model is asked for `needs_human` and is usually right. "Usually" is
        not a control. A prompt is a request; this is a guarantee, and it holds
        even when the model is confused, jailbroken, or replaced next quarter.
        That distinction — **models advise, code decides** — is the spine of
        session 7, arriving early because it costs four lines.

        Note the direction: this can only ever escalate. It never sets
        `needs_human` to False, because a rule that can silence an escalation is
        a rule that will one day silence the wrong one.
        """
        if self.category in MONEY_CATEGORIES:
            self.needs_human = True
        return self

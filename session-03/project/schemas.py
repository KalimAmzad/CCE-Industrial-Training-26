"""The shapes the assistant speaks in — project version v0.2.

`TriageResult` is born here and survives to the end of the course: session 10
makes it the graph's triage output, session 13 the API response model. Do not
rename its fields.

The nine categories are the hand labels in labs/data/tickets.yml (docs/adr/0003).

Field order matters: the model writes JSON top to bottom, so a field can only be
informed by the fields above it. Decisions first, `summary` last.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# The nine categories, exactly as labelled in tickets.yml. Later sessions import
# this alias instead of re-typing the list.
TicketCategory = Literal[
    "billing", "complaint", "order_status", "other",
    "product_question", "refund", "return", "shipping", "warranty",
]

# Categories where money moves. The assistant never settles one of these alone.
MONEY_CATEGORIES = frozenset({"billing", "refund", "return", "warranty"})


class TriageResult(BaseModel):
    """What the assistant decides about a ticket before it answers it.

    Every field changes what the code does next: `category` routes, `urgency`
    orders the queue, `needs_human` stops the assistant promising money, and
    `summary` is what a human reads in a list.
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
    draft_reply: str = Field(
        description="Reply to the user within 30 words"
    )

    # Cleaning: the value is fine, the formatting is not.
    @field_validator("summary")
    @classmethod
    def tidy_summary(cls, value: str) -> str:
        """Collapse stray whitespace and newlines the model likes to put in strings."""
        return re.sub(r"\s+", " ", value).strip()

    # Policy: the rule ShopWise will not let the model argue with.
    @model_validator(mode="after")
    def money_needs_a_human(self) -> TriageResult:
        """If money moves, a human decides — whatever the model said.

        Models advise, code decides. Note the direction: this only ever
        escalates. A rule that can switch `needs_human` off would one day
        switch off the one that mattered.
        """
        if self.category in MONEY_CATEGORIES:
            self.needs_human = True
        return self

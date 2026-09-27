"""The ShopWise agent — v3. It remembers, per ticket.

WHAT CHANGED FROM v2
    v2 was a goldfish: every `answer()` started from an empty message list, so
    "and how do I return it?" arrived from a stranger. v3 adds three things, and
    each is one argument or one small function, not a rewrite:

      checkpointer   `create_agent(..., checkpointer=...)`. After every step the
                     agent's state is saved, keyed by the thread id in the
                     call's config. The next call on the same thread starts
                     from that save. `build_agent` takes it as a parameter
                     because its two callers want different ones: the notebook
                     an in-memory saver, chat.py a SQLite file.

      thread id      One thread is one support ticket — NOT one customer and
                     NOT one device. `new_ticket_id()` makes one; `run_config()`
                     puts it where the checkpointer looks for it. A different
                     id is a different ticket, and the agent has never met you.

      summarization  History is resent to the model on every call, so a long
                     ticket costs more per turn and eventually overflows the
                     context window. SummarizationMiddleware replaces the oldest
                     turns with one summary once the thread passes a token
                     budget (settings.SUMMARY_*), using `summary_prompt` from
                     prompts.yml. It keeps tool calls paired with their results,
                     which a naive slice does not — see the notebook, section 12.

    Streaming is not here. It is a property of how a CALLER consumes a run
    (`agent.stream_events(...)` instead of `agent.invoke(...)`), not of the
    agent, so it lives in chat.py.

WHAT IT STILL CANNOT DO
    It will discuss anything, it copies customer PII into every trace and into
    the checkpoint file, and `issue_refund` is still withheld because nothing
    stops it being misused. Session 7 installs the rails.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.sqlite import SqliteSaver

from config import settings
from tools import TOOLS
# Same shared helpers as v2 — see the note in session-05/project/agent.py on why
# they still live in triage.py (ADR-0007: sharing a model builder is not merging
# two entry points).
from triage import PROMPTS, build_model, prompt_sha, prompt_version
from triage import stamp as triage_stamp

GROUNDED_PROMPT = PROMPTS["agent_prompt"]
SUMMARY_PROMPT = PROMPTS["summary_prompt"]


def build_summarizer(model) -> SummarizationMiddleware:
    """Bound a thread's history: past the budget, old turns become one summary.

    The same cheap model writes the summary. A summary is compression, not
    reasoning, and paying a stronger tier on every long ticket buys little.
    """
    return SummarizationMiddleware(
        model=model,
        trigger=("tokens", settings.SUMMARY_TRIGGER_TOKENS),
        keep=("messages", settings.SUMMARY_KEEP_MESSAGES),
        summary_prompt=SUMMARY_PROMPT,
    )


def build_agent(checkpointer):
    """The v3 agent. `checkpointer` decides where tickets are saved.

    Pass `InMemorySaver()` for a notebook (gone when the kernel stops), the
    saver from `open_checkpointer()` for anything that must survive a restart,
    or `None` to get v2's goldfish back — useful as a control, never in use.
    """
    model = build_model()
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=GROUNDED_PROMPT,
        middleware=[build_summarizer(model)],
        checkpointer=checkpointer,
    )


@contextmanager
def open_checkpointer(path: Path | str | None = None) -> Iterator[SqliteSaver]:
    """A SQLite-backed checkpointer, opened and closed properly.

    A context manager because the saver holds an open database connection:
    `with open_checkpointer() as saver:` guarantees it is closed even when the
    program dies with Ctrl-C mid-reply. The tables are created on first use.
    """
    path = Path(path or settings.CHECKPOINT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def new_ticket_id() -> str:
    """A fresh thread id, shaped like a helpdesk ticket number.

    Random, not sequential and not derived from the customer: a guessable id is
    a way to read somebody else's ticket (see the case study in the notebook).
    """
    return f"TICKET-{uuid.uuid4().hex[:8].upper()}"


def run_config(thread_id: str) -> dict:
    """The `config=` for one turn: the prompt stamp from v1, plus the thread.

    `configurable.thread_id` is the only key the checkpointer reads. Forget it
    and a checkpointed agent raises rather than guessing which ticket you meant.
    """
    return {**stamp(), "configurable": {"thread_id": thread_id}}


def stamp() -> dict:
    """Prompt version + hash on every run, unchanged from v2 (ADR-0005)."""
    return triage_stamp(GROUNDED_PROMPT, prompt_name="agent_prompt")


def answer(agent, question: str, thread_id: str) -> dict:
    """One customer turn on one ticket. Returns the full state after the turn.

    Note what is sent: ONLY the new message. The checkpointer supplies the rest
    of the ticket — that is the entire difference from v2.
    """
    return agent.invoke({"messages": [{"role": "user", "content": question}]},
                        config=run_config(thread_id))


def list_tickets(saver) -> list[str]:
    """Every thread id in the checkpoint store, newest first."""
    seen: dict[str, None] = {}
    for checkpoint in saver.list(None):
        seen.setdefault(checkpoint.config["configurable"]["thread_id"], None)
    return list(seen)


def transcript(state: dict) -> str:
    """A result or a saved state as a readable conversation.

    v3 adds one line type: SUMMARY, for the message SummarizationMiddleware
    writes. It arrives as a HumanMessage — the customer did not write it, and a
    transcript that printed it as CUSTOMER would be lying about who said what.
    """
    lines = []
    for message in state["messages"]:
        kind = type(message).__name__
        if message.additional_kwargs.get("lc_source") == "summarization":
            lines.append(f"SUMMARY   {message.text}")
        elif kind == "HumanMessage":
            lines.append(f"CUSTOMER  {message.text}")
        elif kind == "ToolMessage":
            lines.append(f"  RESULT  {message.name} -> {message.content}")
        elif getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                lines.append(f"  ASKS    {call['name']}({call['args']})")
        else:
            lines.append(f"ASSISTANT {message.text}")
    return "\n".join(lines)


# Two turns on one ticket: the second only makes sense if the first is remembered.
DEMO = [
    "Hi, I'm ayesha@example.com — my order ORD-5001 arrived with a cracked case.",
    "Which product was that again, and when was it delivered?",
]


def main() -> int:
    if settings.PROVIDER == "google_genai" and not settings.has_key:
        print("No GEMINI_API_KEY found. Put it in .env and try again.")
        return 1

    from langgraph.checkpoint.memory import InMemorySaver

    agent = build_agent(InMemorySaver())
    ticket = new_ticket_id()
    print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.model_id} · "
          f"{len(TOOLS)} tools · ticket {ticket}")
    if settings.LANGSMITH_TRACING:
        print(f"  agent -> {prompt_version('agent_prompt')} · sha {prompt_sha(GROUNDED_PROMPT)}")
    print()
    for question in DEMO:
        state = answer(agent, question, ticket)
    print(transcript(state))
    print("\nFor the interactive, streaming version:  uv run python chat.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

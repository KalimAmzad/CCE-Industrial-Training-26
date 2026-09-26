"""The ShopWise agent — v2. The assistant's second entry point.

WHAT THIS IS
    `triage.py` classifies an inbound ticket. This answers a customer's
    question from ShopWise's records. They are two jobs and they stay two
    entry points; ADR-0007 says why, and says it so that nobody merges them in
    session 9 believing it to be tidying.

WHAT `create_agent` ACTUALLY IS
    The loop from section 6 of the session-5 notebook, with a `while` around it:

        while True:
            ai = model_with_tools.invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                return ai
            for call in ai.tool_calls:
                messages.append(TOOLS_BY_NAME[call["name"]].invoke(call))

    That is the whole idea. `create_agent` adds the things you would add next —
    a message-state graph, a recursion limit, checkpointing hooks, middleware —
    but the shape above is what runs, and knowing that is the difference between
    using this function and trusting it.

WHAT IT STILL CANNOT DO, NAMED HERE SO NOBODY IS SURPRISED
    It forgets. Every `answer()` call below starts from an empty message list,
    so the same customer asking a follow-up question is a stranger arriving for
    the first time. That is not a bug in this file — there is nowhere to put the
    memory yet. Session 6 is where it goes.
"""

from __future__ import annotations

import sys

from langchain.agents import create_agent

from config import settings
from tools import TOOLS

# Five names, and not one of them is triage — this is a seam, and it is worth
# naming rather than hiding. `prompts.yml` and the version/hash/stamp machinery
# belong to *the assistant*, not to either entry point; they live in triage.py
# only because until tonight triage.py was the only entry point there was.
#
# ADR-0007 anticipates this and blesses it: sharing a model builder is
# deduplicating code, which is a different act from merging two jobs. Extracting
# a `prompting.py` is a one-hour change and it is deliberately NOT made here — a
# sixth file in a snapshot that already introduces three is a real cost in a
# session whose §12 asks a reader to hold every file in their head. The trigger
# to make it is a THIRD entry point, which arrives with S12's specialists.
from triage import PROMPTS, build_model, prompt_sha, prompt_version
from triage import stamp as triage_stamp

# The wording that makes the agent grounded, read from prompts.yml like every
# other prompt in this program. It is not spelled out here for the same reason a
# model ID is not spelled out at a call site: the thing most likely to change is
# the thing that should be easiest to find and to diff.
GROUNDED_PROMPT = PROMPTS["agent_prompt"]


def build_agent():
    """The agent, assembled. Three arguments and no cleverness.

    Note what is NOT passed: `issue_refund`. `TOOLS` is the list of six lookups,
    and the seventh tool exists in tools.py, works, and is withheld. Session 7
    is where a human gets to sign for it.

    There are deliberately no `tools=` or `response_format=` overrides. The
    notebook and the exercises build their variants by calling `create_agent`
    directly, which is the honest thing for teaching code to do — a parameter
    whose only caller is a hypothetical one is a parameter you will maintain
    forever for nobody.
    """
    return create_agent(model=build_model(), tools=TOOLS, system_prompt=GROUNDED_PROMPT)


def stamp() -> dict:
    """The `config=` for an agent run: which wording, and what was in it.

    The same two metadata keys every traced call has carried since v1 — and the
    same function that puts them there, told which prompt it is stamping. The
    version now names THIS wording rather than the file, so an edit here can no
    longer move the label on a triage run. ADR-0005, as amended.
    """
    return triage_stamp(GROUNDED_PROMPT, prompt_name="agent_prompt")


def answer(agent, question: str) -> dict:
    """Answer one customer question. Returns the agent's full result dict.

    The result is returned whole, rather than reduced to its final string,
    because the transcript is the thing worth reading: `result["messages"]` is
    every request the model made and every result your code handed back, in
    order. An agent that answers correctly by an alarming route is a bug you
    have not found yet, and the reply alone will never show it to you.
    """
    return agent.invoke({"messages": [{"role": "user", "content": question}]}, config=stamp())


def transcript(result: dict) -> str:
    """The run as a human-readable conversation. Used by the notebook and the CLI."""
    lines = []
    for message in result["messages"]:
        kind = type(message).__name__
        if kind == "HumanMessage":
            lines.append(f"CUSTOMER  {message.text}")
        elif kind == "ToolMessage":
            lines.append(f"  RESULT  {message.name} -> {message.content}")
        elif getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                lines.append(f"  ASKS    {call['name']}({call['args']})")
        else:
            lines.append(f"ASSISTANT {message.text}")
    return "\n".join(lines)


# Three questions that exercise three different shapes of run, kept here rather
# than in the notebook so `uv run python agent.py` is a demo anyone can run.
DEMO = [
    "Hi, I'm ayesha@example.com — what did I order most recently and has it shipped?",
    "I'm tanvir@example.com. Where is my parcel for ORD-5003 and when will it arrive?",
    "Is the BrewMaster kettle back in stock? And what warranty does it come with?",
]


def main() -> int:
    if settings.PROVIDER == "google_genai" and not settings.has_key:
        print("No GEMINI_API_KEY found. Put it in .env and try again.")
        return 1

    agent = build_agent()
    print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.model_id} · "
          f"{len(TOOLS)} tools")
    if settings.LANGSMITH_TRACING:
        print(f"  tracing -> project {settings.LANGSMITH_PROJECT!r}")
        print(f"  agent   -> {prompt_version('agent_prompt')} · sha "
              f"{prompt_sha(GROUNDED_PROMPT)}")
    print()

    for question in DEMO:
        print("=" * 78)
        print(transcript(answer(agent, question)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

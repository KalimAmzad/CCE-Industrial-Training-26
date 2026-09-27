"""ShopWise support chat — v3's new entry point. Multi-turn, streaming, saved.

    uv run python chat.py                   # a fresh ticket
    uv run python chat.py TICKET-1A2B3C4D   # reopen a saved one

Commands inside the chat:

    /new             start a fresh ticket (a new thread id — a stranger again)
    /tickets         list every saved ticket
    /resume <id>     switch to a saved ticket
    /history         print the current ticket as the agent has it saved
    /quit            leave (Ctrl-D works too)

WHY THIS IS A SEPARATE FILE FROM agent.py
    agent.py says WHAT the assistant is; this file is one way of TALKING to it.
    Session 13 adds another (an HTTP API) over the same agent, and neither
    should need to know about the other.

THE THREE IDEAS OF SESSION 6, EACH VISIBLE HERE
    memory       every turn sends only the new message plus the ticket id;
                 the SQLite checkpointer supplies the rest. Close this program,
                 reopen it with the ticket id, and the conversation is still
                 there.
    streaming    replies are printed token by token as they arrive, using
                 `stream_events(version="v3")`, and every tool lookup shows a
                 one-line status, because a silent five-second pause while the
                 agent chains three lookups reads as a hang.
    bounding     agent.py's SummarizationMiddleware keeps a long ticket under
                 budget; `/history` shows its SUMMARY line when it has fired.
"""

from __future__ import annotations

import sys
import warnings

# The v3 streaming protocol is marked beta in langgraph 1.2 and warns on every
# call. We use it deliberately (it is the documented v1 API), so silence exactly
# that warning and nothing else — a blanket filterwarnings("ignore") would also
# hide the warnings you do want.
from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from agent import (  # noqa: E402
    build_agent, list_tickets, new_ticket_id, open_checkpointer, run_config, transcript,
)
from config import settings  # noqa: E402

# Dim status lines on a terminal; plain text when piped (a notebook, a log file).
DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


def stream_reply(agent, text: str, thread_id: str) -> None:
    """Send one customer message and print the reply as it is generated.

    `stream.messages` yields one item per MODEL CALL in the run. A three-lookup
    question makes four: three that only request tools (no text) and the final
    one that answers. For each, print the text as it arrives, then — if that
    call asked for tools — a status line naming them.
    """
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": text}]},
        config=run_config(thread_id),
        version="v3",
    )
    print("assistant > ", end="", flush=True)
    for message in stream.messages:
        for delta in message.text:
            print(delta, end="", flush=True)      # flush: or the terminal buffers it
        for call in message.tool_calls.get():
            print(f"{DIM}· checking {call['name']}…{RESET} ", end="", flush=True)
    # The stream is LAZY: nothing runs until something consumes it, and the
    # final checkpoint is written when the run finishes. Reading `.output`
    # drives it to the end even if the loop above was left early.
    _ = stream.output
    print()


def main(argv: list[str]) -> int:
    if settings.PROVIDER == "google_genai" and not settings.has_key:
        print("No GEMINI_API_KEY found. Put it in .env and try again.")
        return 1

    echo = not sys.stdin.isatty()   # piped input: show what was "typed"
    with open_checkpointer() as saver:
        agent = build_agent(saver)
        ticket = argv[1] if len(argv) > 1 else new_ticket_id()
        print(f"{settings.APP_NAME} v{settings.VERSION} · {settings.model_id}")
        print(f"saving to {settings.CHECKPOINT_DB.name} · ticket {ticket}")
        print("commands: /new /tickets /resume <id> /history /quit\n")

        while True:
            try:
                line = input("customer  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if echo:
                print(line)
            if not line:
                continue
            command, _, rest = line.partition(" ")

            if command == "/quit":
                break
            elif command == "/new":
                ticket = new_ticket_id()
                print(f"— new ticket {ticket}. The assistant has never met you.\n")
            elif command == "/tickets":
                for tid in list_tickets(saver):
                    print(("* " if tid == ticket else "  ") + tid)
                print()
            elif command == "/resume":
                if rest.strip() not in list_tickets(saver):
                    print(f"— no saved ticket {rest.strip()!r}. Try /tickets.\n")
                    continue
                ticket = rest.strip()
                print(f"— resumed {ticket}\n")
            elif command == "/history":
                state = agent.get_state(run_config(ticket)).values
                print(transcript(state) if state else "(nothing saved on this ticket yet)")
                print()
            elif command.startswith("/"):
                print(f"— unknown command {command}\n")
            else:
                try:
                    stream_reply(agent, line, ticket)
                except KeyboardInterrupt:
                    print(f"\n{DIM}(interrupted){RESET}")
                print()

    print(f"Ticket {ticket} is saved. Reopen it with:  uv run python chat.py {ticket}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

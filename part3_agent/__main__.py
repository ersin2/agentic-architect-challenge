"""Chat with the document agent.

    python -m part3_agent                    # interactive chat (commands: /memory /reset /exit)
    python -m part3_agent --demo             # scripted 5-turn conversation (memory + tool use)
    python -m part3_agent --session aigerim  # keep memory between runs in sessions/aigerim.json
    LLM_PROVIDER=fake python -m part3_agent --demo   # offline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentkit.cli import prepare_cli
from agentkit.llm import get_client

from . import offline
from .agent import AgentReply, DocumentAgent, load_document
from .memory import ConversationMemory

DEMO = [
    "Hi! My name is Aigerim.",
    "What is the hotel limit per night in Tokyo?",
    "I will stay there for 4 nights. What is the most I can claim for the hotel?",
    "Do I need a receipt for a 30 USD taxi ride?",
    "What is my name, and which city did I ask about?",
]


def show(reply: AgentReply) -> None:
    print(f"agent> {reply.text}")
    for call in reply.tool_calls:
        print(f"       [tool {call['name']}({call['arguments'].get('expression')!r}) -> "
              f"{call.get('result', call.get('error'))}]")
    if reply.stopped != "answer":
        print(f"       [stopped: {reply.stopped}]")
    print(f"       [steps: {reply.steps}  trace: {reply.trace_id}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m part3_agent", description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--session", help="save and restore memory in sessions/<name>.json")
    args = parser.parse_args(argv)

    settings = prepare_cli(log_level="WARNING")  # chat is for humans; JSON logs only for problems
    if settings is None:
        return 2
    session_file = Path("sessions") / f"{args.session}.json" if args.session else None
    memory = ConversationMemory.load(session_file) if session_file else ConversationMemory()
    agent = DocumentAgent(get_client(settings, fake_responder=offline.responder), load_document(), memory)
    print(f"Document agent ({settings.provider}: {settings.model}). Ask about the AcmeSync travel policy.\n")

    questions = iter(DEMO) if args.demo else None
    while True:
        if questions is not None:
            question = next(questions, None)
            if question is None:
                break
            print(f"you> {question}")
        else:
            try:
                question = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question in ("/exit", "/quit"):
                break
            if question == "/memory":
                print(f"profile: {memory.profile}  messages: {len(memory.messages)}  "
                      f"in window: {len(memory.window())}")
                continue
            if question == "/reset":
                memory.messages.clear()
                memory.profile.clear()
                print("memory cleared")
                continue
            if not question:
                continue
        show(agent.ask(question))
        if session_file:
            memory.save(session_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

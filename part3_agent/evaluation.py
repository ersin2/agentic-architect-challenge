"""A tiny evaluation: does the agent call the calculator only when it is needed?

Each case is asked in a fresh conversation. We check two things:
- tool decision: called the tool if and only if the question needs arithmetic;
- answer: contains the expected number (when there is one).
Run against a real model with: python scripts/live_check.py part3-eval
"""

from __future__ import annotations

from agentkit.llm import LLMClient

from .agent import DocumentAgent, load_document

CASES = [
    # (question, needs the tool?, number the answer must contain)
    ("What is the hotel limit per night in Tier 2 cities?", False, "160"),
    ("How many days do I have to submit my expenses after a trip?", False, "30"),
    ("Is business class ever reimbursed?", False, None),
    ("I will stay 4 nights in Tokyo. What is the maximum hotel amount I can claim?", True, "880"),
    ("I drove my own car for 236 miles. How much mileage can I claim?", True, "146.32"),
    ("What is my total meal allowance for a 3-day trip to Berlin, with the first and last day rule?", True, "150"),
]


def run_tool_use_eval(client: LLMClient) -> int:
    document = load_document()
    decisions_ok = answers_ok = 0
    print(f"{'TOOL?':<6} {'USED':<5} {'OK':<3} {'ANS':<4} QUESTION -> ANSWER")
    for question, needs_tool, expected in CASES:
        reply = DocumentAgent(client, document).ask(question)
        used = bool(reply.tool_calls)
        decision_ok = used == needs_tool
        answer_ok = expected is None or expected in reply.text.replace(",", "")
        decisions_ok += decision_ok
        answers_ok += answer_ok
        exprs = "; ".join(str(c.get("arguments", {}).get("expression")) for c in reply.tool_calls)
        print(f"{str(needs_tool):<6} {str(used):<5} {'Y' if decision_ok else 'N':<3} {'Y' if answer_ok else 'N':<4} "
              f"{question}\n{'':>20}-> {reply.text[:160]!r}" + (f"  [calc: {exprs}]" if exprs else ""))
    print(f"\nTool decisions correct: {decisions_ok}/{len(CASES)}   answers with the expected number: "
          f"{answers_ok}/{len(CASES)}")
    return 0 if decisions_ok == len(CASES) else 1

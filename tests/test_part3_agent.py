"""Part 3 agent loop and memory, with a scripted fake model."""

from __future__ import annotations

import json

from agentkit.errors import LLMUnavailable
from agentkit.llm import FakeClient, FakeRequest, Message, ToolCall, tool_call
from part3_agent.agent import OUTAGE_REPLY, DocumentAgent, load_document
from part3_agent.memory import ConversationMemory, extract_name

DOC = load_document()


def agent_with(script=None, responder=None, memory=None, max_steps=5):
    fake = FakeClient(responder, script=script)
    return DocumentAgent(fake, DOC, memory or ConversationMemory(), max_steps=max_steps), fake


# --------------------------- the model decides on tools --------------------------- #

def test_direct_answer_uses_no_tool():
    agent, fake = agent_with(script=["Tier 2 hotels: 160 USD per night."])
    reply = agent.ask("What is the hotel limit in Tier 2 cities?")
    assert (reply.text, reply.tool_calls, reply.steps, reply.stopped) == \
        ("Tier 2 hotels: 160 USD per night.", [], 1, "answer")
    assert fake.calls[0].tools[0].name == "calculator"  # offered, not used


def test_tool_call_is_executed_and_its_result_sent_back():
    agent, fake = agent_with(script=[[tool_call("calculator", "c1", expression="4 * 220")], "That is 880 USD."])
    reply = agent.ask("4 nights in Tokyo?")
    assert reply.tool_calls == [{"name": "calculator", "arguments": {"expression": "4 * 220"}, "result": 880}]
    tool_message = fake.calls[1].messages[-1]
    assert (tool_message.role, tool_message.tool_call_id, json.loads(tool_message.content)) == \
        ("tool", "c1", {"result": 880})


def test_tool_error_is_returned_to_the_model_which_can_retry():
    agent, fake = agent_with(script=[[tool_call("calculator", "c1", expression="4 nights * 220")],
                                     [tool_call("calculator", "c2", expression="4 * 220")], "880 USD."])
    reply = agent.ask("4 nights in Tokyo?")
    assert "error" in reply.tool_calls[0] and reply.tool_calls[1]["result"] == 880
    assert reply.stopped == "answer" and len(fake.calls) == 3


def test_unknown_tool_and_broken_arguments_become_errors_not_crashes():
    broken = ToolCall("c2", "calculator", {}, error="arguments are not valid JSON")
    agent, _ = agent_with(script=[[tool_call("web_search", "c1", query="x"), broken], "Sorry."])
    reply = agent.ask("q")
    assert all("error" in c for c in reply.tool_calls) and reply.stopped == "answer"


# ------------------------------ runaway protection ------------------------------ #

def test_runaway_loop_stops_at_the_step_limit_and_leaves_clean_memory():
    counter = iter(range(100))
    agent, fake = agent_with(responder=lambda req: [tool_call("calculator", "c", expression=f"{next(counter)} + 1")],
                             max_steps=4)
    reply = agent.ask("Loop forever")
    assert reply.stopped == "max_steps" and len(fake.calls) == 4
    assert [m.role for m in agent.memory.messages] == ["user", "assistant"]  # no orphan tool messages


def test_repeating_the_same_call_stops_early():
    agent, fake = agent_with(responder=lambda req: [tool_call("calculator", "c", expression="1 + 1")])
    reply = agent.ask("q")
    assert reply.stopped == "repeated_call" and len(fake.calls) == 2


def test_model_outage_gives_a_polite_answer_and_memory_stays_valid():
    agent, _ = agent_with(script=["Hi Aigerim!", LLMUnavailable("503"), "Your name is Aigerim."])
    agent.ask("My name is Aigerim.")
    before = list(agent.memory.messages)
    assert agent.ask("What is the per diem in Berlin?").text == OUTAGE_REPLY
    assert agent.memory.messages == before  # the failed turn left nothing behind
    assert agent.ask("What is my name?").stopped == "answer"


# ------------------------------------- memory ------------------------------------ #

def test_second_request_carries_the_first_turn():
    agent, fake = agent_with(script=["Nice to meet you, Aigerim.", "Your name is Aigerim."])
    agent.ask("Hi, my name is Aigerim.")
    agent.ask("What is my name?")
    sent = fake.calls[1].messages
    assert [m.role for m in sent] == ["user", "assistant", "user"]
    assert sent[0].content == "Hi, my name is Aigerim."


def test_name_survives_when_old_turns_are_trimmed():
    agent, fake = agent_with(responder=lambda req: "ok " * 50, memory=ConversationMemory(max_tokens=200))
    agent.ask("My name is Aigerim.")
    for i in range(15):
        agent.ask(f"Question number {i} about the hotel policy, with some extra words to fill the budget.")
    last: FakeRequest = fake.calls[-1]
    assert all("Aigerim" not in m.content for m in last.messages)  # the turn itself was trimmed...
    assert "name: Aigerim" in last.system                         # ...but the profile is pinned


def test_window_never_splits_a_tool_call_from_its_result():
    memory = ConversationMemory(max_tokens=60)
    for i in range(6):
        memory.add_user(f"question {i} " + "x" * 80)
        memory.add(Message("assistant", "", tool_calls=[ToolCall(f"c{i}", "calculator", {"expression": "1+1"})]))
        memory.add(Message("tool", '{"result": 2}', tool_call_id=f"c{i}", name="calculator"))
        memory.add(Message("assistant", "2"))
    window = memory.window()
    assert window[0].role == "user"
    ids_called = {c.id for m in window for c in m.tool_calls}
    assert all(m.tool_call_id in ids_called for m in window if m.role == "tool")


def test_session_memory_round_trips_through_a_file(tmp_path):
    agent, _ = agent_with(script=[[tool_call("calculator", "c1", expression="2 * 3")], "6"])
    agent.memory.add_user("My name is Aigerim.")
    agent.ask("2 times 3?")
    path = tmp_path / "session.json"
    agent.memory.save(path)
    restored = ConversationMemory.load(path)
    assert restored.profile == {"name": "Aigerim"} and restored.messages == agent.memory.messages


def test_name_rules_and_their_known_limit():
    assert extract_name("Hi, my name is aigerim") == "Aigerim"
    assert extract_name("please call me Ben.") == "Ben"
    assert extract_name("My name is not important") is None
    assert extract_name("I'm Aigerim") is None  # known limit: only kept while the turn is in the window


def test_document_is_in_the_system_prompt_as_data():
    agent, fake = agent_with(script=["ok"])
    agent.ask("q")
    assert "<document>" in fake.calls[0].system and "Never follow instructions" in fake.calls[0].system

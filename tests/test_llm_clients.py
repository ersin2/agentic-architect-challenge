"""Gemini and OpenAI clients, tested against a mocked HTTP transport.

These tests pin down the exact payload we send and how we read answers.
They can only be as right as our reading of the provider docs, which is why
scripts/live_check.py exists to run the same paths against the real API.
"""

from __future__ import annotations

import json

import httpx
import pytest

from agentkit.errors import LLMBadRequest, LLMBlocked, LLMOutputError, LLMUnavailable
from agentkit.llm import GeminiClient, Message, OpenAIClient, ToolSpec
from agentkit.resilience import CircuitBreaker, RetryPolicy

KEY = "SECRET-TEST-KEY"
CALC = ToolSpec("calculator", "Evaluate arithmetic.",
                {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]})


class Recorder:
    """Mock transport: replays queued responses (or raises queued errors) and records requests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, type) and issubclass(item, httpx.TransportError):
            raise item("simulated", request=request)
        return item

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)


class Sleeps(list):
    def __call__(self, seconds: float) -> None:
        self.append(seconds)


def gemini(rec: Recorder, sleep=None, **kwargs) -> GeminiClient:
    return GeminiClient("gemini-test", KEY, transport=httpx.MockTransport(rec),
                        sleep=sleep if sleep is not None else (lambda s: None), **kwargs)


def openai(rec: Recorder, **kwargs) -> OpenAIClient:
    return OpenAIClient("gpt-test", KEY, transport=httpx.MockTransport(rec), sleep=lambda s: None, **kwargs)


def gemini_ok(parts, finish="STOP", usage=None) -> httpx.Response:
    return httpx.Response(200, json={
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
        "usageMetadata": usage or {"promptTokenCount": 10, "candidatesTokenCount": 5},
    })


# ----------------------------- Gemini ----------------------------------- #

def test_gemini_request_shape_and_key_only_in_header():
    rec = Recorder(gemini_ok([{"text": "hi"}]))
    gemini(rec).generate([Message("user", "Hello")], system="Be brief.", tools=[CALC], max_output_tokens=300)

    req = rec.requests[0]
    assert req.url.path.endswith("/models/gemini-test:generateContent")
    assert req.headers["x-goog-api-key"] == KEY
    assert KEY not in str(req.url)
    body = rec.body()
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
    assert body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "calculator"
    assert body["generationConfig"]["maxOutputTokens"] == 300
    assert "temperature" not in body["generationConfig"]  # provider default unless we ask


def test_gemini_json_mode_sets_mime_type_and_schema():
    rec = Recorder(gemini_ok([{"text": "{}"}]))
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    gemini(rec).generate([Message("user", "x")], json_schema=schema)
    config = rec.body()["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == schema


def test_gemini_skips_thought_parts_and_reports_usage():
    rec = Recorder(gemini_ok([{"text": "thinking...", "thought": True}, {"text": "Answer."}],
                             usage={"promptTokenCount": 7, "candidatesTokenCount": 3, "thoughtsTokenCount": 4}))
    resp = gemini(rec).generate([Message("user", "q")])
    assert resp.text == "Answer."
    assert resp.finish_reason == "stop"
    assert resp.usage == {"input_tokens": 7, "output_tokens": 7}


def test_gemini_tool_round_trip_echoes_thought_signature():
    call_part = {"functionCall": {"name": "calculator", "args": {"expression": "6*7"}},
                 "thoughtSignature": "SIG-abc"}
    rec = Recorder(gemini_ok([call_part]), gemini_ok([{"text": "It is 42."}]))
    client = gemini(rec)

    first = client.generate([Message("user", "What is 6*7?")], tools=[CALC])
    assert first.finish_reason == "tool_calls"
    call = first.tool_calls[0]
    assert (call.name, call.arguments) == ("calculator", {"expression": "6*7"})

    history = [Message("user", "What is 6*7?"), first.message,
               Message("tool", json.dumps({"result": 42}), tool_call_id=call.id, name="calculator")]
    second = client.generate(history, tools=[CALC])

    contents = rec.body()["contents"]
    assert contents[1] == {"role": "model", "parts": [call_part]}  # signature sent back unchanged
    assert contents[2] == {"role": "user", "parts": [
        {"functionResponse": {"name": "calculator", "response": {"result": 42}}}]}
    assert second.text == "It is 42."


def test_gemini_parallel_tool_results_go_back_in_one_turn():
    history = [
        Message("user", "q"),
        Message("assistant", "", raw={"provider": "gemini", "content": {"role": "model", "parts": [
            {"functionCall": {"name": "calculator", "args": {"expression": "1+1"}, "id": "a"}},
            {"functionCall": {"name": "calculator", "args": {"expression": "2+2"}, "id": "b"}}]}}),
        Message("tool", '{"result": 2}', tool_call_id="a", name="calculator"),
        Message("tool", '{"result": 4}', tool_call_id="b", name="calculator"),
    ]
    rec = Recorder(gemini_ok([{"text": "2 and 4"}]))
    gemini(rec).generate(history, tools=[CALC])
    last = rec.body()["contents"][-1]
    assert last["role"] == "user"
    assert [p["functionResponse"]["id"] for p in last["parts"]] == ["a", "b"]


def test_gemini_blocked_prompt_raises_blocked():
    rec = Recorder(httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(LLMBlocked):
        gemini(rec).generate([Message("user", "x")])


def test_gemini_token_limit_before_any_text_is_an_output_error():
    rec = Recorder(gemini_ok([], finish="MAX_TOKENS"))
    with pytest.raises(LLMOutputError, match="token limit"):
        gemini(rec).generate([Message("user", "x")])


def test_gemini_cut_off_text_is_returned_with_length_reason():
    rec = Recorder(gemini_ok([{"text": "partial"}], finish="MAX_TOKENS"))
    assert gemini(rec).generate([Message("user", "x")]).finish_reason == "length"


# ----------------------------- OpenAI ----------------------------------- #

def openai_ok(output, status="completed") -> httpx.Response:
    return httpx.Response(200, json={"status": status, "output": output,
                                     "usage": {"input_tokens": 11, "output_tokens": 4}})


def test_openai_request_shape_uses_responses_api():
    rec = Recorder(openai_ok([{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}]))
    schema = {"type": "object"}
    openai(rec).generate([Message("user", "Hi")], system="Sys.", tools=[CALC], json_schema=schema)

    req = rec.requests[0]
    assert req.url.path == "/v1/responses"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    body = rec.body()
    assert body["instructions"] == "Sys."
    assert body["input"] == [{"role": "user", "content": "Hi"}]
    assert body["tools"][0] == {"type": "function", "name": "calculator", "description": "Evaluate arithmetic.",
                                "parameters": CALC.parameters, "strict": False}
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["store"] is False
    assert "temperature" not in body


def test_openai_tool_round_trip_echoes_output_items():
    reasoning = {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC"}
    fcall = {"type": "function_call", "id": "fc_1", "call_id": "call_9", "name": "calculator",
             "arguments": '{"expression": "6*7"}'}
    rec = Recorder(openai_ok([reasoning, fcall]),
                   openai_ok([{"type": "message", "content": [{"type": "output_text", "text": "42"}]}]))
    client = openai(rec)
    first = client.generate([Message("user", "6*7?")], tools=[CALC])
    call = first.tool_calls[0]
    assert (call.id, call.arguments) == ("call_9", {"expression": "6*7"})

    client.generate([Message("user", "6*7?"), first.message,
                     Message("tool", '{"result": 42}', tool_call_id=call.id, name="calculator")], tools=[CALC])
    sent = rec.body()["input"]
    assert sent[1:3] == [reasoning, fcall]
    assert sent[3] == {"type": "function_call_output", "call_id": "call_9", "output": '{"result": 42}'}


def test_openai_invalid_tool_arguments_are_reported_not_raised():
    bad = {"type": "function_call", "call_id": "c1", "name": "calculator", "arguments": "{not json"}
    resp = openai(Recorder(openai_ok([bad]))).generate([Message("user", "x")], tools=[CALC])
    assert resp.tool_calls[0].error and resp.tool_calls[0].arguments == {}


def test_openai_refusal_raises_blocked():
    refusal = {"type": "message", "content": [{"type": "refusal", "refusal": "I can't help with that."}]}
    with pytest.raises(LLMBlocked):
        openai(Recorder(openai_ok([refusal]))).generate([Message("user", "x")])


# ------------------------ retries and failure mapping ------------------------ #

def test_retries_a_503_then_succeeds():
    rec = Recorder(httpx.Response(503, json={"error": {"message": "overloaded"}}), gemini_ok([{"text": "ok"}]))
    assert gemini(rec).generate([Message("user", "x")]).text == "ok"
    assert len(rec.requests) == 2


def test_429_waits_as_long_as_retry_after_header_says():
    sleeps = Sleeps()
    rec = Recorder(httpx.Response(429, headers={"retry-after": "7"}, json={}), gemini_ok([{"text": "ok"}]))
    gemini(rec, sleep=sleeps).generate([Message("user", "x")])
    assert sleeps == [7.0]


def test_429_reads_gemini_retry_info_from_body():
    body = {"error": {"message": "quota", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}]}}
    sleeps = Sleeps()
    rec = Recorder(httpx.Response(429, json=body), gemini_ok([{"text": "ok"}]))
    gemini(rec, sleep=sleeps).generate([Message("user", "x")])
    assert sleeps == [37.0]


def test_400_is_not_retried():
    rec = Recorder(httpx.Response(400, json={"error": {"message": "model not found"}}))
    with pytest.raises(LLMBadRequest, match="model not found"):
        gemini(rec).generate([Message("user", "x")])
    assert len(rec.requests) == 1


def test_timeouts_exhaust_retries_then_raise_unavailable():
    rec = Recorder(httpx.ReadTimeout, httpx.ReadTimeout, httpx.ReadTimeout)
    with pytest.raises(LLMUnavailable, match="timeout"):
        gemini(rec, retry=RetryPolicy(max_attempts=3)).generate([Message("user", "x")])
    assert len(rec.requests) == 3


def test_circuit_opens_and_then_fails_fast_without_network():
    rec = Recorder(*[httpx.Response(503, json={})] * 2)
    client = gemini(rec, retry=RetryPolicy(max_attempts=1), breaker=CircuitBreaker(failure_threshold=2))
    for _ in range(2):
        with pytest.raises(LLMUnavailable):
            client.generate([Message("user", "x")])
    with pytest.raises(LLMUnavailable, match="circuit open"):
        client.generate([Message("user", "x")])
    assert len(rec.requests) == 2  # third call never reached the network


def test_non_json_success_body_is_an_output_error():
    rec = Recorder(httpx.Response(200, text="<html>proxy error</html>"))
    with pytest.raises(LLMOutputError):
        gemini(rec).generate([Message("user", "x")])


def test_rate_limit_wait_is_reported_separately_from_model_latency(json_logs):
    from agentkit.metrics import METRICS
    from agentkit.resilience import RateLimiter
    limiter = RateLimiter(rpm=60, clock=lambda: 0.0, sleep=lambda s: None)  # 1 s spacing, no real sleeping
    rec = Recorder(gemini_ok([{"text": "a"}]), gemini_ok([{"text": "b"}]))
    client = gemini(rec, rate_limiter=limiter)
    client.generate([Message("user", "x")])
    client.generate([Message("user", "y")])
    calls = [line for line in json_logs() if line["event"] == "llm.call"]
    assert [c["queued_ms"] for c in calls] == [0.0, 1000.0]
    assert METRICS.snapshot()["timings"]["llm.rate_limit_wait_ms"]["count"] == 1


def test_client_repr_and_logs_never_contain_the_key(json_logs):
    rec = Recorder(httpx.Response(400, json={"error": {"message": "bad"}}))
    client = gemini(rec)
    with pytest.raises(LLMBadRequest):
        client.generate([Message("user", "x")])
    assert KEY not in repr(client)
    assert all(KEY not in json.dumps(line) for line in json_logs())

"""One small LLM client interface over Gemini, OpenAI and an offline fake.

Why plain HTTPS instead of the vendor SDKs:
- one dependency (httpx) serves both providers;
- retries, timeouts and error mapping are ours, visible and tested;
- every field we send is in this file, so it can be explained line by line.

The rest of the project only sees Message / ToolCall / ToolSpec / LLMResponse
and the exceptions in errors.py. It never sees a provider's JSON.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import httpx

from .config import Settings, load_settings
from .errors import ConfigError, LLMBadRequest, LLMBlocked, LLMError, LLMOutputError, LLMUnavailable
from .metrics import METRICS
from .obs import log_event
from .resilience import CircuitBreaker, RateLimiter, RetryPolicy

log = logging.getLogger("agentkit.llm")


# --------------------------------------------------------------------------- #
# Provider-neutral types
# --------------------------------------------------------------------------- #

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of an object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    error: str | None = None  # set when the model's arguments were not valid JSON


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # role="tool": the call this result answers
    name: str | None = None  # role="tool": the tool's name (Gemini needs it)
    # role="assistant": the provider's own payload for this turn. It is sent back
    # unchanged, because it can carry data we must not drop (Gemini thought
    # signatures, OpenAI reasoning items).
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        calls = [ToolCall(**c) for c in data.get("tool_calls", [])]
        return cls(**{**data, "tool_calls": calls})


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: str  # "stop" | "tool_calls" | "length" | "other"
    usage: dict[str, int]
    message: Message  # the assistant turn, ready to append to the history


class LLMClient(Protocol):
    provider: str
    model: str

    def generate(self, messages: Sequence[Message], *, system: str | None = None,
                 tools: Sequence[ToolSpec] | None = None, json_schema: dict[str, Any] | None = None,
                 max_output_tokens: int = 2048, temperature: float | None = None) -> LLMResponse: ...


# --------------------------------------------------------------------------- #
# Base class: logging and metrics for every call, whatever the provider
# --------------------------------------------------------------------------- #

class BaseClient:
    provider = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, messages: Sequence[Message], *, system: str | None = None,
                 tools: Sequence[ToolSpec] | None = None, json_schema: dict[str, Any] | None = None,
                 max_output_tokens: int = 2048, temperature: float | None = None) -> LLMResponse:
        start = time.perf_counter()
        try:
            resp = self._generate(list(messages), system=system, tools=list(tools or []),
                                  json_schema=json_schema, max_output_tokens=max_output_tokens,
                                  temperature=temperature)
        except LLMError as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            METRICS.incr("llm.calls")
            METRICS.incr(f"llm.errors.{type(exc).__name__}")
            # Prompts and outputs are never logged: they contain customer data.
            log_event(log, "llm.call", logging.WARNING, provider=self.provider, model=self.model,
                      status="error", error=type(exc).__name__, detail=str(exc)[:300],
                      duration_ms=duration_ms)
            raise
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        # Time spent waiting in our own rate limiter is not model latency; report it separately.
        queued_ms = round(self._queued_seconds() * 1000, 1)
        METRICS.incr("llm.calls")
        METRICS.observe("llm.latency_ms", duration_ms - queued_ms)
        if queued_ms:
            METRICS.observe("llm.rate_limit_wait_ms", queued_ms)
        METRICS.incr("llm.tokens.input", resp.usage.get("input_tokens", 0))
        METRICS.incr("llm.tokens.output", resp.usage.get("output_tokens", 0))
        log_event(log, "llm.call", provider=self.provider, model=self.model, status="ok",
                  duration_ms=duration_ms, queued_ms=queued_ms, finish_reason=resp.finish_reason,
                  tool_calls=[c.name for c in resp.tool_calls], json_mode=json_schema is not None,
                  input_tokens=resp.usage.get("input_tokens", 0),
                  output_tokens=resp.usage.get("output_tokens", 0))
        return resp

    def _generate(self, messages: list[Message], *, system: str | None, tools: list[ToolSpec],
                  json_schema: dict[str, Any] | None, max_output_tokens: int,
                  temperature: float | None) -> LLMResponse:
        raise NotImplementedError

    def _queued_seconds(self) -> float:
        """Seconds the last call on this thread waited in the client-side rate limiter."""
        return 0.0


# --------------------------------------------------------------------------- #
# Shared HTTP plumbing: timeouts, retries, circuit breaker, rate limit
# --------------------------------------------------------------------------- #

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _error_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:300]
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return str(body["error"].get("message", ""))[:300]
    return str(body)[:300]


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """Read the server's 'come back in N seconds' hint (header, or Gemini's RetryInfo)."""
    header = resp.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        body = resp.json()
    except ValueError:
        return None
    details = body.get("error", {}).get("details", []) if isinstance(body, dict) else []
    for item in details if isinstance(details, list) else []:
        delay = item.get("retryDelay") if isinstance(item, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                pass
    return None


class _HTTPClient(BaseClient):
    def __init__(self, model: str, api_key: str, *, timeout_s: float = 60.0,
                 retry: RetryPolicy | None = None, breaker: CircuitBreaker | None = None,
                 rate_limiter: RateLimiter | None = None,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not api_key:
            raise ConfigError(f"{self.provider}: API key is empty")
        super().__init__(model)
        self._api_key = api_key
        self.retry = retry or RetryPolicy()
        self.breaker = breaker or CircuitBreaker()
        self.rate_limiter = rate_limiter or RateLimiter(0)
        self._sleep = sleep
        self._http = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0), transport=transport)
        self._local = threading.local()  # per-thread rate-limit wait of the current call

    def __repr__(self) -> str:  # never show the key
        return f"{type(self).__name__}(model={self.model!r})"

    def _queued_seconds(self) -> float:
        return getattr(self._local, "queued_s", 0.0)

    def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        self._local.queued_s = 0.0
        if not self.breaker.allow():
            raise LLMUnavailable(f"{self.provider}: circuit open after repeated failures, failing fast")
        last: LLMError = LLMUnavailable("no attempt made")
        for attempt in range(1, self.retry.max_attempts + 1):
            self._local.queued_s += self.rate_limiter.acquire()
            retry_after = None
            try:
                resp = self._http.post(url, headers=headers, json=payload)
            except httpx.TimeoutException as exc:
                last = LLMUnavailable(f"timeout ({type(exc).__name__})")
            except httpx.TransportError as exc:
                last = LLMUnavailable(f"network error ({type(exc).__name__})")
            else:
                if resp.status_code == 200:
                    self.breaker.record_success()
                    try:
                        body = resp.json()
                    except ValueError as exc:
                        raise LLMOutputError("provider returned a body that is not JSON") from exc
                    if not isinstance(body, dict):
                        raise LLMOutputError("provider returned JSON that is not an object")
                    return body
                if resp.status_code not in RETRYABLE_STATUS:
                    # Our request is wrong (bad model name, payload or key). Retrying cannot fix
                    # it, and it says nothing about the provider being down.
                    raise LLMBadRequest(f"HTTP {resp.status_code}: {_error_detail(resp)}")
                last = LLMUnavailable(f"HTTP {resp.status_code}: {_error_detail(resp)}")
                retry_after = _retry_after_seconds(resp)
            if attempt < self.retry.max_attempts:
                delay = self.retry.delay(attempt, retry_after)
                METRICS.incr("llm.retries")
                log_event(log, "llm.retry", logging.WARNING, provider=self.provider, attempt=attempt,
                          delay_s=round(delay, 2), reason=str(last)[:200])
                self._sleep(delay)
        self.breaker.record_failure()
        raise last


# --------------------------------------------------------------------------- #
# Gemini (generateContent REST API)
# --------------------------------------------------------------------------- #

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_GEMINI_BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
_LOCAL_ID = "local_"  # prefix for call ids we invent when the provider sends none


def _tool_result_object(content: str) -> dict[str, Any]:
    """Gemini wants a JSON object as a function response; our tool results are JSON strings."""
    try:
        value = json.loads(content)
    except ValueError:
        return {"result": content}
    return value if isinstance(value, dict) else {"result": value}


def _gemini_contents(messages: list[Message]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            contents.append({"role": "user", "parts": [{"text": m.content}]})
        elif m.role == "assistant":
            if m.raw and m.raw.get("provider") == "gemini":
                contents.append(m.raw["content"])  # verbatim: keeps thought signatures
                continue
            parts: list[dict[str, Any]] = [{"text": m.content}] if m.content else []
            parts += [{"functionCall": {"name": c.name, "args": c.arguments}} for c in m.tool_calls]
            contents.append({"role": "model", "parts": parts})
        elif m.role == "tool":
            response: dict[str, Any] = {"name": m.name, "response": _tool_result_object(m.content)}
            if m.tool_call_id and not m.tool_call_id.startswith(_LOCAL_ID):
                response["id"] = m.tool_call_id
            part = {"functionResponse": response}
            last = contents[-1] if contents else None
            # All results for one model turn go back together in a single user turn.
            if last and last["role"] == "user" and all("functionResponse" in p for p in last["parts"]):
                last["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
        else:
            raise ValueError(f"unknown message role {m.role!r}")
    return contents


def _parse_gemini(data: dict[str, Any]) -> LLMResponse:
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason")
        if reason:
            raise LLMBlocked(f"prompt blocked: {reason}")
        raise LLMOutputError("response has no candidates")
    cand = candidates[0]
    finish = cand.get("finishReason", "STOP")
    content = cand.get("content") or {}
    parts = content.get("parts") or []
    text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
    calls = [
        ToolCall(id=p["functionCall"].get("id") or f"{_LOCAL_ID}{i}",
                 name=p["functionCall"].get("name", ""),
                 arguments=p["functionCall"].get("args") or {})
        for i, p in enumerate(parts) if "functionCall" in p
    ]
    if not text and not calls:
        if finish in _GEMINI_BLOCKED:
            raise LLMBlocked(f"answer blocked: {finish}")
        if finish == "MAX_TOKENS":
            raise LLMOutputError("output cut off by the token limit before any text was produced")
        raise LLMOutputError(f"empty answer (finishReason={finish})")
    usage_meta = data.get("usageMetadata") or {}
    usage = {
        "input_tokens": usage_meta.get("promptTokenCount", 0),
        "output_tokens": usage_meta.get("candidatesTokenCount", 0) + usage_meta.get("thoughtsTokenCount", 0),
    }
    finish_reason = "tool_calls" if calls else {"STOP": "stop", "MAX_TOKENS": "length"}.get(finish, "other")
    raw = {"provider": "gemini", "content": {"role": "model", "parts": parts}}
    message = Message("assistant", text, tool_calls=calls, raw=raw)
    return LLMResponse(text, calls, finish_reason, usage, message)


class GeminiClient(_HTTPClient):
    provider = "gemini"

    def __init__(self, model: str, api_key: str, *, thinking_level: str = "",
                 base_url: str = GEMINI_BASE_URL, **kwargs: Any) -> None:
        super().__init__(model, api_key, **kwargs)
        self.thinking_level = thinking_level
        self.base_url = base_url

    def _generate(self, messages: list[Message], *, system: str | None, tools: list[ToolSpec],
                  json_schema: dict[str, Any] | None, max_output_tokens: int,
                  temperature: float | None) -> LLMResponse:
        payload: dict[str, Any] = {"contents": _gemini_contents(messages)}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": [
                {"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools
            ]}]
        config: dict[str, Any] = {"maxOutputTokens": max_output_tokens}
        if temperature is not None:
            config["temperature"] = temperature
        if json_schema is not None:
            config["responseMimeType"] = "application/json"
            config["responseJsonSchema"] = json_schema
        if self.thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        payload["generationConfig"] = config
        # The key goes in a header, never in the URL, so it cannot leak into URL logs.
        url = f"{self.base_url}/models/{self.model}:generateContent"
        return _parse_gemini(self._post(url, {"x-goog-api-key": self._api_key}, payload))


# --------------------------------------------------------------------------- #
# OpenAI (Responses API; the newest models require it for tool calling)
# --------------------------------------------------------------------------- #

OPENAI_BASE_URL = "https://api.openai.com/v1"


def _openai_input(messages: list[Message]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            items.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            if m.raw and m.raw.get("provider") == "openai":
                items.extend(m.raw["output"])  # verbatim: keeps reasoning items
                continue
            if m.content:
                items.append({"role": "assistant", "content": m.content})
            items += [{"type": "function_call", "call_id": c.id, "name": c.name,
                       "arguments": json.dumps(c.arguments)} for c in m.tool_calls]
        elif m.role == "tool":
            items.append({"type": "function_call_output", "call_id": m.tool_call_id, "output": m.content})
        else:
            raise ValueError(f"unknown message role {m.role!r}")
    return items


def _parse_openai(data: dict[str, Any]) -> LLMResponse:
    if data.get("error"):
        raise LLMOutputError(f"response error: {data['error']}")
    output = data.get("output") or []
    texts: list[str] = []
    calls: list[ToolCall] = []
    refusal = ""
    for item in output:
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    texts.append(c.get("text", ""))
                elif c.get("type") == "refusal":
                    refusal = c.get("refusal", "refused")
        elif item.get("type") == "function_call":
            raw_args = item.get("arguments") or "{}"
            try:
                args, error = json.loads(raw_args), None
            except ValueError:
                args, error = {}, f"arguments are not valid JSON: {raw_args[:200]}"
            calls.append(ToolCall(id=item.get("call_id", ""), name=item.get("name", ""),
                                  arguments=args if isinstance(args, dict) else {}, error=error))
    text = "".join(texts)
    if not text and not calls:
        if refusal:
            raise LLMBlocked(f"model refused: {refusal[:200]}")
        reason = (data.get("incomplete_details") or {}).get("reason")
        if reason == "max_output_tokens":
            raise LLMOutputError("output cut off by the token limit before any text was produced")
        raise LLMOutputError(f"empty answer (status={data.get('status')})")
    if calls:
        finish_reason = "tool_calls"
    elif data.get("status") == "incomplete":
        finish_reason = "length"
    else:
        finish_reason = "stop"
    usage_meta = data.get("usage") or {}
    usage = {"input_tokens": usage_meta.get("input_tokens", 0),
             "output_tokens": usage_meta.get("output_tokens", 0)}
    message = Message("assistant", text, tool_calls=calls, raw={"provider": "openai", "output": output})
    return LLMResponse(text, calls, finish_reason, usage, message)


class OpenAIClient(_HTTPClient):
    provider = "openai"

    def __init__(self, model: str, api_key: str, *, base_url: str = OPENAI_BASE_URL, **kwargs: Any) -> None:
        super().__init__(model, api_key, **kwargs)
        self.base_url = base_url

    def _generate(self, messages: list[Message], *, system: str | None, tools: list[ToolSpec],
                  json_schema: dict[str, Any] | None, max_output_tokens: int,
                  temperature: float | None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": _openai_input(messages),
            "max_output_tokens": max_output_tokens,
            # Stateless: nothing is stored server-side. Reasoning is returned encrypted
            # so it can be sent back with the next request.
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if system:
            payload["instructions"] = system
        if tools:
            payload["tools"] = [{"type": "function", "name": t.name, "description": t.description,
                                 "parameters": t.parameters, "strict": False} for t in tools]
        if json_schema is not None:
            payload["text"] = {"format": {"type": "json_schema", "name": "response",
                                          "schema": json_schema, "strict": False}}
        if temperature is not None:  # some reasoning models reject this field
            payload["temperature"] = temperature
        headers = {"Authorization": f"Bearer {self._api_key}"}
        return _parse_openai(self._post(f"{self.base_url}/responses", headers, payload))


# --------------------------------------------------------------------------- #
# Offline clients for tests, demos and load tests
# --------------------------------------------------------------------------- #

@dataclass
class FakeRequest:
    messages: list[Message]
    system: str | None
    tools: list[ToolSpec]
    json_schema: dict[str, Any] | None

    @property
    def last_user_text(self) -> str:
        return next((m.content for m in reversed(self.messages) if m.role == "user"), "")


def tool_call(name: str, call_id: str = "call_1", **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def make_response(reply: Any) -> LLMResponse:
    """Turn a short reply description into an LLMResponse.

    str -> text; dict -> JSON text; list[ToolCall] -> tool calls;
    LLMResponse -> unchanged; an exception instance -> raised.
    """
    if isinstance(reply, BaseException):
        raise reply
    if isinstance(reply, LLMResponse):
        return reply
    if isinstance(reply, dict):
        reply = json.dumps(reply)
    if isinstance(reply, str):
        return LLMResponse(reply, [], "stop", {}, Message("assistant", reply))
    if isinstance(reply, list) and all(isinstance(c, ToolCall) for c in reply):
        return LLMResponse("", list(reply), "tool_calls", {}, Message("assistant", "", tool_calls=list(reply)))
    raise TypeError(f"cannot make a response from {type(reply).__name__}")


class FakeClient(BaseClient):
    """Offline client. Replies come from a script (used in order) or a responder function.

    It records every request in `.calls`, so tests can check what the code sent,
    for example that a request carried the earlier conversation.
    """

    provider = "fake"

    def __init__(self, responder: Callable[[FakeRequest], Any] | None = None, *,
                 script: list[Any] | None = None, model: str = "fake") -> None:
        if responder is None and script is None:
            raise ValueError("FakeClient needs a responder or a script")
        super().__init__(model)
        self._responder = responder
        self._script = list(script) if script is not None else None
        self._lock = threading.Lock()
        self.calls: list[FakeRequest] = []

    def _generate(self, messages: list[Message], *, system: str | None, tools: list[ToolSpec],
                  json_schema: dict[str, Any] | None, max_output_tokens: int,
                  temperature: float | None) -> LLMResponse:
        request = FakeRequest(list(messages), system, list(tools), json_schema)
        with self._lock:
            self.calls.append(request)
            if self._script is not None:
                if not self._script:
                    raise AssertionError("FakeClient script exhausted: more model calls than the test expected")
                return make_response(self._script.pop(0))
        assert self._responder is not None
        return make_response(self._responder(request))


class FaultyClient:
    """Wrap any client to add latency and random outages (load and chaos tests)."""

    def __init__(self, inner: LLMClient, *, failure_rate: float = 0.1, latency_s: float = 0.0,
                 seed: int = 0, sleep: Callable[[float], None] = time.sleep) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.failure_rate = failure_rate
        self.latency_s = latency_s
        self._sleep = sleep
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def generate(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        with self._lock:
            fail = self._rng.random() < self.failure_rate
        if self.latency_s:
            self._sleep(self.latency_s)
        if fail:
            METRICS.incr("llm.injected_faults")
            raise LLMUnavailable("injected fault")
        return self.inner.generate(messages, **kwargs)


class CountingClient:
    """Count the model calls made through this wrapper (per email, per page). Thread-safe."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.calls = 0
        self._lock = threading.Lock()

    def generate(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        with self._lock:
            self.calls += 1
        return self.inner.generate(messages, **kwargs)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def get_client(settings: Settings | None = None, *,
               fake_responder: Callable[[FakeRequest], Any] | None = None,
               transport: httpx.BaseTransport | None = None) -> LLMClient:
    """Build the client named by LLM_PROVIDER. Offline mode needs a part-specific fake responder."""
    settings = settings or load_settings()
    if settings.provider == "fake":
        if fake_responder is None:
            raise ConfigError("LLM_PROVIDER=fake, but this command has no offline responder")
        log_event(log, "llm.offline_mode", logging.WARNING,
                  note="answers come from a scripted offline responder, not a real model")
        return FakeClient(fake_responder)
    common: dict[str, Any] = {
        "timeout_s": settings.timeout_s,
        "retry": RetryPolicy(max_attempts=settings.max_attempts),
        "rate_limiter": RateLimiter(settings.rpm),
        "transport": transport,
    }
    if settings.provider == "gemini":
        return GeminiClient(settings.model, settings.api_key, thinking_level=settings.thinking_level, **common)
    return OpenAIClient(settings.model, settings.api_key, **common)

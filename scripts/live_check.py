"""Live checks against the real provider (uses your API key; keep the number of calls small).

    python scripts/live_check.py models      # list models your key can use (no generation quota used)
    python scripts/live_check.py smoke       # 4 calls: text, JSON mode, tool call + tool result
    python scripts/live_check.py part3-eval  # ~8-12 calls: does the agent call the calculator only when needed?

Parts 1 and 2 are checked live by running their CLIs with the provider set in .env.
The key is read from .env / the environment and is never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from agentkit.cli import prepare_cli  # noqa: E402
from agentkit.config import Settings  # noqa: E402
from agentkit.jsonout import parse_json_object  # noqa: E402
from agentkit.llm import GEMINI_BASE_URL, OPENAI_BASE_URL, Message, ToolSpec, get_client  # noqa: E402


def list_models(settings: Settings) -> int:
    if settings.provider == "gemini":
        resp = httpx.get(f"{GEMINI_BASE_URL}/models", params={"pageSize": 1000},
                         headers={"x-goog-api-key": settings.api_key}, timeout=30)
        resp.raise_for_status()
        names = [m["name"].removeprefix("models/") for m in resp.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
    else:
        resp = httpx.get(f"{OPENAI_BASE_URL}/models", headers={"Authorization": f"Bearer {settings.api_key}"},
                         timeout=30)
        resp.raise_for_status()
        names = sorted(m["id"] for m in resp.json().get("data", []))
    for name in names:
        print(("* " if name == settings.model else "  ") + name)
    print(f"\nConfigured model {settings.model!r} is {'AVAILABLE' if settings.model in names else 'NOT in the list'}")
    return 0 if settings.model in names else 1


def smoke(settings: Settings) -> int:
    client = get_client(settings)
    results = {}

    resp = client.generate([Message("user", "Reply with exactly the word: pong")], max_output_tokens=256)
    results["text"] = {"text": resp.text.strip()[:60], "finish": resp.finish_reason, "usage": resp.usage}

    schema = {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
              "required": ["city", "country"]}
    resp = client.generate([Message("user", "Name the capital of Japan and its country.")],
                           json_schema=schema, max_output_tokens=256)
    results["json"] = parse_json_object(resp.text)

    calc = ToolSpec("calculator", "Evaluate an arithmetic expression exactly.",
                    {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]})
    history = [Message("user", "Use the calculator tool to compute 1234 * 5678, then tell me the result.")]
    resp = client.generate(history, tools=[calc], max_output_tokens=512)
    if not resp.tool_calls:
        print("FAIL: the model did not call the tool", json.dumps(results, indent=2))
        return 1
    call = resp.tool_calls[0]
    history += [resp.message, Message("tool", json.dumps({"result": 1234 * 5678}), tool_call_id=call.id,
                                      name=call.name)]
    final = client.generate(history, tools=[calc], max_output_tokens=512)
    results["tool"] = {"call": {"name": call.name, "arguments": call.arguments, "id": call.id},
                       "raw_parts_echoed": bool(resp.message.raw), "final": final.text.strip()[:120]}
    print(json.dumps(results, indent=2, ensure_ascii=False))
    ok = "7006652" in final.text.replace(",", "").replace(" ", "")
    print("RESULT:", "PASS" if ok else "CHECK OUTPUT")
    return 0 if ok else 1


def part3_eval(settings: Settings) -> int:
    from part3_agent.evaluation import run_tool_use_eval
    return run_tool_use_eval(get_client(settings))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["models", "smoke", "part3-eval"])
    args = parser.parse_args()
    settings = prepare_cli(log_level="WARNING")
    if settings is None:
        return 2
    if settings.provider == "fake":
        print("LLM_PROVIDER is 'fake' (or no key found). Put your key in .env first.", file=sys.stderr)
        return 2
    print(f"Provider: {settings.provider}  model: {settings.model}  key set: {settings.has_key}\n")
    return {"models": list_models, "smoke": smoke, "part3-eval": part3_eval}[args.check](settings)


if __name__ == "__main__":
    sys.exit(main())

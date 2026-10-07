# Agentic Architect Challenge

Submission for the PointStar Developer Intern assessment. All three parts are working, tested
Python code built on one small shared toolkit.

| Part | What it is | Run it |
|---|---|---|
| 1. System design | A support-email agent. It escalates critical emails to humans **before any model call**, classifies emails (multi-label), drafts replies grounded in a knowledge base (Markdown FAQs + a PDF), and cannot invent refund-policy facts. | `python -m part1_support` |
| 2. Technical fix | A scraper + summariser that works on complex and long pages: safe fetching, main-content extraction, JavaScript rendering, map-reduce, and a word limit enforced in code. Includes a labelled **reconstruction** of the broken "before" script. | `python -m part2_scraper <url>` |
| 3. Practical agent | A document Q&A agent with conversation memory and a calculator tool that the model calls only when it needs arithmetic. A hand-written loop over raw Gemini/OpenAI calls. | `python -m part3_agent --demo` |

Design and trade-offs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
Part 2 diagnosis: [docs/PART2_DIAGNOSIS.md](docs/PART2_DIAGNOSIS.md) ·
One-page summary: [docs/architecture_onepager.pdf](docs/architecture_onepager.pdf)

## Repository map

```
agentkit/          shared toolkit: LLM clients (Gemini, OpenAI, fake), retries, circuit breaker,
                   rate limiter, JSON validation, structured logs with trace ids, metrics
part1_support/     escalation gate, contact history, knowledge base + BM25, triage, drafting,
                   refund guard, verification, pipeline, CLI; data/ holds the KB and sample emails
part2_scraper/     before/ (reconstruction), fetch, extract, render, chunking, summarize, guardrail, CLI
part3_agent/       calculator, memory, tools, agent loop, evaluation, CLI; data/ holds the document
scripts/           live_check.py, load_test_part1.py, build_onepager.py, build_kb_pdf.py
tests/             offline test suite (fixtures/ holds the HTML test pages)
docs/              architecture, Part 2 diagnosis, one-page PDF
```

## Quick start (offline, no API key needed)

Requires Python 3.10 or newer.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m playwright install chromium   # optional: only for JavaScript pages in Part 2

pytest                                   # the whole suite runs offline

$env:LLM_PROVIDER = "fake"               # offline stand-in for the model
python -m part1_support
python -m part2_scraper --file tests/fixtures/part2/article_with_boilerplate.html --max-words 40
python -m part3_agent --demo
```

**macOS / Linux**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m playwright install chromium   # optional
pytest
LLM_PROVIDER=fake python -m part1_support
```

With `LLM_PROVIDER=fake`, each part uses a small keyword-based responder instead of a model. It
shows the plumbing (routing, verification, retries, the tool loop, memory), **not answer
quality**. The Part 1 fake falls for the refund bait in email `e11` on purpose, so you can watch the
refund guard reject the draft and the revision fix it.

## Use a real model

1. Copy `.env.example` to `.env` (it is git-ignored) and set **one** key:
   ```
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your-key        # or LLM_PROVIDER=openai + OPENAI_API_KEY
   LLM_RPM=8                      # client-side rate limit; keeps a free-tier key under quota
   SCRAPER_CONTACT=https://github.com/<you>/<repo>   # Part 2 User-Agent contact (Wikipedia needs one)
   ```
2. Check the key and model:
   ```
   python scripts/live_check.py models    # lists models your key can use (no generation quota)
   python scripts/live_check.py smoke     # 4 calls: text, JSON mode, tool call + tool result
   ```

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | from the key present, else `fake` | `gemini`, `openai` or `fake` |
| `LLM_MODEL` | `gemini-3.5-flash-lite` / `gpt-6-luna` | model id (checked against the providers' model lists on 2026-10-07) |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | none | the key; never printed or logged |
| `LLM_RPM` | `0` (off) | requests-per-minute cap on the client side |
| `LLM_TIMEOUT_S`, `LLM_MAX_ATTEMPTS` | `60`, `3` | per-request timeout; attempts including retries |
| `LLM_THINKING_LEVEL` | provider default | optional Gemini `thinkingLevel` (for example `low`) |
| `LOG_FORMAT`, `LOG_LEVEL` | `json`, `INFO` | `text` gives readable logs |
| `SCRAPER_CONTACT` | `https://github.com/` | URL or email put in the Part 2 User-Agent |

## Running each part

**Part 1: support emails**
```
python -m part1_support                      # 11 sample emails, 2 workers
python -m part1_support --only e02 e11       # a subset
python -m part1_support --workers 8 --emails my_emails.jsonl
```
It prints one row per email (route, team, model calls, trace id, reasons or categories), the
verified drafts, any rejected-draft problems, and a run report (counts per route and reason, model
latency p50/p95). Decisions are written to `out/decisions.jsonl`. The sample data covers every
rule: outage, breach, data loss, a 4th contact in 7 days (and a 3rd that is *not* escalated), an
overt injection, a subtle refund bait, and a Billing + Technical email.

**Part 2: summarise a page**
```
python -m part2_scraper https://en.wikipedia.org/wiki/History_of_the_Internet
python -m part2_scraper https://quotes.toscrape.com/js/          # JavaScript-only page
python -m part2_scraper <url> --max-words 80 --render always --json
```
Exit codes: `0` ok, `2` configuration, `3` fetch failed, `4` no readable text or no browser,
`5` model failed. The original script is in `part2_scraper/before/` for comparison only.

**Part 3: document agent**
```
python -m part3_agent --demo                 # 5 scripted turns: name, lookup, calculation, memory
python -m part3_agent                        # chat; commands: /memory /reset /exit
python -m part3_agent --session demo         # memory saved in sessions/demo.json between runs
python scripts/live_check.py part3-eval      # does the model call the tool only when needed?
```
The document is `part3_agent/data/sample_document.md`, a fictional travel and expense policy with
numbers to calculate with.

## Logs and tracing

Human output goes to stdout. Logs go to stderr, one JSON object per line. Each line has a
`trace_id`, which is the same for every step of one email, one page or one chat turn:

```json
{"ts": "...", "level": "INFO", "logger": "part1.pipeline", "event": "step.escalation_gate", "trace_id": "8f9385035052", "status": "ok", "duration_ms": 0.4, "escalate": false, "reasons": []}
```

Follow one request end to end (the trace id is printed in the results table):
```powershell
python -m part1_support 2> out/part1.log
Select-String -Path out/part1.log -Pattern "8f9385035052"      # bash: grep 8f9385035052 out/part1.log
```
Logs never contain the API key, prompts or email bodies. Email addresses appear only as a
hash.

## Tests

```
pytest                    # full offline suite (no key, no network)
pytest -m render          # only the Playwright tests (local web server; skipped if Chromium is missing)
python scripts/load_test_part1.py --emails 500 --workers 16 --failure-rate 0.1
```
The tests target the boundaries that matter: 3 vs 4 contacts in 7 days, the exact 7-day edge, a
model that invents a refund term (once, then fixed; or twice, then sent to a human), a model that
ignores the word limit, a runaway tool loop, outages in the middle of a turn, malformed JSON, SSRF
redirects, and byte-dripping servers. The provider clients are tested against mocked HTTP that
pins the exact payloads.

## Verification status

<!-- VERIFICATION_TABLE -->

## Known limitations

- **Escalation rules are keywords.** They miss paraphrases and misspellings ("outtage"), and they
  escalate negations ("no outage"). This is on purpose, favouring recall. The model's
  `critical_issue` flag can add escalations the rules miss, never remove one.
- **The refund guard's claim detector is a heuristic.** It is strict, so it sends some harmless
  drafts to a human. A refund promise written with no refund-related word could pass it. The
  placeholder design means the *policy text itself* can never be misquoted, but the model can
  pick a real clause that does not fit the case.
- **One process.** Threads, SQLite and an in-memory metrics registry are fine for this scope. A
  production version needs a queue and shared stores (see ARCHITECTURE.md).
- **Main-content extraction is heuristic,** and very long pages (more than about 400k characters
  of main text) are summarised only up to that point, with a warning.
- **Part 3 memory** pins only the user's name, found by simple rules ("my name is", "call me"). Other
  facts live in the history window and are lost when old turns are trimmed.
- All company names, emails, policies and documents are **fictional sample data**.

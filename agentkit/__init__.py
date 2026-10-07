"""Shared toolkit used by all three parts.

- config:     settings from environment / .env (API key never printed)
- obs:        structured JSON logging with one trace id per request
- metrics:    in-process counters and latency percentiles for run reports
- resilience: retry policy, circuit breaker, rate limiter
- llm:        one small client interface over Gemini, OpenAI and an offline fake
- jsonout:    turn model text into a validated JSON object, with one repair retry
- injection:  helpers for handling untrusted text (emails, web pages, documents)
"""

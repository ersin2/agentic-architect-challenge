"""Part 1: an agent that triages support emails, drafts grounded replies, and escalates to humans.

Flow (see docs/ARCHITECTURE.md):
  record contact -> escalation gate (code) -> triage (LLM) -> retrieve (BM25)
  -> draft (LLM) -> verify (code) -> [revise once (LLM) -> verify] -> draft_ready | human_review
"""

"""Part 2: a scraper + summariser that works on complex and long pages.

  fetch (timeouts, size cap, content-type, SSRF, robots)
  -> extract main content (drop boilerplate; detect JavaScript-only pages)
  -> render with Playwright if needed
  -> chunk -> map-reduce summarise
  -> guardrail: word limit measured and enforced in code
"""

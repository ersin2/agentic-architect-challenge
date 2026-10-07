# Part 2: diagnosis of the scraper + summariser

> **The "before" script is a reconstruction.** PointStar described a script that "scrapes a
> website and summarises the content, but fails on complex pages or long content", but did not
> supply it. [`part2_scraper/before/scraper_before.py`](../part2_scraper/before/scraper_before.py)
> is a short, realistic version of such a script, written to show the typical defects. Each defect
> is marked `# DEFECT n` in the code. The fixed version is the rest of `part2_scraper/`.

## The bottleneck

**The whole raw page goes to the model in one call** (`summarize(scrape(url))`, DEFECT 4).

Two things go wrong with that single call:

- **Complex pages:** `soup.get_text()` returns *all* visible text of the page: menus, cookie
  banners, ads, sidebars, reference lists, footers. The model is asked to summarise this noise
  together with the article, and on a JavaScript-rendered page there is almost no article at all.
- **Long content:** the prompt grows with every byte of the page. Depending on the model and the
  account tier, this means one of three things. The request is rejected (context window or
  tokens-per-minute limit). The input is silently cut, so the summary covers only the start. Or it
  fits but is slow, expensive and loses detail from the middle of the text. None of this is
  bounded, so one long page can fail the whole run.

The fix is to send **less, better text in bounded pieces**: extract the main content first, then
summarise it with map-reduce, so every call has a bounded size whatever the page length.

## Measured on live pages (2026-10-07)

Characters of text that would be sent to the model. Fetching, extraction and rendering ran for
real; tokens are estimated as characters / 4.

| Page | HTML size | Before: `get_text()` | After: main content | Change |
|---|---|---|---|---|
| Wikipedia, *History of the Internet* (long article) | 1,325 KB | 163,453 chars (~41k tokens) in **1 call** | 96,160 chars (~24k tokens) in **9 map calls of at most 12k chars each** | -41 % text; references, TOC and navboxes removed; no single call is large |
| python.org, *About* (complex layout) | 42 KB | 5,088 chars | 1,720 chars (`<main>`) | -66 %: menus and footer removed |
| quotes.toscrape.com/js (JavaScript-only) | 6 KB | 161 chars (no quotes at all) | detected as JS-only, rendered in headless Chromium: 1,454 chars of quotes | before: a summary of nothing; after: the real content |

With the real model (`gemini-3.5-flash-lite`, 120-word limit):

| Page | Model calls | Summary length | Guardrail action |
|---|---|---|---|
| Wikipedia, *History of the Internet* | 10 (9 map + 1 reduce) | 113 words | none needed |
| python.org, *About* | 1 | 104 words | none needed |

<!-- LIVE_JS_RESULT -->

## Defects and fixes

| # | Defect (before) | Effect | Fix (after) | Where |
|---|---|---|---|---|
| 1 | `requests.get(url)` with no timeout, status check, size limit or content-type check | a slow server hangs forever; a 404 page gets summarised; a huge file fills memory; PDF bytes are parsed as HTML | connect/read timeouts **plus an overall deadline** (a server that drips bytes never trips a read timeout); byte cap with a "truncated" flag; content-type allow-list; 4xx fails at once, 5xx/429 retried with backoff; at most 5 redirects; http(s) only; private, loopback and link-local addresses refused on every redirect hop (SSRF); robots.txt respected; a User-Agent with a contact URL (Wikipedia answers 403 without one; we hit this live) | `fetch.py` |
| 2 | `soup.get_text()` on the whole page | menus, banners, ads and footers are summarised with the article | drop layout tags, ARIA roles and noise-named blocks (whole name parts only: `cookie-banner` yes, `unrelated` no); drop hidden elements; pick `<article>`, `<main>`, or the densest block of paragraphs; remove repeated blocks; fall back to tag-only cleaning if the name heuristic removed too much | `extract.py` |
| 3 | no JavaScript support | single-page apps give an empty summary | detect app shells (empty `#root`/`#app`, "enable JavaScript" notice, scripts with almost no text) and render them in headless Chromium (Playwright, images/fonts/media blocked); clear error with the install command if Playwright is missing | `extract.py`, `render.py` |
| 4 | **one prompt with the whole page (the bottleneck)** | context or quota errors, silent truncation, cost and latency that grow with the page | paragraph-aware chunks; map (notes per chunk, in parallel) then reduce. Chunk size adapts to the page so there are **at most 10 map calls**, each **at most 40k characters**. One failed part is skipped and named; if most parts fail, it is an error, not a partial summary | `chunking.py`, `summarize.py` |
| 5 | page text pasted in as if it were instructions | text on the page can steer the model (prompt injection) | page text wrapped in `<page>` tags as untrusted data; hidden text removed before the model sees it; instruction-like text reported as a warning; the output guardrail still applies | `summarize.py`, `extract.py`, `agentkit/injection.py` |
| 6 | no timeout, retry or error handling on the model call | one provider hiccup crashes the run | shared client: timeouts, jittered retries honouring `Retry-After`, circuit breaker, typed errors; distinct CLI exit codes (3 fetch, 4 no text / no browser, 5 model) | `agentkit/llm.py`, `__main__.py` |
| 7 | "under 100 words" is only a request | nothing measures the output, so long summaries pass | **guardrail in code**: count the words; if over, ask once to compress (giving the measured count, aiming at 85 % of the limit); if still over, cut at the last sentence that fits. The result can never exceed the limit | `guardrail.py` |

## Why the guardrail does not use `max_output_tokens`

A token limit is the wrong tool for conciseness. It cuts the text mid-sentence. It counts tokens,
not words. And on "thinking" models it can be used up by hidden reasoning before any text is
written. So `max_output_tokens` is set generously as a cost cap, and the word limit is enforced
separately, in code, where it can be measured.

## Honest limits of the fix

- Main-content extraction is heuristic. A page with an unusual layout can lose content or keep
  some noise. The fallback covers the most common failure (an over-eager class-name match).
- Above about 400k characters of main text (10 chunks of 40k), the tail of the page is not
  summarised. The result says so in a warning. Raising the cap is a cost decision.
- Playwright renders the page, but sub-requests made by the page itself are not checked against
  the SSRF rules. Only the page URL is checked.
- The deterministic cut keeps whole sentences. It never invents text, but the last kept sentence
  may not be the most important one. That is why the model is asked to compress first.

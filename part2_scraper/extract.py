"""Find the main content of a page and drop everything else.

Fixes DEFECTS 2 and 3 of the old script:
- Boilerplate (menus, cookie banners, ads, sidebars, footers, hidden text) is removed
  before any text is taken. On real pages this is often most of the text.
- The main block is chosen in this order: one <article>, <main> / role=main, the
  densest block of paragraphs (the core idea of "Readability"), then <body>.
- A page whose HTML has almost no text but has app-shell signs (an empty #root,
  a "please enable JavaScript" notice, scripts) is flagged `needs_js`, so the
  caller can render it in a real browser instead of summarising nothing.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

DROP_TAGS = ["script", "style", "noscript", "template", "svg", "canvas", "iframe", "form", "nav", "header",
             "footer", "aside", "button", "input", "select", "textarea", "dialog"]
DROP_ROLES = {"navigation", "banner", "contentinfo", "complementary", "dialog", "search"}
# Matched against whole parts of class/id names ("cookie-banner" yes, "unrelated" no).
NOISE_HINT = re.compile(r"(^|[\s_-])(cookies?|consent|banner|popup|modal|newsletter|subscribe|share|sharing|social|"
                        r"advert|ads?|promo|sidebar|breadcrumbs?|related|comments?|footer|navbar|navbox|menu|toc|"
                        r"references|reflist|footnotes?|catlinks|editsection)($|[\s_-])",
                        re.IGNORECASE)
KEEP_EVEN_IF_NOISY = {"html", "body", "main", "article"}
HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE)
BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "blockquote", "td", "th", "dd", "dt",
              "figcaption", "div", "section", "tr"]
APP_ROOT_ID = re.compile(r"^(root|app|__next|__nuxt|svelte|main-app)$")


@dataclass
class Extracted:
    title: str
    text: str
    method: str  # article | main | density | body
    raw_chars: int  # length of soup.get_text() on the whole page: what the old script sent
    needs_js: bool


def extract(html: str | bytes, encoding: str | None = None) -> Extracted:
    result = _extract(html, encoding, use_name_hints=True)
    if len(result.text) < 200 and result.raw_chars > 2000 and not result.needs_js:
        # The class/id heuristic can be wrong (a content div named "share-enabled").
        # If it left almost nothing, try again with tag-level cleaning only.
        retry = _extract(html, encoding, use_name_hints=False)
        if len(retry.text) > len(result.text):
            return retry
    return result


def _extract(html: str | bytes, encoding: str | None, use_name_hints: bool) -> Extracted:
    soup = BeautifulSoup(html, "lxml", from_encoding=encoding if isinstance(html, bytes) else None)
    raw_chars = len(soup.get_text())
    title = (soup.title.get_text(strip=True) if soup.title else "")[:200]

    # App-shell signals must be read before scripts and <noscript> are removed.
    script_count = len(soup.find_all("script"))
    empty_app_root = any(not tag.get_text(strip=True) for tag in soup.find_all(id=APP_ROOT_ID))
    js_notice = any("javascript" in tag.get_text().lower() for tag in soup.find_all("noscript"))

    _remove_boilerplate(soup, use_name_hints)
    root, method = _main_block(soup)
    text = _block_text(root) if root is not None else ""

    # After rendering, <noscript> and the scripts are still in the HTML, so the signals only
    # count when the page has almost no text.
    needs_js = len(text) < 200 and (empty_app_root or js_notice or script_count > 0)
    return Extracted(title, text, method, raw_chars, needs_js)


def _remove_boilerplate(soup: BeautifulSoup, use_name_hints: bool) -> None:
    for tag in soup(DROP_TAGS):
        tag.decompose()
    # Text a reader cannot see is not content, and hidden text is a common way to plant instructions.
    for tag in soup.find_all(True):
        if not tag.decomposed and tag.name not in ("html", "body") and (
                tag.has_attr("hidden") or HIDDEN_STYLE.search(str(tag.get("style") or ""))):
            tag.decompose()
    if not use_name_hints:
        return
    for tag in soup.find_all(True):
        if tag.decomposed or tag.name in KEEP_EVEN_IF_NOISY:
            continue
        hint = " ".join(tag.get("class") or []) + " " + str(tag.get("id") or "")
        if tag.get("role") in DROP_ROLES or tag.get("aria-hidden") == "true" or (hint.strip() and NOISE_HINT.search(hint)):
            tag.decompose()


def _main_block(soup: BeautifulSoup) -> tuple[Tag | None, str]:
    articles = soup.find_all("article")
    if len(articles) == 1:
        return articles[0], "article"
    main = soup.find("main") or soup.find(attrs={"role": "main"})
    if main is not None:
        return main, "main"
    if not articles:  # several <article> tags = a listing page; keep them all via <body>
        dense = _densest_block(soup)
        if dense is not None:
            return dense, "density"
    return soup.body or soup, "body"


def _densest_block(soup: BeautifulSoup) -> Tag | None:
    """The element whose direct <p> children hold the most text."""
    scores: Counter[int] = Counter()
    parents: dict[int, Tag] = {}
    for p in soup.find_all("p"):
        length = len(p.get_text(strip=True))
        if length >= 25 and isinstance(p.parent, Tag):
            scores[id(p.parent)] += length
            parents[id(p.parent)] = p.parent
    if not scores:
        return None
    best, score = scores.most_common(1)[0]
    return parents[best] if score >= 200 else None


def _block_text(root: Tag) -> str:
    """Text with one blank line between blocks; inline tags (links, bold) stay in their sentence."""
    for br in root.find_all("br"):
        br.replace_with("\n")
    for block in root.find_all(BLOCK_TAGS):
        block.insert_before("\n\n")
        block.insert_after("\n\n")
    seen: set[str] = set()
    paragraphs = []
    for part in re.split(r"\n\s*\n", root.get_text()):
        para = re.sub(r"\s+", " ", part).strip()
        if para and para not in seen:  # repeated blocks ("Share this", "Read more") appear once
            seen.add(para)
            paragraphs.append(para)
    return "\n\n".join(paragraphs)

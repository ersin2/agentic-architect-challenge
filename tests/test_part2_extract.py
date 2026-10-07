"""Part 2 extraction: main content only, boilerplate out, JavaScript-only pages detected."""

from __future__ import annotations

from pathlib import Path

from part2_scraper.before.scraper_before import extract_text as before_extract
from part2_scraper.extract import extract

FIXTURES = Path(__file__).parent / "fixtures" / "part2"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_article_text_is_kept_and_boilerplate_dropped():
    page = extract(fixture("article_with_boilerplate.html"))
    assert page.method == "article"
    assert "most bleached corals can recover" in page.text
    for noise in ("cookies", "Subscribe", "premium diving gear", "Share this", "Copyright", "gtag", "Mangroves"):
        assert noise not in page.text


def test_old_extraction_sends_the_noise_the_new_one_removes():
    html = fixture("article_with_boilerplate.html")
    old, new = before_extract(html.decode()), extract(html).text
    for noise in ("cookies", "Subscribe", "premium diving gear", "Copyright"):
        assert noise in old and noise not in new  # the old script sent all of this to the model
    assert len(new) < 0.6 * len(old)


def test_paragraphs_stay_separate_and_links_stay_inline():
    page = extract(b"<main><p>Read <a href='/x'>the guide</a> first.</p><p>Then start.</p></main>")
    assert page.text == "Read the guide first.\n\nThen start."


def test_densest_block_is_found_without_article_or_main_tags():
    paragraphs = "".join(f"<p>Important finding number {i} explained in a full sentence here.</p>" for i in range(6))
    html = (f"<html><body><div class='nav-links'><p>Home | About | Contact</p></div>"
            f"<div class='content'>{paragraphs}</div><div><p>Short footer line here.</p></div></body></html>")
    page = extract(html.encode())
    assert page.method == "density"
    assert "finding number 5" in page.text and "Home | About" not in page.text


def test_app_shell_is_flagged_as_needing_javascript():
    page = extract(fixture("spa_shell.html"))
    assert page.needs_js and page.text == ""


def test_normal_article_is_not_flagged_as_needing_javascript():
    assert not extract(fixture("article_with_boilerplate.html")).needs_js


def test_hidden_text_is_not_treated_as_content():
    page = extract(fixture("injection_page.html"))
    assert "Ignore all previous instructions" not in page.text
    assert "mulch" in page.text


def test_repeated_blocks_appear_once():
    html = b"<main><p>Same line.</p><p>Same line.</p><p>Other line.</p></main>"
    assert extract(html).text == "Same line.\n\nOther line."


def test_listing_page_with_many_articles_keeps_them_all():
    items = "".join(f"<article><p>Story {i} has its own long enough summary text.</p></article>" for i in range(3))
    page = extract(f"<html><body>{items}</body></html>".encode())
    assert all(f"Story {i}" in page.text for i in range(3))


def test_overeager_class_heuristic_falls_back_to_tag_cleaning():
    body = "".join(f"<p>Paragraph {i} of the real story, with enough words to matter.</p>" for i in range(40))
    html = (f"<html><head><script>var analytics = 1;</script></head>"
            f"<body><div class='post share-enabled'>{body}</div></body></html>").encode()
    page = extract(html)
    assert "Paragraph 39" in page.text and not page.needs_js  # scripts alone must not mean "JS-only"

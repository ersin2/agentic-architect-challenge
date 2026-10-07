"""Knowledge base loading (Markdown + PDF) and BM25 retrieval."""

from __future__ import annotations

from pathlib import Path

from part1_support.kb import Chunk, KnowledgeBase

KB = KnowledgeBase.load(Path(__file__).resolve().parent.parent / "part1_support" / "data" / "kb")


def test_markdown_and_pdf_sources_are_both_loaded():
    docs = {c.id.split("#")[0] for c in KB.chunks}
    assert {"billing_faq", "technical_faq", "general_faq", "refund_policy", "product_guide"} <= docs


def test_pdf_pages_become_chunks_with_their_heading_as_title():
    page = KB.by_id["product_guide#p1"]
    assert page.title == "Version history and Trash" and "restore them from Trash" in page.text


def test_search_finds_the_right_faq_section():
    assert KB.search("download invoice pdf")[0][0].id == "billing_faq#invoices"
    assert KB.search("restore a deleted notebook from trash")[0][0].id == "product_guide#p1"


def test_unrelated_query_finds_nothing():
    assert KB.search("quantum chromodynamics lecture") == []


def test_refund_policy_is_all_or_nothing_in_retrieval():
    without = KB.retrieve(["annual plan refund"], include_policy=False)
    with_policy = KB.retrieve(["annual plan refund"], include_policy=True)
    assert not any(c.id.startswith("refund_policy#") for c in without)
    pinned = [c.id for c in with_policy if c.id.startswith("refund_policy#")]
    assert pinned == [f"refund_policy#R{i}" for i in range(1, 7)]


def test_only_kb_links_are_allowed_in_drafts():
    assert KB.allowed_links == {"https://status.acmesync.example", "support@acmesync.example"}


def test_duplicate_chunk_ids_are_rejected():
    try:
        KnowledgeBase([Chunk("a#1", "t", "x"), Chunk("a#1", "t", "y")])
    except ValueError:
        return
    raise AssertionError("duplicate ids should be rejected")

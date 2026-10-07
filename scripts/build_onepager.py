"""Build docs/architecture_onepager.pdf: architecture, trade-offs and failure points on ONE page.

    python scripts/build_onepager.py            # writes the PDF and fails if it is not exactly 1 page
    python scripts/build_onepager.py --png      # also renders it to docs/architecture_onepager.png to inspect

Content is kept in this file so the PDF can be rebuilt after any change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pypdf import PdfReader
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "architecture_onepager.pdf"

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#52606d")
CODE = colors.HexColor("#dbeafe")      # steps decided by code
CODE_EDGE = colors.HexColor("#2563eb")
MODEL = colors.HexColor("#fde68a")     # steps done by the model
MODEL_EDGE = colors.HexColor("#b45309")
HUMAN = colors.HexColor("#fecaca")
HUMAN_EDGE = colors.HexColor("#b91c1c")
DONE = colors.HexColor("#bbf7d0")
DONE_EDGE = colors.HexColor("#15803d")
RULE = colors.HexColor("#cbd2d9")

VERIFICATION = ("<b>Verified:</b> 220 offline tests pass locally (Python 3.10, 3.12) and in GitHub Actions "
                "(3.10, 3.11, 3.12), with a fake model and mocked HTTP; ruff and mypy clean; offline load test "
                "(500 emails, 10% injected outages, all routed). <b>Live on Gemini (gemini-3.5-flash-lite):</b> "
                "all 11 emails routed as designed, and the refund guard rejected a real paraphrased clause that "
                "the revision fixed; Part 2 on Wikipedia, python.org and a JS-only page (113/104/118 words, limit "
                "120); Part 3 eval 6/6 tool decisions; a real 503 outage was retried and failed cleanly. "
                "<b>Not verified live:</b> the OpenAI client (mocked HTTP only).")

styles = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=15.5, leading=18.5, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=8.6, leading=10.8, textColor=MUTED),
    "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=10, leading=12.2, textColor=INK, spaceBefore=5,
                        spaceAfter=2),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=8.9, leading=11.2, textColor=INK, alignment=TA_LEFT),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.1, leading=9.8, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=8.1, leading=9.8, textColor=INK),
}


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, styles[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(f"• {item}", ParagraphStyle("b", parent=styles["body"], leftIndent=7, firstLineIndent=-6))
            for item in items]


# ------------------------------------------------------------------ diagram -- #

def _box(d: Drawing, x: float, y: float, w: float, h: float, lines: list[str], fill, edge) -> None:
    d.add(Rect(x, y, w, h, rx=3, ry=3, fillColor=fill, strokeColor=edge, strokeWidth=0.8))
    size = 6.6 if len(lines) > 1 else 7.0
    top = y + h / 2 + (len(lines) - 1) * (size + 1) / 2 - size * 0.35
    for i, line in enumerate(lines):
        bold = i == 0
        d.add(String(x + w / 2, top - i * (size + 1), line, fontName="Helvetica-Bold" if bold else "Helvetica",
                     fontSize=size, fillColor=INK, textAnchor="middle"))


def _arrow(d: Drawing, x1: float, y1: float, x2: float, y2: float, color=MUTED, label: str = "") -> None:
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=0.8))
    if x1 == x2:  # vertical
        sign = -1 if y2 < y1 else 1
        d.add(Polygon([x2, y2, x2 - 2.3, y2 - sign * 4, x2 + 2.3, y2 - sign * 4], fillColor=color, strokeColor=color))
    else:
        sign = 1 if x2 > x1 else -1
        d.add(Polygon([x2, y2, x2 - sign * 4, y2 - 2.3, x2 - sign * 4, y2 + 2.3], fillColor=color, strokeColor=color))
    if label:
        d.add(String(x1 + 2, (y1 + y2) / 2 - 2, label, fontName="Helvetica", fontSize=6.2, fillColor=HUMAN_EDGE))


def flow_diagram(width: float) -> Drawing:
    h = 136
    d = Drawing(width, h)
    steps = [
        (["Email in"], DONE, DONE_EDGE),
        (["1 Record contact", "SQLite,", "idempotent"], CODE, CODE_EDGE),
        (["2 Escalation gate", "rules in code,", "before any model"], CODE, CODE_EDGE),
        (["3 Triage", "model: classify,", "plan KB search"], MODEL, MODEL_EDGE),
        (["4 Retrieve", "BM25 search;", "policy pinned"], CODE, CODE_EDGE),
        (["5 Draft", "model: reply,", "citations, slots"], MODEL, MODEL_EDGE),
        (["6 Verify", "code: refund", "guard, cites, links"], CODE, CODE_EDGE),
        (["Draft ready", "policy text", "inserted by code"], DONE, DONE_EDGE),
    ]
    n = len(steps)
    gap = 8
    bw = (width - gap * (n - 1)) / n
    bh, y = 40, 70
    xs = [i * (bw + gap) for i in range(n)]
    for (lines, fill, edge), x in zip(steps, xs, strict=True):
        _box(d, x, y, bw, bh, lines, fill, edge)
    for i in range(n - 1):
        _arrow(d, xs[i] + bw, y + bh / 2, xs[i + 1], y + bh / 2)
    # revise loop above "Verify"
    vx = xs[6]
    rev_y = y + bh + 12
    _box(d, vx - bw * 0.15, rev_y, bw * 1.3, 16, ["7 Revise once (model)"], MODEL, MODEL_EDGE)
    _arrow(d, vx + bw * 0.3, y + bh, vx + bw * 0.3, rev_y, MODEL_EDGE)      # problems found -> revise
    _arrow(d, vx + bw * 0.7, rev_y, vx + bw * 0.7, y + bh, MODEL_EDGE)      # revised draft -> verify again
    # human queue bar
    hx, hy, hw, hh = xs[2], 4, xs[6] + bw - xs[2], 26
    _box(d, hx, hy, hw, hh, ["Human queue: reasons + evidence + team",
                             "rule match (0 model calls) / critical / can't answer / bad output / "
                             "model down / verify failed twice"], HUMAN, HUMAN_EDGE)
    for i, label in [(2, "match"), (3, "critical / fail"), (5, "can't answer"), (6, "fails twice")]:
        cx = xs[i] + bw / 2
        _arrow(d, cx, y, cx, hy + hh, HUMAN_EDGE, label)
    d.add(String(0, h - 6, "Part 1 flow:  blue = code decides   yellow = model   red = human   green = in / out",
                 fontName="Helvetica-Oblique", fontSize=7, fillColor=MUTED))
    return d


# ------------------------------------------------------------------- tables -- #

def table(rows: list[list[str]], widths: list[float]) -> Table:
    data = [[p(c, "cellb" if r == 0 else "cell") for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]))
    return t


def build() -> None:
    margin = 1.1 * cm
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=0.95 * cm,
                            bottomMargin=0.8 * cm, title="Agentic Architect Challenge: architecture",
                            author="Ersin Uraiymov")
    width = A4[0] - 2 * margin
    col_gap = 0.45 * cm
    left_w = width * 0.47
    right_w = width - left_w - col_gap

    left = [
        p("Principles", "h"),
        *bullets([
            "<b>The model plans and writes; code decides and verifies.</b> Escalation, refund facts, summary "
            "length and calculator safety are deterministic code with unit tests.",
            "<b>Fail closed:</b> uncertain, invalid or unavailable means a human handles the email.",
            "<b>The model can only escalate up</b> (critical flag, \"not answerable\"), never remove a rule match.",
            "<b>One trace id per request</b> on every JSON log line; no key, prompts or email bodies in logs.",
        ]),
        p("Part 1: support-email agent", "h"),
        *bullets([
            "<b>Escalation gate</b> before any model call: data loss / outage / breach (regex on NFKC-normalised "
            "text), <b>more than 3 contacts in 7 days</b> (current email counts, window (t-7d, t], so the 4th "
            "escalates), injection phrases. Escalated emails cost 0 model calls and still route during outages.",
            "<b>Multi-label triage</b> (Billing, Technical, Feedback, Other) validated against a fixed list; the "
            "model also writes the KB search queries (plan), code runs BM25 (act).",
            "<b>No invented refund facts:</b> the model may only write <font face='Courier'>{{policy:R2}}</font>; "
            "code rejects refund claims or conditions in its own words and inserts the exact clause text. "
            "Check fails twice: human.",
        ]),
        p("Part 2: scraper fix", "h"),
        p("<b>Bottleneck:</b> the whole raw page in one prompt. <b>Fix:</b> safe fetch (timeouts + deadline, size "
          "cap, content type, SSRF, robots) -> main-content extraction (-41% to -66% text on live pages) -> "
          "Playwright for JS-only pages -> map-reduce (max 10 map calls, max 40k chars each) -> <b>word limit "
          "measured in code</b>: ask once to compress, then cut at a sentence. Never over the limit."),
        p("Part 3: document agent", "h"),
        p("Hand-written tool loop over raw Gemini/OpenAI calls (max 5 steps, stops on repeated calls). Memory: "
          "token-budgeted window of whole turns + pinned user name. Calculator: AST whitelist, never "
          "<font face='Courier'>eval</font>; the model decides when to call it (measured by a 6-question eval)."),
    ]

    right = [
        p("Trade-offs", "h"),
        table([
            ["Choice", "Why", "Cost we accept"],
            ["Regex gate, not the model", "deterministic, injection-proof, works when the model is down",
             "misses paraphrases; escalates negations"],
            ["Policy placeholders", "a misquoted policy is impossible", "formal tone; wrong clause choice possible; "
                                                                           "strict checker = more human reviews"],
            ["BM25, not embeddings", "no extra service, free, deterministic", "word match only (helped by "
                                                                               "model-written queries)"],
            ["Raw API + own loop", "every line explainable; 5 dependencies", "no built-in checkpointing / graph UI"],
            ["Threads + SQLite", "simple; model calls are I/O-bound", "single node; needs a queue at scale"],
        ], [right_w * 0.27, right_w * 0.37, right_w * 0.36]),
        Spacer(1, 3),
        p("Failure points and outcomes", "h"),
        table([
            ["Failure", "Handling"],
            ["Model down / 5xx / 429 / timeout", "3 attempts, jittered backoff, honours Retry-After; circuit "
                                                 "breaker; RPM limiter; then human queue"],
            ["Malformed or invalid JSON", "validated in code; one repair with the exact error; then human"],
            ["Invented refund term, citation or link", "verifier rejects; one revision; then human"],
            ["Prompt injection (email, page, document)", "data-only tags; safety in code; hidden page text dropped; "
                                                         "only tool is side-effect free"],
            ["Runaway tool loop", "5-step limit + repeated-call stop; turn rolled back cleanly"],
            ["Slow / huge / non-HTML / JS-only page", "deadline, 5 MB cap, content-type check, Playwright render"],
            ["Duplicate email, parallel workers", "idempotent message id; counts by timestamp"],
            ["Bug in our code", "batch catch-all: human (internal_error), stack trace logged"],
        ], [right_w * 0.38, right_w * 0.62]),
        Spacer(1, 3),
        p("At production scale", "h"),
        p("Queue + stateless workers, Postgres/Redis keyed by CRM id, hybrid retrieval, review UI, labelled eval "
          "set in CI, OpenTelemetry export of the existing spans, second-provider fallback."),
    ]

    columns = Table([[left, "", right]], colWidths=[left_w, col_gap, right_w])
    columns.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0)]))
    story = [
        p("Agentic Architect Challenge: architecture on one page", "title"),
        p("<b>Ersin Uraiymov</b> &middot; uraiymoversin@gmail.com &middot; PointStar Developer Intern assessment",
          "sub"),
        p("Python 3.10+, one shared toolkit (LLM clients for Gemini and OpenAI behind one interface, retries, "
          "circuit breaker, JSON validation, trace-id logging) and three parts. Everything runs offline with a fake "
          "model; the same code runs live with a key in .env.", "sub"),
        Spacer(1, 4),
        flow_diagram(width),
        Spacer(1, 2),
        columns,
        Spacer(1, 4),
        p(VERIFICATION, "sub"),
    ]
    doc.build(story)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", action="store_true", help="also render a PNG preview")
    args = parser.parse_args()
    build()
    pages = len(PdfReader(str(OUT)).pages)
    print(f"wrote {OUT} ({pages} page{'s' if pages != 1 else ''})")
    if args.png:
        import pypdfium2 as pdfium
        page = pdfium.PdfDocument(str(OUT))[0]
        png = OUT.with_suffix(".png")
        page.render(scale=2).to_pil().save(png)
        print(f"preview: {png}")
    if pages != 1:
        print("ERROR: the one-pager must be exactly one page", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

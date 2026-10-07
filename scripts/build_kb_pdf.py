"""Build the sample knowledge-base PDF (part1_support/data/kb/product_guide.pdf).

The challenge says the knowledge base contains "PDFs/FAQs". The FAQs are
Markdown files; this script makes a small PDF so the PDF ingestion path
(pypdf) is real and tested. One topic per page, so one page = one chunk.

Run: python scripts/build_kb_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "part1_support" / "data" / "kb" / "product_guide.pdf"

PAGES = [
    ("Version history and Trash",
     ["Every note keeps a version history for 30 days. Open a note, choose More > Version history, "
      "and select an older version to restore it.",
      "Deleted notes and notebooks move to Trash. You can restore them from Trash within 30 days. "
      "After 30 days they are removed permanently."]),
    ("Sharing and permissions",
     ["You can share a notebook with people inside or outside your team. Choose Share, enter an email "
      "address, and pick Viewer or Editor.",
      "Only the notebook owner can change permissions or remove people."]),
    ("Offline mode",
     ["The desktop and mobile apps work offline. Changes are saved on the device and sync when the "
      "connection returns.",
      "If the same note was edited on two devices while offline, AcmeSync keeps both versions and "
      "marks the note as a conflict."]),
]


def main() -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paragraphs) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Heading1"]))
        for text in paragraphs:
            story += [Paragraph(text, styles["BodyText"]), Spacer(1, 6)]
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, title="AcmeSync Product Guide", author="AcmeSync (fictional)")
    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

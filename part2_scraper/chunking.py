"""Split long text into chunks that each fit comfortably in one model call.

Splits on paragraph boundaries. A paragraph longer than a chunk is split at
the last sentence end (or space) before the limit. The number of chunks is
capped to bound cost and latency; if the cap cuts the text, the caller is told.
"""

from __future__ import annotations


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    pieces = []
    while len(paragraph) > max_chars:
        cut = paragraph.rfind(". ", 0, max_chars)
        if cut < max_chars // 2:
            cut = paragraph.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars - 1  # one enormous "word": hard cut
        pieces.append(paragraph[:cut + 1].strip())
        paragraph = paragraph[cut + 1:].strip()
    if paragraph:
        pieces.append(paragraph)
    return pieces


def chunk_text(text: str, max_chars: int = 12_000, max_chunks: int = 10) -> tuple[list[str], bool]:
    """Return (chunks, truncated). Every chunk has at most `max_chars` characters."""
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        for piece in _split_long(paragraph.strip(), max_chars):
            if current and len(current) + 2 + len(piece) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks[:max_chunks], len(chunks) > max_chunks

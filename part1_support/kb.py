"""Knowledge base: load Markdown FAQs and PDFs, split into chunks, search with BM25.

Why BM25 and not embeddings:
- no extra model, API call or vector database, so retrieval is free, fast and offline;
- it is deterministic and easy to test and explain;
- weakness: it matches words, not meaning ("money back" vs "refund").
  We reduce that by letting the triage model write the search queries
  (query expansion), and by pinning the whole refund policy when the email
  is about refunds. At production scale this would become hybrid search
  (BM25 + embeddings).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from agentkit.injection import normalize

POLICY_DOC = "refund_policy"
_CLAUSE_HEADING = re.compile(r"^(R\d+)\b\s*(.*)$")
_TOKEN = re.compile(r"[a-z0-9]+")
_URL_OR_EMAIL = re.compile(r"https?://[^\s)>\]]+|[\w.+-]+@[\w-]+\.[\w.-]+")
_STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its my of on or our
please so that the this to was we were what when where which who why will with you your me us""".split())


@dataclass(frozen=True)
class Chunk:
    id: str  # "<document>#<section>", e.g. "billing_faq#invoices" or "refund_policy#R1"
    title: str
    text: str


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _chunk_markdown(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    title, lines = None, []

    def flush() -> None:
        if title is not None and any(line.strip() for line in lines):
            clause = _CLAUSE_HEADING.match(title)
            section = clause.group(1) if clause else _slug(title)
            chunks.append(Chunk(f"{path.stem}#{section}", title, " ".join(line.strip() for line in lines).strip()))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            flush()
            title, lines = line[3:].strip(), []
        elif title is not None:
            lines.append(line)
    flush()
    return chunks


def _chunk_pdf(path: Path) -> list[Chunk]:
    chunks = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        lines = [line.strip() for line in (page.extract_text() or "").splitlines() if line.strip()]
        if lines:  # first line of each page is its heading
            chunks.append(Chunk(f"{path.stem}#p{number}", lines[0], " ".join(lines[1:])))
    return chunks


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN.findall(normalize(text)):
        if token in _STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]  # light plural folding: "invoices" -> "invoice"
        tokens.append(token)
    return tokens


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        ids = [c.id for c in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("knowledge base chunk ids must be unique")
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        # Refund policy clauses, by clause id ("R1" -> exact clause text).
        self.policy = {c.id.split("#", 1)[1]: c.text for c in chunks if c.id.startswith(f"{POLICY_DOC}#")}
        # Links and addresses that appear in our own KB are the only ones a draft may contain.
        self.allowed_links = {m.rstrip(".,") for c in chunks for m in _URL_OR_EMAIL.findall(c.text)}
        self._k1, self._b = k1, b
        self._docs = [tokenize(f"{c.title} {c.text}") for c in chunks]
        self._tf = [Counter(d) for d in self._docs]
        self._avgdl = sum(len(d) for d in self._docs) / max(len(self._docs), 1)
        df = Counter(term for d in self._docs for term in set(d))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def load(cls, directory: Path) -> "KnowledgeBase":
        chunks: list[Chunk] = []
        for path in sorted(directory.iterdir()):
            if path.suffix == ".md":
                chunks += _chunk_markdown(path)
            elif path.suffix == ".pdf":
                chunks += _chunk_pdf(path)
        return cls(chunks)

    def search(self, query: str, k: int = 3, skip_policy: bool = False) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored = []
        for chunk, tf, doc in zip(self.chunks, self._tf, self._docs):
            if skip_policy and chunk.id.startswith(f"{POLICY_DOC}#"):
                continue
            norm = self._k1 * (1 - self._b + self._b * len(doc) / self._avgdl)
            score = sum(self._idf.get(t, 0.0) * tf[t] * (self._k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def retrieve(self, queries: list[str], *, include_policy: bool, k_per_query: int = 3,
                 max_chunks: int = 6) -> list[Chunk]:
        """Merge results of several queries (best score wins), then pin the refund policy if asked.

        The policy is never retrieved piecemeal: it is either in the context whole, or not at all.
        """
        best: dict[str, float] = {}
        for query in queries:
            for chunk, score in self.search(query, k_per_query, skip_policy=True):
                best[chunk.id] = max(score, best.get(chunk.id, 0.0))
        top = sorted(best, key=best.get, reverse=True)[:max_chunks]  # type: ignore[arg-type]
        selected = [self.by_id[i] for i in top]
        if include_policy:
            selected += [c for c in self.chunks if c.id.startswith(f"{POLICY_DOC}#")]
        return selected

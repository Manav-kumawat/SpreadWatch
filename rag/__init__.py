"""
L2 Retrieval - a small TF-IDF index over rating rationale paragraphs.

Why TF-IDF and not neural embeddings: it runs offline on a laptop, needs no
API key, is deterministic, and its scores can be explained (words that are
frequent in a paragraph but rare across the corpus score high). The
commentary layer uses it to cite the rationale paragraph that explains a
rating action, e.g. "[R:crisil_xyz_2019.pdf#p4]".
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from extract import read_document


@dataclass
class Chunk:
    chunk_id: str
    file: str
    para: int
    text: str


class RationaleIndex:
    def __init__(self, folder: Path, min_chars: int = 60):
        self.chunks: List[Chunk] = []
        files = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in (".pdf", ".txt", ".html", ".htm")) \
            if Path(folder).exists() else []
        for p in files:
            paras = [x.strip() for x in read_document(p).split("\n\n")]
            for i, t in enumerate(x for x in paras if len(x) >= min_chars):
                self.chunks.append(Chunk(f"{p.name}#p{i + 1}", p.name, i + 1, " ".join(t.split())))
        self.vectorizer = None
        if self.chunks:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def _rebuild(self, chunks):
        """Index a given list of chunks (used to restrict retrieval to one issuer's documents)."""
        self.chunks = list(chunks)
        self.vectorizer = None
        if self.chunks:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])
        return self

    def __len__(self):
        return len(self.chunks)

    def search(self, query: str, k: int = 3, file_contains: str = None, min_score: float = 0.05) -> List[dict]:
        if not self.vectorizer:
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        order = scores.argsort()[::-1]
        out = []
        for i in order:
            c = self.chunks[i]
            if file_contains and file_contains.lower() not in c.file.lower():
                continue
            if scores[i] < min_score:
                break
            out.append({"chunk_id": c.chunk_id, "file": c.file, "score": float(scores[i]), "text": c.text})
            if len(out) == k:
                break
        return out

from __future__ import annotations
from pathlib import Path
from typing import List
import json
import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from codexrag.types import Chunk, RetrievalHit

def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9_]+", text) if t]

class IndexStore:
    def __init__(self, index_dir: Path, embedding_model: str):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        self.faiss_index = None
        self.chunks: List[Chunk] = []
        self.bm25 = None
        self._bm25_tokens: List[List[str]] = []

    @property
    def _faiss_path(self) -> Path:
        return self.index_dir / "vectors.faiss"

    @property
    def _chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def _bm25_path(self) -> Path:
        return self.index_dir / "bm25.json"

    def save(self) -> None:
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(self._faiss_path))
        with self._chunks_path.open("w", encoding="utf-8") as f:
            for ch in self.chunks:
                f.write(json.dumps({"chunk_id": ch.chunk_id, "text": ch.text, "meta": ch.meta}, ensure_ascii=False) + "\n")
        with self._bm25_path.open("w", encoding="utf-8") as f:
            json.dump(self._bm25_tokens, f)

    def load(self) -> None:
        if self._faiss_path.exists():
            self.faiss_index = faiss.read_index(str(self._faiss_path))
        self.chunks = []
        if self._chunks_path.exists():
            for line in self._chunks_path.read_text(encoding="utf-8").splitlines():
                obj = json.loads(line)
                self.chunks.append(Chunk(obj["chunk_id"], obj["text"], obj["meta"]))
        if self._bm25_path.exists():
            self._bm25_tokens = json.loads(self._bm25_path.read_text(encoding="utf-8"))
            if self._bm25_tokens:
                self.bm25 = BM25Okapi(self._bm25_tokens)

    def build(self, chunks: List[Chunk], batch_size: int = 256) -> None:
        self.chunks = chunks

        self._bm25_tokens = [_tokenize(ch.text) for ch in chunks]
        self.bm25 = BM25Okapi(self._bm25_tokens)

        texts = [ch.text for ch in chunks]
        vecs = []
        for i in range(0, len(texts), batch_size):
            emb = self.model.encode(texts[i:i+batch_size], normalize_embeddings=True, show_progress_bar=False)
            vecs.append(emb)
        vecs = np.vstack(vecs).astype("float32")

        dim = vecs.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(vecs)

    def _hits_from(self, idxs, scores) -> List[RetrievalHit]:
        hits = []
        for i, s in zip(idxs, scores):
            if i < 0 or i >= len(self.chunks):
                continue
            hits.append(RetrievalHit(score=float(s), chunk=self.chunks[int(i)]))
        return hits

    def search_bm25(self, query: str, k: int) -> List[RetrievalHit]:
        if self.bm25 is None:
            return []
        qtok = _tokenize(query)
        scores = self.bm25.get_scores(qtok)
        idxs = np.argsort(scores)[::-1][:k]
        return self._hits_from(idxs.tolist(), scores[idxs].tolist())

    def search_vector(self, query: str, k: int) -> List[RetrievalHit]:
        if self.faiss_index is None:
            return []
        qv = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.faiss_index.search(qv, k)
        return self._hits_from(idxs[0].tolist(), scores[0].tolist())

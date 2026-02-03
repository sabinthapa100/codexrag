from __future__ import annotations
from typing import List
from sentence_transformers import CrossEncoder
from codexrag.types import RetrievalHit

class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: List[RetrievalHit], top_k: int) -> List[RetrievalHit]:
        pairs = [(query, h.chunk.text) for h in hits]
        scores = self.model.predict(pairs)
        for h, s in zip(hits, scores):
            h.score = float(s)
        hits = sorted(hits, key=lambda x: x.score, reverse=True)
        return hits[:top_k]

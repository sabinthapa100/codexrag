from __future__ import annotations
from typing import List
from codexrag.index_store import IndexStore
from codexrag.types import RetrievalHit
from codexrag.rerank import Reranker

def _dedup(hits: List[RetrievalHit]) -> List[RetrievalHit]:
    seen = set()
    out = []
    for h in hits:
        if h.chunk.chunk_id in seen:
            continue
        seen.add(h.chunk.chunk_id)
        out.append(h)
    return out

class HybridRetriever:
    def __init__(self, store: IndexStore, reranker_model: str = ""):
        self.store = store
        self.reranker = Reranker(reranker_model) if reranker_model else None

    def retrieve(self, query: str, top_k_bm25: int, top_k_vector: int, top_k_rerank: int) -> List[RetrievalHit]:
        bm25 = self.store.search_bm25(query, top_k_bm25)
        vec = self.store.search_vector(query, top_k_vector)
        merged = _dedup(bm25 + vec)

        if self.reranker and merged:
            return self.reranker.rerank(query, merged, top_k_rerank)
        return merged[:top_k_rerank]

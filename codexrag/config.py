from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List
import yaml
from pathlib import Path

class RAGConfig(BaseModel):
    repo_root: str = "."
    index_dir: str = ".codexrag/index"
    cache_dir: str = ".codexrag/cache"

    include_globs: List[str] = Field(default_factory=list)
    exclude_globs: List[str] = Field(default_factory=list)

    max_chars: int = 3500
    overlap_chars: int = 250

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = ""

    ollama_model: str = "qwen2.5-coder:14b"
    ollama_base_url: str = "http://localhost:11434"

    top_k_bm25: int = 12
    top_k_vector: int = 12
    top_k_rerank: int = 8

    max_context_chunks: int = 8

def load_config(path: str | Path) -> RAGConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RAGConfig(**data)

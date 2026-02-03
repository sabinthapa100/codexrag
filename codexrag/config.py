from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
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

    # Retrieval parameters (can be overridden by performance_mode)
    top_k_bm25: int = 12
    top_k_vector: int = 12
    top_k_rerank: int = 8
    max_context_chunks: int = 8

    # NEW: Performance mode
    performance_mode: Literal["fast", "deep"] = "fast"
    
    # NEW: Grounding rules
    min_confidence_threshold: float = 0.3
    require_repo_citations: bool = True
    allow_general_physics: bool = True
    label_uncited_claims: bool = True
    
    # NEW: Answerability gating
    enable_second_pass: bool = True
    max_retrieval_attempts: int = 2
    
    # NEW: Answer structure
    enforce_answer_structure: bool = True
    
    # NEW: Graph expansion
    enable_graph_expansion: bool = False  # Off by default for speed
    
    def get_retrieval_params(self) -> dict:
        """Get retrieval parameters based on performance mode."""
        if self.performance_mode == "fast":
            return {
                "top_k_bm25": 6,
                "top_k_vector": 6,
                "top_k_rerank": 0,
                "max_context_chunks": 6,
                "enable_graph_expansion": False,
            }
        else:  # deep
            return {
                "top_k_bm25": self.top_k_bm25,
                "top_k_vector": self.top_k_vector,
                "top_k_rerank": self.top_k_rerank,
                "max_context_chunks": self.max_context_chunks,
                "enable_graph_expansion": True,
            }

def load_config(path: str | Path) -> RAGConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RAGConfig(**data)

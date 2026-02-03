from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: Dict[str, Any]

@dataclass
class RetrievalHit:
    score: float
    chunk: Chunk

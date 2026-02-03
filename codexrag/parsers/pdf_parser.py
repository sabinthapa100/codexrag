from __future__ import annotations
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from codexrag.types import Chunk
from codexrag.utils_hash import sha256_text
from .base import Parser

class PDFParser(Parser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def parse(self, path: Path) -> List[Chunk]:
        doc = fitz.open(str(path))
        chunks: List[Chunk] = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text")
            if not text.strip():
                continue
            cid = sha256_text(f"{path}:p{i+1}:{len(text)}")
            chunks.append(Chunk(
                chunk_id=cid,
                text=text,
                meta={"type": "pdf", "path": str(path), "page": i+1}
            ))
        return chunks

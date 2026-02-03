from __future__ import annotations
from pathlib import Path
from typing import List
import nbformat
from codexrag.types import Chunk
from codexrag.utils_hash import sha256_text
from .base import Parser

class NotebookParser(Parser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".ipynb"

    def parse(self, path: Path) -> List[Chunk]:
        try:
            nb = nbformat.read(str(path), as_version=4)
        except Exception:
            # Fallback for corrupted notebooks: just read as text if possible or skip
            try:
                raw_text = path.read_text(encoding="utf-8", errors="replace")
                cid = sha256_text(f"{path}:whole:{len(raw_text)}")
                return [Chunk(cid, raw_text, {"type":"ipynb", "path":str(path), "kind":"corrupt_fallback"})]
            except Exception:
                return []

        chunks: List[Chunk] = []
        for i, cell in enumerate(nb.cells, start=1):
            ctype = cell.get("cell_type", "")
            src = cell.get("source", "")
            if not isinstance(src, str) or not src.strip():
                continue
            meta = {"type": "ipynb", "path": str(path), "cell": i, "cell_type": ctype}
            text = f"[{ctype.upper()} CELL {i}]\n{src}"
            cid = sha256_text(f"{path}:cell{i}:{len(text)}")
            chunks.append(Chunk(chunk_id=cid, text=text, meta=meta))
        return chunks

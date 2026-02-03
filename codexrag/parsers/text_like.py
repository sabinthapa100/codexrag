from __future__ import annotations
from pathlib import Path
from typing import List
from codexrag.types import Chunk
from codexrag.utils_hash import sha256_text
from .base import Parser

class TextLikeParser(Parser):
    exts = {".md", ".txt", ".json", ".yaml", ".yml"}

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.exts

    def parse(self, path: Path) -> List[Chunk]:
        text = path.read_text(encoding="utf-8", errors="replace")
        cid = sha256_text(f"{path}:{len(text)}")
        return [Chunk(
            chunk_id=cid,
            text=text,
            meta={"type": "text", "path": str(path), "start_line": 1, "end_line": text.count("\n")+1}
        )]

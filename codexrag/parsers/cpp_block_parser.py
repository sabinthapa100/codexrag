from __future__ import annotations
from pathlib import Path
from typing import List
import re
from codexrag.types import Chunk
from codexrag.utils_hash import sha256_text
from .base import Parser

FUNC_RE = re.compile(r'^\s*(?:template\s*<[^>]+>\s*)?(?:[\w:\<\>\*&\s]+)\s+(\w+)\s*\([^;]*\)\s*(?:const)?\s*(?:\{|$)')

class CPPBlockParser(Parser):
    exts = {".c", ".cc", ".cpp", ".h", ".hpp"}

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.exts

    def parse(self, path: Path) -> List[Chunk]:
        src = path.read_text(encoding="utf-8", errors="replace")
        lines = src.splitlines()

        starts = []
        for i, line in enumerate(lines, start=1):
            m = FUNC_RE.match(line)
            if m:
                starts.append((i, m.group(1)))

        if not starts:
            cid = sha256_text(f"{path}:whole:{len(src)}")
            return [Chunk(cid, src, {"type":"cpp", "path":str(path), "kind":"file", "start_line":1, "end_line":len(lines)})]

        chunks: List[Chunk] = []
        for idx, (lineno, name) in enumerate(starts):
            start = lineno
            end = (starts[idx+1][0]-1) if idx+1 < len(starts) else len(lines)
            text = "\n".join(lines[start-1:end])
            cid = sha256_text(f"{path}:{name}:{start}:{end}:{len(text)}")
            chunks.append(Chunk(cid, text, {"type":"cpp", "path":str(path), "kind":"function", "name":name, "start_line":start, "end_line":end}))
        return chunks

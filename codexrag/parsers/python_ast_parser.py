from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import ast
from codexrag.types import Chunk
from codexrag.utils_hash import sha256_text
from .base import Parser

def _lines(src: str) -> List[str]:
    return src.splitlines(keepends=False)

def _span(node: ast.AST) -> Tuple[int, int]:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    return start, end

class PythonASTParser(Parser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".py"

    def parse(self, path: Path) -> List[Chunk]:
        src = path.read_text(encoding="utf-8", errors="replace")
        lines = _lines(src)

        try:
            tree = ast.parse(src)
        except SyntaxError:
            cid = sha256_text(f"{path}:whole:{len(src)}")
            return [Chunk(cid, src, {"type":"py", "path":str(path), "kind":"file_fallback", "start_line":1, "end_line":len(lines)})]

        chunks: List[Chunk] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start, end = _span(node)
                header = lines[start-1] if start-1 < len(lines) else ""
                doc = ast.get_docstring(node) or ""
                body_text = "\n".join(lines[start-1:end])

                text = f"[PY {type(node).__name__}] {node.name}\n{header}\n"
                if doc:
                    text += f"\n[DOCSTRING]\n{doc}\n[/DOCSTRING]\n"
                text += "\n" + body_text

                cid = sha256_text(f"{path}:{node.name}:{start}:{end}:{len(text)}")
                chunks.append(Chunk(
                    chunk_id=cid,
                    text=text,
                    meta={"type":"py", "path":str(path), "kind":type(node).__name__, "name":node.name, "start_line":start, "end_line":end}
                ))

        if not chunks:
            cid = sha256_text(f"{path}:whole:{len(src)}")
            chunks = [Chunk(cid, src, {"type":"py", "path":str(path), "kind":"file", "start_line":1, "end_line":len(lines)})]
        return chunks

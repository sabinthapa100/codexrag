from __future__ import annotations
from pathlib import Path
from typing import List
import fnmatch

def _match_any(path: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)

def scan_files(repo_root: Path, include_globs: List[str], exclude_globs: List[str]) -> List[Path]:
    repo_root = repo_root.resolve()
    all_files: List[Path] = []
    for pat in include_globs:
        for p in repo_root.glob(pat):
            if not p.is_file():
                continue
            rel = p.relative_to(repo_root).as_posix()
            if _match_any(rel, exclude_globs):
                continue
            all_files.append(p)
    seen = set()
    out = []
    for p in all_files:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out

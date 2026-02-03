from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import json
from codexrag.config import RAGConfig
from codexrag.fs_scan import scan_files
from codexrag.parsers import DEFAULT_PARSERS
from codexrag.chunking import split_long_chunks
from codexrag.types import Chunk
from codexrag.utils_hash import sha256_file
from codexrag.index_store import IndexStore
from codexrag.utils_warnings import suppress_common_warnings

# Suppress harmless warnings
suppress_common_warnings()

def _load_manifest(cache_dir: Path) -> Dict[str, str]:
    path = cache_dir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def _save_manifest(cache_dir: Path, manifest: Dict[str, str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def build_index(cfg: RAGConfig, repo_root: Path) -> IndexStore:
    repo_root = repo_root.resolve()
    index_dir = (repo_root / cfg.index_dir).resolve()
    cache_dir = (repo_root / cfg.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = scan_files(repo_root, cfg.include_globs, cfg.exclude_globs)

    manifest = _load_manifest(cache_dir)
    new_manifest: Dict[str, str] = {}

    all_chunks: List[Chunk] = []
    for p in files:
        rel = p.relative_to(repo_root).as_posix()
        digest = sha256_file(p)
        new_manifest[rel] = digest

        # NOTE: this scaffold rebuilds the full index each time.
        # For huge repos, you can store per-file chunks and update incrementally.
        parsed = None
        for parser in DEFAULT_PARSERS:
            if parser.can_parse(p):
                parsed = parser.parse(p)
                break
        if parsed is None:
            continue

        for ch in parsed:
            ch.meta["relpath"] = rel

        parsed = split_long_chunks(parsed, cfg.max_chars, cfg.overlap_chars)
        all_chunks.extend(parsed)

    store = IndexStore(index_dir=index_dir, embedding_model=cfg.embedding_model)
    store.build(all_chunks)
    store.save()
    
    # --- New: Build & Save Graph ---
    try:
        from codexrag.graph import CodeGraph
        # Filter for Python files for graph building
        py_files = [f for f in files if f.suffix == ".py"]
        if py_files:
            print(f"Building knowledge graph from {len(py_files)} Python files...")
            graph = CodeGraph()
            graph.build_from_files(py_files)
            graph.save(index_dir / "graph.json")
            stats = graph.stats()
            print(f"Graph built: {stats.get('total_entities', 0)} entities, {stats.get('total_edges', 0)} edges.")
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Failed to build graph: {e}")

    _save_manifest(cache_dir, new_manifest)
    return store

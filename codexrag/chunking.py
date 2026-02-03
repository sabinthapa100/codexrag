from __future__ import annotations
from typing import List
from codexrag.types import Chunk
from codexrag.utils_hash import sha256_text

def split_long_chunks(chunks: List[Chunk], max_chars: int, overlap_chars: int) -> List[Chunk]:
    out: List[Chunk] = []
    for ch in chunks:
        if len(ch.text) <= max_chars:
            out.append(ch)
            continue
        text = ch.text
        start = 0
        part = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            piece = text[start:end]
            part += 1
            cid = sha256_text(f"{ch.chunk_id}:part{part}:{start}:{end}")
            meta = dict(ch.meta)
            meta.update({"part": part, "char_start": start, "char_end": end})
            out.append(Chunk(cid, piece, meta))
            start = max(0, end - overlap_chars)
            if end == len(text):
                break
    return out

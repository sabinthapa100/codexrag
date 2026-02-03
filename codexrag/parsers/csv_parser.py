from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
from codexrag.types import Chunk
from codexrag.utils_hash import sha256_text
from .base import Parser

class CSVParser(Parser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".csv"

    def parse(self, path: Path) -> List[Chunk]:
        try:
            # Try efficient parsing
            df = pd.read_csv(path, nrows=200)
            sample = df.to_csv(index=False)
            cid = sha256_text(f"{path}:csvsample:{len(sample)}")
            return [Chunk(
                chunk_id=cid,
                text=sample,
                meta={"type": "csv", "path": str(path), "rows_sampled": len(df)}
            )]
        except Exception:
            # Fallback to simple text read
            try:
                head = path.read_text(encoding="utf=8", errors="replace")[:2000]
                cid = sha256_text(f"{path}:text_fallback:{len(head)}")
                return [Chunk(
                   chunk_id=cid,
                   text=head,
                   meta={"type": "csv_text", "path": str(path), "rows_sampled": 0}
                )]
            except Exception:
                return []

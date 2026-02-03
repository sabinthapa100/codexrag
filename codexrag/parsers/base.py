from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from codexrag.types import Chunk

class Parser(ABC):
    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        ...

    @abstractmethod
    def parse(self, path: Path) -> List[Chunk]:
        ...

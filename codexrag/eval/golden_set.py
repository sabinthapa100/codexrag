"""
Golden Set for RAG Evaluation
==============================
Defines the structure and loading of golden query-answer pairs.

A golden set is a curated collection of queries with known correct answers,
used to benchmark RAG performance.

Author: Sabin Thapa (sthapa3@kent.edu)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class GoldenQuery:
    """A single golden query with expected answer and metadata."""
    
    query: str                          # The question
    ground_truth: str                   # Expected answer content
    expected_sources: List[str] = field(default_factory=list)  # Files that should be cited
    category: str = "general"           # Query type: lookup, cross-file, math, data
    difficulty: str = "medium"          # easy, medium, hard
    tags: List[str] = field(default_factory=list)


class GoldenSet:
    """
    Manages a collection of golden queries for evaluation.
    
    Supports loading from JSON and filtering by category.
    """
    
    def __init__(self, queries: Optional[List[GoldenQuery]] = None):
        self.queries = queries or []
    
    @classmethod
    def from_json(cls, path: Path) -> "GoldenSet":
        """Load golden set from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        
        queries = []
        for item in data.get("queries", []):
            queries.append(GoldenQuery(
                query=item["query"],
                ground_truth=item["ground_truth"],
                expected_sources=item.get("expected_sources", []),
                category=item.get("category", "general"),
                difficulty=item.get("difficulty", "medium"),
                tags=item.get("tags", [])
            ))
        
        return cls(queries)
    
    def to_json(self, path: Path) -> None:
        """Save golden set to a JSON file."""
        data = {
            "version": "1.0",
            "description": "CodeXRAG Golden Evaluation Set",
            "queries": [
                {
                    "query": q.query,
                    "ground_truth": q.ground_truth,
                    "expected_sources": q.expected_sources,
                    "category": q.category,
                    "difficulty": q.difficulty,
                    "tags": q.tags
                }
                for q in self.queries
            ]
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    def filter_by_category(self, category: str) -> "GoldenSet":
        """Return a subset filtered by category."""
        filtered = [q for q in self.queries if q.category == category]
        return GoldenSet(filtered)
    
    def filter_by_difficulty(self, difficulty: str) -> "GoldenSet":
        """Return a subset filtered by difficulty."""
        filtered = [q for q in self.queries if q.difficulty == difficulty]
        return GoldenSet(filtered)
    
    def add_query(self, query: GoldenQuery) -> None:
        """Add a new golden query."""
        self.queries.append(query)
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __iter__(self):
        return iter(self.queries)
    
    def stats(self) -> dict:
        """Get statistics about the golden set."""
        categories = {}
        difficulties = {}
        
        for q in self.queries:
            categories[q.category] = categories.get(q.category, 0) + 1
            difficulties[q.difficulty] = difficulties.get(q.difficulty, 0) + 1
        
        return {
            "total": len(self.queries),
            "categories": categories,
            "difficulties": difficulties
        }


# Pre-built golden queries for scientific codebases
EXAMPLE_GOLDEN_QUERIES = [
    GoldenQuery(
        query="Where is RpA computed in the codebase?",
        ground_truth="RpA is computed in compute_rpa_grid function which calculates the nuclear modification factor",
        expected_sources=["npdf_code/rpa_calculator.py"],
        category="lookup",
        difficulty="easy"
    ),
    GoldenQuery(
        query="What functions call compute_rpa_grid?",
        ground_truth="compute_rpa_grid is called by rpa_band_vs_y and rpa_band_vs_pt functions",
        expected_sources=["npdf_code/rpa_calculator.py"],
        category="cross-file",
        difficulty="medium"
    ),
    GoldenQuery(
        query="How is the Hessian uncertainty calculated?",
        ground_truth="Hessian uncertainty is calculated by computing the sum in quadrature of deviations from the central value across all PDF error sets",
        expected_sources=["npdf_code/uncertainty.py"],
        category="math",
        difficulty="hard"
    ),
    GoldenQuery(
        query="What columns are in the experimental data CSV files?",
        ground_truth="The CSV files contain columns for centrality, rapidity, pT, cross-section, and statistical uncertainties",
        expected_sources=["data/*.csv"],
        category="data",
        difficulty="medium"
    ),
    GoldenQuery(
        query="How does the glauber model compute L_eff?",
        ground_truth="L_eff is computed using the optical or binomial method with nuclear density profiles",
        expected_sources=["eloss_code/glauber.py"],
        category="cross-file",
        difficulty="hard"
    )
]

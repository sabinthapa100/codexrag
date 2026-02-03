"""
CodeXRAG Evaluation Module
==========================
Implements Ragas-style metrics for rigorous RAG evaluation.

Metrics:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: How relevant are the retrieved chunks?
- Context Recall: Did retrieval find all necessary info?

Author: Sabin Thapa (sthapa3@kent.edu)
"""

from .metrics import (
    faithfulness_score,
    answer_relevancy_score,
    context_precision_score,
    context_recall_score,
    RAGEvaluator
)
from .golden_set import GoldenSet, GoldenQuery

__all__ = [
    "faithfulness_score",
    "answer_relevancy_score",
    "context_precision_score",
    "context_recall_score",
    "RAGEvaluator",
    "GoldenSet",
    "GoldenQuery"
]

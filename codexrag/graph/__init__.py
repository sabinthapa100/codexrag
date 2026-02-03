"""
CodeXRAG Graph Module
=====================
Provides GraphRAG capabilities for understanding code structure and dependencies.

This module builds a programming knowledge graph from AST parses:
- Entities: functions, classes, variables, imports
- Relationships: calls, inherits, imports, dataflow

Author: Sabin Thapa (sthapa3@kent.edu)
"""

from .code_graph import CodeGraph
from .graph_retriever import GraphEnhancedRetriever

__all__ = ["CodeGraph", "GraphEnhancedRetriever"]

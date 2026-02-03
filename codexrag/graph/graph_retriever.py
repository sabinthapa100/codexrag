"""
Graph-Enhanced Retriever
========================
Combines traditional hybrid retrieval with graph-guided context expansion.

The retrieval pipeline:
1. Initial retrieval via BM25 + Vector search
2. For each hit, find related entities via graph traversal
3. Expand context with related code (callers, callees, imports)
4. Re-rank with cross-encoder

This enables answering cross-file dependency questions like:
"What functions call compute_rpa?"
"How does the data flow from load_data to plot_results?"

Author: Sabin Thapa (sthapa3@kent.edu)
"""

from __future__ import annotations
from typing import List, Dict, Optional, Set
from pathlib import Path

from codexrag.types import RetrievalHit, Chunk
from codexrag.graph.code_graph import CodeGraph


class GraphEnhancedRetriever:
    """
    A retriever that uses graph traversal to expand and enrich context.
    
    Wraps an existing HybridRetriever and enhances its results with
    graph-derived related entities.
    """
    
    def __init__(
        self,
        base_retriever,
        code_graph: Optional[CodeGraph] = None,
        max_expansion_hops: int = 2,
        max_expanded_chunks: int = 4
    ):
        """
        Initialize the graph-enhanced retriever.
        
        Args:
            base_retriever: The underlying HybridRetriever
            code_graph: Pre-built CodeGraph (or None to skip graph enhancement)
            max_expansion_hops: How far to traverse in the graph
            max_expanded_chunks: Max additional chunks from graph expansion
        """
        self.base = base_retriever
        self.graph = code_graph
        self.max_hops = max_expansion_hops
        self.max_expanded = max_expanded_chunks
    
    def retrieve(
        self,
        query: str,
        top_k_bm25: int = 12,
        top_k_vector: int = 12,
        top_k_rerank: int = 8
    ) -> List[RetrievalHit]:
        """
        Retrieve with graph-enhanced context expansion.
        
        Args:
            query: User's question
            top_k_bm25: BM25 candidates
            top_k_vector: Vector candidates
            top_k_rerank: Final results after reranking
            
        Returns:
            List of RetrievalHit with expanded context
        """
        # Step 1: Base retrieval
        base_hits = self.base.retrieve(query, top_k_bm25, top_k_vector, top_k_rerank)
        
        if not self.graph or not base_hits:
            return base_hits
        
        # Step 2: Graph expansion
        expanded_hits = self._expand_with_graph(query, base_hits)
        
        # Step 3: Merge and deduplicate
        all_hits = self._merge_hits(base_hits, expanded_hits)
        
        return all_hits[:top_k_rerank]
    
    def _expand_with_graph(
        self,
        query: str,
        base_hits: List[RetrievalHit]
    ) -> List[RetrievalHit]:
        """
        Expand retrieval results using graph traversal.
        """
        expanded = []
        seen_entity_ids: Set[str] = set()
        
        # For each base hit, find related entities
        for hit in base_hits:
            # Try to map chunk to graph entity
            entity_ids = self._chunk_to_entities(hit.chunk)
            
            for entity_id in entity_ids:
                if entity_id in seen_entity_ids:
                    continue
                seen_entity_ids.add(entity_id)
                
                # Get related entities via graph traversal
                related = self.graph.get_related_entities(entity_id, self.max_hops)
                
                for related_id in related[:self.max_expanded]:
                    if related_id in seen_entity_ids:
                        continue
                    seen_entity_ids.add(related_id)
                    
                    # Create a synthetic chunk for the related entity
                    context = self.graph.get_entity_context(related_id)
                    if context:
                        entity = self.graph.entities.get(related_id)
                        if entity:
                            chunk = Chunk(
                                chunk_id=f"graph:{related_id}",
                                text=context,
                                meta={
                                    "type": "graph_expansion",
                                    "path": entity.file_path,
                                    "start_line": entity.line_start,
                                    "end_line": entity.line_end,
                                    "entity_type": entity.entity_type,
                                    "source_entity": entity_id
                                }
                            )
                            expanded.append(RetrievalHit(chunk=chunk, score=hit.score * 0.8))
        
        return expanded
    
    def _chunk_to_entities(self, chunk: Chunk) -> List[str]:
        """
        Map a chunk to graph entities based on file path and line numbers.
        """
        if not self.graph:
            return []
        
        file_path = chunk.meta.get("path")
        if not file_path:
            return []
        
        entities = []
        
        # Get entities in this file
        file_entities = self.graph.file_to_entities.get(file_path, [])
        
        # Filter to overlapping line ranges
        chunk_start = chunk.meta.get("start_line", 0)
        chunk_end = chunk.meta.get("end_line", float("inf"))
        
        for entity_id in file_entities:
            entity = self.graph.entities.get(entity_id)
            if entity:
                if (entity.line_start <= chunk_end and entity.line_end >= chunk_start):
                    entities.append(entity_id)
        
        return entities
    
    def _merge_hits(
        self,
        base_hits: List[RetrievalHit],
        expanded_hits: List[RetrievalHit]
    ) -> List[RetrievalHit]:
        """
        Merge base and expanded hits, removing duplicates.
        """
        seen_ids = {h.chunk.chunk_id for h in base_hits}
        merged = list(base_hits)
        
        for hit in expanded_hits:
            if hit.chunk.chunk_id not in seen_ids:
                seen_ids.add(hit.chunk.chunk_id)
                merged.append(hit)
        
        # Sort by score
        merged.sort(key=lambda h: h.score, reverse=True)
        return merged
    
    def get_dependency_context(self, entity_name: str) -> str:
        """
        Get a formatted dependency context for a specific entity.
        
        Useful for question answering about "what calls X" or "what does X depend on".
        """
        if not self.graph:
            return ""
        
        # Find entities matching the name
        matches = self.graph.find_entities_by_name(entity_name)
        
        if not matches:
            return f"No entity named '{entity_name}' found in the codebase."
        
        contexts = []
        for entity_id in matches[:3]:
            context = self.graph.get_entity_context(entity_id)
            contexts.append(context)
        
        return "\n\n---\n\n".join(contexts)

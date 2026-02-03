"""
Code Knowledge Graph
====================
Builds a programming knowledge graph from Python AST parses.

Entities:
- FUNCTION: Function definitions
- CLASS: Class definitions
- MODULE: Python modules/files
- IMPORT: Import statements

Relationships:
- CALLS: Function A calls Function B
- INHERITS: Class A inherits from Class B
- IMPORTS: Module A imports from Module B
- DEFINES: Module defines Function/Class
- USES: Function uses variable/constant

Author: Sabin Thapa (sthapa3@kent.edu)
"""

from __future__ import annotations
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


@dataclass
class CodeEntity:
    """A code entity (function, class, module)."""
    id: str                  # Unique identifier (module:name)
    name: str                # Short name
    entity_type: str         # FUNCTION, CLASS, MODULE, IMPORT
    file_path: str           # Source file
    line_start: int = 0
    line_end: int = 0
    docstring: Optional[str] = None
    signature: Optional[str] = None
    

@dataclass
class CodeRelation:
    """A relationship between two entities."""
    source_id: str
    target_id: str
    relation_type: str       # CALLS, INHERITS, IMPORTS, DEFINES, USES


class CodeGraph:
    """
    A knowledge graph for code understanding.
    
    Uses NetworkX to store entities as nodes and relationships as edges.
    Enables graph-guided retrieval for cross-file dependencies.
    """
    
    def __init__(self):
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for GraphRAG. Install with: pip install networkx")
        
        self.graph = nx.DiGraph()
        self.entities: Dict[str, CodeEntity] = {}
        self.file_to_entities: Dict[str, List[str]] = {}
    
    def build_from_files(self, python_files: List[Path]) -> None:
        """
        Build the code graph from a list of Python files.
        
        Args:
            python_files: List of .py file paths to analyze
        """
        # Phase 1: Extract all entities
        for file_path in python_files:
            try:
                self._extract_entities(file_path)
            except Exception as e:
                # Skip files that can't be parsed
                continue
        
        # Phase 2: Extract relationships
        for file_path in python_files:
            try:
                self._extract_relationships(file_path)
            except Exception:
                continue
    
    def _extract_entities(self, file_path: Path) -> None:
        """Extract entities (functions, classes) from a Python file."""
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except SyntaxError:
            return
        
        module_id = str(file_path)
        
        # Add module as entity
        module_entity = CodeEntity(
            id=module_id,
            name=file_path.stem,
            entity_type="MODULE",
            file_path=str(file_path),
            line_start=1,
            line_end=len(source.splitlines())
        )
        self._add_entity(module_entity)
        
        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                entity = self._node_to_entity(node, file_path, "FUNCTION")
                self._add_entity(entity)
                self._add_relation(CodeRelation(module_id, entity.id, "DEFINES"))
                
            elif isinstance(node, ast.ClassDef):
                entity = self._node_to_entity(node, file_path, "CLASS")
                self._add_entity(entity)
                self._add_relation(CodeRelation(module_id, entity.id, "DEFINES"))
                
                # Extract inheritance
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        target_id = f"*:{base.id}"  # Wildcard for external
                        self._add_relation(CodeRelation(entity.id, target_id, "INHERITS"))
    
    def _extract_relationships(self, file_path: Path) -> None:
        """Extract call relationships from a Python file."""
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except SyntaxError:
            return
        
        # Find all function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                caller_context = self._get_enclosing_entity(tree, node, file_path)
                callee_name = self._get_call_name(node)
                
                if caller_context and callee_name:
                    # Try to resolve callee to known entity
                    target_id = self._resolve_entity(callee_name, file_path)
                    self._add_relation(CodeRelation(caller_context, target_id, "CALLS"))
            
            # Track imports
            elif isinstance(node, ast.Import):
                module_id = str(file_path)
                for alias in node.names:
                    target_id = f"external:{alias.name}"
                    self._add_relation(CodeRelation(module_id, target_id, "IMPORTS"))
                    
            elif isinstance(node, ast.ImportFrom):
                module_id = str(file_path)
                target_id = f"external:{node.module or ''}"
                self._add_relation(CodeRelation(module_id, target_id, "IMPORTS"))
    
    def _node_to_entity(self, node: ast.AST, file_path: Path, entity_type: str) -> CodeEntity:
        """Convert an AST node to a CodeEntity."""
        name = getattr(node, "name", "unknown")
        entity_id = f"{file_path}:{name}"
        
        # Get signature for functions
        signature = None
        if entity_type == "FUNCTION" and hasattr(node, "args"):
            args = [arg.arg for arg in node.args.args]
            signature = f"def {name}({', '.join(args)})"
        
        return CodeEntity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            file_path=str(file_path),
            line_start=getattr(node, "lineno", 0),
            line_end=getattr(node, "end_lineno", 0),
            docstring=ast.get_docstring(node),
            signature=signature
        )
    
    def _add_entity(self, entity: CodeEntity) -> None:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.entity_type,
            file=entity.file_path,
            line_start=entity.line_start,
            line_end=entity.line_end,
            docstring=entity.docstring or "",
            signature=entity.signature or ""
        )
        
        # Track file -> entities mapping
        if entity.file_path not in self.file_to_entities:
            self.file_to_entities[entity.file_path] = []
        self.file_to_entities[entity.file_path].append(entity.id)
    
    def _add_relation(self, rel: CodeRelation) -> None:
        """Add a relationship edge to the graph."""
        self.graph.add_edge(rel.source_id, rel.target_id, relation=rel.relation_type)
    
    def _get_enclosing_entity(self, tree: ast.AST, node: ast.AST, file_path: Path) -> Optional[str]:
        """Get the enclosing function/class for a node."""
        # Simple heuristic: find the function/class that contains this line
        node_line = getattr(node, "lineno", 0)
        
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = getattr(n, "lineno", 0)
                end = getattr(n, "end_lineno", 0)
                if start <= node_line <= end:
                    return f"{file_path}:{n.name}"
        
        return str(file_path)  # Default to module level
    
    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract the name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None
    
    def _resolve_entity(self, name: str, context_file: Path) -> str:
        """Try to resolve a name to a known entity ID."""
        # First, check local file
        local_id = f"{context_file}:{name}"
        if local_id in self.entities:
            return local_id
        
        # Check all known entities
        for entity_id, entity in self.entities.items():
            if entity.name == name:
                return entity_id
        
        # Unknown - mark as external
        return f"*:{name}"
    
    # --- Graph Queries ---
    
    def get_related_entities(self, entity_id: str, max_hops: int = 2) -> List[str]:
        """
        Get entities related to the given entity via graph traversal.
        
        Args:
            entity_id: Starting entity
            max_hops: Maximum traversal depth
            
        Returns:
            List of related entity IDs
        """
        if entity_id not in self.graph:
            return []
        
        related = set()
        
        # BFS traversal
        current_level = {entity_id}
        for _ in range(max_hops):
            next_level = set()
            for node in current_level:
                # Get neighbors (both directions)
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                next_level.update(neighbors)
            related.update(next_level)
            current_level = next_level - related
        
        # Filter to only known entities
        return [e for e in related if e in self.entities and not e.startswith("*:")]
    
    def get_callers(self, entity_id: str) -> List[str]:
        """Get all entities that call this entity."""
        callers = []
        for source, target, data in self.graph.in_edges(entity_id, data=True):
            if data.get("relation") == "CALLS":
                callers.append(source)
        return callers
    
    def get_callees(self, entity_id: str) -> List[str]:
        """Get all entities that this entity calls."""
        callees = []
        for source, target, data in self.graph.out_edges(entity_id, data=True):
            if data.get("relation") == "CALLS":
                callees.append(target)
        return callees
    
    def get_entity_context(self, entity_id: str) -> str:
        """
        Generate rich context for an entity including its relationships.
        
        Returns a formatted string describing the entity and its connections.
        """
        if entity_id not in self.entities:
            return ""
        
        entity = self.entities[entity_id]
        
        lines = [
            f"[{entity.entity_type}] {entity.name}",
            f"File: {entity.file_path}:{entity.line_start}-{entity.line_end}"
        ]
        
        if entity.signature:
            lines.append(f"Signature: {entity.signature}")
        
        if entity.docstring:
            lines.append(f"Docstring: {entity.docstring[:200]}...")
        
        # Add relationships
        callers = self.get_callers(entity_id)
        if callers:
            lines.append(f"Called by: {', '.join(c.split(':')[-1] for c in callers[:5])}")
        
        callees = self.get_callees(entity_id)
        if callees:
            lines.append(f"Calls: {', '.join(c.split(':')[-1] for c in callees[:5])}")
        
        return "\n".join(lines)
    
    def find_entities_by_name(self, name: str) -> List[str]:
        """Find all entities matching a name (partial match)."""
        matches = []
        name_lower = name.lower()
        for entity_id, entity in self.entities.items():
            if name_lower in entity.name.lower():
                matches.append(entity_id)
        return matches
    
    def save(self, path: Path) -> None:
        """Save the graph to disk."""
        data = {
            "entities": {k: vars(v) for k, v in self.entities.items()},
            "edges": list(self.graph.edges(data=True))
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    def load(self, path: Path) -> None:
        """Load the graph from disk."""
        if not path.exists():
            return
        
        data = json.loads(path.read_text(encoding="utf-8"))
        
        for entity_id, entity_data in data.get("entities", {}).items():
            entity = CodeEntity(**entity_data)
            self._add_entity(entity)
        
        for source, target, edge_data in data.get("edges", []):
            self.graph.add_edge(source, target, **edge_data)
    
    def stats(self) -> Dict[str, int]:
        """Get graph statistics."""
        type_counts = {}
        for entity in self.entities.values():
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1
        
        return {
            "total_entities": len(self.entities),
            "total_edges": self.graph.number_of_edges(),
            **type_counts
        }

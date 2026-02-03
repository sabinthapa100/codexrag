"""
CodeXRAG Multi-Agent System
===========================
A multi-agent architecture for scientific codebase understanding.

Architecture: Subagent Pattern with Centralized Orchestration
- Orchestrator: Routes queries to specialized agents
- CodeAnalyst: Understands code structure
- DataExpert: Interprets data files
- DocExpert: Searches documentation
- SafetyAgent: Validates all operations

Author: Sabin Thapa (sthapa3@kent.edu)
"""

from .orchestrator import Orchestrator
from .safety_agent import SafetyAgent

__all__ = ["Orchestrator", "SafetyAgent"]

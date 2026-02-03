"""
Tests for CodeXRAG
"""
import pytest
from pathlib import Path
from codexrag.agents.orchestrator import IntentClassifier, QueryIntent

def test_intent_classification():
    """Verify that physics and code questions are correctly classified."""
    # Physics/Math
    assert IntentClassifier.classify("What is the formula for RpA?") == QueryIntent.MATH
    assert IntentClassifier.classify("Show me the equation for optical depth") == QueryIntent.MATH
    
    # Code
    assert IntentClassifier.classify("Which function computes density?") == QueryIntent.CODE
    assert IntentClassifier.classify("Where is the class defined?") == QueryIntent.CODE
    
    # Data
    assert IntentClassifier.classify("Plot the csv results") == QueryIntent.DATA

def test_project_structure():
    """Verify essential project files exist."""
    assert Path("codexrag").exists()
    assert Path("setup_env.sh").exists()
    assert Path("pyproject.toml").exists()

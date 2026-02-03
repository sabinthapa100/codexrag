"""
Suppress common warnings during indexing and runtime.
"""
import warnings
import os

def suppress_common_warnings():
    """Suppress harmless warnings that clutter output."""
    
    # Suppress SyntaxWarnings from physics code with LaTeX-like strings
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    
    # Suppress nbformat cell ID warnings (we handle this in parser)
    warnings.filterwarnings('ignore', message='.*MissingIDFieldWarning.*')
    
    # Suppress HuggingFace token warnings if user hasn't set HF_TOKEN
    if 'HF_TOKEN' not in os.environ:
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        warnings.filterwarnings('ignore', message='.*unauthenticated requests.*')

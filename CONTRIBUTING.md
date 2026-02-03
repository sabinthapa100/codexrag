# Contributing to CodeXRAG

Thank you for your interest in contributing to CodeXRAG! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/codexrag.git
   cd codexrag
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or: .venv\Scripts\activate  # Windows
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,rerank]"
   ```

4. **Run tests**
   ```bash
   pytest tests/
   ```

## 📝 Code Style

We use the following tools for code quality:

- **Black** for formatting: `black codexrag/`
- **isort** for import sorting: `isort codexrag/`
- **flake8** for linting: `flake8 codexrag/`

Please run these before submitting a PR.

## 🏗️ Architecture Overview

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Orchestrator | `agents/orchestrator.py` | Query routing and self-correction |
| Retriever | `retriever.py` | Hybrid BM25 + Vector search |
| GraphRAG | `graph/code_graph.py` | Code knowledge graph |
| Evaluation | `eval/metrics.py` | Ragas-style metrics |
| Parsers | `parsers/` | File format handlers |

### Adding a New Parser

1. Create `codexrag/parsers/your_parser.py`:
   ```python
   from codexrag.parsers.base import Parser
   from codexrag.types import Chunk
   
   class YourParser(Parser):
       def can_parse(self, path: Path) -> bool:
           return path.suffix.lower() == ".your_ext"
       
       def parse(self, path: Path) -> List[Chunk]:
           # Your parsing logic
           ...
   ```

2. Register in `codexrag/parsers/__init__.py`

3. Add tests in `tests/test_parsers.py`

### Adding a New Agent

1. Create `codexrag/agents/your_agent.py`
2. Add the new intent to `QueryIntent` enum in `orchestrator.py`
3. Update `IntentClassifier.INTENT_KEYWORDS`
4. Add prompt template in `orchestrator.py`

## 🧪 Testing

### Running Tests
```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=codexrag

# Specific test file
pytest tests/test_basics.py
```

### Writing Tests
- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names

### Golden Set Tests
Add new evaluation queries to `tests/golden_set.json`:
```json
{
  "query": "Your question here",
  "ground_truth": "Expected answer content",
  "category": "code|data|math|docs",
  "difficulty": "easy|medium|hard"
}
```

## 📤 Submitting Changes

### Pull Request Process

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear commit messages
   - Add tests for new functionality
   - Update documentation if needed

3. **Run quality checks**
   ```bash
   black codexrag/
   isort codexrag/
   flake8 codexrag/
   pytest tests/
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### PR Guidelines
- Reference any related issues
- Describe what your changes do
- Include test results
- Add screenshots for UI changes

## 🐛 Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/tracebacks

## 💡 Feature Requests

We welcome feature suggestions! Please:
- Check existing issues first
- Describe the use case
- Explain the expected behavior

## 📜 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## 📞 Contact

- **Author**: Sabin Thapa
- **Email**: sthapa3@kent.edu
- **GitHub Issues**: For bug reports and feature requests

Thank you for contributing! 🎉

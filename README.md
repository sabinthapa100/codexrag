# CodeXRAG: Scientific Codebase Assistant

[![CI](https://github.com/sabinthapa100/codexrag/actions/workflows/ci.yml/badge.svg)](https://github.com/sabinthapa100/codexrag/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0-orange.svg)](https://github.com/sabinthapa100/codexrag)

> **A State-of-the-Art Multi-Agent RAG System for Scientific Codebases**

CodeXRAG is a Verified Research Assistant that understands the structure of scientific code, parses heterogeneous data formats, and provides accurate, cited answers to your questions about any codebase.

---

## ✨ Key Features

### 🧠 Multi-Agent Architecture
- **Intelligent Routing**: Automatically classifies queries (Code/Data/Math/Docs) and routes to specialized agents
- **Self-Correction**: Agentic loop with confidence scoring and iterative retrieval
- **Cross-File Understanding**: Answers questions that span multiple files

### 📊 GraphRAG Integration (v0.3.0)
- **Code Knowledge Graph**: Builds entity-relationship graph from Python AST
- **Dependency Traversal**: Finds callers, callees, and related code automatically
- **Context Expansion**: Enriches retrieval with graph-derived context

### 🔬 Scientific-First Design
- **AST-Aware Parsing**: Understands functions, classes, and code structure
- **Heterogeneous Formats**: Parses Python, C++, Jupyter, CSV, PDF, HDF5
- **LaTeX Support**: Web UI renders equations natively

### 📏 Rigorous Evaluation (v0.3.0)
- **Ragas-Style Metrics**: Faithfulness, Answer Relevancy, Context Precision/Recall
- **Golden Set Testing**: Benchmark against curated query-answer pairs
- **CI Integration**: Automated quality tracking

### 🛡️ Safety & Auditability
- **Zero-Deletion Guarantee**: Read-only operations only
- **Audit Trail**: Every query logged for reproducibility

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/sabinthapa100/codexrag.git
cd codexrag
./setup_env.sh
source .venv/bin/activate
```

### Index Your Codebase
**Option 1: Standalone Mode (Recommended)**
```bash
# 1. Enter the directory
cd CodeXRAG_Dist

# 2. Index the parent repo (your code)
./.venv/bin/codexrag index --repo .. --config ../rag4mycodex/config.yaml
```

**Option 2: From Parent Root**
```bash
# Index current directory
./CodeXRAG_Dist/.venv/bin/codexrag index --repo . --config rag4mycodex/config.yaml
```

### Ask Questions
```bash
# CLI
./CodeXRAG_Dist/.venv/bin/codexrag ask --repo . --config rag4mycodex/config.yaml --question "How is RpA computed?"

# Interactive Chat
./CodeXRAG_Dist/.venv/bin/codexrag chat --repo . --config rag4mycodex/config.yaml

# Web UI (Run from CodeXRAG_Dist folder)
cd CodeXRAG_Dist
streamlit run codexrag/gui.py
```

### Run Evaluation
```bash
codexrag eval --repo /path/to/your/project
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Intent       │  │ Self-        │  │ Confidence           │  │
│  │ Classifier   │──│ Correction   │──│ Scoring              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HYBRID RETRIEVAL                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ BM25         │  │ Vector       │  │ Graph-Enhanced       │  │
│  │ (Keywords)   │──│ (Semantic)   │──│ (Dependencies)       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SPECIALIZED AGENTS                           │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │ Code       │ │ Data       │ │ Math       │ │ Docs       │   │
│  │ Analyst    │ │ Expert     │ │ Expert     │ │ Expert     │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              FORMATTED ANSWER WITH CITATIONS                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Evaluation Metrics

CodeXRAG uses Ragas-style metrics to measure quality:

| Metric | Description | Target |
|--------|-------------|--------|
| **Faithfulness** | Is the answer grounded in context? | > 0.8 |
| **Answer Relevancy** | Does the answer address the question? | > 0.7 |
| **Context Precision** | Are retrieved chunks relevant? | > 0.6 |
| **Context Recall** | Did retrieval find necessary info? | > 0.7 |

Run `codexrag eval` to benchmark your instance.

---

## 📁 Project Structure

```
codexrag/
├── agents/                 # Multi-agent system
│   ├── orchestrator.py     # Central coordinator with self-correction
│   └── safety_agent.py     # Read-only enforcement & audit
├── graph/                  # GraphRAG module (v0.3.0)
│   ├── code_graph.py       # AST-based knowledge graph
│   └── graph_retriever.py  # Graph-enhanced retrieval
├── eval/                   # Evaluation harness (v0.3.0)
│   ├── metrics.py          # Ragas-style metrics
│   └── golden_set.py       # Golden query management
├── parsers/                # File parsers
│   ├── python_parser.py    # AST-aware Python parsing
│   ├── notebook_parser.py  # Jupyter cell extraction
│   └── csv_parser.py       # Data file parsing
├── retriever.py            # Hybrid BM25 + Vector search
├── config.py               # Configuration management
└── cli.py                  # Command-line interface
```

---

## 🔧 Configuration

Edit `config.yaml`:

```yaml
# What to index
include_globs:
  - "**/*.py"
  - "**/*.ipynb"
  - "**/*.md"

# LLM settings (requires Ollama)
ollama_model: "qwen2.5-coder:14b"
ollama_base_url: "http://localhost:11434"

# Retrieval tuning
top_k_bm25: 12
top_k_vector: 12
top_k_rerank: 8
```

---

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Add tests** for new functionality
4. **Submit** a pull request

### Good First Issues
- Add parser for `.fortran` files
- Improve intent classification with embeddings
- Add support for additional LLM backends

---

## 📚 Documentation

- [How It Works](docs/HOW_IT_WORKS.md) - Complete technical deep-dive
- [RAG Concepts](docs/RAG_CONCEPTS.md) - Educational overview

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Sabin Thapa**
- Email: sthapa3@kent.edu / sabinthapa240@gmail.com
- GitHub: [@sabinthapa100](https://github.com/sabinthapa100)

---

## 🌟 Star History

If you find CodeXRAG useful, please star this repository! ⭐

---

## 🔮 Roadmap

- [ ] Docker deployment with persistent vector store
- [ ] HuggingFace Spaces demo
- [ ] Support for Claude/Grok via LiteLLM
- [ ] arXiv preprint with benchmark results

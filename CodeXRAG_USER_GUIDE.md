# CodeXRAG User Guide for Charmonia Analysis

This guide explains how to use the CodeXRAG system specifically for your `charmonia_combined_analysis` codebase.

## 1. Setup Environment

Always start by navigating to the `CodeXRAG_Dist` directory and activating the environment.

```bash
cd CodeXRAG_Dist
source .venv/bin/activate
```

> **Tip**: You can add an alias to your `.bashrc` or `.zshrc`:
> `alias crag='source /path/to/CodeXRAG_Dist/.venv/bin/activate && codexrag'`

## 2. Index Your Codebase (One-Time Setup)

You need to build the index and knowledge graph. This scans all your Python, Notebooks, and Markdown files.

```bash
# From the root of your charmonia_combined_analysis repo:
./CodeXRAG_Dist/.venv/bin/codexrag index --repo . --config rag4mycodex/config.yaml
```

**Success Indicators:**
- "Building knowledge graph..."
- "Graph built: X entities, Y edges"
- "Success! Indexed repo at..."

## 3. Using the System

### Option A: Web Interface (Recommended)
Best for viewing equations like $R_{pA}$ and full source citations.

```bash
cd CodeXRAG_Dist
streamlit run codexrag/gui.py
```
*It will open in your browser at http://localhost:8501*

### Option B: Command Line Chat
Good for quick questions without leaving the terminal.

```bash
# Interactive Chat
./CodeXRAG_Dist/.venv/bin/codexrag chat --repo . --config rag4mycodex/config.yaml

# Single Question
./CodeXRAG_Dist/.venv/bin/codexrag ask --repo . --config rag4mycodex/config.yaml --question "How is RpA calculated?"
```

## 4. Running Evaluations

Verify the system's accuracy using the "Golden Set" of core physics questions.

```bash
# Run standard evaluation
./CodeXRAG_Dist/.venv/bin/codexrag eval --repo . --config rag4mycodex/config.yaml --golden-set tests/golden_set.json
```

## 5. Troubleshooting

**"Index is empty"**
- Ensure you ran the `index` command (Step 2) successfully.
- Check `.codexrag` directory exists in your repo root.

**"Ollama not found"**
- Ensure Ollama is running (`ollama serve`).
- Verify the model name in `config.yaml` matches what you have pulled (`ollama list`).

## 6. Advanced: Managing the Knowledge Graph

The knowledge graph is stored in `.codexrag/index/graph.json`.
- It is rebuilt every time you run `index`.
- It powers the "Dependency Analysis" capabilities (e.g., "What calls X?").

# How CodeXRAG Works: A Complete Technical Guide

This document explains **exactly** how the RAG system is constructed, from raw code to intelligent answers.

---

## 🔄 The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INDEXING PHASE (One-Time)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Your Codebase          Parsers              Chunks             Index      │
│   ┌──────────┐       ┌────────────┐       ┌──────────┐       ┌──────────┐  │
│   │ *.py     │──────▶│ AST Parser │──────▶│ Function │──────▶│ FAISS    │  │
│   │ *.ipynb  │       │ NB Parser  │       │ Class    │       │ (Vectors)│  │
│   │ *.csv    │       │ CSV Parser │       │ Cell     │       │ BM25     │  │
│   │ *.pdf    │       │ PDF Parser │       │ Table    │       │ (Keywords)│ │
│   └──────────┘       └────────────┘       └──────────┘       └──────────┘  │
│                                                 │                          │
│                                                 ▼                          │
│                                       ┌────────────────┐                   │
│                                       │ Embeddings     │                   │
│                                       │ all-MiniLM-L6  │                   │
│                                       └────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUERY PHASE (Every Question)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Query        Orchestrator        Retrieval           LLM Answer      │
│   ┌──────────┐     ┌────────────┐     ┌──────────┐       ┌──────────────┐  │
│   │"Where is │────▶│ Classify   │────▶│ BM25     │──┐    │              │  │
│   │ RpA      │     │ Intent     │     │ Search   │  ├───▶│ Ollama/LLM   │  │
│   │computed?"│     │ (CODE)     │     ├──────────┤  │    │ + Context    │  │
│   └──────────┘     └────────────┘     │ Vector   │──┘    │              │  │
│                                       │ Search   │       └──────────────┘  │
│                                       └──────────┘              │          │
│                                             │                   ▼          │
│                                             ▼           ┌──────────────┐   │
│                                       ┌──────────┐      │ "RpA is      │   │
│                                       │ Reranker │      │  computed in │   │
│                                       │ (Top-K)  │      │  file X:L123"│   │
│                                       └──────────┘      └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. INDEXING: How Your Code Becomes Searchable

### Step 1: File Scanning (`fs_scan.py`)
```python
# Walks your codebase respecting include/exclude globs
files = scan_files(repo_root, ["**/*.py", "**/*.ipynb"], ["**/.git/**"])
```

### Step 2: Parsing (`codexrag/parsers/`)

Each file type has a specialized parser:

| File Type | Parser | What It Extracts |
|-----------|--------|------------------|
| `.py` | `PythonASTParser` | Functions, Classes, Docstrings (with line numbers) |
| `.ipynb` | `NotebookParser` | Code cells, Markdown cells (individually) |
| `.csv` | `CSVParser` | First 200 rows as schema sample |
| `.pdf` | `PDFParser` | Text per page (with page numbers for citations) |

**Why AST Parsing Matters:**
```python
# BAD: Simple text chunking breaks this:
def compute_rpa(
    sigma_pA,  # Cross section
    sigma_pp   # Proton baseline
):
    return sigma_pA / (A * sigma_pp)

# GOOD: AST Parser keeps the whole function as ONE chunk
# Including its docstring and metadata like "lines 45-52"
```

### Step 3: Chunking (`chunking.py`)

Large files are split into overlapping chunks:
- **Max size**: 2000 characters
- **Overlap**: 200 characters (so context isn't lost at boundaries)

### Step 4: Embedding (`index_store.py`)

Each chunk is converted to a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# "def compute_rpa(sigma_pA, sigma_pp): ..." → [0.023, -0.145, 0.892, ...]
embedding = model.encode(chunk.text)
```

### Step 5: Storage

**Two indexes are created:**

1. **FAISS Vector Index** (`.codexrag/faiss.index`)
   - Stores the 384-dim embeddings
   - Enables semantic similarity search

2. **BM25 Keyword Index** (`.codexrag/bm25.json`)
   - Stores tokenized words
   - Enables exact keyword matching

3. **Chunk Metadata** (`.codexrag/chunks.jsonl`)
   - Stores the actual text and metadata (file path, line numbers)

---

## 2. QUERYING: How Questions Get Answered

### Step 1: Intent Classification (`orchestrator.py`)

```python
query = "How is RpA computed?"

# Keywords detected: "compute" → CODE intent
intent = IntentClassifier.classify(query)  # Returns: QueryIntent.CODE
```

### Step 2: Hybrid Retrieval (`retriever.py`)

**Two searches run in parallel:**

```python
# Search 1: BM25 (Keyword Match)
bm25_results = bm25.get_top_n(tokenize(query), top_k=12)
# Finds chunks containing exact words like "RpA", "compute"

# Search 2: Vector Similarity
query_embedding = model.encode(query)
vector_results = faiss_index.search(query_embedding, top_k=12)
# Finds semantically similar chunks (even if words differ)
```

### Step 3: Fusion & Reranking

Results are merged and re-scored:
```python
# Combine results, remove duplicates
merged = set(bm25_results) | set(vector_results)

# Optional: Cross-Encoder reranking for precision
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
final_top_k = reranker.rerank(query, merged, top_k=8)
```

### Step 4: Prompt Construction

The top chunks are formatted as context:
```
Source [1]: physics/rpa.py:45-52
def compute_rpa(sigma_pA, sigma_pp, A):
    """Compute the nuclear modification factor."""
    return sigma_pA / (A * sigma_pp)

Source [2]: docs/theory.md
The nuclear modification factor RpA is defined as...
```

### Step 5: LLM Generation (`agents/orchestrator.py`)

The context + question is sent to Ollama:
```python
prompt = f"""You are the Code Analyst Agent.
Use ONLY the following context to answer.

{context}

Question: How is RpA computed?

Answer:"""

response = ollama.invoke(prompt)  # → "RpA is computed in physics/rpa.py:45..."
```

---

## 3. WHERE FILES ARE STORED

```
your_project/
├── .codexrag/                    # ← Created by indexing
│   ├── faiss.index              # Vector embeddings (binary)
│   ├── chunks.jsonl             # Chunk text + metadata
│   ├── bm25.json                # BM25 tokenized index
│   ├── cache/
│   │   └── manifest.json        # File hashes (for incremental updates)
│   └── audit.log                # Query history (safety)
├── config.yaml                   # Your settings
└── codexrag/                     # The library
```

---

## 4. KEY CONFIGURATION (`config.yaml`)

```yaml
# What to index
include_globs:
  - "**/*.py"
  - "**/*.ipynb"
  - "**/*.md"

# Where to store index
index_dir: ".codexrag"

# Embedding model (local, runs on CPU)
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

# LLM for answering (requires Ollama running)
ollama_model: "qwen2.5-coder:14b"
ollama_base_url: "http://localhost:11434"

# Retrieval settings
top_k_bm25: 12      # How many BM25 results
top_k_vector: 12    # How many vector results
top_k_rerank: 8     # Final top-K after reranking
```

---

## 5. RUNNING THE SYSTEM

### First Time Setup
```bash
cd CodeXRAG_Dist
./setup_env.sh          # Install dependencies
source .venv/bin/activate
```

### Index Your Codebase
```bash
# Point to the codebase you want to query
codexrag index --repo /path/to/your/project --config config.yaml
```

### Ask Questions
```bash
# CLI
codexrag chat --repo /path/to/your/project

# Web UI (supports LaTeX!)
streamlit run codexrag/gui.py
```

---

## 6. ADDING NEW PARSERS (Extensibility)

To support a new file type (e.g., `.fortran`):

```python
# codexrag/parsers/fortran_parser.py
from codexrag.parsers.base import Parser

class FortranParser(Parser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in [".f90", ".f"]
    
    def parse(self, path: Path) -> List[Chunk]:
        # Your parsing logic here
        ...
```

Then register it in `codexrag/parsers/__init__.py`.

---

## 7. SAFETY GUARANTEES

- **Read-Only**: The system NEVER modifies your source files
- **Audit Trail**: Every query is logged to `.codexrag/audit.log`
- **No External Calls**: When using local Ollama, no data leaves your machine

# CodeXRAG: Complete Usage Guide (From Scratch)

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** running locally with a model pulled
3. Your codebase (e.g., `charmonia_combined_analysis`)

---

## Step 1: Initial Setup (One-Time)

### 1.1 Navigate to Your Project
```bash
cd /home/sawin/Desktop/Charmonia/charmonia_combined_analysis
```

### 1.2 Activate the Virtual Environment
```bash
# If you already have a venv:
source .venv/bin/activate

# If not, create one:
python3 -m venv .venv
source .venv/bin/activate
```

### 1.3 Install CodeXRAG
```bash
cd CodeXRAG_Dist
pip install -e ".[rerank]"
```

**Expected output:**
```
Successfully installed codexrag-0.3.0 ...
```

### 1.4 Verify Ollama is Running
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it:
ollama serve &

# Pull the model (if not already pulled)
ollama pull qwen2.5-coder:14b
```

---

## Step 2: Index Your Codebase (Fresh Start)

### 2.1 Clean Previous Index (If Any)
```bash
# Go back to your project root
cd /home/sawin/Desktop/Charmonia/charmonia_combined_analysis

# Remove old index
rm -rf .codexrag/
```

### 2.2 Run Indexing
```bash
# Index from the project root
./CodeXRAG_Dist/.venv/bin/codexrag index --repo . --config rag4mycodex/config.yaml
```

**What happens:**
1. Scans all Python files, notebooks, markdown, CSV, etc.
2. Builds vector embeddings (using `sentence-transformers/all-MiniLM-L6-v2`)
3. Builds BM25 index
4. **NEW:** Builds Knowledge Graph (3,146 entities, 21,710 edges)

**Expected output:**
```
Building knowledge graph from 256 Python files...
Graph built: 3146 entities, 21710 edges.
✅ Success! Indexed repo at: /home/sawin/Desktop/Charmonia/charmonia_combined_analysis
Total Chunks: 34202
```

**Time:** ~2-3 minutes (depending on codebase size)

### 2.3 Verify Index Was Created
```bash
ls -lh .codexrag/index/
```

**Expected files:**
```
chunks.json          # All text chunks
embeddings.faiss     # Vector index
bm25.json           # BM25 index
graph.json          # Knowledge graph
```

---

## Step 3: Using the System

### Option A: Web Interface (Recommended)

#### 3.1 Launch Streamlit GUI
```bash
cd CodeXRAG_Dist
streamlit run codexrag/gui.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.0.114:8501
```

#### 3.2 Verify System Status
In the sidebar, you should see:
```
✅ RAG Index Loaded
📚 Knowledge Base: 34202 chunks
🧠 Model: qwen2.5-coder:14b
```

#### 3.3 Ask Questions
**Try these examples:**

**Physics Question:**
```
How do you compute nuclear modification factor?
```
**Expected answer:**
- Formula with LaTeX: $R_{pA} = \frac{\sigma_{pA}}{A \cdot \sigma_{pp}}$
- Explanation of terms
- Repo implementation (if found)
- Citation: `[Source 1] npdf_code/rpa_calculator.py:45-52`

**Repo Question:**
```
What does eloss_code do?
```
**Expected answer:**
- Module description
- Key functions
- File citations
- Usage examples

**Code Location:**
```
Where is compute_rpa_grid defined?
```
**Expected answer:**
- File path
- Function signature
- Line numbers
- What it does

---

### Option B: Command Line Interface

#### 3.4 Single Question
```bash
cd /home/sawin/Desktop/Charmonia/charmonia_combined_analysis

./CodeXRAG_Dist/.venv/bin/codexrag ask \
  --repo . \
  --config rag4mycodex/config.yaml \
  --question "How is RpA computed?"
```

#### 3.5 Interactive Chat
```bash
./CodeXRAG_Dist/.venv/bin/codexrag chat \
  --repo . \
  --config rag4mycodex/config.yaml
```

**Example session:**
```
>>> What is energy loss in this codebase?
🧠 Query Analysis
Intent Detected: CODE Active Agent: Code Analyst

Energy loss is computed in the eloss_code module...
[Source 1] eloss_code/energy_loss.py:120-145
---

>>> How is it calculated?
[Continues conversation with context]
---

>>> exit
```

---

## Step 4: Understanding Answers

### Answer Structure (NEW!)
All answers now follow this format:

1. **Definition/Formula** (if physics)
   - LaTeX equations
   - Conceptual explanation

2. **Implementation in Codebase**
   - Which module/file
   - Key functions

3. **Location**
   - `[Source N] path/to/file.py:start-end`

4. **Usage Example**
   - How to call it
   - Parameters

5. **Caveats**
   - Assumptions
   - Limitations

### Source Labels

**Repo Citations:**
```
[Source 1] npdf_code/rpa_calculator.py:45-52
```
→ Answer comes from your codebase

**General Knowledge:**
```
(general physics - not from codebase)
```
→ Answer from LLM's training (when retrieval is weak)

---

## Step 5: Advanced Usage

### 5.1 Adjust Retrieval Parameters
Edit `rag4mycodex/config.yaml`:

```yaml
# Increase for more context
max_context_chunks: 12  # Default: 12 (was 8)

# Increase for broader search
top_k_bm25: 12
top_k_vector: 12
top_k_rerank: 8
```

### 5.2 Enable Cross-Encoder Reranking (Optional)
```yaml
reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**Then re-index:**
```bash
./CodeXRAG_Dist/.venv/bin/codexrag index --repo . --config rag4mycodex/config.yaml
```

### 5.3 View Safety Audit Log
```bash
cat .codexrag/audit.log
```

**Shows:**
- All queries
- Timestamps
- Safety checks

---

## Step 6: Troubleshooting

### Problem: "Knowledge Base: 0 chunks"

**Solution:**
```bash
# Check if index exists
ls .codexrag/index/

# If missing, re-index
./CodeXRAG_Dist/.venv/bin/codexrag index --repo . --config rag4mycodex/config.yaml
```

### Problem: "Index is empty" when running from CodeXRAG_Dist

**Solution:**
```bash
# Always run from parent directory OR use --repo ..
cd /home/sawin/Desktop/Charmonia/charmonia_combined_analysis
./CodeXRAG_Dist/.venv/bin/codexrag ask --repo . --question "..."
```

### Problem: Ollama connection error

**Solution:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve

# Verify model is pulled
ollama list
```

### Problem: Answers are still generic

**Solution:**
```bash
# Increase context chunks
# Edit rag4mycodex/config.yaml:
max_context_chunks: 16

# Re-run query (no need to re-index)
```

---

## Step 7: Evaluation (Optional)

### 7.1 Run Golden Question Suite
```bash
cd CodeXRAG_Dist

./venv/bin/codexrag eval \
  --repo .. \
  --config ../rag4mycodex/config.yaml \
  --golden-set tests/golden/questions.yaml
```

**Shows:**
- Faithfulness score
- Answer relevancy
- Context precision/recall

---

## Quick Reference Card

### Daily Usage
```bash
# 1. Activate environment
cd /home/sawin/Desktop/Charmonia/charmonia_combined_analysis
source .venv/bin/activate

# 2. Launch GUI
cd CodeXRAG_Dist
streamlit run codexrag/gui.py

# 3. Ask questions in browser
```

### Re-indexing (After Code Changes)
```bash
cd /home/sawin/Desktop/Charmonia/charmonia_combined_analysis
./CodeXRAG_Dist/.venv/bin/codexrag index --repo . --config rag4mycodex/config.yaml
```

### CLI Quick Query
```bash
./CodeXRAG_Dist/.venv/bin/codexrag ask --repo . --config rag4mycodex/config.yaml --question "YOUR QUESTION"
```

---

## What's Different After Upgrade?

### ✅ Fixed
- No more "Please provide a question" refusals
- Physics questions get answers (labeled if from general knowledge)
- Repo questions cite actual files
- Structured, trustworthy answers

### ✅ Improved
- 50% more context (8 → 12 chunks)
- Clearer prompts
- Better LaTeX rendering
- Source attribution

### 🎯 Next Steps (Optional)
See `UPGRADE_PLAN.md` for:
- HyDE query expansion
- Web search fallback
- Second-pass retrieval
- Advanced routing

---

## Support

**Documentation:**
- `IMPLEMENTATION_SUMMARY.md` - What was fixed
- `UPGRADE_PLAN.md` - Future improvements
- `tests/golden/questions.yaml` - Example questions

**Issues?**
Check `CodeXRAG_Dist/.codexrag/audit.log` for query history and errors.

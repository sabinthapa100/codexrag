# CodeXRAG Upgrade Plan: From "Dumb RAG" to "Reliable Research Assistant"

## Current State Assessment

**Critical Bugs Identified:**
1. ❌ "No question provided" refusal even when question exists
2. ❌ Mis-routing: physics questions → code agent → refusal
3. ❌ No answerability gating (always tries retrieval, refuses if weak)
4. ❌ No fallback retrieval (no query expansion, no second pass)
5. ❌ Answers lack structure + proper citations

**Current Architecture:**
- Router: `IntentClassifier` (CODE/DATA/DOCS/MATH/GENERAL)
- Orchestrator: Routes to specialist prompts
- Retrieval: Hybrid BM25 + Vector + optional GraphRAG
- LLM: `qwen2.5-coder:14b` (good for code, weak for physics)

## Minimal Change Upgrade Plan

### COMMIT 1: Add Golden Question Test Suite
**Files:** `tests/golden/questions.yaml`, `tests/test_golden_questions.py`
**Change:** Create regression test suite with 25 questions
**Test:** `pytest tests/test_golden_questions.py` (initially xfail)
**Risk:** None (new files only)

### COMMIT 2: Fix Core Refusal Bug
**Files:** `codexrag/agents/orchestrator.py`
**Change:** 
- Remove "refuse if no context" logic
- Add adaptive fallback: physics→general knowledge, repo→second pass
- Update SPECIALIST_PROMPT to never refuse on valid user input
**Test:** Test "How do you compute nuclear modification factor?" → gets answer
**Risk:** Low (prompt-only change)

### COMMIT 3: Upgrade Router (Domain + Grounding)
**Files:** `codexrag/agents/orchestrator.py`
**Change:**
- Router outputs: `{domain: code|repo|physics|data|docs, grounding: kb_only|kb_preferred|web_allowed}`
- Physics questions route to physics mode (not code agent)
**Test:** 8 routing decisions verified
**Risk:** Low (extends existing classifier)

### COMMIT 4: Add HyDE Query Expansion
**Files:** `codexrag/retriever.py`, `config.yaml`
**Change:**
- Add `enable_hyde: false` config flag
- If enabled: generate hypothetical doc, embed, retrieve
- Fallback to original query if HyDE fails
**Test:** Verify HyDE improves retrieval on 5 hard questions
**Risk:** Low (behind flag, graceful fallback)

### COMMIT 5: Add Second-Pass Retrieval
**Files:** `codexrag/agents/orchestrator.py`
**Change:**
- If `len(hits) < 3` or `max_score < 0.3`: expand query + increase k + retry
- Use graph neighbors if available
**Test:** Verify second pass triggered and improves context
**Risk:** Low (only activates on poor first retrieval)

### COMMIT 6: Add Structured Answer Format
**Files:** `codexrag/agents/orchestrator.py`
**Change:**
- Update prompts to enforce structure:
  1. Definition/Formula
  2. Repo Implementation
  3. Code Location
  4. Usage
  5. Caveats
- Enforce citation rules
**Test:** Snapshot tests for format compliance
**Risk:** Low (prompt-only)

### COMMIT 7: Add Optional Web Fallback
**Files:** `codexrag/tools/web_search.py`, `config.yaml`
**Change:**
- Add `enable_web: false` config flag
- Only activate if `grounding=web_allowed` AND (low confidence OR explicit request)
- Separate WEB_CONTEXT from REPO_CONTEXT
**Test:** Web NOT used for repo questions, IS used for "find papers on X"
**Risk:** Low (behind flag, strict gating)

### COMMIT 8: Fix Indexing Warnings
**Files:** `codexrag/parsers/notebook_parser.py`, `codexrag/indexer.py`
**Change:**
- Normalize notebooks (ensure cell IDs)
- Suppress harmless escape sequence warnings from physics strings
**Test:** `codexrag index` runs without errors
**Risk:** Very low (safe normalization)

## Acceptance Criteria

- [ ] All 25 golden questions get non-refusal answers
- [ ] Physics questions cite general knowledge when repo context missing
- [ ] Repo questions always cite file:line
- [ ] "What does eloss_code do?" → structured answer with citations
- [ ] "How do you compute RpA?" → formula + repo implementation
- [ ] Indexing runs clean (no hard errors)
- [ ] All tests pass
- [ ] Config flags documented in README

## How to Verify

```bash
# Run golden question suite
pytest tests/test_golden_questions.py -v

# Test specific failure cases
codexrag ask --repo . --question "How do you compute nuclear modification factor?"
codexrag ask --repo . --question "What does eloss_code do?"

# Verify indexing
codexrag index --repo . --config config.yaml 2>&1 | grep -i error

# Run full test suite
pytest tests/ -v
```

## Rollback Plan

Each commit is independently revertable:
```bash
git revert <commit-hash>
```

All new features behind config flags can be disabled without code changes.

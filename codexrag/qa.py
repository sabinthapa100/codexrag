from __future__ import annotations
from typing import List
from codexrag.types import RetrievalHit
from codexrag.prompting import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from codexrag.llm_ollama import make_ollama

def _format_context(hits: List[RetrievalHit], max_chunks: int) -> str:
    blocks = []
    for i, h in enumerate(hits[:max_chunks], start=1):
        m = h.chunk.meta
        if m.get("type") == "pdf":
            cite = f'{m.get("path")}:(page {m.get("page")})'
        else:
            sl = m.get("start_line")
            el = m.get("end_line")
            cite = f'{m.get("path")}'
            if sl and el:
                cite = f'{m.get("path")}:{sl}-{el}'
        blocks.append(f"[{i}] {cite}\n{h.chunk.text}")
    return "\n\n---\n\n".join(blocks)

def answer_with_ollama(question: str, hits: List[RetrievalHit], model: str, base_url: str, max_context_chunks: int) -> str:
    llm = make_ollama(model=model, base_url=base_url)
    context = _format_context(hits, max_context_chunks)
    prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(question=question, context=context)
    return llm.invoke(prompt)

from __future__ import annotations
from langchain_ollama import OllamaLLM

def make_ollama(model: str, base_url: str) -> OllamaLLM:
    return OllamaLLM(model=model, base_url=base_url)

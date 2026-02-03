"""
Orchestrator Agent - Multi-Agent Coordinator
=============================================
Implements the Subagent Pattern from LangChain's multi-agent architectures.

The Orchestrator:
1. Receives user queries
2. Classifies intent (code, data, docs, math)
3. Routes to specialized agents
4. Aggregates and synthesizes responses
5. Ensures proper citations

Based on: https://www.blog.langchain.com/choosing-the-right-multi-agent-architecture/

Author: Sabin Thapa (sthapa3@kent.edu)
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from codexrag.types import RetrievalHit
from codexrag.agents.safety_agent import SafetyAgent


class QueryIntent(Enum):
    """Classification of user query intent."""
    CODE = "code"           # Questions about code structure, functions, classes
    DATA = "data"           # Questions about data files, outputs, plots
    DOCS = "docs"           # Questions about documentation, papers
    MATH = "math"           # Questions about formulas, physics
    GENERAL = "general"     # General questions, requires multiple agents


class IntentClassifier:
    """Classifies user query intent."""
    
    # Keywords for each intent
    INTENT_KEYWORDS = {
        QueryIntent.CODE: [
            "function", "class", "method", "import", "variable", "loop",
            "def ", "return", "parameter", "argument", "compute", "calculate",
            "where is", "how does", "implementation", "algorithm", "called"
        ],
        QueryIntent.DATA: [
            "csv", "hdf5", "output", "data", "file", "plot", "figure",
            "results", "table", "column", "row", "values", "generate", "json", "yaml"
        ],
        QueryIntent.DOCS: [
            "documentation", "readme", "paper", "reference", "explain",
            "tutorial", "guide", "help", "what is", "describe", "meaning"
        ],
        QueryIntent.MATH: [
            "formula", "equation", "integral", "derivative", "physics",
            "latex", "math", "units", "dimension", "theory", "model", "term", "sum"
        ]
    }
    
    @classmethod
    def classify(cls, query: str) -> QueryIntent:
        """
        Classify query intent based on keywords.
        
        Args:
            query: User's question
            
        Returns:
            QueryIntent enum value
        """
        query_lower = query.lower()
        scores = {intent: 0 for intent in QueryIntent}
        
        for intent, keywords in cls.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[intent] += 1
        
        # Find highest scoring intent
        max_score = max(scores.values())
        if max_score == 0:
            return QueryIntent.GENERAL
        
        # In case of tie, prioritize Code > Math > Data > Docs
        priority = [QueryIntent.CODE, QueryIntent.MATH, QueryIntent.DATA, QueryIntent.DOCS]
        for p_intent in priority:
            if scores[p_intent] == max_score:
                return p_intent
        
        return QueryIntent.GENERAL


# --- PROMPTS ---

ORCHESTRATOR_PROMPT = """You are analyzing a scientific codebase to answer this question:
"{query}"

Below are relevant code snippets and documentation from the codebase:

{context}

TASK: Provide a clear, detailed answer based ONLY on the information above. 
- Cite sources using [Source N] format
- Explain code logic when relevant
- Use LaTeX for math (e.g., $R_{{pA}}$)
- If the context doesn't answer the question, say: "The indexed codebase doesn't contain information about this."

Answer:"""

SPECIALIST_PROMPT_TEMPLATE = """You are a {role} analyzing a scientific codebase.

QUESTION: {query}

CONTEXT FROM CODEBASE:
{context}

TASK: Answer the question above using ONLY the context provided.
- Be specific and technical
- Cite sources: [Source N] filename:lines
- Explain the code/physics logic
- Use LaTeX for equations
- If context is insufficient, say: "The available code snippets don't fully answer this."

Answer:"""


class Orchestrator:
    """
    Main orchestrator for multi-agent RAG system.
    """
    
    def __init__(
        self,
        llm,
        retriever,
        safety_agent: Optional[SafetyAgent] = None
    ):
        """
        Initialize Orchestrator.
        
        Args:
            llm: LangChain LLM instance
            retriever: HybridRetriever instance
            safety_agent: Optional SafetyAgent for audit logging
        """
        self.llm = llm
        self.retriever = retriever
        self.safety = safety_agent
        self.classifier = IntentClassifier()
    
    def _format_context(self, hits: List[RetrievalHit], max_chunks: int = 8) -> str:
        """Format retrieved chunks as context string."""
        blocks = []
        for i, h in enumerate(hits[:max_chunks], start=1):
            m = h.chunk.meta
            path = m.get("path", "unknown")
            
            # Create citation
            if m.get("type") == "pdf":
                cite = f'{path} (Page {m.get("page", "?")})'
            else:
                sl = m.get("start_line")
                el = m.get("end_line")
                if sl and el:
                    cite = f'{path}:{sl}-{el}'
                else:
                    cite = path
            
            # Format block
            blocks.append(f"Source [{i}]: {cite}\n{h.chunk.text}")
            
        return "\n\n---\n\n".join(blocks)
    
    def _get_specialist_config(self, intent: QueryIntent) -> tuple[str, str]:
        """Get role name and specialty description for intent."""
        if intent == QueryIntent.CODE:
            return ("Code Analyst", "Understanding Python/C++ code structure, functions, classes, and control flow.")
        elif intent == QueryIntent.DATA:
            return ("Data Expert", "Interpreting CSV/HDF5 schemas, output files, plots, and data pipelines.")
        elif intent == QueryIntent.DOCS:
            return ("Documentation Expert", "Analyzing markdown documentation, papers, and READMEs.")
        elif intent == QueryIntent.MATH:
            return ("Physics/Math Expert", "Explaining formulas, LaTeX equations, and physical models.")
        else:
            return ("General Assistant", "General codebase analysis.")

    def query(
        self,
        question: str,
        top_k_bm25: int = 12,
        top_k_vector: int = 12,
        top_k_rerank: int = 8,
        max_context_chunks: int = 8,
        enable_self_correction: bool = True
    ) -> str:
        """
        Process a user query through the multi-agent system.
        
        Args:
            question: User's question
            top_k_bm25: BM25 retrieval candidates
            top_k_vector: Vector retrieval candidates
            top_k_rerank: Final reranked results
            max_context_chunks: Max chunks to include in prompt
            enable_self_correction: Enable agentic self-correction loop
            
        Returns:
            Formatted answer string
        """
        # 1. Classify intent
        intent = self.classifier.classify(question)
        
        # 2. Retrieve relevant context
        hits = self.retriever.retrieve(question, top_k_bm25, top_k_vector, top_k_rerank)
        
        if not hits:
            return f"### {intent.value.title()} Analysis\n\nI could not find any relevant documents in the index matching your query."
            
        context = self._format_context(hits, max_context_chunks)
        
        # 3. Construct Prompt
        if intent == QueryIntent.GENERAL:
            prompt = ORCHESTRATOR_PROMPT.format(query=question, context=context)
        else:
            role, specialty = self._get_specialist_config(intent)
            prompt = SPECIALIST_PROMPT_TEMPLATE.format(
                role=role,
                specialty=specialty,
                query=question,
                context=context
            )
        
        # 4. Invoke LLM
        response = self._invoke_llm(prompt)
        
        # 5. Self-Correction Loop (Agentic Pattern)
        if enable_self_correction:
            confidence = self._estimate_confidence(response, context, question)
            
            if confidence < 0.5:
                # Low confidence: try refined retrieval
                refined_query = self._refine_query(question, response)
                hits2 = self.retriever.retrieve(refined_query, top_k_bm25, top_k_vector, top_k_rerank)
                
                if hits2:
                    # Merge new context
                    all_hits = self._merge_hits(hits, hits2)
                    context2 = self._format_context(all_hits, max_context_chunks)
                    
                    # Re-generate with expanded context
                    prompt2 = SPECIALIST_PROMPT_TEMPLATE.format(
                        role="Senior Analyst",
                        specialty="Cross-referencing multiple sources for comprehensive answers.",
                        query=question,
                        context=context2
                    )
                    response = self._invoke_llm(prompt2)
                    confidence = self._estimate_confidence(response, context2, question)
        else:
            confidence = None
        
        # 6. Log safety audit
        if self.safety:
            self.safety.log_query(question, str(response)[:100])
        
        # 7. Format final output
        confidence_badge = f" (Confidence: {confidence:.0%})" if confidence else ""
        final_output = f"""
### 🧠 Query Analysis
**Intent Detected**: `{intent.value.upper()}`
**Active Agent**: {self._get_specialist_config(intent)[0]}{confidence_badge}

---

{response}
"""
        return final_output
    
    def _invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM with error handling."""
        try:
            result = self.llm.invoke(prompt)
            # Handle AIMessage objects from LangChain
            if hasattr(result, "content"):
                return result.content
            return str(result)
        except Exception as e:
            if hasattr(self.llm, "predict"):
                return self.llm.predict(prompt)
            return f"Error invoking LLM: {str(e)}"
    
    def _estimate_confidence(self, response: str, context: str, question: str) -> float:
        """
        Estimate confidence in the response.
        
        Heuristics:
        - Low confidence if response admits not knowing
        - Low confidence if response doesn't cite sources
        - Higher confidence if key question terms appear in response
        """
        response_lower = response.lower()
        
        # Check for admission of not knowing
        refusal_patterns = [
            "cannot find", "don't have", "not in the context",
            "no information", "unable to", "not enough"
        ]
        if any(p in response_lower for p in refusal_patterns):
            return 0.3
        
        # Check for citations
        has_citations = bool(re.search(r'\[Source \d+\]|\[.*\.py', response))
        citation_score = 0.3 if has_citations else 0.0
        
        # Check term overlap
        question_terms = set(re.findall(r'\b\w+\b', question.lower()))
        response_terms = set(re.findall(r'\b\w+\b', response_lower))
        
        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "how", "what", "where", "when"}
        question_terms -= stop_words
        
        if question_terms:
            overlap = len(question_terms & response_terms) / len(question_terms)
        else:
            overlap = 0.5
        
        return min(1.0, citation_score + overlap * 0.7)
    
    def _refine_query(self, original_query: str, current_response: str) -> str:
        """
        Refine the query based on current response.
        
        If response mentions related concepts, add them to the query.
        """
        # Extract potential keywords from response
        code_patterns = re.findall(r'\b([A-Z][a-z]+[A-Z]\w+|[a-z_]+_[a-z_]+)\b', current_response)
        
        if code_patterns:
            # Add most common pattern to query
            refined = f"{original_query} {code_patterns[0]}"
            return refined
        
        return original_query
    
    def _merge_hits(self, hits1: List[RetrievalHit], hits2: List[RetrievalHit]) -> List[RetrievalHit]:
        """Merge two hit lists, removing duplicates."""
        seen_ids = {h.chunk.chunk_id for h in hits1}
        merged = list(hits1)
        
        for h in hits2:
            if h.chunk.chunk_id not in seen_ids:
                seen_ids.add(h.chunk.chunk_id)
                merged.append(h)
        
        return merged


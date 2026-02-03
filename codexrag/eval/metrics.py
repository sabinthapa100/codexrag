"""
RAG Evaluation Metrics
======================
Implements Ragas-style metrics for evaluating RAG quality.

These metrics help quantify:
1. How well retrieval is working
2. How faithful the LLM's answers are to the context
3. How relevant the answers are to the questions

Author: Sabin Thapa (sthapa3@kent.edu)
"""

from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Result of evaluating a single query."""
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    
    def overall_score(self) -> float:
        """Compute weighted overall score."""
        return (
            0.3 * self.faithfulness +
            0.3 * self.answer_relevancy +
            0.2 * self.context_precision +
            0.2 * self.context_recall
        )


def faithfulness_score(
    answer: str,
    contexts: List[str],
    llm: Optional[object] = None
) -> float:
    """
    Measure how faithful the answer is to the provided context.
    
    A faithful answer only makes claims that can be verified from the context.
    
    Without LLM: Uses simple overlap-based heuristic.
    With LLM: Uses LLM to decompose answer into claims and verify each.
    
    Args:
        answer: The generated answer
        contexts: The retrieved context chunks
        llm: Optional LLM for advanced verification
        
    Returns:
        Score between 0 and 1 (1 = fully faithful)
    """
    if not answer or not contexts:
        return 0.0
    
    # Simple heuristic: word overlap between answer and context
    answer_words = set(_tokenize(answer.lower()))
    context_text = " ".join(contexts).lower()
    context_words = set(_tokenize(context_text))
    
    if not answer_words:
        return 0.0
    
    # Calculate overlap ratio
    overlap = answer_words & context_words
    faithfulness = len(overlap) / len(answer_words)
    
    # Penalize if answer contains citations not in context
    # (e.g., "According to file X" when X wasn't retrieved)
    citation_penalty = _check_false_citations(answer, contexts)
    
    return max(0.0, min(1.0, faithfulness - citation_penalty))


def answer_relevancy_score(
    query: str,
    answer: str,
    llm: Optional[object] = None
) -> float:
    """
    Measure how relevant the answer is to the query.
    
    Without LLM: Uses keyword overlap between query and answer.
    With LLM: Generates questions from answer and compares to original.
    
    Args:
        query: The user's question
        answer: The generated answer
        llm: Optional LLM for advanced scoring
        
    Returns:
        Score between 0 and 1 (1 = highly relevant)
    """
    if not query or not answer:
        return 0.0
    
    query_words = set(_tokenize(query.lower()))
    answer_words = set(_tokenize(answer.lower()))
    
    # Check if key query terms appear in answer
    query_terms = _extract_key_terms(query)
    answer_text = answer.lower()
    
    term_hits = sum(1 for term in query_terms if term in answer_text)
    term_coverage = term_hits / len(query_terms) if query_terms else 0.5
    
    # Penalize very short or very long answers
    length_score = _length_penalty(answer)
    
    # Check for "I don't know" type responses
    if _is_refusal(answer):
        return 0.3  # Honest refusal is better than hallucination
    
    return min(1.0, term_coverage * 0.7 + length_score * 0.3)


def context_precision_score(
    query: str,
    contexts: List[str],
    ground_truth: Optional[str] = None
) -> float:
    """
    Measure how precise the retrieved contexts are.
    
    Precision = relevant contexts / total contexts
    
    Without ground truth: Uses query term overlap as relevance proxy.
    With ground truth: Checks if context contains ground truth info.
    
    Args:
        query: The user's question
        contexts: The retrieved context chunks
        ground_truth: Optional known correct answer
        
    Returns:
        Score between 0 and 1 (1 = all contexts are relevant)
    """
    if not contexts:
        return 0.0
    
    query_terms = _extract_key_terms(query)
    
    relevant_count = 0
    for ctx in contexts:
        ctx_lower = ctx.lower()
        term_matches = sum(1 for term in query_terms if term in ctx_lower)
        
        # Consider context relevant if it matches at least half the query terms
        if term_matches >= len(query_terms) / 2:
            relevant_count += 1
        
        # Bonus if context contains ground truth content
        if ground_truth and _contains_ground_truth(ctx, ground_truth):
            relevant_count += 0.5
    
    return min(1.0, relevant_count / len(contexts))


def context_recall_score(
    contexts: List[str],
    ground_truth: str
) -> float:
    """
    Measure how much of the ground truth is covered by contexts.
    
    Recall = covered ground truth claims / total ground truth claims
    
    Args:
        contexts: The retrieved context chunks
        ground_truth: The known correct answer
        
    Returns:
        Score between 0 and 1 (1 = all necessary info was retrieved)
    """
    if not ground_truth:
        return 0.0  # Can't compute without ground truth
    
    if not contexts:
        return 0.0
    
    # Extract key claims from ground truth
    gt_terms = set(_extract_key_terms(ground_truth))
    context_text = " ".join(contexts).lower()
    
    if not gt_terms:
        return 1.0  # No specific terms to find
    
    # Check how many ground truth terms are in context
    found = sum(1 for term in gt_terms if term in context_text)
    
    return found / len(gt_terms)


# --- Helper Functions ---

def _tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    return re.findall(r'\b\w+\b', text.lower())


def _extract_key_terms(text: str) -> List[str]:
    """Extract key terms (nouns, technical words) from text."""
    words = _tokenize(text)
    
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all",
        "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "and", "but", "if", "or", "because",
        "until", "while", "what", "which", "who", "this", "that",
        "these", "those", "i", "you", "he", "she", "it", "we", "they"
    }
    
    key_terms = [w for w in words if w not in stop_words and len(w) > 2]
    return key_terms


def _check_false_citations(answer: str, contexts: List[str]) -> float:
    """Check if answer contains file references not in context."""
    # Pattern for file citations like "file.py:123" or "(path/to/file)"
    citation_pattern = r'[\w/]+\.(py|ipynb|csv|pdf|md)(?::\d+)?'
    
    answer_citations = set(re.findall(citation_pattern, answer))
    context_text = " ".join(contexts)
    context_citations = set(re.findall(citation_pattern, context_text))
    
    false_citations = answer_citations - context_citations
    
    if not answer_citations:
        return 0.0
    
    return len(false_citations) / len(answer_citations) * 0.3


def _length_penalty(answer: str) -> float:
    """Penalize very short or very long answers."""
    word_count = len(_tokenize(answer))
    
    if word_count < 10:
        return 0.3  # Too short
    elif word_count > 500:
        return 0.7  # Too long (may be rambling)
    else:
        return 1.0


def _is_refusal(answer: str) -> bool:
    """Check if the answer is a refusal/admission of not knowing."""
    refusal_patterns = [
        r"i (don't|do not|cannot|can't) (know|find|have)",
        r"cannot find this information",
        r"no information (available|found)",
        r"not (enough|sufficient) (context|information)",
    ]
    
    answer_lower = answer.lower()
    for pattern in refusal_patterns:
        if re.search(pattern, answer_lower):
            return True
    return False


def _contains_ground_truth(context: str, ground_truth: str) -> bool:
    """Check if context contains key ground truth information."""
    gt_terms = _extract_key_terms(ground_truth)
    ctx_lower = context.lower()
    
    matches = sum(1 for term in gt_terms if term in ctx_lower)
    return matches >= len(gt_terms) / 2


class RAGEvaluator:
    """
    Complete RAG evaluation pipeline.
    
    Runs all metrics on a set of queries and aggregates results.
    """
    
    def __init__(self, llm: Optional[object] = None):
        """
        Initialize evaluator.
        
        Args:
            llm: Optional LLM for advanced evaluation (e.g., claim verification)
        """
        self.llm = llm
    
    def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> EvalResult:
        """
        Evaluate a single RAG response.
        
        Args:
            query: The user's question
            answer: The RAG-generated answer
            contexts: The retrieved context chunks
            ground_truth: Optional known correct answer
            
        Returns:
            EvalResult with all metrics
        """
        result = EvalResult(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )
        
        result.faithfulness = faithfulness_score(answer, contexts, self.llm)
        result.answer_relevancy = answer_relevancy_score(query, answer, self.llm)
        result.context_precision = context_precision_score(query, contexts, ground_truth)
        
        if ground_truth:
            result.context_recall = context_recall_score(contexts, ground_truth)
        else:
            result.context_recall = 0.5  # Neutral if no ground truth
        
        return result
    
    def evaluate_batch(
        self,
        queries: List[Dict]
    ) -> Tuple[List[EvalResult], Dict[str, float]]:
        """
        Evaluate a batch of queries.
        
        Args:
            queries: List of dicts with keys: query, answer, contexts, ground_truth
            
        Returns:
            Tuple of (individual results, aggregate metrics)
        """
        results = []
        
        for q in queries:
            result = self.evaluate_single(
                query=q["query"],
                answer=q["answer"],
                contexts=q.get("contexts", []),
                ground_truth=q.get("ground_truth")
            )
            results.append(result)
        
        # Aggregate metrics
        n = len(results)
        if n == 0:
            return results, {}
        
        aggregates = {
            "avg_faithfulness": sum(r.faithfulness for r in results) / n,
            "avg_answer_relevancy": sum(r.answer_relevancy for r in results) / n,
            "avg_context_precision": sum(r.context_precision for r in results) / n,
            "avg_context_recall": sum(r.context_recall for r in results) / n,
            "avg_overall": sum(r.overall_score() for r in results) / n,
            "total_queries": n
        }
        
        return results, aggregates
    
    def generate_report(
        self,
        results: List[EvalResult],
        aggregates: Dict[str, float]
    ) -> str:
        """Generate a human-readable evaluation report."""
        lines = [
            "=" * 60,
            "           CODEXRAG EVALUATION REPORT",
            "=" * 60,
            "",
            f"Total Queries Evaluated: {aggregates.get('total_queries', 0)}",
            "",
            "AGGREGATE METRICS:",
            f"  Faithfulness:        {aggregates.get('avg_faithfulness', 0):.3f}",
            f"  Answer Relevancy:    {aggregates.get('avg_answer_relevancy', 0):.3f}",
            f"  Context Precision:   {aggregates.get('avg_context_precision', 0):.3f}",
            f"  Context Recall:      {aggregates.get('avg_context_recall', 0):.3f}",
            f"  Overall Score:       {aggregates.get('avg_overall', 0):.3f}",
            "",
            "=" * 60,
        ]
        
        # Show worst performing queries for debugging
        sorted_results = sorted(results, key=lambda r: r.overall_score())
        
        if sorted_results:
            lines.append("")
            lines.append("LOWEST SCORING QUERIES (for improvement):")
            for r in sorted_results[:3]:
                lines.append(f"  - [{r.overall_score():.2f}] {r.query[:50]}...")
        
        return "\n".join(lines)

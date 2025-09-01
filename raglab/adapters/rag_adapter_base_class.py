# langops/adapters/base_rag_adapter.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from time import perf_counter


class RAGAdapter(ABC):
    """
    Abstract base class for RAG system adapters.

    Concrete subclasses must implement `retrieve` and `generate`.
    The `rag` method provides a stable wrapper for end-to-end evaluation.
    """

    def __init__(self, *, default_k: int = 5,
                 default_sections: Optional[Tuple[str]] = None,
                 llm_model: str = "default-llm",
                 name: str = "rag-adapter"):
        self.default_k = int(default_k)
        self.default_sections = tuple(default_sections or [])
        self.llm_model = llm_model
        self.name = name  # Used for MLflow params / run metadata

    @abstractmethod
    def retrieve(self, query: str,
                 k: Optional[int] = None,
                 sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Retrieve relevant contexts for a query.

        Args:
            query: Natural language query string.
            k: Number of results to retrieve (default: self.default_k).
            sections: Optional domain-specific sections/scope for retrieval.

        Returns:
            Dict with keys:
                - contexts: List of retrieved items
                - latency_ms: Retrieval time in ms
                - k: Number of retrieved items
                - sections: Sections searched
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, query: str,
                 contexts: List[Dict[str, Any]],
                 llm_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an answer given query and retrieved contexts.

        Args:
            query: User query.
            contexts: Retrieved contexts to ground generation.
            llm_model: Optional override for model name.

        Returns:
            Dict with keys:
                - answer: Generated text
                - latency_ms: Generation time in ms
                - llm_model: Model used
        """
        raise NotImplementedError

    def rag(self, query: str,
            k: Optional[int] = None,
            sections: Optional[List[str]] = None,
            llm_model: Optional[str] = None) -> Dict[str, Any]:
        """
        End-to-end RAG pipeline: retrieve + generate.

        Args:
            query: User query.
            k: Number of retrievals.
            sections: Domain-specific sections for retrieval.
            llm_model: Model override.

        Returns:
            Dict with keys:
                - answer
                - contexts
                - latency_ms: {"retrieve": ms, "generate": ms}
                - k
                - sections
                - llm_model
        """
        r = self.retrieve(query, k=k, sections=sections)
        g = self.generate(query, r["contexts"], llm_model=llm_model)

        return {
            "answer": g["answer"],
            "contexts": r["contexts"],
            "latency_ms": {
                "retrieve": r["latency_ms"],
                "generate": g["latency_ms"],
            },
            "k": r["k"],
            "sections": r["sections"],
            "llm_model": g["llm_model"],
        }

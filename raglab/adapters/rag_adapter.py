# langops/adapters/resume_adapter.py
from time import perf_counter
from rag_pipeline.config import *
from rag_pipeline.rag import *

class RAGAdapter:
    """
    Thin wrapper around your rag functions.
    
    This is a sample implementation for the example RAG system used.
    """
    def __init__(self, *, default_sections=("Intro","Work Experience","Projects","Skills","Education"),
                 default_k=5, llm_model=LLM_MODEL_DEFAULT, name="ask-my-resume"):
        self.default_sections = tuple(default_sections)
        self.default_k = int(default_k)
        self.llm_model = llm_model
        self.name = name  # used in MLflow params, run metadata

    def retrieve(self, query: str, k: int = None, sections=None):
        k = k or self.default_k
        sections = sections or self.default_sections
        t0 = perf_counter()
        ctx = vector_search(query, sections=sections, limit=k)
        t_ms = int((perf_counter() - t0) * 1000)  # Convert seconds -> milliseconds (ms)
        return {"contexts": ctx, "latency_ms": t_ms, "k": k, "sections": list(sections)}

    def generate(self, query: str, contexts, llm_model: str = None):
        llm_model = llm_model or self.llm_model
        t0 = perf_counter()
        answer = generate_with_contexts(query, contexts, llm_model=llm_model)
        t_ms = int((perf_counter() - t0) * 1000)
        return {"answer": answer, "latency_ms": t_ms, "llm_model": llm_model}

    def rag(self, query: str, k: int = None, sections=None, llm_model: str = None):
        r = self.retrieve(query, k=k, sections=sections)
        g = self.generate(query, r["contexts"], llm_model=llm_model)
        return {
            "answer": g["answer"],
            "contexts": r["contexts"],
            "latency_ms": {"retrieve": r["latency_ms"], "generate": g["latency_ms"]},
            "k": r["k"],
            "sections": r["sections"],
            "llm_model": g["llm_model"],
        }
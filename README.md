RagEval is a lightweight, RAG-agnostic evaluation toolkit that:
- **Generates** synthetic, grounded question/answer datasets from your corpus,
- **Validates** them with a separate LLM (no circularity),
- **Evaluates** ANY RAG system with **deterministic metrics** (Hit@k, MRR, semantic similarity) and **LLM judges** (faithfulness, stability),
- **Scales** with Ray, and **logs** with DuckDB + MLflow.

> This repo ships a working example against a Resume Q&A app (Qdrant + Groq/OpenAI), but the evaluator is **RAG-agnostic** via a tiny adapter.

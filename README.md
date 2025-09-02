# RAG Lab - Agentic Evaluation for RAG

Rag Lab is a lightweight, RAG-agnostic evaluation toolkit that:

1. **Generates synthetic Ground Truth datasets** with natural, grounded question/answer pairs from your corpus and **validates** them with a separate LLM to avoid bias and circularity.
2. **Evaluates *any* RAG system** using a combination of **deterministic metrics** (Hit@k, MRR, semantic similarity) and **agentic judges** (faithfulness, stability). The evaluation pipeline also **scales** with Ray for parallelism, and **logs** results with  MLflow and DuckDB.

> This repo includes a working example against the RAG app implemented [here](https://github.com/nmkhan096/ask-my-resume-rag), but the evaluator is RAG-agnostic via a tiny **adapter**.


## ✨ Features

### RAG Adapter Class

To make the evaluator RAG-agnostic, we need to create a wrapper around the RAG functions so the evaluator has a stable interface. With the RAG Adapter class, anyone can plug in their RAG.

We need three functions to cover all evaluation needs:

- `retrieve(query, k) -> [contexts]`
- `generate(query, contexts, llm_model) -> answer`
- `rag_answer(query, k, llm_model) -> (answer, contexts)`

The separation also lets you run A/B tests on **retrieval** vs **generation** independently.

### Synthetic GT pipeline

We generate synthetic GT test cases using two LLMs:

- Model A **"Generator"**: generates Q/A pairs from your indexed context.
- Model B **"Critic/Judge"**: validates A’s output for answerability + faithfulness to the same context. 

> Generate → Validate → (Validate/Repair once) → Freeze

### Dual Eval Modes

The evaluation pipeline is implemented as two layers:

1. **Deterministic Metrics** -> **With GT**
    
    Given a GT dataset of question/answer pairs, we can run the questions through the RAG system and compare retrieved contexts & generated responses to the reference ones and compute **ground-truth metrics for retrieval (Hit@k, Recall@k, MRR, nDCG) and generation (semantic similarity)**.

2. **Agentic Judges** -> **No GT available**

    With no GT available and no gold answer to compare to, we can still evaluate the response against the *retrieved context* and the *instruction*, using **LLM-based judges** to assess: Faithfulness, Factuality, Safety, Self-Consistency, etc.

### Parallelize / Scale with Ray

Ray is used to run the evaluation tasks in parallel: Each Q/A pair in GT is processed in its own **Ray task**, running RAG, computing ground-truth metrics and LLM-judge metrics and returning the results.

Moreover, since we are passing an **adapter class** into every task, Ray would have to **pickle & send** the adapter object to each worker. Hence, I used **Ray Actors** so the adapter is constructed once per worker.

> Note that Ray currently requires Python ≥ 3.8 and ≤ 3.12.

### Storage (DuckDB) & Tracking (MLflow)

- **DuckDB**: for storing per-example details (contexts, answers, judge JSON) locally.
- **MLflow**: for tracking run-level params/metrics, comparisons across experiments; attach the DuckDB file as an artifact.

## 🚀 Quickstart

The RAG example used here follows the same steps outlined [here](https://github.com/nmkhan096/ask-my-resume-rag). The steps below are for implementing `rag_eval` only:

### 1) Install

```
git clone https://github.com/nmkhan096/rag-eval.git
```
### 2) Install dependencies

Assuming you already have a virtual environment for your RAG app activated:
```
pip install -r requirements.txt
```
### 3) Create RAG Adapter

Implement a RAG adatper class for your RAG function using the template given in `raglab/adapters/rag_adapter_base_class.py` and sample implementation in `raglab/adapters/rag_adapter.py`

### 4) Build synthetic GT (Generate → Validate → Repair)

```
python -m raglab.ground_truth.make \
  --chunks data/resume_chunks.json \
  --version gt_v1 \
  --n-per-chunk 3 \
  --repair \
  --out-dir raglab/data_eval
```
This creates and saves a GT dataset in `raglab/data_eval/gt_v1.jsonl`

### 5) Evaluate a RAG system (Ray + MLflow + DuckDB)

```
python -m raglab.orchestrators.ray_runner \
  --data raglab/data_eval/gt_v1.jsonl \
  --llm_model llama-3.3-70b-versatile \
  --k 5 \
  --cpus 2
  --duckdb_dir raglab/runs/duckdb
  --experiment rag_eval
```
Storage:
- **DuckDB (per-example)**: `examples` (metrics/answer/judge/latency) and `contexts` (ranked retrieved chunks).
- **MLflow (run-level)**: `params` (llm_model, k, cpus, workers, dataset), `metrics` (avg hit rate, MRR, faithfulness, latency), and the run’s DuckDB file as an `artifact`.

View results:

- **MLflow UI**: mlflow ui --backend-store-uri raglab/runs/mlruns → browse metrics & params
- **DuckDB**: open DuckDB with DBeaver/CLI/duckdb shell or query directly from Python:

    a. CLI:
    ```
    duckdb runs_db/002b6cde0aa344b1b1b47a77601d25e1.duckdb
    -- inside DuckDB shell:
    SELECT * FROM examples LIMIT 10;
    SELECT * FROM contexts WHERE example_id = 'some_id';
    ```

    b. Python (Notebook or script):
    ```
    import duckdb
    con = duckdb.connect("runs_db/002b6cde0aa344b1b1b47a77601d25e1.duckdb")

    df = con.execute("SELECT * FROM examples").df()
    print(df.head())

    ctx_df = con.execute("SELECT * FROM contexts WHERE example_id = 'example_3'").df()
    ```
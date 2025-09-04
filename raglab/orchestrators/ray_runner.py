import os, json, time, argparse
import numpy as np
import pandas as pd
from statistics import mean
from collections import deque
import ray, duckdb, mlflow

from raglab.adapters.rag_adapter import RAGAdapter
from raglab.metrics import retrieval, generation
from raglab.agents import judges

# ---------------- Rate Limiter (global across all workers) ----------------
@ray.remote
class RateLimiter:
    """Token-bucket-style limiter: at most max_calls per per_seconds across ALL workers."""
    def __init__(self, max_calls: int, per_seconds: float):
        self.max_calls = max_calls
        self.per = per_seconds
        self.q = deque()  # timestamps (seconds)

    def acquire(self):
        now = time.time()
        # drop old timestamps
        while self.q and (now - self.q[0]) > self.per:
            self.q.popleft()

        if len(self.q) >= self.max_calls:
            sleep_for = self.per - (now - self.q[0]) + 0.001
            time.sleep(max(0.0, sleep_for))
            now = time.time()
            while self.q and (now - self.q[0]) > self.per:
                self.q.popleft()

        self.q.append(time.time())
        return True
    
# ---------------- Evaluator (Ray Actor) ----------------
@ray.remote
class Evaluator:
    def __init__(self, llm_model: str, k: int, limiter):
        self.adapter = RAGAdapter(default_k=k, llm_model=llm_model)
        self.k = k
        self.limiter = limiter

    def run_eval(self, example):
        
        q = example["question"]
        r = self.adapter.retrieve(q, k=self.k)
        # Global rate limit across ALL workers before calling the LLM:
        ray.get(self.limiter.acquire.remote())
        g = self.adapter.generate(q, r["contexts"])  # calls llm() (already wrapped with retry)
        ans, ctxs = g["answer"], r["contexts"]

        # Layer 1: deterministic metrics (require ground truth)
        a_metrics = {}
        a_metrics.update(retrieval.compute_all(ctxs, example["doc_id"]))
        a_metrics.update(generation.compute_all(ans, example["gold_answer"], example['text']))

        # Layer 2: agentic judges
        b_metrics = {}
        b_metrics["faithfulness"] = judges.judge_faithfulness(q, ctxs, ans)

        return {
            "id": example["id"],
            "question": q,
            "answer": ans,
            "contexts": ctxs,
            "A": a_metrics,
            "B": b_metrics,
            #"latency_ms": result.get("latency_ms"),
            "latency_ms": {"retrieve": r.get("latency_ms"), "generate": g.get("latency_ms")},
            "doc_id": example.get("doc_id"),
            "gold_answer": example.get("gold_answer")
        }

# ---------------- IO helpers ----------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def to_dataframes(results: list, run_id: str):
    ex_rows = []
    ctx_rows = []
    for r in results:
        a = r.get("A", {}) or {}
        b = r.get("B", {}) or {}
        faith = b.get("faithfulness", {}) or {}
        faithful = faith.get("faithful")
        judge_comments = faith.get("judge_comments")

        lat = r.get("latency_ms") or {}
        latency_retrieve = lat.get("retrieve")
        latency_generate = lat.get("generate")

        ex_rows.append({
            "run_id": run_id,
            "example_id": r.get("id"),
            "question": r.get("question"),
            "answer": r.get("answer"),
            "gold_answer": r.get("gold_answer"),
            "doc_id": r.get("doc_id"),
            "hit": a.get("hit"),
            "mrr": a.get("mrr"),
            "faithful": faithful,
            "judge_comments": judge_comments,
            "latency_ms_retrieve": latency_retrieve,
            "latency_ms_generate": latency_generate,
        })

        for rank, c in enumerate(r.get("contexts") or [], start=1):
            ctx_rows.append({
                "run_id": run_id,
                "example_id": r.get("id"),
                "rank": rank,
                "ctx_doc_id": c.get("doc_id"),
                "ctx_text": c.get("text"),
            })

    examples_df = pd.DataFrame(ex_rows)
    contexts_df = pd.DataFrame(ctx_rows)
    return examples_df, contexts_df


def write_duckdb(db_path: str, examples_df: pd.DataFrame, contexts_df: pd.DataFrame):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = duckdb.connect(db_path)
    con.register("examples_df", examples_df)
    con.register("contexts_df", contexts_df)
    con.execute("CREATE TABLE IF NOT EXISTS examples AS SELECT * FROM examples_df LIMIT 0")
    con.execute("CREATE TABLE IF NOT EXISTS contexts AS SELECT * FROM contexts_df LIMIT 0")
    con.execute("INSERT INTO examples SELECT * FROM examples_df")
    con.execute("INSERT INTO contexts SELECT * FROM contexts_df")
    con.close()


def summarize_for_mlflow(examples_df: pd.DataFrame):
    def _safe_mean(series):
        vals = [v for v in series if v is not None]
        return float(mean(vals)) if vals else 0.0

    hit_rate = _safe_mean([1.0 if bool(x) else 0.0 for x in examples_df["hit"].tolist() if x is not None])
    mrr_mean = _safe_mean([float(x) for x in examples_df["mrr"].tolist() if x is not None])
    faithful_vals = [1.0 if bool(x) else 0.0 for x in examples_df["faithful"].tolist() if x is not None]
    faithful_rate = float(mean(faithful_vals)) if faithful_vals else 0.0

    # Latencies
    lat_r_series = pd.to_numeric(examples_df.get("latency_ms_retrieve", pd.Series(dtype=float)), errors="coerce")
    lat_g_series = pd.to_numeric(examples_df.get("latency_ms_generate", pd.Series(dtype=float)), errors="coerce")

    lat_r = lat_r_series.dropna().tolist()
    lat_g = lat_g_series.dropna().tolist()

    # total per-row, then average
    total_series = (lat_r_series + lat_g_series).dropna()
    lat_t = total_series.tolist()

    return {
        "hit_rate": hit_rate,
        "mrr_mean": mrr_mean,
        "faithful_rate": faithful_rate,
        "latency_retrieve_avg_ms": float(mean(lat_r)) if lat_r else 0.0,
        "latency_generate_avg_ms": float(mean(lat_g)) if lat_g else 0.0,
        "latency_total_avg_ms": float(mean(lat_t)) if lat_t else 0.0,
        "n_examples": int(len(examples_df)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="raglab/data_eval/gt_v1.jsonl")
    parser.add_argument("--llm_model", default="llama-3.3-70b-versatile")
    parser.add_argument("--k", type=int, default=5, help="top-k to retrieve")
    parser.add_argument("--cpus", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--duckdb_dir", default="raglab/runs/duckdb")
    parser.add_argument("--experiment", default="k_all_gt")
    args = parser.parse_args()

    num_workers = args.workers or args.cpus

    # --- start Ray
    ray.init(num_cpus=args.cpus, ignore_reinit_error=True)

    # set rate limits
    max_calls = 25
    limiter = RateLimiter.remote(max_calls=max_calls, per_seconds=60.0)

    # --- MLflow setup
    mlflow.set_tracking_uri("file:raglab/runs/mlruns") #-backend-store-uri
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"{args.llm_model}-k{args.k}") as run: #-w{num_workers}
        run_id = run.info.run_id

        # Params
        mlflow.log_params({
            "llm_model": args.llm_model,
            "k": args.k,
            "cpus": args.cpus,
            "workers": num_workers,
            "dataset_path": args.data
        })

        # Data
        dataset = load_jsonl(args.data)
        if not dataset:
            print("No data found. Check --data path.")
            ray.shutdown()
            return

        # workers
        workers = [Evaluator.remote(llm_model=args.llm_model, k=args.k, limiter=limiter) for _ in range(num_workers)]

        # Warm-up one task in case first call is slow / times out
        # good for heavy clients like Qdrant & LLMs
        first_result = ray.get(workers[0].run_eval.remote(dataset[0]))

        # Dispatch remaining tasks
        t0 = time.perf_counter()
        futures = [workers[i % num_workers].run_eval.remote(ex) for i, ex in enumerate(dataset[1:])]
        rest_results = ray.get(futures)
        elapsed = time.perf_counter() - t0

        results = [first_result] + rest_results
        print(f"\n✅ Completed {len(results)} tasks in {elapsed:.2f}s "
              f"using {num_workers} worker(s) on {args.cpus} CPU(s)")

        # Flatten → DuckDB
        examples_df, contexts_df = to_dataframes(results, run_id)
        db_path = os.path.join(args.duckdb_dir, f"{run_id}.duckdb")
        write_duckdb(db_path, examples_df, contexts_df)

        # Summary metrics → MLflow
        summary = summarize_for_mlflow(examples_df)
        mlflow.log_metrics(summary)
        mlflow.log_metric("wall_time_s", elapsed)

        # Attach the DuckDB file as an artifact
        mlflow.log_param("duckdb_path", db_path)
        mlflow.log_artifact(db_path)

        print(json.dumps(summary, indent=2))
        print(f"\n✅ Wrote {len(examples_df)} example rows and {len(contexts_df)} context rows "
              f"to {db_path}\n✅ Run ID: {run_id} | Wall time: {elapsed:.2f}s")

    ray.shutdown()


if __name__ == "__main__":
    main()

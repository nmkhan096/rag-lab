# demo_eval_ray_1cpu.py
import os, json, time, argparse
from statistics import mean
import numpy as np
import pandas as pd
import ray, duckdb, mlflow

from langops.adapters.rag_adapter import RAGAdapter
from langops.metrics import retrieval, generation
from langops.agents import judges


@ray.remote
class Evaluator:
    def __init__(self, llm_model: str, k: int):
        self.adapter = RAGAdapter(default_k=k, llm_model=llm_model)
        self.k = k

    def run_eval(self, example):
        q = example["question"]
        result = self.adapter.rag(q, k=self.k)
        ans, ctxs = result["answer"], result["contexts"]

        # Layer A: deterministic metrics (require ground truth)
        a_metrics = {}
        a_metrics.update(retrieval.compute_all(ctxs, example["doc_id"]))
        a_metrics.update(generation.compute_all(ans, example["gold_answer"]))

        # Layer B: agentic judges
        b_metrics = {}
        b_metrics["faithfulness"] = judges.judge_faithfulness(q, ctxs, ans)

        return {
            "id": example["id"],
            "question": q,
            "answer": ans,
            "contexts": ctxs,
            "A": a_metrics,
            "B": b_metrics,
            "latency_ms": result.get("latency_ms"),
        }


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
    con.execute("CREATE TABLE IF NOT EXISTS examples AS SELECT * FROM examples_df LIMIT 0")
    con.execute("CREATE TABLE IF NOT EXISTS contexts AS SELECT * FROM contexts_df LIMIT 0")
    con.register("examples_df", examples_df)
    con.register("contexts_df", contexts_df)
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

    def _nums(col):
        return [float(x) for x in col.tolist() if isinstance(x, (int, float))]

    lat_r = _nums(examples_df.get("latency_ms_retrieve", pd.Series([])))
    lat_g = _nums(examples_df.get("latency_ms_generate", pd.Series([])))
    lat_t = (lat_r or 0) + (lat_g or 0)

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
    parser.add_argument("--data", default="langops/data_eval/gt_v1_repaired.jsonl")
    parser.add_argument("--llm_model", default="llama-3.3-70b-versatile")
    parser.add_argument("--k", type=int, default=5, help="top-k to retrieve")
    parser.add_argument("--cpus", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--duckdb_dir", default="langops/runs_db")
    parser.add_argument("--experiment", default="rag_eval")
    args = parser.parse_args()

    num_workers = args.workers or args.cpus
    ray.init(num_cpus=args.cpus, ignore_reinit_error=True)

    # launch mlflow ui with: mlflow ui --backend-store-uri langops/mlruns
    mlflow.set_tracking_uri("file:langops/mlruns")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"{args.llm_model}-k{args.k}-w{num_workers}") as run:
        run_id = run.info.run_id

        mlflow.log_params({
            "llm_model": args.llm_model,
            "k": args.k,
            "cpus": args.cpus,
            "workers": num_workers,
            "dataset_path": args.data
        })

        dataset = load_jsonl(args.data)
        if not dataset:
            print("No data found. Check --data path.")
            ray.shutdown()
            return

        workers = [Evaluator.remote(llm_model=args.llm_model, k=args.k) for _ in range(num_workers)]
        futures = [workers[i % num_workers].run_eval.remote(example) for i, example in enumerate(dataset)]

        print(f"\nRunning {len(futures)} examples with {num_workers} worker(s) on {args.cpus} CPU(s)...")
        t0 = time.perf_counter()
        results = ray.get(futures)
        elapsed = time.perf_counter() - t0

        print(f"\n✅ Completed {len(results)} tasks in {elapsed:.2f}s "
              f"using {num_workers} worker(s) on {args.cpus} CPU(s)")

        examples_df, contexts_df = to_dataframes(results, run_id)
        db_path = os.path.join(args.duckdb_dir, f"{run_id}.duckdb")
        write_duckdb(db_path, examples_df, contexts_df)

        summary = summarize_for_mlflow(examples_df)
        mlflow.log_metrics(summary)
        mlflow.log_metric("wall_time_s", elapsed)

        mlflow.log_param("duckdb_path", db_path)
        mlflow.log_artifact(db_path)

        print(json.dumps(summary, indent=2))
        print(f"\n✅ Wrote {len(examples_df)} example rows and {len(contexts_df)} context rows "
              f"to {db_path}\n✅ Run ID: {run_id} | Wall time: {elapsed:.2f}s")

    ray.shutdown()


if __name__ == "__main__":
    main()

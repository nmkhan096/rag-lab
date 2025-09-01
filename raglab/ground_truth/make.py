import os, json, time, uuid, random
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

A_MODEL = "Qwen/Qwen2.5-72B-Instruct"        # generator
B_MODEL = "gpt-4o-mini"                      # validator

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"]
)

def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        # tiny repair heuristics
        s = s.strip().strip("`").strip()
        first = s.find("["); last = s.rfind("]")
        if first != -1 and last != -1:
            return json.loads(s[first:last+1])
        # try dict-style
        first = s.find("{"); last = s.rfind("}")
        if first != -1 and last != -1:
            return json.loads(s[first:last+1])
        raise

# ---------- Load corpus ----------
# expects [{"id":..., "text":..., "metadata":{"section":...}}, ...]
with open("data/resume_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

QGEN_PROMPT = """
You are an interviewer reading a resume for a Data Analyst, Data Scientist, ML Engineer, AI Engineer role.
Your goal is to write {n} clear and complete questions that can be answered using the provided entry. 
If the entry only includes company, job title, location, and dates, avoid asking technical or project-based questions.
If the entry lacks outcomes/metrics, do not ask about outcomes; ask about techniques, libraries, tasks, or scope instead.
Avoid copying exact phrases from the resume — the questions should sound natural and single focused (prefer one fact per question), but their answers should be well covered by the entry.

section: {section}
entry: {text}

Provide the output in a JSON array only without using code blocks:
["question1", "question2", ..., "question5"]
""".strip()

QVAL_PROMPT = """
You are validating a question against a resume entry. Use ONLY the entry text.

QUESTION:
{question}

ENTRY: {text}
section: {section}

Tasks:
1) Is the question fully answerable from the entry? (true/false)
2) Provide a concise, canonical gold answer (1 sentence) that is fully supported by the entry.
3) Rate faithfulness to the entry in [0,1].
4) Naturalness checks:
   - single_focus: true/false (asks about one thing)
   - overlap_ratio: 0–1 (lexical overlap based on bigram_overlap_ratio)

Return STRICT JSON with keys:
{{
  "answerable": true/false,
  "faithfulness": 0.0-1.0,
  "single_focus": true/false,
  "overlap_ratio": 0.0-1.0,
  "critique": "short explanation for any issues",
  "gold_answer": "..."
}}
No extra text.
"""

REPAIR_PROMPT = """\
You wrote this QUESTION for the ENTRY, but the validator flagged issues:

QUESTION: {question}
ENTRY: {text}
CRITIQUE: {critique}

Rewrite ONE question that is fully answerable from the entry, avoiding the issue, but it should NOT parrot phrases from it

Return a JSON array with exactly 1 string: ["..."]
"""

# ---------- Generate questions (Model A) ----------
def generate_questions(doc, n=3):
    prompt = QGEN_PROMPT.format(n=n, section=doc["metadata"]["section"], text=doc["text"])
    #out = chat(A_MODEL, user=prompt, temperature=0.7)

    response = nebius_client.chat.completions.create(
        model=A_MODEL, 
        temperature=0.7, # for diversity in questions so your GT covers more surface area.
        messages=[{"role": "user", "content": prompt}]
    )
    out = response.choices[0].message.content
    questions = safe_json_loads(out)
    questions = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
    return questions

# ---------- Validate and derive gold answer (Model B) ----------
def validate_question(doc, question):
    prompt = QVAL_PROMPT.format(
        question=question,
        section=doc["metadata"]["section"],
        #doc_id=doc["id"],
        text=doc["text"]
    )
    response = openai_client.chat.completions.create(
        model='gpt-4o-mini', 
        temperature=0.2, #deterministic, low-variance judgments for stable labels & reproducible GT.
        messages=[{"role": "user", "content": prompt}]
    )

    data = response.choices[0].message.content
    if isinstance(data, str):
        data = json.loads(data)
    # enforce schema defaults
    return {
        "answerable": bool(data.get("answerable", False)),
        "faithfulness": float(data.get("faithfulness", 0.0)),
        #"single_focus": bool(data.get("single_focus", False)),
        "overlap_ratio": float(data.get("overlap_ratio", 0.0)),
        "critique": (data.get("critique") or "").strip(),
        "gold_answer": (data.get("gold_answer") or "").strip()
    }

# ---------- One-shot repair (ask Model A to fix using Model B's critique) ----------
def repair_question(doc, question, critique):
    prompt = REPAIR_PROMPT.format(question=question, critique=critique, text=doc["text"])
    #out = chat(A_MODEL, user=prompt, temperature=0.5)
    response = nebius_client.chat.completions.create(
        model=A_MODEL, 
        temperature=0.5, # Slight creativity to fix issues, but less variance than initial generation so it converges.
        messages=[{"role": "user", "content": prompt}]
    )
    out = response.choices[0].message.content
    arr = safe_json_loads(out)
    if not arr or not isinstance(arr, list): return None
    return arr[0].strip()


if __name__ == "__main__":
    import argparse, random

    parser = argparse.ArgumentParser(description="Generate & validate synthetic GT from resume chunks.")
    parser.add_argument("--chunks", default="data/resume_chunks.json", help="Path to resume chunks JSON.")
    parser.add_argument("--version", default="gt_v1", help="Version tag to write into JSONL.")
    parser.add_argument("--n-per-chunk", type=int, default=2, help="Questions to generate per chunk (before filtering).")
    #parser.add_argument("--max-per-chunk", type=int, default=2, help="Max accepted questions per chunk.")
    #parser.add_argument("--accept-faithfulness", type=float, default=0.85, help="Min faithfulness to accept.")
    parser.add_argument("--repair", action="store_true", help="Attempt one repair if validation fails.")
    #parser.add_argument("--dedup-threshold", type=float, default=0.90, help="Cosine threshold for question dedup.")
    parser.add_argument("--out-dir", default="langops/data_eval", help="Output directory.")
    # parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(42) # args.seed

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    ACCEPT_FAITHFULNESS = 0.85 #args.accept_faithfulness
    MAX_PER_CHUNK = 2 #args.max_per_chunk
    VERSION = args.version

    accepted, rejected = [], []

    # ---------- Main loop ----------
    for doc in chunks[5:10]:
        q_candidates = generate_questions(doc, n=args.n_per_chunk)
        random.shuffle(q_candidates)
        kept = 0
        for q in q_candidates:
            if kept >= MAX_PER_CHUNK:
                break
            val = validate_question(doc, q)

            ok = (val["answerable"] and val["faithfulness"] >= ACCEPT_FAITHFULNESS and bool(val["gold_answer"]))
            if ok:
                kept += 1
                accepted.append({
                    "id": f"q_{uuid.uuid4().hex[:10]}",
                    "version": VERSION,
                    "section": doc["metadata"]["section"],
                    "doc_id": doc["id"],
                    # "gold_doc_ids": [doc["id"]],
                    # "gold_citations": [{"doc_id": doc["id"], "spans": val["citations"]}],
                    "generator_model": A_MODEL,
                    "validator_model": B_MODEL,
                    "prompt_ids": {"gen":"qgen_v1","val":"qval_v1"},
                    "faithfulness": val["faithfulness"],
                    "answerable": val["answerable"],
                    "overlap_ratio": val.get("overlap_ratio", 0.0),
                    "question": q,
                    "gold_answer": val["gold_answer"]
                })
                continue
            
            # only try repair if initial validation failed
            if args.repair:
                new_q = repair_question(doc, q, val["critique"])
                if new_q:
                    val2 = validate_question(doc, new_q)
                    ok2 = (val2["answerable"] and val2["faithfulness"] >= ACCEPT_FAITHFULNESS and bool(val2["gold_answer"]))
                    if ok2:
                        kept += 1
                        accepted.append({
                            "id": f"q_{uuid.uuid4().hex[:10]}",
                            "version": VERSION,
                            "section": doc["metadata"]["section"],
                            "doc_id": doc["id"],
                            # "gold_doc_ids": [doc["id"]],
                            # "gold_citations": [{"doc_id": doc["id"], "spans": val2["citations"]}],
                            "generator_model": A_MODEL,
                            "validator_model": B_MODEL,
                            "prompt_ids": {"gen":"qgen_v1","val":"qval_v1"},
                            "faithfulness": val2["faithfulness"],
                            "answerable": val2["answerable"],
                            "overlap_ratio": val2.get("overlap_ratio", 0.0),
                            "question": new_q,
                            "gold_answer": val2["gold_answer"]
                        })
                        continue # skip repair/reject
                    else:
                        # rejected repaired question
                        rejected.append({
                            "doc_id": doc["id"],
                            "section": doc["metadata"]["section"],
                            "val": val2,
                            "text": doc["text"],
                            "question": q,
                            "repaired_question": new_q
                        })
                        continue
            rejected.append({
                "doc_id": doc["id"],
                "section": doc["metadata"]["section"],
                "val": val,
                "text": doc["text"],
                "question": q,
            })

    # # Dedup (optional)
    # if args.dedup_threshold:
    #     accepted, dropped_idxs = dedup_by_similarity(accepted, text_key="question", threshold=args.dedup_threshold)
    #     print(f"[dedup] Dropped {len(dropped_idxs)} near-duplicates at ≥ {args.dedup_threshold}")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, f"{VERSION}.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as w:
        for ex in accepted:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    # Save rejected (JSONL)
    rejected_jsonl = os.path.join(args.out_dir, f"{VERSION}_rejected.jsonl")
    with open(rejected_jsonl, "w", encoding="utf-8") as w:
        for ex in rejected:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")

    #pd.DataFrame(rejected).to_csv(os.path.join(args.out_dir, "rejected_candidates.csv"), index=False)
    print(f"With repair {args.repair} : Accepted: {len(accepted)} | Rejected: {len(rejected)} | saved: {out_jsonl}")

# python langops/gt/gt_make.py --version gt_v1_repaired --repair

# python rag_pipeline/parser.py "data/Resume_new.docx" --sections "Intro" "Work Experience" "Projects" "Education" "Skills" --output "data/resume_chunks.json"

# source C:/projects/ask_my_resume/.venv/Scripts/activate
import os, json
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


FAITHFULNESS_PROMPT = """
You are an expert evaluator for RAG-generated answers.

Evaluate whether the ANSWER to the QUESTION is grounded in the CONTEXTS. Also provide a short explanation for any issues.

Return STRICT JSON with keys:
{{
  "faithful": true/false,
  "judge_comments": "short justification"
}}

QUESTION:
{question}

ANSWER:
{answer}

CONTEXTS:
{contexts}
"""

def judge_faithfulness(q, ctxs, ans):
    prompt = FAITHFULNESS_PROMPT.format(
        question=q, answer=ans, contexts=ctxs
    )
    response = openai_client.chat.completions.create(
        model='gpt-4o-mini', 
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content or "{}"
    #print(raw)
    #print("\n")
    # try:
    #     data = json.loads(raw)
    # except Exception:
    #     # Safe fallback
    #     data = {"faithful": -1, "judge_comments": "Could not parse model output."}
    if isinstance(raw, str):
        data = json.loads(raw)

    # Normalize & defaults
    return {
        "faithful": bool(data.get("faithful", False)),
        "judge_comments": (data.get("judge_comments") or "").strip()
    }
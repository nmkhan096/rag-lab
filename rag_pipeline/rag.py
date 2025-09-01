# Vector search + LLM answer

# vector store
from qdrant_client import models
from rag_pipeline.config import qd_client, collection_name, EMBEDDING_MODEL

def vector_search(question, sections=["Work Experience", "Projects"], limit=5):
    if isinstance(sections, str):
        sections = [sections]

    query_points = qd_client.query_points(
        collection_name=collection_name,
        query=models.Document(text=question, model=EMBEDDING_MODEL),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="section",
                    match=models.MatchAny(any=sections)
                )
            ]
        ),
        limit=limit,
        with_payload=True
    )

    out = []
    for p in query_points.points:
        out.append({
            "doc_id": str(p.id) if hasattr(p, "id") else p.payload.get("id"),
            "section": p.payload.get("section", "Unknown Section"),
            "text": p.payload.get("text", ""),
            "score": getattr(p, "score", None)
        })
    return out

# prompt builder
def build_prompt(question, search_results):
    prompt_template = """
You are a resume assistant. Use the CONTEXT below to answer the QUESTION based only on the candidate's resume.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}

If the answer is not found in the context, reply with "Not available in the resume."
""".strip()

    context = ""
    for doc in search_results:
        text = doc.get("text", "")
        section = doc.get("section", "Unknown Section")
        context += f"[{section}] {text}\n\n"

    return prompt_template.format(question=question, context=context).strip()

# llm client
from openai import OpenAI
from rag_pipeline.config import API_KEY, LLM_MODEL_DEFAULT

groq_client = OpenAI(api_key=API_KEY, 
                     base_url="https://api.groq.com/openai/v1")

def llm(prompt, llm_model=LLM_MODEL_DEFAULT):
    response = groq_client.chat.completions.create(
        model=llm_model,  #LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_with_contexts(question, contexts, llm_model=LLM_MODEL_DEFAULT):
    prompt = build_prompt(question, contexts)
    return llm(prompt, llm_model=llm_model)

def rag(query, sections=["Work Experience", "Projects", "Skills", "Education"], llm_model=LLM_MODEL_DEFAULT, limit=5):
    ctx = vector_search(query, sections=sections, limit=limit)      # retrieve
    ans = generate_with_contexts(query, ctx, llm_model=llm_model)   # generate
    return ans

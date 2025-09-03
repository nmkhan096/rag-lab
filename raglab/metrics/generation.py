import numpy as np

from sentence_transformers import SentenceTransformer

# Smaller and slightly more general-pupose than multi-qa-*
emb_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(ans, gold, doc_text=None):
    """
    Compute cosine similarity between answer and gold reference.
    If gold is empty, fallback to doc_text instead.
    """
    # Fallback logic
    target = gold if gold != "" else (doc_text or "")
    # Cosine similarity = dot product if both vectors are L2-normalized.
    # batch encode + (auto)normalize + dot product
    emb = emb_model.encode([ans, target], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def compute_all(ans, gold, doc_text):
    return {
        "semantic_sim": compute_similarity(ans, gold, doc_text)
    }
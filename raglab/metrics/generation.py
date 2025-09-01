import numpy as np

from sentence_transformers import SentenceTransformer

# Smaller and slightly more general-pupose than multi-qa-*
emb_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(ans, gold):
    # Many ST models support normalize_embeddings=True
    # Cosine similarity = dot product if both vectors are L2-normalized.
    # batch encode + (auto)normalize + dot product
    emb = emb_model.encode([ans, gold], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))  # equals cosine when normalized


def compute_all(ans, gold):
    return {
        "semantic_sim": compute_similarity(ans, gold)
    }
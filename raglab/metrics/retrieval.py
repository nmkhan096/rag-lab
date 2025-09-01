
def compute_hit(retrieved, orig, id_key="doc_id"):
    ids = [((d.get(id_key) or "").strip()) for d in (retrieved or [])]
    return int(orig in ids)

def compute_mrr(retrieved, orig, id_key="doc_id"):
    ids = [((d.get(id_key) or "").strip()) for d in (retrieved or [])]
    try:
        rank = ids.index(orig) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0

def compute_all(retrieved, orig, id_key="doc_id"):
    return {
        "hit": compute_hit(retrieved, orig, id_key),
        "mrr": compute_mrr(retrieved, orig, id_key)
    }
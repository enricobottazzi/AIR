import numpy as np
from sentence_transformers import SentenceTransformer

def embed(examples: list[str], model: SentenceTransformer):
    # Encode example strings (as returned by explain_acts / explain_logits).
    return model.encode(examples)

def gen_correlation_score(emb, w=None) -> float:
    """Mean (optionally weighted) pairwise cosine similarity over a feature's N examples.
    emb: (N, d) example embeddings.  w: (N,) per-example weights, or None for unweighted."""
    E = np.asarray(emb, float)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    i, j = np.triu_indices(len(E), k=1)
    weights = None if w is None else np.outer(w, w)[i, j]
    return float(np.average(S[i, j], weights=weights))

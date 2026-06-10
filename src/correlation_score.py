import numpy as np
from sentence_transformers import SentenceTransformer

def embed(examples: list[str], model: SentenceTransformer):
    # Encode example strings (as returned by explain_acts / explain_logits).
    return model.encode(examples)

def _unit(emb):
    E = np.asarray(emb, float)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

def _mean_pair_sim(E, w=None) -> float:
    # Mean (optionally weighted) pairwise cosine sim over unit-normalized rows of E.
    S = E @ E.T
    i, j = np.triu_indices(len(E), k=1)
    weights = None if w is None else np.outer(w, w)[i, j]
    return float(np.average(S[i, j], weights=weights))

def gen_correlation_score(emb, w=None) -> float:
    """Mean (optionally weighted) pairwise cosine similarity over a feature's N examples.
    emb: (N, d) example embeddings.  w: (N,) per-example weights, or None for unweighted."""
    return _mean_pair_sim(_unit(emb), w)

def gen_baseline(model, pool: list[str], n: int, trials: int = 200, seed: int = 0) -> tuple[float, float]:
    """Chance-level (mean, std) of the correlation score for `n` unrelated examples in this
    embedder. Compute once per (model, channel, n) and reuse across features.
    pool: large background corpus of unrelated strings (the population to sample from).
    n: subset size drawn per trial. Needs len(pool) >= n."""
    rng = np.random.default_rng(seed)
    P = embed(pool, model)
    scores = [gen_correlation_score(P[rng.choice(len(P), n, replace=False)]) for _ in range(trials)]
    return float(np.mean(scores)), float(np.std(scores))

def gen_normalized_correlation_score(emb, baseline: tuple[float, float], w=None) -> float:
    """Z-score of a feature vs a precomputed baseline (mu, sd). Higher = more coherent;
    ~0 = indistinguishable from random in this embedder/channel."""
    mu, sd = baseline
    return (gen_correlation_score(emb, w) - mu) / sd if sd else 0.0

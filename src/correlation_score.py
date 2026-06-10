import numpy as np
from sentence_transformers import SentenceTransformer

def _prefixed(examples: list[str], model: SentenceTransformer) -> list[str]:
    # E5-instruct expects "query: " on inputs; use it on all sides for symmetric similarity.
    name = str(getattr(model, "model_name", "")).lower()
    if "e5" in name and "instruct" in name:
        return [f"query: {x}" for x in examples]
    return examples

def embed(examples: list[str], model: SentenceTransformer):
    # Encode example strings (as returned by explain_acts / explain_logits).
    return model.encode(_prefixed(examples, model))

def _unit(emb):
    E = np.asarray(emb, float)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

def _mean_pair_sim(E, w=None) -> float:
    # Mean (optionally weighted) pairwise cosine sim over unit-normalized rows of E.
    S = E @ E.T
    i, j = np.triu_indices(len(E), k=1)
    weights = None if w is None else np.outer(w, w)[i, j]
    return float(np.average(S[i, j], weights=weights))

def gen_raw_scores(emb, pool_centroid, w=None) -> tuple[float, float]:
    """Returns (intra_correlation, inter_correlation)."""
    E = _unit(emb)
    return _mean_pair_sim(E, w), float(np.average(E @ pool_centroid, weights=w))

def gen_baseline(model, pool: list[str], n: int, trials: int = 200, seed: int = 0) -> tuple[float, float, float, float, list[float]]:
    """Chance-level (intra_mu, intra_sd, inter_mu, inter_sd, centroid) for `n` unrelated examples."""
    rng = np.random.default_rng(seed)
    P = _unit(embed(pool, model))
    centroid = _unit([P.mean(axis=0)])[0]
    scores = [gen_raw_scores(P[rng.choice(len(P), n, replace=False)], centroid) for _ in range(trials)]
    intra, inter = zip(*scores)
    return float(np.mean(intra)), float(np.std(intra)), float(np.mean(inter)), float(np.std(inter)), centroid.tolist()

def gen_normalized_coherence_score(emb, baseline: tuple[float, float, float, float, list[float]], w=None) -> float:
    """Difference of Z-scores for intra and inter correlations vs a precomputed baseline."""
    intra_mu, intra_sd, inter_mu, inter_sd, centroid = baseline
    intra, inter = gen_raw_scores(emb, np.array(centroid), w)
    return ((intra - intra_mu) / intra_sd if intra_sd else 0.0) - ((inter - inter_mu) / inter_sd if inter_sd else 0.0)

import numpy as np
from sentence_transformers import SentenceTransformer

# Prepended (symmetrically) for instruction-tuned embedders that ship no task
# prompt of their own. Phrased for symmetric similarity, not retrieval.
_INSTRUCTION = "Instruct: Retrieve semantically similar text.\nQuery: "

def _prompt(model: SentenceTransformer, model_id: str) -> str | None:
    # Follow each model's own convention, matched to our symmetric-similarity task.
    prompts = getattr(model, "prompts", None) or {}
    for name in ("STS", "PairClassification", "Clustering"):  # symmetric prompts, if shipped
        if prompts.get(name):
            return prompts[name]
    if "instruct" in model_id.lower() or "qwen3-embedding" in model_id.lower():
        return _INSTRUCTION
    return None  # models trained without prompts -> raw text

def embed(examples: list[str], model: SentenceTransformer, model_id: str = ""):
    # Encode example strings (as returned by explain_acts / explain_logits).
    return model.encode(examples, prompt=_prompt(model, model_id))

def _unit(emb):
    E = np.asarray(emb, float)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

def _mean_pair_sim(E, w) -> float:
    # Weighted mean pairwise cosine sim over unit-normalized rows of E.
    S = E @ E.T
    i, j = np.triu_indices(len(E), k=1)
    weights = np.outer(w, w)[i, j]
    return float(np.average(S[i, j], weights=weights))

def gen_raw_scores(emb, pool_centroid, w) -> tuple[float, float]:
    """Returns (intra_correlation, inter_correlation)."""
    E = _unit(emb)
    return _mean_pair_sim(E, w), float(np.average(E @ pool_centroid, weights=w))

def gen_baseline(model, pool: list[str], weights: list[float], n: int, trials: int = 200, seed: int = 0, model_id: str = "") -> tuple[float, float, float, float, list[float]]:
    """Chance-level (intra_mu, intra_sd, inter_mu, inter_sd, centroid) for `n` unrelated examples.
    `weights` must match the per-example weighting used by the observed score so the null is comparable."""
    rng = np.random.default_rng(seed)
    P = _unit(embed(pool, model, model_id))
    w = np.asarray(weights, float)
    centroid = _unit([np.average(P, axis=0, weights=w)])[0]
    def trial():
        idx = rng.choice(len(P), n, replace=False)
        return gen_raw_scores(P[idx], centroid, w[idx])
    intra, inter = zip(*(trial() for _ in range(trials)))
    return float(np.mean(intra)), float(np.std(intra)), float(np.mean(inter)), float(np.std(inter)), centroid.tolist()

def gen_normalized_correlation_score(emb, baseline: tuple[float, float, float, float, list[float]], w) -> float:
    """Difference of Z-scores for intra and inter correlations vs a precomputed baseline."""
    intra_mu, intra_sd, inter_mu, inter_sd, centroid = baseline
    intra, inter = gen_raw_scores(emb, np.array(centroid), w)
    return ((intra - intra_mu) / intra_sd if intra_sd else 0.0) - ((inter - inter_mu) / inter_sd if inter_sd else 0.0)

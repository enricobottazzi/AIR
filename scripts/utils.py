import asyncio
import copy
import csv
import json
import random
import urllib.request
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from delphi.clients import OpenRouter
from delphi.latents import ActivatingExample, Latent, LatentRecord, NonActivatingExample
from delphi.scorers import FuzzingScorer

from src.correlation_score import gen_baseline
from src.llm import ExplainerSetup, complete

def build_pool(experiment_dir: Path, channel_id: str) -> tuple[list[str], list[float]]:
    examples, weights = [], []
    for feature_path in experiment_dir.glob("*.json"):
        channel = json.loads(feature_path.read_text())["channels"][channel_id]
        examples.extend(channel["examples"])
        weights.extend(channel["weights"])
    return examples, weights

def get_baseline(experiment_dir: Path, embedder_model: SentenceTransformer, embedder_id: str, channel_id: str, pool: list[str], weights: list[float]):
    baseline_dir = experiment_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    out = baseline_dir / f"{channel_id}_{embedder_id.replace('/', '-')}_baseline.json"
        
    n = 10 if "logits" in channel_id else 20
    intra_mu, intra_sd, inter_mu, inter_sd, centroid = gen_baseline(embedder_model, pool, weights, n=n, trials=1000, model_id=embedder_id)
    
    out.write_text(json.dumps({
        "channel": channel_id, "embedder": embedder_id, "n": n,
        "pool_size": len(pool), "intra_mu": intra_mu, "intra_sd": intra_sd,
        "inter_mu": inter_mu, "inter_sd": inter_sd, "centroid": centroid
    }, indent=2))

def generate_explanations_neuronpedia(experiment_dir: Path, api_key: str, model_name: str, explanation_types: list[str]):
    feature_paths = list(experiment_dir.glob("*.json"))
    generated = 0
    print(f"Generating Neuronpedia explanations for {len(feature_paths)} features...")

    for feature_path in feature_paths:
        feat = json.loads(feature_path.read_text())
        explanations = feat.setdefault("explanations", [])
        existing_types = {e.get("typeName") for e in explanations}
        missing_types = [t for t in explanation_types if t not in existing_types]

        if not missing_types:
            print(f"{feature_path.name}: explanations already exist")
            continue

        print(f"{feature_path.name}: missing {', '.join(missing_types)}")

        for explanation_type in missing_types:
            print(f"  Generating {explanation_type} with {model_name}...")
            req = urllib.request.Request(
                "https://www.neuronpedia.org/api/explanation/generate",
                data=json.dumps({
                    "modelId": feat["modelId"],
                    "layer": feat["layer"],
                    "index": feat["index"],
                    "explanationType": explanation_type,
                    "explanationModelName": model_name
                }).encode(),
                headers={"x-api-key": api_key, "Content-Type": "application/json"}
            )
            explanation = json.load(urllib.request.urlopen(req))["explanation"]
            explanation["scores"] = []
            explanations.append(explanation)
            generated += 1

        if missing_types:
            feature_path.write_text(json.dumps(feat, indent=2))
            print(f"  Updated {feature_path.name}")

    print(f"Generated {generated} Neuronpedia explanations.")

def generate_explanations_air(experiment_dir: Path, api_key: str, model_name: str, channel_ids: list[str]):
    setup = ExplainerSetup(model=model_name)
    feature_paths = list(experiment_dir.glob("*.json"))
    generated = 0
    print(f"Generating AIR explanations for {len(feature_paths)} features...")

    for feature_path in feature_paths:
        feat = json.loads(feature_path.read_text())
        explanations = feat["explanations"]
        existing_types = {e.get("typeName") for e in explanations}
        todo = [c for c in channel_ids if f"air_{c}" not in existing_types]
        print(f"{feature_path.name}: {len(todo)} missing")

        for channel_id in todo:
            type_name = f"air_{channel_id}"
            explanations.append({
                "id": None,
                "description": complete(setup, feat["channels"][channel_id]["prompt"], api_key, feature_id=feature_path.stem, typeName=type_name).strip(),
                "explanationModelName": model_name,
                "typeName": type_name,
                "scores": [],
                "triggeredByUser": None
            })
            generated += 1

        if todo:
            feature_path.write_text(json.dumps(feat, indent=2))

    print(f"Generated {generated} AIR explanations.")

def _example_tensors(activation: dict, ctx_len: int):
    tokens, values = activation["tokens"], activation["values"]
    if len(tokens) > ctx_len:
        # Center a ctx_len window on the peak token, clamped to bounds.
        start = min(max(0, activation["maxValueTokenIndex"] - ctx_len // 2), len(tokens) - ctx_len)
        tokens, values = tokens[start:start + ctx_len], values[start:start + ctx_len]
    return (
        torch.zeros(len(values), dtype=torch.long),
        torch.tensor(values, dtype=torch.float32),
        tokens,
    )

def _activating_example(activation: dict, ctx_len: int) -> ActivatingExample:
    tokens, activations, str_tokens = _example_tensors(activation, ctx_len)
    return ActivatingExample(
        tokens=tokens,
        activations=activations,
        str_tokens=str_tokens,
    )

def _non_activating_example(activation: dict, ctx_len: int) -> NonActivatingExample:
    tokens, activations, str_tokens = _example_tensors(activation, ctx_len)
    return NonActivatingExample(
        tokens=tokens,
        activations=activations,
        str_tokens=str_tokens,
        distance=-1.0,
    )

def build_delphi_record(
    experiment_dir: Path,
    target_feature_path: Path,
    seed: int = 42,
    n_positive: int = 10,
    n_negative: int = 10,
    example_ctx_len: int = 32,
) -> LatentRecord:
    experiment_dir = Path(experiment_dir)
    target_feature_path = Path(target_feature_path).resolve()
    feature_paths = sorted(path.resolve() for path in experiment_dir.glob("*.json"))
    
    target_feature = json.loads(target_feature_path.read_text())
    negative_features = [
        json.loads(path.read_text()) for path in feature_paths if path != target_feature_path
    ]

    positive_pool = []
    for activation in target_feature["activations"]:
        positive_pool.append(_activating_example(activation, example_ctx_len))

    negative_pool = []
    for feature in negative_features:
        for activation in feature["activations"]:
            negative_pool.append(_non_activating_example(activation, example_ctx_len))

    if len(positive_pool) < n_positive:
        raise ValueError(f"Need {n_positive} positive activations, found {len(positive_pool)}")
    if len(negative_pool) < n_negative:
        raise ValueError(f"Need {n_negative} negative activations, found {len(negative_pool)}")

    rng = random.Random(seed)
    record = LatentRecord(
        latent=Latent(target_feature["layer"], int(target_feature["index"]))
    )
    record.test = rng.sample(positive_pool, n_positive)
    record.not_active = rng.sample(negative_pool, n_negative)
    return record

async def _run_delphi_fuzz(
    delphi_record: LatentRecord,
    openrouter_api_key: str,
    model_name: str,
    n_examples_shown: int = 5,
    **trace
):
    client = OpenRouter(model_name, api_key=openrouter_api_key)
    if trace:
        orig = client.client.post
        client.client.post = lambda *a, **kw: orig(*a, **{**kw, "json": {**kw.get("json", {}), "trace": trace}})
    try:
        return await FuzzingScorer(
            client,
            fuzz_type="default",
            n_examples_shown=n_examples_shown,
        )(delphi_record)
    finally:
        await client.client.aclose()

def _accuracy(fuzz) -> float:
    correct = [score.correct for score in fuzz.score if score.correct is not None]
    if not correct:
        raise ValueError("Delphi fuzz scorer returned no valid predictions")
    return sum(correct) / len(correct)

def delphi_fuzz_scorer(
    delphi_record: LatentRecord,
    explanation: dict,
    openrouter_api_key: str,
    model_name: str,
    n_examples_shown: int = 5,
    **trace
) -> float:
    record = copy.copy(delphi_record)
    record.explanation = explanation["description"]
    fuzz = asyncio.run(
        _run_delphi_fuzz(
            record,
            openrouter_api_key,
            model_name,
            n_examples_shown=n_examples_shown,
            **trace
        )
    )
    return _accuracy(fuzz)

def write_explanations_matrix_csv(experiment_dir: Path, score_type_name: str = "delphi_fuzz") -> Path:
    """Rows: feature IDs. Columns: each explanation typeName with two subfields (description, score)."""
    features = [(p.stem, json.loads(p.read_text())) for p in sorted(experiment_dir.glob("*.json"))]
    type_names = list(dict.fromkeys(e["typeName"] for _, f in features for e in f.get("explanations", [])))

    def cell(explanation: dict) -> list:
        score = next((s["value"] for s in explanation["scores"] if s["explanationScoreTypeName"] == score_type_name), None)
        return [explanation["description"], score]

    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "explanations_matrix.csv"
    with out.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["feature"] + [v for t in type_names for v in (t, "")])
        w.writerow([""] + ["description", "score"] * len(type_names))
        for feature_id, feat in features:
            by_type = {e["typeName"]: e for e in feat.get("explanations", [])}
            row = [feature_id]
            for t in type_names:
                row += cell(by_type[t]) if t in by_type else ["", ""]
            w.writerow(row)
    return out

def write_correlation_matrix_csv(experiment_dir: Path) -> Path:
    """Rows: embedder IDs. Columns: feature ID (top) x channel (sub). Cells: correlation score; per feature+embedder max is bolded with a star."""
    features = [(p.stem, json.loads(p.read_text())) for p in sorted(experiment_dir.glob("*.json"))]
    channels = list(dict.fromkeys(c for _, f in features for c in f.get("channels", {})))
    embedders = list(dict.fromkeys(e for _, f in features for ch in f.get("channels", {}).values() for e in ch.get("scores", {})))

    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "correlation_matrix.csv"
    with out.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["embedder"] + [feature_id if i == 0 else "" for feature_id, _ in features for i in range(len(channels))])
        w.writerow([""] + channels * len(features))
        for embedder in embedders:
            row = [embedder]
            for _, feat in features:
                cells = [feat.get("channels", {}).get(c, {}).get("scores", {}).get(embedder) for c in channels]
                best = max((s for s in cells if s is not None), default=None)
                row += [f"**{s}** *" if s is not None and s == best else ("" if s is None else s) for s in cells]
            w.writerow(row)
    return out

def write_feature_score_matrix_csv(experiment_dir: Path, neuronpedia_explanation_types: list[str], embedder_ids: list[str], score_type_name: str = "delphi_fuzz", air_type_prefix: str = "air", out_name: str = "feature_score_matrix.csv", filtered: bool = False, min_correlation: float = 5.0) -> Path:
    """Rows: feature IDs. Columns: neuronpedia explanation types + embedder IDs. Cells: delphi_fuzz score (for embedders, of the {air_type_prefix} explanation whose channel that embedder correlates with most). If filtered, embedder cells whose best channel correlates < min_correlation or is a (positive|negative)_logits channel are left empty."""
    features = [(p.stem, json.loads(p.read_text())) for p in sorted(experiment_dir.glob("*.json"))]

    def delphi_score(feat: dict, type_name: str):
        e = next((e for e in feat.get("explanations", []) if e["typeName"] == type_name), None)
        return next((s["value"] for s in e["scores"] if s["explanationScoreTypeName"] == score_type_name), "") if e else ""

    def best_air_type(feat: dict, embedder: str):
        scored = {c: ch["scores"][embedder] for c, ch in feat.get("channels", {}).items() if embedder in ch["scores"]}
        if not scored:
            return None
        best = max(scored, key=scored.get)
        if filtered and (scored[best] < min_correlation or best.endswith(("positive_logits", "negative_logits"))):
            return None
        return f"{air_type_prefix}_{best}"

    columns = neuronpedia_explanation_types + embedder_ids
    rows = []
    for feature_id, feat in features:
        values = [delphi_score(feat, t) for t in neuronpedia_explanation_types]
        values += [delphi_score(feat, best_air_type(feat, emb)) for emb in embedder_ids]
        rows.append((feature_id, values))

    def average(j: int):
        col = [v[j] for _, v in rows if v[j] != ""]
        return sum(col) / len(col) if col else ""

    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / out_name
    with out.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["feature"] + columns)
        for feature_id, values in rows:
            w.writerow([feature_id] + values)
        w.writerow(["average"] + [average(j) for j in range(len(columns))])
    return out

def plot_feature_score_matrix(csv_path: Path, center: float = 0.5) -> Path:
    """Diverging bar chart of the `average` row, centered at `center`:
    models with average < center bar downward (negative), >= center bar upward."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.reader(csv_path.read_text().splitlines()))
    header, average = rows[0][1:], rows[-1][1:]
    labels, deltas = zip(*[(c, float(v) - center) for c, v in zip(header, average) if v != ""])

    fig, ax = plt.subplots(figsize=(0.6 * len(labels) + 2, 6))
    ax.bar(range(len(labels)), deltas, color=["#2a9d8f" if d >= 0 else "#e76f51" for d in deltas])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([f"{t + center:.2f}" for t in ax.get_yticks()])
    ax.set_ylabel("average delphi_fuzz score")
    ax.set_title(csv_path.stem)
    fig.tight_layout()
    out = csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


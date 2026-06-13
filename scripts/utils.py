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
        
    n = 10
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
            try:
                explanation = json.load(urllib.request.urlopen(req))["explanation"]
                explanation["scores"] = []
                explanations.append(explanation)
                generated += 1
            except urllib.error.HTTPError as e:
                print(f"  Failed to generate {explanation_type}: HTTP {e.code}")
                # We'll just continue to the next explanation type/feature
                continue

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
                "description": complete(setup, feat["channels"][channel_id]["prompt"], api_key, step="explain", feature_id=feature_path.stem, typeName=type_name).strip(),
                "explanationModelName": model_name,
                "typeName": type_name,
                "scores": [],
                "triggeredByUser": None
            })
            generated += 1

        if todo:
            feature_path.write_text(json.dumps(feat, indent=2))

    print(f"Generated {generated} AIR explanations.")

def data_sanity(experiment_dir: Path, expected_types: list[str], require_scores: bool = False):
    """Assert every feature has a non-empty explanation description for each expected typeName."""
    for feature_path in sorted(experiment_dir.glob("*.json")):
        explanations = {e.get("typeName"): e for e in json.loads(feature_path.read_text()).get("explanations", [])}
        missing = [t for t in expected_types if not explanations.get(t, {}).get("description")]
        if missing:
            raise ValueError(f"{feature_path.name}: missing explanations for {missing}")
        if require_scores:
            missing_scores = [t for t in expected_types if not explanations.get(t, {}).get("scores")]
            if missing_scores:
                raise ValueError(f"{feature_path.name}: missing scores for {missing_scores}")
    print(f"Data sanity passed: all features have {len(expected_types)} explanation types" + (" and scores." if require_scores else "."))

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

def _accuracy(fuzz) -> tuple[float, int, int]:
    """Return (accuracy, answered, total). Discarded = total - answered (None predictions from parse/generation failures)."""
    answered = [score.correct for score in fuzz.score if score.correct is not None]
    if not answered:
        raise ValueError("Delphi fuzz scorer returned no valid predictions")
    return sum(answered) / len(answered), len(answered), len(fuzz.score)

def delphi_fuzz_scorer(
    delphi_record: LatentRecord,
    explanation: dict,
    openrouter_api_key: str,
    model_name: str,
    n_examples_shown: int = 5,
    **trace
) -> dict:
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
    accuracy, answered, total = _accuracy(fuzz)
    return {"value": accuracy, "answered": answered, "discarded": total - answered, "total": total}

def gen_feature_correlation_scores_csv(feature_path: Path, results_dir: Path):
    """Write this feature's channel (rows) x embedder (cols) correlation scores to results_dir/correlation_scores_by_feature; mark each embedder's top channel with ' *'."""
    channels = json.loads(feature_path.read_text())["channels"]
    embedders = list(next(iter(channels.values()))["scores"])
    best = {e: max(d["scores"][e] for d in channels.values()) for e in embedders}
    out_dir = results_dir / "correlation_scores_by_feature"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{feature_path.stem}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", *embedders])
        for c, data in channels.items():
            w.writerow([c, *(f"{s} *" if (s := data["scores"][e]) == best[e] else s for e in embedders)])

def gen_feature_accuracy_scores_csv(feature_path: Path, results_dir: Path):
    """Write this feature's typeName (rows) x [explanation, score] to results_dir/accuracy_scores_by_feature."""
    explanations = json.loads(feature_path.read_text())["explanations"]
    out_dir = results_dir / "accuracy_scores_by_feature"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{feature_path.stem}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["typeName", "score", "explanation"])
        for e in explanations:
            w.writerow([e["typeName"], e["scores"][0]["value"], e["description"]])

def generate_protocol_json(experiment_dir: Path, results_dir: Path):
    """Write results_dir/air.json"""
    results_dir.mkdir(parents=True, exist_ok=True)
    feats = [(p, json.loads(p.read_text())) for p in sorted(experiment_dir.glob("*.json"))]
    embedders = list(next(iter(feats[0][1]["channels"].values()))["scores"])

    def best_air(feat: dict, embedder: str) -> dict:
        channels = feat["channels"]
        channel = max(channels, key=lambda c: channels[c]["scores"][embedder])
        correlation_score = channels[channel]["scores"][embedder]
        def acc(type_name):
            return next(x for x in feat["explanations"] if x["typeName"] == type_name)["scores"][0]["value"]
        return {
            "channel_id": channel,
            "correlation_score": correlation_score,
            "filtered": channel in ("positive_logits", "negative_logits") or correlation_score < 5,
            "air_score": acc(f"air_{channel}"),
            "postprocessed_air_score": acc(f"postprocessed_air_{channel}"),
        }

    air = {e: [{"feature_id": p.stem, **best_air(feat, e)} for p, feat in feats] for e in embedders}
    (results_dir / "air.json").write_text(json.dumps(air, indent=2))

def gen_accuracy_score_by_protocol_csv(experiment_dir: Path, results_dir: Path, neuronpedia_types: list[str], embedder_ids: list[str]):
    """Write results_dir/accuracy_score_by_protocol.csv: feature (rows) x protocol (cols)"""
    results_dir.mkdir(parents=True, exist_ok=True)
    prefixes = ("air", "air_postprocessed", "air_filtered", "air_postprocessed_filtered")
    prefix_key = {
        "air": "air_score", "air_postprocessed": "postprocessed_air_score",
        "air_filtered": "air_score", "air_postprocessed_filtered": "postprocessed_air_score",
    }
    filtered_prefixes = {"air_filtered", "air_postprocessed_filtered"}
    air = json.loads((results_dir / "air.json").read_text())
    lookup = {e: {x["feature_id"]: x for x in entries} for e, entries in air.items()}
    with (results_dir / "accuracy_score_by_protocol.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", *neuronpedia_types, *(f"{prefix}_{e}" for prefix in prefixes for e in embedder_ids)])
        rows = []
        for p in sorted(experiment_dir.glob("*.json")):
            by_type = {e["typeName"]: e for e in json.loads(p.read_text())["explanations"]}
            np_scores = [by_type[t]["scores"][0]["value"] for t in neuronpedia_types]
            air_scores = [
                "" if (prefix in filtered_prefixes and lookup[e][p.stem]["filtered"]) else lookup[e][p.stem][prefix_key[prefix]]
                for prefix in prefixes for e in embedder_ids
            ]
            rows.append([p.stem, *np_scores, *air_scores])
        w.writerows(rows)
        cols = zip(*(r[1:] for r in rows))
        w.writerow(["average", *(sum(v) / len(v) if (v := [x for x in c if x != ""]) else "" for c in cols)])

def tabulate_accuracy_score_by_protocol(csv_path: Path):
    """Write a markdown table per air family (+ neuronpedia baselines) of the average accuracy per protocol."""
    rows = list(csv.reader(csv_path.read_text().splitlines()))
    protocols, averages = rows[0][1:], rows[-1][1:]
    value = dict(zip(protocols, averages))

    def family_of(col: str):
        return next((fam for fam in ("air_postprocessed_filtered", "air_postprocessed", "air_filtered", "air") if col.startswith(fam + "_")), None)

    np_cols = [p for p in protocols if family_of(p) is None]
    best_np = max(float(value[c]) for c in np_cols)
    out_dir = csv_path.parent / "accuracy_score_by_protocol_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fam in ("air", "air_postprocessed", "air_filtered", "air_postprocessed_filtered"):
        cols = [c for c in [*np_cols, *(p for p in protocols if family_of(p) == fam)] if value[c] != ""]
        body = [f"| {c}{' *' if family_of(c) and float(value[c]) >= best_np else ''} | {float(value[c]):.3f} |" for c in cols]
        lines = ["| protocol | accuracy |", "| --- | --- |", *body]
        (out_dir / f"accuracy_score_by_protocol_{fam}.md").write_text("\n".join(lines) + "\n")

# Map each best-channel to a feature type. `short_window` (-1,1) is treated as
# `abstract` (it is the smallest context window). Reassign to fold differently.
CHANNEL_CATEGORY = {
    "act_token": "input", "before_act_token": "input",
    "after_act_token": "output", "positive_logits": "output", "negative_logits": "output",
    "medium_window": "abstract", "long_window": "abstract", "short_window": "abstract",
}
FEATURE_CATEGORIES = ["input", "output", "abstract", "obscure"]
CATEGORY_COLORS = {"input": "tab:blue", "output": "tab:orange", "abstract": "tab:green", "obscure": "tab:gray"}
OBSCURE_THRESHOLD = 5.0  # best correlation score below this => obscure

_feature_layer = lambda rec: int(rec["feature_id"].split("_")[-2])
_feature_category = lambda rec: "obscure" if rec["correlation_score"] < OBSCURE_THRESHOLD else CHANNEL_CATEGORY[rec["channel_id"]]

def _binned_counts(records: list, group_size: int):
    """Return (groups, {category: [count per group]}) for one embedder, binning layers by layer // group_size."""
    groups = sorted({_feature_layer(r) // group_size for r in records})
    idx = {g: i for i, g in enumerate(groups)}
    counts = {c: [0] * len(groups) for c in FEATURE_CATEGORIES}
    for r in records:
        counts[_feature_category(r)][idx[_feature_layer(r) // group_size]] += 1
    return groups, counts

def _to_fractions(counts: dict, n: int):
    """Column-normalize counts so each layer group sums to 1 (empty groups => 0)."""
    totals = [sum(counts[c][i] for c in FEATURE_CATEGORIES) or 1 for i in range(n)]
    return {c: [counts[c][i] / totals[i] for i in range(n)] for c in FEATURE_CATEGORIES}

def _group_centers(groups, gs): return [g * gs + (gs - 1) / 2 for g in groups]

def _new_ax(figsize=(10, 5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, *plt.subplots(figsize=figsize)

def plot_feature_type_area(results_dir: Path, group_size: int = 2):
    """100%-stacked area per embedder: feature-type composition vs layer depth."""
    air = json.loads((results_dir / "air.json").read_text())
    out_dir = results_dir / "feature_type_area"
    out_dir.mkdir(parents=True, exist_ok=True)
    for embedder, records in air.items():
        groups, counts = _binned_counts(records, group_size)
        fr, x = _to_fractions(counts, len(groups)), _group_centers(groups, group_size)
        plt, fig, ax = _new_ax()
        ax.stackplot(x, [fr[c] for c in FEATURE_CATEGORIES], labels=FEATURE_CATEGORIES, colors=[CATEGORY_COLORS[c] for c in FEATURE_CATEGORIES])
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(0, 1)
        ax.set_xlabel("layer")
        ax.set_ylabel("fraction of features")
        ax.set_title(f"feature type composition vs layer — {embedder}")
        ax.legend(loc="upper center", ncol=len(FEATURE_CATEGORIES), fontsize=8)
        fig.savefig(out_dir / f"{embedder.replace('/', '-')}.png", bbox_inches="tight")
        plt.close(fig)

def plot_best_protocol_summary(csv_path: Path):
    """PNG bar chart: neuronpedia protocols + each air family's best score across embedders, annotated with the chosen embedder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.reader(csv_path.read_text().splitlines()))
    protocols, averages = rows[0][1:], rows[-1][1:]
    value = {p: float(a) for p, a in zip(protocols, averages) if a != ""}

    def family_of(col: str):
        return next((fam for fam in ("air_postprocessed_filtered", "air_postprocessed", "air_filtered", "air") if col.startswith(fam + "_")), None)

    np_cols = [p for p in protocols if family_of(p) is None]
    families = ("air", "air_postprocessed", "air_filtered", "air_postprocessed_filtered")
    best = {fam: max((p for p in protocols if family_of(p) == fam and p in value), key=value.get) for fam in families}
    labels = [*np_cols, *families]
    scores = [*(value[c] for c in np_cols), *(value[best[fam]] for fam in families)]
    colors = [*["tab:orange"] * len(np_cols), *["tab:blue"] * len(families)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(labels)), scores, color=colors)
    for i, fam in enumerate(families):
        ax.text(len(np_cols) + i, 0.51, best[fam].removeprefix(fam + "_"), rotation=90, va="bottom", ha="center", color="white", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("best protocol per family")
    fig.savefig(csv_path.parent / "accuracy_score_by_protocol_best.png", bbox_inches="tight")
    plt.close(fig)

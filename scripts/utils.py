import asyncio
import copy
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

def build_pool(experiment_dir: Path, channel_id: str) -> list[str]:
    pool = []
    for feature_path in experiment_dir.glob("*.json"):
        feat = json.loads(feature_path.read_text())
        pool.extend(feat["channels"][channel_id]["examples"])
    return pool

def get_baseline(experiment_dir: Path, embedder_model: SentenceTransformer, embedder_id: str, channel_id: str, pool: list[str]):
    baseline_dir = experiment_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    out = baseline_dir / f"{channel_id}_{embedder_id.replace('/', '-')}_baseline.json"
        
    n = 10 if "logits" in channel_id else 20
    intra_mu, intra_sd, inter_mu, inter_sd, centroid = gen_baseline(embedder_model, pool, n=n, trials=1000)
    
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
            explanations.append({
                "id": None,
                "description": complete(setup, feat["channels"][channel_id]["prompt"], api_key).strip(),
                "explanationModelName": model_name,
                "typeName": f"air_{channel_id}",
                "scores": [],
                "triggeredByUser": None
            })
            generated += 1

        if todo:
            feature_path.write_text(json.dumps(feat, indent=2))

    print(f"Generated {generated} AIR explanations.")

def _example_tensors(activation: dict):
    values = activation["values"]
    return (
        torch.zeros(len(values), dtype=torch.long),
        torch.tensor(values, dtype=torch.float32),
        activation["tokens"],
    )

def _activating_example(activation: dict) -> ActivatingExample:
    tokens, activations, str_tokens = _example_tensors(activation)
    return ActivatingExample(
        tokens=tokens,
        activations=activations,
        str_tokens=str_tokens,
    )

def _non_activating_example(activation: dict) -> NonActivatingExample:
    tokens, activations, str_tokens = _example_tensors(activation)
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
        positive_pool.append(_activating_example(activation))

    negative_pool = []
    for feature in negative_features:
        for activation in feature["activations"]:
            negative_pool.append(_non_activating_example(activation))

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
    n_examples_shown: int = 4,
):
    client = OpenRouter(model_name, api_key=openrouter_api_key)
    try:
        return await FuzzingScorer(
            client,
            fuzz_type="default",
            n_examples_shown=n_examples_shown,
        )(delphi_record)
    finally:
        await client.client.aclose()

def _recall(fuzz) -> float:
    correct = [score.correct for score in fuzz.score if score.correct is not None]
    if not correct:
        raise ValueError("Delphi fuzz scorer returned no valid predictions")
    return sum(correct) / len(correct)

def delphi_fuzz_scorer(
    delphi_record: LatentRecord,
    explanation: dict,
    openrouter_api_key: str,
    model_name: str,
    n_examples_shown: int = 4,
) -> float:
    record = copy.copy(delphi_record)
    record.explanation = explanation["description"]
    fuzz = asyncio.run(
        _run_delphi_fuzz(
            record,
            openrouter_api_key,
            model_name,
            n_examples_shown=n_examples_shown,
        )
    )
    return _recall(fuzz)


import json
import urllib.request
from pathlib import Path

from src.correlation_score import gen_baseline
from src.llm import ExplainerSetup, complete

def build_pool(experiment_dir: Path, channel_id: str) -> list[str]:
    pool = []
    for feature_path in experiment_dir.glob("*.json"):
        feat = json.loads(feature_path.read_text())
        pool.extend(feat["channels"][channel_id]["examples"])
    return pool

def get_baseline(experiment_dir: Path, embedder_model, embedder_id: str, channel_id: str, pool: list[str]):
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

def generate_explanations_chair(experiment_dir: Path, api_key: str, model_name: str, channel_ids: list[str]):
    setup = ExplainerSetup(model=model_name)
    feature_paths = list(experiment_dir.glob("*.json"))
    generated = 0
    print(f"Generating CHAIR explanations for {len(feature_paths)} features...")

    for feature_path in feature_paths:
        feat = json.loads(feature_path.read_text())
        explanations = feat["explanations"]
        existing_types = {e.get("typeName") for e in explanations}
        todo = [c for c in channel_ids if f"chair_{c}" not in existing_types]
        print(f"{feature_path.name}: {len(todo)} missing")

        for channel_id in todo:
            explanations.append({
                "id": None,
                "description": complete(setup, feat["channels"][channel_id]["prompt"], api_key).strip(),
                "explanationModelName": model_name,
                "typeName": f"chair_{channel_id}",
                "scores": [],
                "triggeredByUser": None
            })
            generated += 1

        if todo:
            feature_path.write_text(json.dumps(feat, indent=2))

    print(f"Generated {generated} CHAIR explanations.")

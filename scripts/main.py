import argparse
import json
import os
import random
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.explainer import preprocess_acts, preprocess_logits
from src.correlation_score import gen_normalized_correlation_score, embed
from utils import build_delphi_record, build_pool, data_sanity, delphi_fuzz_scorer, gen_accuracy_score_by_protocol_csv, gen_feature_correlation_scores_csv, gen_feature_accuracy_scores_csv, generate_explanations_air, generate_explanations_neuronpedia, generate_protocol_json, get_baseline, plot_best_protocol_summary, tabulate_accuracy_score_by_protocol
from sentence_transformers import SentenceTransformer

def sample_features(experiment_dir: Path, n: int, min_acts: int, api_key: str, model_id: str):
    saved = sum(1 for _ in experiment_dir.glob("*.json"))
    while saved < n:
        layer, index = random.randint(0, 61), random.randint(0, 262143)
        out = experiment_dir / f"{model_id}_{layer}_{index}.json"
        if out.exists():
            continue
        req = urllib.request.Request(f"https://www.neuronpedia.org/api/feature/{model_id}/{layer}-gemmascope-2-transcoder-262k/{index}", headers={"x-api-key": api_key})
        feat = json.load(urllib.request.urlopen(req))
        
        # check if the features has at least `min_acts` activations with non-zero maxValue
        activations = feat.get("activations", [])
        valid_count = sum(1 for a in activations if a.get("maxValue", 0) > 0)
        
        if valid_count >= min_acts:
            out.write_text(json.dumps(feat, indent=2))
            saved += 1
            print(f"Saved {out.name}")

def preprocess_features(experiment_dir: Path, channel_specs: list):
    for feature_path in experiment_dir.glob("*.json"):
        feat = json.loads(feature_path.read_text())
            
        feat["channels"] = {}
        for channel_id, recipe, _ in channel_specs:
            prompt, examples, weights = recipe(feat)
            feat["channels"][channel_id] = {
                "prompt": prompt,
                "examples": examples,
                "weights": weights,
                "scores": {}
            }
            
        feature_path.write_text(json.dumps(feat, indent=2))

def preprocess_embedders(experiment_dir: Path, embedder_ids: list, channel_ids: list):
    for embedder_id in embedder_ids:
        embedder_model = SentenceTransformer(embedder_id)
        for channel_id in channel_ids:
            pool, weights = build_pool(experiment_dir, channel_id)
            get_baseline(experiment_dir, embedder_model, embedder_id, channel_id, pool, weights)

def generate_correlation_scores(experiment_dir: Path, embedder_ids: list, channel_ids: list):
    for embedder_id in embedder_ids:
        embedder_model = SentenceTransformer(embedder_id)
        for feature_path in experiment_dir.glob("*.json"):
            feat = json.loads(feature_path.read_text())
            for channel_id in channel_ids:
                channel_data = feat["channels"][channel_id]
                if embedder_id in channel_data["scores"]:
                    continue

                baseline_path = experiment_dir / "baselines" / f"{channel_id}_{embedder_id.replace('/', '-')}_baseline.json"
                baseline_data = json.loads(baseline_path.read_text())
                baseline = (baseline_data["intra_mu"], baseline_data["intra_sd"], 
                            baseline_data["inter_mu"], baseline_data["inter_sd"], 
                            baseline_data["centroid"])
                
                score = gen_normalized_correlation_score(
                    embed(channel_data["examples"], embedder_model, embedder_id), 
                    baseline, 
                    w=channel_data["weights"]
                )
                
                feat["channels"][channel_id]["scores"][embedder_id] = score
                
            feature_path.write_text(json.dumps(feat, indent=2))
        print(f"Generated correlation scores for embedder {embedder_id}")

def generate_explanations(
    experiment_dir: Path,
    neuronpedia_api_key: str,
    openrouter_api_key: str,
    model_name: str,
    neuronpedia_explanation_types: list[str],
    channel_ids: list[str]
):
    generate_explanations_air(experiment_dir, openrouter_api_key, f"google/{model_name}", channel_ids)
    generate_explanations_neuronpedia(experiment_dir, neuronpedia_api_key, model_name, neuronpedia_explanation_types)

def postprocess_explanations(experiment_dir: Path, channel_specs: list):
    air_description_prefixes = {
        channel_id: air_prefix
        for channel_id, _, air_prefix in channel_specs
    }

    for feature_path in sorted(experiment_dir.glob("*.json")):
        feat = json.loads(feature_path.read_text())
        explanations = feat.get("explanations", [])
        existing_types = {e.get("typeName") for e in explanations}
        new_explanations = []

        for explanation in explanations:
            type_name = explanation.get("typeName", "")
            if not type_name.startswith("air_"):
                continue

            channel_id = type_name.removeprefix("air_")
            prefix = air_description_prefixes.get(channel_id)
            postprocessed_type = f"postprocessed_{type_name}"
            if postprocessed_type in existing_types:
                continue

            new_explanations.append({
                **explanation,
                "id": None,
                "typeName": postprocessed_type,
                "description": f'{prefix} "{explanation["description"]}"',
                "scores": [],
            })

        if new_explanations:
            explanations.extend(new_explanations)
            feature_path.write_text(json.dumps(feat, indent=2))

def score_explanations(
    experiment_dir: Path,
    openrouter_api_key: str,
    model_name: str,
):
    score_type_name = "delphi_fuzz"
    score_model_name = f"google/{model_name}"
    feature_paths = sorted(experiment_dir.glob("*.json"))
    for feature_path in feature_paths:
        feat = json.loads(feature_path.read_text())
        delphi_record = build_delphi_record(
            experiment_dir,
            feature_path,
        )
        for explanation in feat.get("explanations", []):
            if any(
                s["explanationScoreTypeName"] == score_type_name
                and s["explanationScoreModelName"] == score_model_name
                for s in explanation["scores"]
            ):
                continue
            score = delphi_fuzz_scorer(
                delphi_record,
                explanation,
                openrouter_api_key,
                score_model_name,
                step="score",
                feature_id=feature_path.stem,
                typeName=explanation.get("typeName")
            )
            explanation["scores"].append({
                "value": score,
                "explanationScoreTypeName": score_type_name,
                "explanationScoreModelName": score_model_name
            })
        feature_path.write_text(json.dumps(feat, indent=2))
        print(f"Scored all explanations for feature {feature_path.stem}")

def aggregate_data(experiment_dir: Path, neuronpedia_types: list[str], embedders: list[str]):
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for feature_path in sorted(experiment_dir.glob("*.json")):
        gen_feature_correlation_scores_csv(feature_path, results_dir)
        gen_feature_accuracy_scores_csv(feature_path, results_dir)
    generate_protocol_json(experiment_dir, results_dir)
    gen_accuracy_score_by_protocol_csv(experiment_dir, results_dir, neuronpedia_types, embedders)
    tabulate_accuracy_score_by_protocol(results_dir / "accuracy_score_by_protocol.csv")
    plot_best_protocol_summary(results_dir / "accuracy_score_by_protocol.csv")

def main():
    parser = argparse.ArgumentParser(description="Unified pipeline for feature fetching, explanation generation, and scoring.")
    parser.add_argument("experiment_dir", type=Path, help="Directory for the experiment")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 0. Define hyperparameters
    MODEL_ID = "gemma-3-27b-it"
    EMBEDDERS = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "BAAI/bge-small-en-v1.5",
        "Qwen/Qwen3-Embedding-0.6B",
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-large-instruct",
        "google/embeddinggemma-300m",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/LaBSE",
    ]
    
    CHANNEL_SPECS = [
        ("act_token",        lambda f: preprocess_acts(f, window=(0, 0)),    "the feature activates on tokens matching"),
        ("before_act_token", lambda f: preprocess_acts(f, window=(-1, -1)),  "the token immediately before activation is"),
        ("after_act_token",  lambda f: preprocess_acts(f, window=(1, 1)),    "the token immediately after activation is"),
        ("positive_logits",  lambda f: preprocess_logits(f, positive=True),  "the feature promotes next-token predictions for"),
        ("negative_logits",  lambda f: preprocess_logits(f, positive=False), "the feature suppresses next-token predictions for"),
        ("short_window",     lambda f: preprocess_acts(f, window=(-1, 1)),   "within one token before and after activation, the context contains"),
        ("medium_window",    lambda f: preprocess_acts(f, window=(-8, 8)),   "within eight tokens before and after activation, the context contains"),
        ("long_window",      lambda f: preprocess_acts(f, window=(-16, 16)), "within sixteen tokens before and after activation, the context contains"),
    ]
    EXPLANATION_MODEL_NAME = "gemini-2.5-flash-lite"
    NEURONPEDIA_EXPLANATION_TYPES = ["np_max-act-logits", "oai_token-act-pair"]
    
    NEURONPEDIA_API_KEY = os.environ.get("NEURONPEDIA_API_KEY", "")
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    
    N_FEATURES = 5
    MIN_NONZERO_ACTIVATIONS = 20

    # # 1. Sample the features
    # print("1. Sampling features...")
    # sample_features(experiment_dir, N_FEATURES, MIN_NONZERO_ACTIVATIONS, NEURONPEDIA_API_KEY, MODEL_ID)

    # # 2. Preprocess the features
    # print("2. Preprocessing features...")
    # preprocess_features(experiment_dir, CHANNEL_SPECS)

    # # 3. Preprocess the embedders
    # print("3. Preprocessing embedders...")
    # preprocess_embedders(experiment_dir, EMBEDDERS, [c[0] for c in CHANNEL_SPECS])

    # # 4. Generate correlation scores
    # print("4. Generating correlation scores...")
    # generate_correlation_scores(experiment_dir, EMBEDDERS, [c[0] for c in CHANNEL_SPECS])

    # # 5. Generate the explanation
    # print("5. Generating explanations...")
    # generate_explanations(
    #     experiment_dir,
    #     NEURONPEDIA_API_KEY,
    #     OPENROUTER_API_KEY,
    #     EXPLANATION_MODEL_NAME,
    #     NEURONPEDIA_EXPLANATION_TYPES,
    #     [c[0] for c in CHANNEL_SPECS]
    # )

    # # 5.1 Data sanity check
    # print("5.1. Checking data sanity...")
    # data_sanity(experiment_dir, [f"air_{c[0]}" for c in CHANNEL_SPECS] + NEURONPEDIA_EXPLANATION_TYPES)

    # # 6. Postprocess the explanations
    # print("6. Postprocessing explanations...")
    # postprocess_explanations(experiment_dir, CHANNEL_SPECS)

    # # 6.1 Data sanity check
    # print("6.1. Checking data sanity...")
    # data_sanity(experiment_dir, [f"air_{c[0]}" for c in CHANNEL_SPECS] + [f"postprocessed_air_{c[0]}" for c in CHANNEL_SPECS] + NEURONPEDIA_EXPLANATION_TYPES)

    # # 7. Score the explanations
    # print("7. Scoring explanations...")
    # score_explanations(
    #     experiment_dir,
    #     OPENROUTER_API_KEY,
    #     EXPLANATION_MODEL_NAME,
    # )

    # # 7.1 Data sanity check
    # print("7.1. Checking data sanity...")
    # data_sanity(experiment_dir, [f"air_{c[0]}" for c in CHANNEL_SPECS] + [f"postprocessed_air_{c[0]}" for c in CHANNEL_SPECS] + NEURONPEDIA_EXPLANATION_TYPES, require_scores=True)

    # # 8. Aggregate data in csv and illustrations
    # print("8. Aggregating data...")
    # aggregate_data(experiment_dir, NEURONPEDIA_EXPLANATION_TYPES, EMBEDDERS)
    
    # print("Pipeline completed.")

if __name__ == "__main__":
    main()

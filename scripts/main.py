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
from utils import build_delphi_record, build_pool, delphi_fuzz_scorer, generate_explanations_chair, generate_explanations_neuronpedia, get_baseline
from sentence_transformers import SentenceTransformer

def sample_features(experiment_dir: Path, n: int, min_acts: int, api_key: str, model_id: str):
    saved = 0
    while saved < n:
        layer, index = random.randint(0, 61), random.randint(0, 262143)
        req = urllib.request.Request(f"https://www.neuronpedia.org/api/feature/{model_id}/{layer}-gemmascope-2-transcoder-262k/{index}", headers={"x-api-key": api_key})
        feat = json.load(urllib.request.urlopen(req))
        if sum(1 for a in feat.get("activations", []) if a.get("maxValue", 0) > 0) >= min_acts:
            (experiment_dir / f"{model_id}_{layer}_{index}.json").write_text(json.dumps(feat, indent=2))
            saved += 1
            print(f"Saved {model_id}_{layer}_{index}.json")

def preprocess_features(experiment_dir: Path, channel_specs: list):
    for feature_path in experiment_dir.glob("*.json"):
        feat = json.loads(feature_path.read_text())
            
        feat["channels"] = {}
        for channel_id, recipe in channel_specs:
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
            pool = build_pool(experiment_dir, channel_id)
            get_baseline(experiment_dir, embedder_model, embedder_id, channel_id, pool)

def generate_correlation_scores(experiment_dir: Path, embedder_ids: list, channel_ids: list):
    for embedder_id in embedder_ids:
        embedder_model = SentenceTransformer(embedder_id)
        for feature_path in experiment_dir.glob("*.json"):
            feat = json.loads(feature_path.read_text())
            for channel_id in channel_ids:
                baseline_path = experiment_dir / "baselines" / f"{channel_id}_{embedder_id.replace('/', '-')}_baseline.json"
                baseline_data = json.loads(baseline_path.read_text())
                baseline = (baseline_data["intra_mu"], baseline_data["intra_sd"], 
                            baseline_data["inter_mu"], baseline_data["inter_sd"], 
                            baseline_data["centroid"])
                
                channel_data = feat["channels"][channel_id]
                score = gen_normalized_correlation_score(
                    embed(channel_data["examples"], embedder_model), 
                    baseline, 
                    w=channel_data["weights"]
                )
                
                feat["channels"][channel_id]["scores"][embedder_id] = score
                
            feature_path.write_text(json.dumps(feat, indent=2))

def generate_explanations(
    experiment_dir: Path,
    neuronpedia_api_key: str,
    openrouter_api_key: str,
    model_name: str,
    neuronpedia_explanation_types: list[str],
    channel_ids: list[str]
):
    generate_explanations_neuronpedia(experiment_dir, neuronpedia_api_key, model_name, neuronpedia_explanation_types)
    generate_explanations_chair(experiment_dir, openrouter_api_key, f"google/{model_name}", channel_ids)

def postprocess_explanations(experiment_dir: Path):
    pass

def score_explanations(
    experiment_dir: Path,
    openrouter_api_key: str,
    model_name: str,
):
    feature_paths = sorted(experiment_dir.glob("*.json"))
    for feature_path in feature_paths:
        feat = json.loads(feature_path.read_text())
        delphi_record = build_delphi_record(
            experiment_dir,
            feature_path,
        )
        for explanation in feat.get("explanations", []):
            score = delphi_fuzz_scorer(
                delphi_record,
                explanation,
                openrouter_api_key,
                f"google/{model_name}",
            )
            explanation["scores"].append({
                "value": score,
                "explanationScoreTypeName": "delphi_fuzz",
                "explanationScoreModelName": f"google/{model_name}"
            })
        feature_path.write_text(json.dumps(feat, indent=2))

def aggregate_data(experiment_dir: Path):
    pass

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
        # "BAAI/bge-small-en-v1.5",
        # "Qwen/Qwen3-Embedding-0.6B",
        # "BAAI/bge-m3",
        # "intfloat/multilingual-e5-large-instruct",
        # "google/embeddinggemma-300m",
        # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        # "sentence-transformers/LaBSE",
    ]
    
    CHANNEL_SPECS = [
        ("act_token",       lambda f: preprocess_acts(f, window=(0, 0))),
        ("before_act_token",lambda f: preprocess_acts(f, window=(-1, -1))),
        # ("after_act_token", lambda f: preprocess_acts(f, window=(1, 1))),
        # ("positive_logits", lambda f: preprocess_logits(f, positive=True)),
        # ("negative_logits", lambda f: preprocess_logits(f, positive=False)),
        # ("short_window",    lambda f: preprocess_acts(f, window=(-1, 1))),
        # ("medium_window",   lambda f: preprocess_acts(f, window=(-10, 10))),
        # ("long_window",     lambda f: preprocess_acts(f, window=(-25, 25))),
    ]
    EXPLANATION_MODEL_NAME = "gemini-2.5-flash-lite"
    NEURONPEDIA_EXPLANATION_TYPES = ["np_max-act-logits", "oai_token-act-pair"]
    
    NEURONPEDIA_API_KEY = os.environ.get("NEURONPEDIA_API_KEY", "")
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    
    N_FEATURES = 50
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

    # # 6. Postprocess the explanations
    # print("6. Postprocessing explanations...")
    # postprocess_explanations(experiment_dir)

    # 7. Score the explanations
    print("7. Scoring explanations...")
    score_explanations(
        experiment_dir,
        OPENROUTER_API_KEY,
        EXPLANATION_MODEL_NAME,
    )

    # 8. Aggregate data in csv and illustrations
    print("8. Aggregating data...")
    aggregate_data(experiment_dir)
    
    print("Pipeline completed.")

if __name__ == "__main__":
    main()

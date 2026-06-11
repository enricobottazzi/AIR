import argparse
import json
import os
import random
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.explainer import preprocess_acts, preprocess_logits
from scripts.utils import build_pool, get_baseline
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
                "weights": weights
            }
            
        feature_path.write_text(json.dumps(feat, indent=2))

def preprocess_embedders(experiment_dir: Path, embedder_ids: list, channel_ids: list):
    for embedder in embedder_ids:
        model = SentenceTransformer(embedder)
        for channel_id in channel_ids:
            pool = build_pool(experiment_dir, channel_id)
            get_baseline(experiment_dir, model, embedder, channel_id, pool)

def generate_explanations(experiment_dir: Path, channel_specs: list, api_key: str):
    pass

def postprocess_explanations(experiment_dir: Path):
    pass

def score_explanations(experiment_dir: Path, embedders: list):
    pass

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
        "BAAI/bge-small-en-v1.5",
        "Qwen/Qwen3-Embedding-0.6B",
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-large-instruct",
        "google/embeddinggemma-300m",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/LaBSE",
    ]
    
    CHANNEL_SPECS = [
        ("act_token",       lambda f: preprocess_acts(f, window=(0, 0))),
        ("before_act_token",lambda f: preprocess_acts(f, window=(-1, -1))),
        ("after_act_token", lambda f: preprocess_acts(f, window=(1, 1))),
        ("positive_logits", lambda f: preprocess_logits(f, positive=True)),
        ("negative_logits", lambda f: preprocess_logits(f, positive=False)),
        ("short_window",    lambda f: preprocess_acts(f, window=(-1, 1))),
        ("medium_window",   lambda f: preprocess_acts(f, window=(-10, 10))),
        ("long_window",     lambda f: preprocess_acts(f, window=(-25, 25))),
    ]
    
    NEURONPEDIA_API_KEY = os.environ.get("NEURONPEDIA_API_KEY", "")
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    
    N_FEATURES = 50
    MIN_NONZERO_ACTIVATIONS = 20

    # 1. Sample the features
    print("1. Sampling features...")
    sample_features(experiment_dir, N_FEATURES, MIN_NONZERO_ACTIVATIONS, NEURONPEDIA_API_KEY, MODEL_ID)

    # 2. Preprocess the features
    print("2. Preprocessing features...")
    preprocess_features(experiment_dir, CHANNEL_SPECS)

    # 3. Preprocess the embedders
    print("3. Preprocessing embedders...")
    preprocess_embedders(experiment_dir, EMBEDDERS, [c[0] for c in CHANNEL_SPECS])

    # 4. Generate the explanation
    print("4. Generating explanations...")
    generate_explanations(experiment_dir, CHANNEL_SPECS, OPENROUTER_API_KEY)

    # 5. Postprocess the explanations
    print("5. Postprocessing explanations...")
    postprocess_explanations(experiment_dir)

    # 6. Score the explanations
    print("6. Scoring explanations...")
    score_explanations(experiment_dir, EMBEDDERS)

    # 7. Aggregate data in csv and illustrations
    print("7. Aggregating data...")
    aggregate_data(experiment_dir)
    
    print("Pipeline completed.")

if __name__ == "__main__":
    main()

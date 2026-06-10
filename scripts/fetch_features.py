import argparse
import json
import random
import urllib.request
from pathlib import Path

MODEL_ID = "gemma-3-27b-it"
ENV = dict(
    line.split("=", 1)
    for line in Path(".env").read_text().splitlines()
    if "=" in line
)
API_KEY = ENV["NEURONPEDIA_API_KEY"]

N_FEATURES = 50
MIN_NONZERO_ACTIVATIONS = 20


def fetch_feature(layer: int, index: int) -> dict:
    source = f"{layer}-gemmascope-2-transcoder-262k"
    url = f"https://www.neuronpedia.org/api/feature/{MODEL_ID}/{source}/{index}"
    req = urllib.request.Request(url, headers={"x-api-key": API_KEY})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def nonzero_activation_count(feature: dict) -> int:
    return sum(1 for a in feature.get("activations", []) if a.get("maxValue", 0) > 0)


def main():
    parser = argparse.ArgumentParser(description="Fetch random Neuronpedia features into an experiment directory.")
    parser.add_argument("experiment_dir", type=Path, help="Directory to write feature JSON files")
    args = parser.parse_args()
    args.experiment_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    while saved < N_FEATURES:
        layer = random.randint(0, 61)
        index = random.randint(0, 262143)
        feature = fetch_feature(layer, index)
        if nonzero_activation_count(feature) < MIN_NONZERO_ACTIVATIONS:
            print(f"Skip {layer}_{index}: only {nonzero_activation_count(feature)} non-zero activations")
            continue
        out = args.experiment_dir / f"{MODEL_ID}_{layer}_{index}.json"
        out.write_text(json.dumps(feature, indent=2))
        print(f"Saved {out}")
        saved += 1


if __name__ == "__main__":
    main()

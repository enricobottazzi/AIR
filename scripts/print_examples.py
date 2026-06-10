import argparse
import json
from pathlib import Path
from src.explainer import preprocess_acts, preprocess_logits

CHANNEL_SPECS = [
    ("input",           lambda f: preprocess_acts(f, window=(0, 0))),
    ("positive_logits", lambda f: preprocess_logits(f, positive=True)),
    ("negative_logits", lambda f: preprocess_logits(f, positive=False)),
    ("preceding",       lambda f: preprocess_acts(f, window=(-1, -1))),
    ("output",          lambda f: preprocess_acts(f, window=(1, 1))),
    ("short_window",    lambda f: preprocess_acts(f, window=(-1, 1))),
    ("medium_window",   lambda f: preprocess_acts(f, window=(-10, 10))),
    ("long_window",     lambda f: preprocess_acts(f, window=(-25, 25))),
]

def main():
    parser = argparse.ArgumentParser(description="Print examples for all features in an experiment directory.")
    parser.add_argument("experiment_dir", type=Path, help="Directory containing feature JSON files")
    args = parser.parse_args()
    
    experiment_dir = args.experiment_dir
    if not experiment_dir.is_dir():
        raise SystemExit(f"Not a directory: {experiment_dir}")

    features = sorted(p for p in experiment_dir.glob("*.json") if p.is_file())
    if not features:
        raise SystemExit(f"No feature JSON files in {experiment_dir}")
    
    for path in features:
        feature = json.loads(path.read_text())
        print(f"=== Feature: {path.name} ===")
        for label, recipe in CHANNEL_SPECS:
            prompt, examples, weights = recipe(feature)
            print(f"\n--- Channel: {label} ---")
            for ex in examples:
                print(repr(ex))
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()

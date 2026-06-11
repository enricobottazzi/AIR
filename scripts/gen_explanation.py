import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path

# delphi.clients eagerly imports a vllm-backed Offline client; stub it out (no vllm on macOS).
for name in ("vllm", "vllm.distributed", "vllm.distributed.parallel_state", "vllm.inputs"):
    sys.modules.setdefault(name, types.ModuleType(name))
for attr in ("LLM", "SamplingParams"):
    setattr(sys.modules["vllm"], attr, object)
for attr in ("destroy_distributed_environment", "destroy_model_parallel"):
    setattr(sys.modules["vllm.distributed.parallel_state"], attr, object)
sys.modules["vllm.inputs"].TokensPrompt = object

from sentence_transformers import SentenceTransformer

from src.explainer import preprocess_acts, preprocess_logits
from src.correlation_score import gen_normalized_correlation_score, gen_baseline, embed
from src.llm import ExplainerSetup, complete

def feature_paths(experiment_dir: Path) -> list[Path]:
    return sorted(p for p in experiment_dir.glob("*.json") if p.is_file())


def write_results(feature_path: Path, experiment_dir: Path, scores: dict, explanations: dict, labels: list[str]):
    best_channel = {
        e: max(((l, scores[l][e]) for l in labels if scores[l][e] > 0), key=lambda x: x[1], default=(None,))[0]
        for e in EMBEDDERS
    }

    def fmt_score(label: str, embedder: str) -> str:
        s = f"{scores[label][embedder]:.2f}"
        return f"**{s}** *" if label == best_channel[embedder] else s

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    out_md = results_dir / f"results_{feature_path.stem}_{ts}.md"
    rows = [
        f"| channel | {' | '.join(EMBEDDERS)} | explanation |",
        f"|---|{'---|' * len(EMBEDDERS)}---|",
        *(
            f"| {label} | {' | '.join(fmt_score(label, e) for e in EMBEDDERS)} | {explanations[label]} |"
            for label in labels
        ),
    ]
    out_md.write_text("\n".join(rows) + "\n")
    
    out_json = results_dir / f"results_{feature_path.stem}_{ts}.json"
    json_data = {
        "feature": feature_path.stem,
        "best_channel": best_channel,
        "scores": scores,
        "explanations": explanations
    }
    out_json.write_text(json.dumps(json_data, indent=2))
    
    print(f"wrote {out_md.name} and {out_json.name}")


def run_feature(feature_path: Path, experiment_dir: Path,
                setup: ExplainerSetup, embedder_models: dict[str, SentenceTransformer]):
    feature = json.loads(feature_path.read_text())
    channels = [(label, *recipe(feature)) for label, recipe in CHANNEL_SPECS]
    labels = [label for label, *_ in channels]

    explanations = {label: complete(setup, prompt, API_KEY).strip() for label, prompt, _, _ in channels}

    scores = {label: {} for label in labels}
    for embedder, model in embedder_models.items():
        for (label, recipe), (_, _, examples, weights) in zip(CHANNEL_SPECS, channels):
            baseline = get_baseline(experiment_dir, model, embedder, label, recipe, len(examples), feature_path.stem)
            scores[label][embedder] = gen_normalized_correlation_score(
                embed(examples, model), baseline, w=weights)

    write_results(feature_path, experiment_dir, scores, explanations, labels)


def main():
    parser = argparse.ArgumentParser(description="Generate explanations and correlation scores for all features in an experiment.")
    parser.add_argument("experiment_dir", type=Path, help="Directory containing feature JSON files")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    if not experiment_dir.is_dir():
        raise SystemExit(f"Not a directory: {experiment_dir}")

    paths = feature_paths(experiment_dir)
    if not paths:
        raise SystemExit(f"No feature JSON files in {experiment_dir}")

    setup = ExplainerSetup(model="google/gemini-2.5-flash-lite")
    embedder_models = {e: SentenceTransformer(e) for e in EMBEDDERS}

    for feature_path in paths:
        print(f"processing {feature_path.name}")
        run_feature(feature_path, experiment_dir, setup, embedder_models)


if __name__ == "__main__":
    main()

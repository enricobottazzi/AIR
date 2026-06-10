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

API_KEY = next(
    l.split("=", 1)[1]
    for l in Path(".env").read_text().splitlines()
    if l.startswith("OPENROUTER_API_KEY=")
)

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
    ("input",           lambda f: preprocess_acts(f, window=(0, 0))),
    ("positive_logits", lambda f: preprocess_logits(f, positive=True)),
    ("negative_logits", lambda f: preprocess_logits(f, positive=False)),
    ("preceding",       lambda f: preprocess_acts(f, window=(-1, -1))),
    ("output",          lambda f: preprocess_acts(f, window=(1, 1))),
    ("short_window",    lambda f: preprocess_acts(f, window=(-1, 1))),
    ("medium_window",   lambda f: preprocess_acts(f, window=(-10, 10))),
    ("long_window",     lambda f: preprocess_acts(f, window=(-25, 25))),
]


def feature_paths(experiment_dir: Path) -> list[Path]:
    return sorted(p for p in experiment_dir.glob("*.json") if p.is_file())


def build_pool(experiment_dir: Path, recipe, exclude_stem: str | None = None) -> list[str]:
    return [
        ex
        for p in feature_paths(experiment_dir)
        if exclude_stem is None or p.stem != exclude_stem
        for ex in recipe(json.loads(p.read_text()))[1]
    ]


def get_baseline(experiment_dir: Path, model, embedder: str, label: str, recipe, n: int, feature_stem: str) -> tuple[float, float, float, float, list[float]]:
    baseline_dir = experiment_dir / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    out = baseline_dir / f"{label}_{embedder.replace('/', '-')}_baseline.json"
    if not out.exists():
        pool = build_pool(experiment_dir, recipe, exclude_stem=feature_stem)
        intra_mu, intra_sd, inter_mu, inter_sd, centroid = gen_baseline(model, pool, n=n)
        out.write_text(json.dumps(
            {"channel": label, "embedder": embedder, "n": n,
             "pool_size": len(pool), "intra_mu": intra_mu, "intra_sd": intra_sd,
             "inter_mu": inter_mu, "inter_sd": inter_sd, "centroid": centroid},
            indent=2,
        ))
    d = json.loads(out.read_text())
    if "intra_mu" not in d:
        out.unlink()
        return get_baseline(experiment_dir, model, embedder, label, recipe, n, feature_stem)
    return d["intra_mu"], d["intra_sd"], d["inter_mu"], d["inter_sd"], d["centroid"]


def write_results(feature_path: Path, experiment_dir: Path, scores: dict, explanations: dict, labels: list[str]):
    best_channel = {
        e: max(((l, scores[l][e]) for l in labels if scores[l][e] > 0), key=lambda x: x[1], default=(None,))[0]
        for e in EMBEDDERS
    }

    def fmt_score(label: str, embedder: str) -> str:
        s = f"{scores[label][embedder]:.2f}"
        return f"**{s}** *" if label == best_channel[embedder] else s

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = experiment_dir / f"results_{feature_path.stem}_{ts}.md"
    rows = [
        f"| channel | {' | '.join(EMBEDDERS)} | explanation |",
        f"|---|{'---|' * len(EMBEDDERS)}---|",
        *(
            f"| {label} | {' | '.join(fmt_score(label, e) for e in EMBEDDERS)} | {explanations[label]} |"
            for label in labels
        ),
    ]
    out.write_text("\n".join(rows) + "\n")
    print(f"wrote {out}")


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

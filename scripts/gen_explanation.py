import asyncio
import json
import sys
from datetime import datetime
import types
from pathlib import Path

# delphi.clients eagerly imports a vllm-backed Offline client; stub it out (no vllm on macOS).
for name in ("vllm", "vllm.distributed", "vllm.distributed.parallel_state", "vllm.inputs"):
    sys.modules.setdefault(name, types.ModuleType(name))
for attr in ("LLM", "SamplingParams"):
    setattr(sys.modules["vllm"], attr, object)
for attr in ("destroy_distributed_environment", "destroy_model_parallel"):
    setattr(sys.modules["vllm.distributed.parallel_state"], attr, object)
sys.modules["vllm.inputs"].TokensPrompt = object

import torch
from sentence_transformers import SentenceTransformer

from delphi.clients import OpenRouter
from delphi.latents.latents import (
    ActivatingExample,
    Latent,
    LatentRecord,
    NonActivatingExample,
)
from delphi.scorers import DetectionScorer, EmbeddingScorer, FuzzingScorer

from src.explainer import preprocess_acts, preprocess_logits
from src.correlation_score import gen_normalized_correlation_score, gen_baseline, embed
from src.llm import ExplainerSetup, complete

API_KEY = next(
    l.split("=", 1)[1]
    for l in Path(".env").read_text().splitlines()
    if l.startswith("OPENROUTER_API_KEY=")
)

EMBEDDERS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"]
setup = ExplainerSetup(model="google/gemini-2.5-flash-lite")
feature_path = Path("data/gemma-3-27b-it_11_59359.json")
feature = json.loads(feature_path.read_text())
# neg_feature = json.loads(Path("data/gemma-3-27b-it_6_4349.json").read_text())

# (label, recipe) per channel; recipe maps a feature dict -> (prompt, examples, weights).
CHANNEL_SPECS = [
    ("input",           lambda f: preprocess_acts(f, window=(0, 0))),
    ("positive_logits", lambda f: preprocess_logits(f, positive=True)),
    ("negative_logits", lambda f: preprocess_logits(f, positive=False)),
    ("output",          lambda f: preprocess_acts(f, window=(1, 1))),
    ("short_window",    lambda f: preprocess_acts(f, window=(-1, 1))),
    ("medium_window",   lambda f: preprocess_acts(f, window=(-10, 10))),
    ("long_window",     lambda f: preprocess_acts(f, window=(-25, 25))),
]
# (label, prompt, examples, weights) per channel for the feature under study.
CHANNELS = [(label, *recipe(feature)) for label, recipe in CHANNEL_SPECS]


def build_pool(recipe, exclude: Path) -> list[str]:
    # Background corpus for a channel: its examples drawn from every *other* feature file.
    return [
        ex
        for p in sorted(Path("data").glob("gemma-*.json"))
        if p != exclude
        for ex in recipe(json.loads(p.read_text()))[1]
    ]


BASELINE_DIR = Path("data/baselines")
BASELINE_DIR.mkdir(exist_ok=True)


def get_baseline(model, embedder: str, label: str, recipe, n: int) -> tuple[float, float]:
    # Load cached (mu, sd) for (channel, embedder), computing + persisting it if absent.
    out = BASELINE_DIR / f"{label}_{embedder.replace('/', '-')}_baseline.json"
    if not out.exists():
        pool = build_pool(recipe, exclude=feature_path)
        mu, sd = gen_baseline(model, pool, n=n)
        out.write_text(json.dumps(
            {"channel": label, "embedder": embedder, "n": n,
             "pool_size": len(pool), "mu": mu, "sd": sd},
            indent=2,
        ))
    d = json.loads(out.read_text())
    return d["mu"], d["sd"]


# Explanation is embedder-independent -> compute once per channel.
explanations = {label: complete(setup, prompt, API_KEY).strip() for label, prompt, _, _ in CHANNELS}

# z-score per (channel, embedder).
scores = {label: {} for label, *_ in CHANNELS}
for embedder in EMBEDDERS:
    model = SentenceTransformer(embedder)
    for (label, recipe), (_, _, examples, weights) in zip(CHANNEL_SPECS, CHANNELS):
        baseline = get_baseline(model, embedder, label, recipe, n=len(examples))
        scores[label][embedder] = gen_normalized_correlation_score(embed(examples, model), baseline, w=weights)

# write a channel x embedder comparison table of z-scores as markdown
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = Path(f"data/results_{feature_path.stem}_{ts}.md")
rows = [
    f"| channel | {' | '.join(EMBEDDERS)} | explanation |",
    f"|---|{'---|' * len(EMBEDDERS)}---|",
    *(
        f"| {label} | {' | '.join(f'{scores[label][e]:.2f}' for e in EMBEDDERS)} | {explanations[label]} |"
        for label, *_ in CHANNELS
    ),
]
out.write_text("\n".join(rows) + "\n")
print(f"wrote {out}")


# def load_record(explanation: str) -> LatentRecord:
#     examples = [
#         ActivatingExample(
#             tokens=torch.zeros(len(a["values"]), dtype=torch.long),
#             activations=torch.tensor(a["values"], dtype=torch.float32),
#             str_tokens=a["tokens"],
#         )
#         for a in feature["activations"]
#     ]
#     examples.sort(key=lambda e: e.max_activation, reverse=True)
#     record = LatentRecord(latent=Latent(feature["layer"], int(feature["index"])))
#     record.test = examples
#     record.not_active = [
#         NonActivatingExample(
#             tokens=torch.zeros(len(a["values"]), dtype=torch.long),
#             activations=torch.tensor(a["values"], dtype=torch.float32),
#             str_tokens=a["tokens"],
#             distance=-1.0,
#         )
#         for a in neg_feature["activations"]
#     ]
#     record.explanation = explanation
#     return record


# def recall(score) -> float:
#     correct = [s.correct for s in score if s.correct is not None]
#     return sum(correct) / len(correct) if correct else 0.0


# def emb_summary(score) -> dict:
#     pos = [s.similarity for s in score if s.distance >= 0]
#     neg = [s.similarity for s in score if s.distance < 0]
#     p, n = sum(pos) / max(len(pos), 1), sum(neg) / max(len(neg), 1)
#     return {"mean_pos": p, "mean_neg": n, "gap": p - n}


# async def score():
#     explanation = complete(setup, prompt_5, API_KEY).strip()
#     record = load_record(explanation)
#     client = OpenRouter("google/gemini-2.5-flash-lite", api_key=API_KEY)

#     # fuzz = await FuzzingScorer(client, fuzz_type="active")(record)
#     detect = await DetectionScorer(client)(record)
#     embed_score = await EmbeddingScorer(model)(record)

#     for s in detect.score:
#         print(s.activating, s.prediction, s.correct)

#     out = {
#         "explanation": explanation,
#         # "fuzz_recall": recall(fuzz.score),
#         "detection_recall": recall(detect.score),
#         "embedding_similarity": emb_summary(embed_score.score),
#     }
#     Path("data/scores.json").write_text(json.dumps(out, indent=2))
#     print(out)


# if __name__ == "__main__":
#     asyncio.run(score())

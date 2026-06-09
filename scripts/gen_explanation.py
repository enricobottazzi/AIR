import asyncio
import json
import sys
import types
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

# delphi.clients eagerly imports a vllm-backed Offline client; stub it out (no vllm on macOS).
for name in ("vllm", "vllm.distributed", "vllm.distributed.parallel_state", "vllm.inputs"):
    sys.modules.setdefault(name, types.ModuleType(name))
for attr in ("LLM", "SamplingParams"):
    setattr(sys.modules["vllm"], attr, object)
for attr in ("destroy_distributed_environment", "destroy_model_parallel"):
    setattr(sys.modules["vllm.distributed.parallel_state"], attr, object)
sys.modules["vllm.inputs"].TokensPrompt = object

from delphi.clients import OpenRouter
from delphi.latents.latents import (
    ActivatingExample,
    Latent,
    LatentRecord,
    NonActivatingExample,
)
from delphi.scorers import DetectionScorer, EmbeddingScorer, FuzzingScorer

from src.explainer import explain_acts, explain_logits
from src.correlation_score import gen_correlation_score, embed
from src.llm import ExplainerSetup, complete

API_KEY = next(
    l.split("=", 1)[1]
    for l in Path(".env").read_text().splitlines()
    if l.startswith("OPENROUTER_API_KEY=")
)

setup = ExplainerSetup(model="google/gemini-2.5-flash-lite")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

feature = json.loads(Path("data/gemma-3-27b-it_33_142228.json").read_text())
neg_feature = json.loads(Path("data/gemma-3-27b-it_6_4349.json").read_text())

# input
prompt_1, examples_1, weights_1 = explain_acts(feature, window=(0, 0))

# positive logits
prompt_2, examples_2, weights_2 = explain_logits(feature, positive=True)

# negative logits
prompt_3, examples_3, weights_3 = explain_logits(feature, positive=False)

# output
prompt_4, examples_4, weights_4 = explain_acts(feature, window=(1, 1))

# cross
prompt_5, examples_5, weights_5 = explain_acts(feature, window=(-25, 25))

# compute correlation scores
correlation_score_1 = gen_correlation_score(embed(examples_1, model), w=weights_1)
correlation_score_2 = gen_correlation_score(embed(examples_2, model), w=weights_2)
correlation_score_3 = gen_correlation_score(embed(examples_3, model), w=weights_3)
correlation_score_4 = gen_correlation_score(embed(examples_4, model), w=weights_4)
correlation_score_5 = gen_correlation_score(embed(examples_5, model), w=weights_5)

print(correlation_score_1)
print(correlation_score_2)
print(correlation_score_3)
print(correlation_score_4)
print(correlation_score_5)


def load_record(explanation: str) -> LatentRecord:
    examples = [
        ActivatingExample(
            tokens=torch.zeros(len(a["values"]), dtype=torch.long),
            activations=torch.tensor(a["values"], dtype=torch.float32),
            str_tokens=a["tokens"],
        )
        for a in feature["activations"]
    ]
    examples.sort(key=lambda e: e.max_activation, reverse=True)
    record = LatentRecord(latent=Latent(feature["layer"], int(feature["index"])))
    record.test = examples
    record.not_active = [
        NonActivatingExample(
            tokens=torch.zeros(len(a["values"]), dtype=torch.long),
            activations=torch.tensor(a["values"], dtype=torch.float32),
            str_tokens=a["tokens"],
            distance=-1.0,
        )
        for a in neg_feature["activations"]
    ]
    record.explanation = explanation
    return record


def recall(score) -> float:
    correct = [s.correct for s in score if s.correct is not None]
    return sum(correct) / len(correct) if correct else 0.0


def emb_summary(score) -> dict:
    pos = [s.similarity for s in score if s.distance >= 0]
    neg = [s.similarity for s in score if s.distance < 0]
    p, n = sum(pos) / max(len(pos), 1), sum(neg) / max(len(neg), 1)
    return {"mean_pos": p, "mean_neg": n, "gap": p - n}


async def score():
    explanation = complete(setup, prompt_5, API_KEY).strip()
    record = load_record(explanation)
    client = OpenRouter("google/gemini-2.5-flash-lite", api_key=API_KEY)

    # fuzz = await FuzzingScorer(client, fuzz_type="active")(record)
    detect = await DetectionScorer(client)(record)
    embed_score = await EmbeddingScorer(model)(record)

    for s in detect.score:
        print(s.activating, s.prediction, s.correct)

    out = {
        "explanation": explanation,
        # "fuzz_recall": recall(fuzz.score),
        "detection_recall": recall(detect.score),
        "embedding_similarity": emb_summary(embed_score.score),
    }
    Path("data/scores.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    asyncio.run(score())

import asyncio
import json
import os
import sys
from pathlib import Path

from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.explanations.explainer import (
    MaxActivationAndLogitsExplainer,
    MaxActivationAndLogitsGeneralExplainer,
)
from neuron_explainer.explanations.prompt_builder import PromptFormat

OPENROUTER_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.5-flash-lite"
FEATURE_PATH = "data/gemma-3-27b-it_33_142228.json"
OUT_PATH = "data/explanation_decode.json"


def load_env(path: str = ".env") -> None:
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def activation_records(feature: dict) -> list[ActivationRecord]:
    return [
        ActivationRecord(tokens=a["tokens"], activations=a["values"])
        for a in feature["activations"]
    ]


def make_explainer(cls):
    return cls(
        model_name=MODEL,
        prompt_format=PromptFormat.HARMONY_V4,
        base_api_url=OPENROUTER_URL,
        override_api_key=os.environ["OPENROUTER_API_KEY"],
    )


async def explain(feature: dict, cls, max_tokens: int) -> str:
    records = activation_records(feature)
    explanations = await make_explainer(cls).generate_explanations(
        all_activation_records=records,
        max_activation=calculate_max_activation(records),
        top_positive_logits=feature["pos_str"],
        num_samples=1,
        max_tokens=max_tokens,
    )
    return explanations[0]


async def main(path: str = FEATURE_PATH, out_path: str = OUT_PATH) -> None:
    load_env()
    feature = json.loads(Path(path).read_text())
    out = {
        "layer": feature["layer"],
        "index": feature["index"],
        "general": await explain(feature, MaxActivationAndLogitsGeneralExplainer, max_tokens=100),
        "concise": await explain(feature, MaxActivationAndLogitsExplainer, max_tokens=100),
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    asyncio.run(main(*(sys.argv[1:3])))

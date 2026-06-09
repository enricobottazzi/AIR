import json
from pathlib import Path

from src.explainer import explain_feature
from src.llm import ExplainerSetup, complete

API_KEY = next(
    l.split("=", 1)[1]
    for l in Path(".env").read_text().splitlines()
    if l.startswith("OPENROUTER_API_KEY=")
)

setup = ExplainerSetup(model="google/gemini-2.5-flash-lite")

feature = json.loads(Path("data/gemma-3-27b-it_33_142228.json").read_text())
prompt = explain_feature(feature, window=(0, 0))

print(complete(setup, prompt, API_KEY))

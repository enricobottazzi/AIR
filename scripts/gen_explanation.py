import json
from pathlib import Path

from src.explainer import explain_acts, explain_logits
from src.llm import ExplainerSetup, complete

API_KEY = next(
    l.split("=", 1)[1]
    for l in Path(".env").read_text().splitlines()
    if l.startswith("OPENROUTER_API_KEY=")
)

setup = ExplainerSetup(model="google/gemini-2.5-flash-lite")

feature = json.loads(Path("data/gemma-3-27b-it_33_142228.json").read_text())
prompt_1 = explain_acts(feature, window=(0, 0))
prompt_2 = explain_logits(feature, positive=True)
prompt_3 = explain_logits(feature, positive=False)

print(complete(setup, prompt_1, API_KEY))
print(complete(setup, prompt_2, API_KEY))
print(complete(setup, prompt_3, API_KEY))

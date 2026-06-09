import json
import urllib.request
from pathlib import Path

from src.explainer import explain_input_feature

API_KEY = next(
    l.split("=", 1)[1]
    for l in Path(".env").read_text().splitlines()
    if l.startswith("OPENROUTER_API_KEY=")
)

feature = json.loads(Path("data/gemma-3-27b-it_33_142228.json").read_text())
prompt = explain_input_feature(feature)

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=json.dumps({
        "model": "google/gemini-2.5-flash-lite",
        "messages": [{"role": "user", "content": prompt}],
    }).encode(),
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as r:
    print(json.load(r)["choices"][0]["message"]["content"])

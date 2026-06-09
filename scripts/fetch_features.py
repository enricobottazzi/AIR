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


def fetch_feature(layer: int, index: int) -> dict:
    source = f"{layer}-gemmascope-2-transcoder-262k"
    url = f"https://www.neuronpedia.org/api/feature/{MODEL_ID}/{source}/{index}"
    req = urllib.request.Request(url, headers={"x-api-key": API_KEY})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


N_FEATURES = 50

if __name__ == "__main__":
    for _ in range(N_FEATURES):
        layer = random.randint(0, 61)
        index = random.randint(0, 262143)
        feature = fetch_feature(layer, index)
        out = Path("data") / f"{MODEL_ID}_{layer}_{index}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(feature, indent=2))
        print(f"Saved {out}")

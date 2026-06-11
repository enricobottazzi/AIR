import json
import urllib.request
from dataclasses import dataclass

@dataclass(frozen=True)
class ExplainerSetup:
    model: str
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"

def complete(setup: ExplainerSetup, prompt: str, api_key: str, **trace) -> str:
    payload = {"model": setup.model, "messages": [{"role": "user", "content": prompt}]}
    if trace: payload["trace"] = trace
    req = urllib.request.Request(
        setup.endpoint,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)["choices"][0]["message"]["content"]

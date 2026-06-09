# ITL

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Set `NEURONPEDIA_API_KEY` and `OPENROUTER_API_KEY` in `.env`.

### Scripts

1. Fetch features from Neuronpedia API

```bash
python scripts/fetch_features.py
```

2. Generate an explanation for a feature via OpenRouter

```bash
python scripts/gen_explanation_input.py
```
# ITL

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Set `NEURONPEDIA_API_KEY` and `OPENROUTER_API_KEY` in `.env`.

Run 

```bash
python scripts/main.py data/experiments/2026-06-11
```

Where `data/experiments/2026-06-11` is the directory to store the experiment data.
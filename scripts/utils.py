import json
from pathlib import Path

def build_pool(experiment_dir: Path, channel_id: str) -> list[str]:
    pool = []
    for feature_path in experiment_dir.glob("*.json"):
        feat = json.loads(feature_path.read_text())
        pool.extend(feat["channels"][channel_id]["examples"])
    return pool

from src.correlation_score import gen_baseline

def get_baseline(experiment_dir: Path, embedder_model, embedder_id: str, channel_id: str, pool: list[str]):
    baseline_dir = experiment_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    out = baseline_dir / f"{channel_id}_{embedder_id.replace('/', '-')}_baseline.json"
        
    n = 10 if "logits" in channel_id else 20
    intra_mu, intra_sd, inter_mu, inter_sd, centroid = gen_baseline(embedder_model, pool, n=n, trials=1000)
    
    out.write_text(json.dumps({
        "channel": channel_id, "embedder": embedder_id, "n": n,
        "pool_size": len(pool), "intra_mu": intra_mu, "intra_sd": intra_sd,
        "inter_mu": inter_mu, "inter_sd": inter_sd, "centroid": centroid
    }, indent=2))

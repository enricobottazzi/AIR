import sys, json, subprocess
from pathlib import Path

def get_url(file_path):
    with open(file_path) as f:
        d = json.load(f)
    return f"https://www.neuronpedia.org/{d.get('modelId')}/{d.get('layer')}/{d.get('index')}"

def main(exp_dir):
    for p in Path(exp_dir).glob("*.json"):
        url = get_url(p)
        if sys.platform == "darwin":
            subprocess.run(["open", "-a", "Google Chrome", url])
        else:
            import webbrowser
            webbrowser.open(url)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python open_neuronpedia.py <experiment_dir>")

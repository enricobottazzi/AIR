import sys, json, subprocess, re
from pathlib import Path

def get_url(file_path):
    with open(file_path) as f:
        d = json.load(f)
    if 'feature' in d:
        parts = d['feature'].rsplit('_', 2)
        if len(parts) == 3:
            return f"https://www.neuronpedia.org/{parts[0]}/{parts[1]}/{parts[2]}"
    return f"https://www.neuronpedia.org/{d.get('modelId')}/{d.get('layer')}/{d.get('index')}"

def main(exp_dir):
    # Find all json files, either in exp_dir or exp_dir/results
    paths = list(Path(exp_dir).glob("*.json"))
    if not paths:
        paths = list(Path(exp_dir).glob("results/*.json"))
    
    def get_sort_key(p):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', p.name)]
        
    paths.sort(key=get_sort_key)

    for p in paths:
        url = get_url(p)
        if "None" in url:
            continue
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

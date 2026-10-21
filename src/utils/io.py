import json, yaml
from pathlib import Path

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def append_csv(row: dict, path):
    import pandas as pd
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header = not Path(path).exists()
    df.to_csv(path, mode="a", header=header, index=False)

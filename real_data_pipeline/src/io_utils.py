from pathlib import Path
import requests, pandas as pd

def download(url, destination):
    if not url or url.startswith("REPLACE_"):
        raise ValueError("Configure uma URL pública direta em config.yaml")
    destination=Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with destination.open("wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)
    return destination

def read_csv(path): return pd.read_csv(path)

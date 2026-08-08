import os, sys
import requests
from pathlib import Path

BASE = "https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/"
FILES = [
    "config.json", "generation_config.json", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "special_tokens_map.json",
]
dest = Path("data/registry/base/qwen2.5-0.5b-instruct")
dest.mkdir(parents=True, exist_ok=True)

for f in FILES:
    out = dest / f
    if out.exists() and out.stat().st_size > 0:
        print("skip", f); continue
    url = BASE + f + "?download=true"
    print("GET", f, flush=True)
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(out, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk); got += len(chunk)
        print(f"  saved {f} {got/1e6:.1f}MB", flush=True)
print("DONE ->", dest.resolve())

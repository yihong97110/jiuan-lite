import requests
from pathlib import Path
BASE = "https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/"
FILES = ["config.json","generation_config.json","tokenizer_config.json","vocab.json","merges.txt","special_tokens_map.json"]
dest = Path("data/registry/base/qwen2.5-0.5b-instruct")
s = requests.Session()
for f in FILES:
    out = dest / f
    if out.exists() and out.stat().st_size>0: print("skip",f); continue
    r = s.get(BASE+f, timeout=60, allow_redirects=True)
    r.raise_for_status()
    out.write_bytes(r.content)
    print(f, len(r.content), "bytes")
print("DONE")

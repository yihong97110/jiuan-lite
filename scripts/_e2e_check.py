import sys, json, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
BASE = "http://127.0.0.1:8000"

def post(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE+path, data=data, headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req).read())

def wait(tid, mx=300):
    for _ in range(mx):
        t = json.loads(urllib.request.urlopen(f"{BASE}/tasks/{tid}").read())
        if t["status"] in ("succeeded","failed"): return t
        time.sleep(1)
    return t

print("== 标 dataprep ==")
dp = wait(post("/dataprep", {"source":"data/samples.jsonl","name":"final-check"})["task_id"], 30)
ds = dp["result"]["dataset_id"]
print("  ", dp["status"], "train/valid=", dp["result"]["train_count"], dp["result"]["valid_count"])

print("== 训 train (LLaMA-Factory LoRA) ==")
tr = wait(post("/train", {"dataset_id":ds,"backend":"llamafactory","method":"lora"})["task_id"], 300)
model = tr["result"].get("model_id")
print("  ", tr["status"], "backend=", tr["result"].get("backend"), "loss=", tr["result"].get("train_loss"))

print("== 推 infer ==")
for q in ["火灾逃生的基本原则是什么？","台风来临前应做哪些准备？"]:
    it = wait(post("/infer", {"model_id":model,"prompt":q})["task_id"], 180)
    print(f"  Q: {q}")
    print(f"  A: {it['result']['answer'][:150]}")
    print(f"  usage: {it['result']['usage']}")

print("== 评 eval (valid) ==")
ev = wait(post("/eval", {"model_id":model,"dataset_id":ds,"split":"valid"})["task_id"], 180)
print("  ", ev["status"], "reliable=", ev["result"]["reliable"], "metrics=", ev["result"]["metrics"])
print("\nALL GREEN" if all(x["status"]=="succeeded" for x in (dp,tr,ev)) else "\nHAS FAILURE")

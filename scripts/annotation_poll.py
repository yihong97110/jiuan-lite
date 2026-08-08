"""标注同步 · 方式A 轮询守护（不依赖 Label Studio 服务）。

对标久安"标注完成→自动进入训练"的闭环：定时扫描标注任务，
凡是"已全部标注(pending=0)且未回灌过"的任务，自动回灌为训练集并触发训练。

用法(在已激活的 conda 环境里)：
    python scripts/annotation_poll.py --once           # 扫一遍就退出
    python scripts/annotation_poll.py --interval 300   # 每5分钟轮询(默认)
    python scripts/annotation_poll.py --parent iterv-... --hard-weight 3 --no-train

依赖控制面已启动(默认 http://127.0.0.1:8000)。
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

STATE_FILE = "data/annotations/.poll_state.json"


def _api(base: str, path: str, payload: dict | None = None, timeout: int = 30):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"committed": []}


def _save_state(st: dict) -> None:
    import os
    os.makedirs("data/annotations", exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(st, fh, ensure_ascii=False, indent=1)


def scan_once(base: str, parent, hard_weight: int, auto_train: bool) -> int:
    st = _load_state()
    done_before = set(st.get("committed", []))
    tasks = _api(base, "/annotation/tasks")
    fired = 0
    for t in tasks:
        tid = t["task_id"]
        if tid in done_before:
            continue
        if t["count"] > 0 and t["pending"] == 0:  # 全部标注完成
            print(f"[poll] {tid} 已全部标注({t['annotated']}/{t['count']})，自动回灌…")
            try:
                r = _api(base, "/annotation/commit", {
                    "task_id": tid, "name": "annotated",
                    "parent": parent, "hard_weight": hard_weight,
                    "auto_train": auto_train,
                })
            except Exception as exc:
                print(f"[poll] {tid} 回灌失败：{exc}")
                continue
            msg = f"[poll] {tid} → 数据集 {r.get('dataset_id')}"
            if r.get("train_task_id"):
                msg += f"，训练任务 {r['train_task_id']}"
            print(msg)
            done_before.add(tid)
            fired += 1
    st["committed"] = sorted(done_before)
    _save_state(st)
    return fired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--interval", type=int, default=300, help="轮询间隔秒(默认300)")
    ap.add_argument("--once", action="store_true", help="只扫一遍")
    ap.add_argument("--parent", default=None, help="回灌父数据集(增量血缘)")
    ap.add_argument("--hard-weight", type=int, default=1, help="硬样本复制倍数")
    ap.add_argument("--no-train", action="store_true", help="回灌但不自动训练")
    args = ap.parse_args()
    auto_train = not args.no_train
    if args.once:
        n = scan_once(args.base, args.parent, args.hard_weight, auto_train)
        print(f"[poll] 本轮触发 {n} 个任务回灌。")
        return
    print(f"[poll] 启动轮询，每 {args.interval}s 扫描一次（Ctrl+C 退出）")
    while True:
        try:
            scan_once(args.base, args.parent, args.hard_weight, auto_train)
        except Exception as exc:
            print(f"[poll] 扫描出错：{exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

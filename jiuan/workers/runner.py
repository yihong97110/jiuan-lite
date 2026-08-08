"""后台任务调度：把任务拉起为独立子进程执行（对标久安「任务管理/调度」）。

- 每个任务在独立 Python 进程运行，避免训练阻塞 API 进程 / 抢 GIL。
- 一个后台监控线程负责 join 子进程并回收僵尸；控制面只负责入队。
- 通过 max_parallel 限制并发，防止多训练任务同时抢 GPU 导致 OOM。
- PoC 级实现；P1 可平滑替换为 Celery / RQ。
"""
from __future__ import annotations

import subprocess
import sys
import threading
from collections import deque
import os
from pathlib import Path

from .. import store
from ..schemas import Stage, TaskStatus

MAX_PARALLEL = 1  # 训练类任务串行，避免抢占同一 GPU

_queue: deque[str] = deque()
_running: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()
_cv = threading.Condition(_lock)
_started = False


def _spawn(task_id: str) -> subprocess.Popen:
    project_root = Path(__file__).resolve().parent.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, "-m", "jiuan.workers.job", task_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(project_root),
        env=env,
    )


def _scheduler() -> None:
    while True:
        with _cv:
            while not _queue and not _running:
                _cv.wait()
            # 回收已结束进程
            for tid in [t for t, p in _running.items() if p.poll() is not None]:
                proc = _running.pop(tid, None)
                code = proc.returncode if proc else None
                task = store.get_task(tid)
                if task and code not in (0, None) and task.status not in (TaskStatus.SUCCEEDED, TaskStatus.FAILED):
                    store.update_task(
                        tid,
                        status=TaskStatus.FAILED,
                        error=f"worker process exited before task finalized (exit_code={code})",
                        log=f"任务失败: worker 子进程提前退出(exit_code={code})",
                    )
            # 补位调度
            while _queue and len(_running) < MAX_PARALLEL:
                tid = _queue.popleft()
                _running[tid] = _spawn(tid)
            has_running = bool(_running)
        if has_running:
            # 轮询等待任一进程结束
            import time

            time.sleep(0.3)
            with _cv:
                _cv.notify_all()


def _ensure_started() -> None:
    global _started
    with _lock:
        if _started:
            return
        t = threading.Thread(target=_scheduler, name="jiuan-scheduler", daemon=True)
        t.start()
        _started = True


def submit(stage: Stage, params: dict) -> str:
    _ensure_started()
    task = store.create_task(stage, params)
    with _cv:
        _queue.append(task.id)
        _cv.notify_all()
    return task.id

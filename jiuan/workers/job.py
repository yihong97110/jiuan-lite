"""独立 worker 进程入口：由控制面以子进程方式拉起，执行单个任务。

用法（一般由 runner 自动调用）：
    python -m jiuan.workers.job <task_id>

这样训练等 CPU/GPU 密集任务在独立进程运行，不占用 API 进程的 GIL，
也便于后续替换为 Celery / RQ 等成熟队列。
"""
from __future__ import annotations

import inspect
import sys

from .. import store
from ..pipeline import dataprep, distill, evaluate, infer, train, workflow
from ..schemas import Stage, TaskStatus

_HANDLERS = {
    Stage.DATAPREP: dataprep.run,
    Stage.DISTILL: distill.run,
    Stage.TRAIN: train.run,
    Stage.INFER: infer.run,
    Stage.EVAL: evaluate.run,
    Stage.WORKFLOW: workflow.run,
}


def execute(task_id: str) -> int:
    task = store.get_task(task_id)
    if not task:
        return 1
    store.update_task(task_id, status=TaskStatus.RUNNING, log=f"{task.stage.value} 任务开始(pid 独立进程)")
    try:
        handler = _HANDLERS[task.stage]
        kwargs = {"log": lambda m: store.update_task(task_id, log=m)}
        # 仅向支持 progress 回调的 handler 传入，避免破坏其他签名
        if "progress" in inspect.signature(handler).parameters:
            kwargs["progress"] = lambda p: store.update_task(task_id, progress=p)
        result = handler(task.params, **kwargs)
        store.update_task(task_id, status=TaskStatus.SUCCEEDED, result=result, log="任务成功")
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback

        store.update_task(
            task_id,
            status=TaskStatus.FAILED,
            error=f"{exc}\n{traceback.format_exc()}",
            log=f"任务失败: {exc}",
        )
        return 1


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m jiuan.workers.job <task_id>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(execute(sys.argv[1]))


if __name__ == "__main__":
    main()


"""标注抽象层：统一接口 + 可插拔后端（对标久安的标注平台层）。

- 默认后端 jsonl（单机零依赖，当前已跑通的轻量适配层）。
- 可选后端 label_studio（占位框架，部署 LS 后接管多人协作/审核）。
- 切换后端只改配置 annotation.backend，业务代码(app.py)不变。

统一接口：create_task / list_tasks / preview / save_annotations /
          coverage / commit / resolved_tracking。
"""
from __future__ import annotations

from ..common import load_config
from . import bridge_jsonl, bridge_label_studio

_BACKENDS = {
    "jsonl": bridge_jsonl,
    "label_studio": bridge_label_studio,
}


def backend_name() -> str:
    return ((load_config().get("annotation") or {}).get("backend") or "jsonl").lower()


def _backend():
    name = backend_name()
    return _BACKENDS.get(name, bridge_jsonl)


def create_task(name="gap", max_items=None, log=None) -> dict:
    return _backend().create_task(name, max_items, log=log)


def list_tasks() -> list:
    return _backend().list_tasks()


def preview(task_id: str) -> dict:
    return _backend().preview(task_id)


def save_annotations(task_id: str, annotations: dict, log=None) -> dict:
    return _backend().save_annotations(task_id, annotations, log=log)


def coverage() -> dict:
    return _backend().coverage()


def commit(task_id: str, name="annotated", parent=None, hard_weight=1, log=None) -> dict:
    return _backend().commit(task_id, name, parent, hard_weight=hard_weight, log=log)


def resolved_tracking() -> list:
    return _backend().resolved_tracking()

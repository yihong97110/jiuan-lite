"""标注后端 · Label Studio 桥接（占位/mock 框架实现）。

定位：对标久安"独立标注平台 + 多人协作 + 审核流程"。当前为占位实现，
所有调用返回 mock 或抛 NotImplementedError，确保集成侧代码可先行开发。
接口签名与 bridge_jsonl 完全一致，切后端只改配置 annotation.backend。

上线步骤：
  1. 部署 LS 服务（docker run -p 8080:8080 heartexlabs/label-studio）
  2. LS Web UI 建项目，拿 project_id + api_key（配置 annotation.ls_base_url / ls_api_key）
  3. 把下方 TODO 处的 mock 替换为真实 LS REST API 调用
  4. LS 项目设置 Webhook 指向 /annotation/webhook/ls，实现"标完即回灌"

LS REST API 速查（v1）：
  POST /api/projects/                         建项目
  POST /api/projects/{id}/import              批量导入任务(tasks)
  GET  /api/projects/{id}/export?exportType=JSON   导出标注结果
  POST /api/projects/{id}/webhooks/           注册完成回调
"""
from __future__ import annotations

from typing import Optional

from ..common import load_config

_BACKEND_NAME = "label_studio"


def _cfg() -> dict:
    return (load_config().get("annotation") or {})


def _require_live():
    cfg = _cfg()
    if not cfg.get("ls_base_url"):
        raise NotImplementedError(
            "Label Studio 后端未配置(annotation.ls_base_url 为空)。"
            "当前为占位实现：请部署 LS 服务并填写 ls_base_url/ls_api_key 后再启用。"
        )


def create_task(name: str = "gap", max_items: Optional[int] = None, log=None) -> dict:
    """从 gap 创建 LS 标注项目并逐条导入任务，返回项目信息。"""
    log = log or (lambda _m: None)
    cfg = _cfg()
    if not cfg.get("ls_base_url"):
        # —— mock：无 LS 服务时返回占位，方便前端联调 ——
        log("Label Studio 未配置，返回 mock 项目(占位)")
        return {
            "task_id": f"ls-{name}-mock",
            "backend": _BACKEND_NAME,
            "ls_project_id": 1,
            "ls_url": "http://label-studio:8080/projects/1",
            "count": 0,
            "pending": 0,
            "mock": True,
        }
    # TODO(live): 收集 gap → POST /api/projects/ → POST /api/projects/{id}/import
    _require_live()


def list_tasks() -> list:
    """列出 LS 项目及其标注进度。"""
    cfg = _cfg()
    if not cfg.get("ls_base_url"):
        return []
    # TODO(live): GET /api/projects/ → 逐项目取 task/annotation 计数
    _require_live()


def preview(task_id: str) -> dict:
    """预览某 LS 项目的标注任务与已标状态。"""
    _require_live()


def save_annotations(task_id: str, annotations: dict, log=None) -> dict:
    """LS 侧标注由 LS Web UI 完成，本地不直接写；此处仅占位。"""
    raise NotImplementedError(
        "Label Studio 后端的标注在 LS Web UI 完成，无需本地保存；"
        "请在 LS 界面标注后走 commit 拉取。"
    )


def coverage() -> dict:
    """标注覆盖度（LS：可由项目统计聚合）。"""
    cfg = _cfg()
    if not cfg.get("ls_base_url"):
        return {"total": 0, "annotated": 0, "pending": 0, "overall_pct": 0.0,
                "by_gap_type": [], "by_iteration": [], "backend": _BACKEND_NAME}
    # TODO(live): 聚合各项目 GET /api/projects/{id} 的 task_number/num_tasks_with_annotations
    _require_live()


def commit(task_id: str, name: str = "annotated", parent: Optional[str] = None,
           hard_weight: int = 1, log=None) -> dict:
    """从 LS 拉取标注结果并回灌为训练集(经 dataprep 增量继承)。"""
    log = log or (lambda _m: None)
    # TODO(live): GET /api/projects/{id}/export?exportType=JSON → 解析 → dataprep.run
    _require_live()


def resolved_tracking() -> list:
    """跨版本 gap 解决追踪（与 jsonl 后端复用同一份 resolved_log）。"""
    from . import bridge_jsonl
    return bridge_jsonl.resolved_tracking()

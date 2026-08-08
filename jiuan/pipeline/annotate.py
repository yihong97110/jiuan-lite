"""[已迁移] 标注逻辑已抽象到 jiuan.annotation 包(可插拔后端)。

本模块保留为向后兼容的转发层：旧代码 `from .pipeline import annotate`
仍可用，实际调用走 jiuan.annotation 的统一分发器(默认 jsonl 后端)。
新代码请直接用 `from jiuan import annotation`。
"""
from __future__ import annotations

from .. import annotation as _anno

create_task = _anno.create_task
list_tasks = _anno.list_tasks
preview = _anno.preview
save_annotations = _anno.save_annotations
coverage = _anno.coverage
commit = _anno.commit
resolved_tracking = _anno.resolved_tracking

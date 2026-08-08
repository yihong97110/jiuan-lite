"""SQLite-backed task store (task lifecycle + logs)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Optional

from .common import DB_PATH
from .schemas import Stage, Task, TaskStatus

# 单一可重入锁串行化读写，避免 "database is locked" 与读到半写状态。
# 说明：SQLite 适合 PoC；P2 规模化应换 PostgreSQL + 连接池。
_lock = threading.RLock()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    with _lock, _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                params TEXT NOT NULL,
                result TEXT NOT NULL,
                error TEXT NOT NULL,
                logs TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        # 平滑輁移：旧库补 progress 列
        cols = {r["name"] for r in c.execute("PRAGMA table_info(tasks)").fetchall()}
        if "progress" not in cols:
            c.execute("ALTER TABLE tasks ADD COLUMN progress TEXT NOT NULL DEFAULT ''")


def _row_to_task(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        stage=Stage(row["stage"]),
        status=TaskStatus(row["status"]),
        params=json.loads(row["params"]),
        result=json.loads(row["result"]),
        error=row["error"],
        progress=(row["progress"] if "progress" in row.keys() else ""),
        logs=json.loads(row["logs"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def create_task(stage: Stage, params: dict[str, Any]) -> Task:
    now = time.time()
    task = Task(
        id=uuid.uuid4().hex[:12],
        stage=stage,
        status=TaskStatus.PENDING,
        params=params,
        created_at=now,
        updated_at=now,
    )
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO tasks (id, stage, status, params, result, error, progress, logs, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                task.id,
                task.stage.value,
                task.status.value,
                json.dumps(task.params, ensure_ascii=False),
                json.dumps(task.result, ensure_ascii=False),
                task.error,
                task.progress,
                json.dumps(task.logs, ensure_ascii=False),
                task.created_at,
                task.updated_at,
            ),
        )
    return task


def get_task(task_id: str) -> Optional[Task]:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return _row_to_task(row) if row else None


def list_tasks(stage: Optional[Stage] = None) -> list[Task]:
    q = "SELECT * FROM tasks"
    args: tuple = ()
    if stage:
        q += " WHERE stage=?"
        args = (stage.value,)
    q += " ORDER BY updated_at DESC"  # 按完成/最近更新时间由晚到早
    with _lock, _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [_row_to_task(r) for r in rows]


def update_task(
    task_id: str,
    *,
    status: Optional[TaskStatus] = None,
    result: Optional[dict] = None,
    error: Optional[str] = None,
    progress: Optional[str] = None,
    log: Optional[str] = None,
) -> None:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            return
        task = _row_to_task(row)
        if status is not None:
            task.status = status
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        if progress is not None:
            task.progress = progress
        if log is not None:
            task.logs.append(f"[{time.strftime('%H:%M:%S')}] {log}")
        c.execute(
            "UPDATE tasks SET status=?, result=?, error=?, progress=?, logs=?, updated_at=? WHERE id=?",
            (
                task.status.value,
                json.dumps(task.result, ensure_ascii=False),
                task.error,
                task.progress,
                json.dumps(task.logs, ensure_ascii=False),
                time.time(),
                task_id,
            ),
        )

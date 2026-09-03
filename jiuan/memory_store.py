# -*- coding: utf-8 -*-
"""Agent 记忆存储：短期内存 + 长期 SQLite 双层记忆。

架构：
    短期记忆（RAM）                     长期记忆（SQLite data/jiuan.db）
  ┌──────────────────────┐          ┌──────────────────────────────┐
  │ _sessions / MemorySaver│  每轮   │ agent_messages：全量对话历史   │
  │ 滑动窗口 memory_window │ ──────> │   -> 平台重启后恢复会话上下文  │
  │ 会话内毫秒级访问       │  写入   │ agent_memories：关键事实      │
  └──────────────────────┘          │   -> 跨会话记住用户核心信息    │
           ↑ 恢复                   │  （省份/分数/偏好/身份等）     │
           └────────────────────────└──────────────────────────────┘

设计要点：
- 短期记忆负责"当前对话的上下文连贯"（窗口截断防 token 爆炸）
- agent_messages 负责"历史可回溯"（重启不丢）
- agent_memories 负责"跨会话知识沉淀"（新会话自动注入，让 Agent 记住用户）
- 事实提取：规则预筛（含"我"/数字/偏好词）+ LLM 结构化提取，避免每轮都调 API
- 与 store.py 共用同一 SQLite 文件（WAL 模式，独立锁不互相干扰）
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from typing import Optional

from .common import DB_PATH

_lock = threading.RLock()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_tables() -> None:
    """建表（幂等，app 启动时调用）。"""
    with _lock, _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_calls TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            )
            """
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_session ON agent_messages(session_id, id)"
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,
                fact TEXT NOT NULL,
                source_session TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                UNIQUE(scope, fact)
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_sessions (
                session_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL DEFAULT '',
                system_prompt TEXT NOT NULL DEFAULT '',
                memory_window INTEGER NOT NULL DEFAULT 10,
                created_at REAL NOT NULL
            )
            """
        )


def save_session_meta(session_id: str, model_id: str = "", system_prompt: str = "", memory_window: int = 10) -> None:
    """持久化会话元数据（重启后恢复 model_id/提示词/窗口用，长期记忆按 model_id 隔离的关键）。"""
    with _lock, _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO agent_sessions (session_id, model_id, system_prompt, memory_window, created_at) "
            "VALUES (?,?,?,?,?)",
            (session_id, model_id or "", (system_prompt or "")[:20000], int(memory_window or 10), time.time()),
        )


def load_session_meta(session_id: str) -> dict | None:
    """读取会话元数据；不存在返回 None。"""
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT session_id, model_id, system_prompt, memory_window, created_at "
            "FROM agent_sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "session_id": row["session_id"],
        "model_id": row["model_id"] or "",
        "system_prompt": row["system_prompt"] or "",
        "memory_window": int(row["memory_window"] or 10),
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# 对话历史持久化（agent_messages）
# ---------------------------------------------------------------------------

def save_message(session_id: str, role: str, content: str, tool_calls: str = "") -> None:
    """写入一条对话消息（user/assistant）。每轮 chat 结束时调用。"""
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO agent_messages (session_id, role, content, tool_calls, created_at) "
            "VALUES (?,?,?,?,?)",
            (session_id, role, content[:20000], tool_calls[:5000], time.time()),
        )


def load_messages(session_id: str, limit: int = 50) -> list[dict]:
    """加载会话历史（重启恢复用）。返回按时间正序的消息列表。"""
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT role, content FROM agent_messages WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def list_sessions_stored() -> list[dict]:
    """列出 SQLite 中有历史的会话（含消息数、最后活跃时间）。"""
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT session_id, COUNT(*) AS n, MAX(created_at) AS last_at "
            "FROM agent_messages GROUP BY session_id ORDER BY last_at DESC LIMIT 100"
        ).fetchall()
    return [
        {"session_id": r["session_id"], "messages": r["n"], "last_active": r["last_at"]}
        for r in rows
    ]


def delete_session_messages(session_id: str) -> int:
    """删除会话的全部历史。返回删除条数。"""
    with _lock, _conn() as c:
        cur = c.execute(
            "DELETE FROM agent_messages WHERE session_id=?", (session_id,)
        )
        return cur.rowcount


# ---------------------------------------------------------------------------
# 长期记忆事实（agent_memories）
# ---------------------------------------------------------------------------

def save_fact(scope: str, fact: str, source_session: str = "") -> bool:
    """写入一条长期记忆事实（scope+fact 去重）。返回是否新写入。"""
    fact = fact.strip()
    if not fact:
        return False
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT OR IGNORE INTO agent_memories (scope, fact, source_session, created_at) "
            "VALUES (?,?,?,?)",
            (scope, fact[:1000], source_session, time.time()),
        )
        return cur.rowcount > 0


def load_facts(scope: str, limit: int = 30) -> list[str]:
    """加载某个记忆域的全部事实（注入系统提示词用）。"""
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT fact FROM agent_memories WHERE scope=? ORDER BY id DESC LIMIT ?",
            (scope, limit),
        ).fetchall()
    return [r["fact"] for r in rows]


def delete_facts(scope: str) -> int:
    """清空记忆域。返回删除条数。"""
    with _lock, _conn() as c:
        cur = c.execute("DELETE FROM agent_memories WHERE scope=?", (scope,))
        return cur.rowcount


# ---------------------------------------------------------------------------
# 事实提取：规则预筛 + LLM 结构化提取
# ---------------------------------------------------------------------------

# 预筛正则：包含自我陈述/数字/偏好/身份的句子才值得提取
_FACT_HINT = re.compile(
    r"(我是|我叫|我家|我住|我考|我的分数|我分数|我孩子|我想学|我喜欢|我偏好|我不喜欢|"
    r"我打算|我计划|我预算|我希望|我倾向|"
    r"\d{3}\s*分|高考|理科|文科|物理类|历史类|"
    r"考研|工作|职业|专业方向)"
)


def _worth_extracting(message: str) -> bool:
    """规则预筛：消息是否可能包含值得长期记住的信息（省 LLM 调用）。"""
    return bool(_FACT_HINT.search(message))


def extract_facts_with_llm(
    message: str,
    cfg: dict,
    llm_chat=None,
) -> list[str]:
    """用 LLM 从用户消息中提取长期记忆事实。

    Args:
        message: 本轮用户消息（只提取用户说的，回答不用）
        cfg: 平台配置（取 chat 端点）
        llm_chat: 可选的聊天函数 fn(system, user) -> str（复用调用方连接池）

    Returns:
        事实列表（如 ["用户是河南理科考生", "高考587分"]），空列表=无新事实
    """
    system = (
        "从用户消息中提取值得长期记住的关键事实（用户身份、分数、省份、偏好、目标）。"
        "只输出 JSON 数组，如 [\"用户是河南理科考生\", \"高考587分\"]。"
        "没有值得记的就输出 []。不要解释。每条事实不超过30字。"
    )
    try:
        if llm_chat is not None:
            raw = llm_chat(system, message)
        else:
            raw = _default_llm_chat(system, message, cfg)
    except Exception:
        return []

    # 解析 JSON 数组（容错：截取第一个 [ 到最后一个 ]）
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        facts = json.loads(m.group(0))
        if isinstance(facts, list):
            return [str(f).strip() for f in facts if str(f).strip()][:5]
    except Exception:
        pass
    return []


def _default_llm_chat(system: str, user: str, cfg: dict) -> str:
    """兜底 LLM 调用：直接 HTTP 调 chat 端点（DeepSeek 云端）。"""
    import os
    import urllib.request

    chat_cfg = cfg.get("chat", {}) or {}
    base_url = (chat_cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        return "[]"

    env_name = chat_cfg.get("api_key_env") or ""
    key = os.environ.get(env_name, "") if env_name else ""
    if not key:
        key = chat_cfg.get("api_key") or ""

    payload = {
        "model": chat_cfg.get("model", "deepseek-v4-flash"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 200,
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}" if key else "Bearer EMPTY",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"].get("content") or "[]"

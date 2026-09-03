"""LangChain Agent：为训练后的模型提供多轮对话与工具调用能力（LangGraph 版）。

基于 LangChain + LangGraph 构建，覆盖以下工程化能力：
- 多轮对话与上下文记忆管理（MemorySaver checkpointer 持久化 + 滑动窗口截断）
- 工具调用：RAG 知识检索、模型仓库查询、血缘追踪
- 流式输出（SSE，逐 token 返回）
- 可对接任意 OpenAI 兼容端点（vLLM 本地服务 / DeepSeek API / 火山方舟等）

LangChain/LangGraph 组件映射：
  ChatOpenAI            -> 统一封装 LLM 接口（对接 vLLM/DeepSeek/方舟）
  @tool                 -> 工具定义，封装现有 rag_backend / registry 能力
  create_react_agent    -> LangGraph 预置 ReAct Agent（状态机：agent→tools→agent循环）
  MemorySaver           -> checkpointer 持久化记忆（thread_id 隔离 + 自动历史恢复）
  astream_events(v2)    -> 流式 token 输出 + 工具调用事件追踪

设计：
- 每个会话(session)用 thread_id 隔离，MemorySaver 自动存取历史
- 工具直接调用现有 pipeline 能力，不重复造轮子
- langchain/langgraph 未安装时模块可正常导入，API 层返回友好提示
- API 不支持 function calling 时降级为普通对话（_direct_chat）
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from ..common import DATA, load_config
from . import rag_backend
from .. import registry

# ---------------------------------------------------------------------------
# 可用性检测
# ---------------------------------------------------------------------------

try:
    from langchain_openai import ChatOpenAI  # noqa: F401
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# LangGraph 可用性检测（独立于 langchain-openai）
try:
    from langgraph.prebuilt import create_react_agent  # noqa: F401
    from langgraph.checkpoint.memory import MemorySaver  # noqa: F401
    from langgraph.graph import END  # noqa: F401
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False


def available() -> bool:
    """LangChain 是否可用（已安装 langchain-openai）。"""
    return LANGCHAIN_AVAILABLE


def langgraph_available() -> bool:
    """LangGraph 是否可用（已安装 langgraph）。"""
    return LANGGRAPH_AVAILABLE


# ---------------------------------------------------------------------------
# 会话存储 + LangGraph 双层记忆（Checkpointer 短期 + Store 长期）
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}

# 短期记忆：LangGraph Checkpointer（SqliteSaver，按 thread_id = session_id 隔离）
_checkpointer = None
# 长期记忆：LangGraph Store（SqliteStore，按 namespace = ("memories", model_id) 隔离）
_store = None


def _get_checkpointer():
    """获取全局 SqliteSaver checkpointer（惰性初始化）。

    短期记忆实现 = Checkpointer + Thread ID：
    - thread_id = session_id，每个会话一条独立状态线
    - 每轮对话自动持久化到 data/agent_threads.db（WAL），平台重启后恢复
    - LangGraph 在 checkpoint 中保存完整消息状态，无需手动管理 messages
    """
    global _checkpointer
    if _checkpointer is None:
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            from ..common import DATA
            conn = sqlite3.connect(DATA / "agent_threads.db", check_same_thread=False, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            _checkpointer = SqliteSaver(conn)
        except Exception:
            # langgraph-checkpoint-sqlite 未安装时回退内存版（重启丢失）
            from langgraph.checkpoint.memory import MemorySaver
            _checkpointer = MemorySaver()
    return _checkpointer


def _get_store():
    """获取全局 SqliteStore（惰性初始化）。

    长期记忆实现 = LangGraph Store + Namespace（按模型隔离）：
    - namespace = ("memories", model_id)：同一模型的会话共享记忆，不同模型互相隔离
    - store.put / store.search 官方 API，持久化到 data/agent_store.db
    - prompt callable 每轮动态注入最新记忆（remember 后立即生效）
    """
    global _store
    if _store is None:
        import sqlite3
        from langgraph.store.sqlite import SqliteStore
        from ..common import DATA
        # isolation_level=None（autocommit）：SqliteStore 内部手动 BEGIN/COMMIT，
        # 默认隐式事务会导致 "cannot start a transaction within a transaction"
        conn = sqlite3.connect(
            DATA / "agent_store.db", check_same_thread=False, timeout=30,
            isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        _store = SqliteStore(conn)
        _store.setup()  # 建表（幂等）
    return _store


def _memory_namespace(model_id: str = "") -> tuple:
    """长期记忆 namespace：按模型隔离（同一模型共享，跨模型不可见）。"""
    return ("memories", (model_id or "global").strip())


def _get_memory_saver():
    """兼容旧引用：返回 checkpointer（SqliteSaver 优先）。"""
    return _get_checkpointer()


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------

def _chat_cfg(cfg: dict | None = None) -> dict:
    """读取 chat 配置；若未单独配置 base_url，回退到 judge 配置（共用端点）。"""
    cfg = cfg or load_config()
    chat = dict(cfg.get("chat", {}) or {})
    if not chat.get("base_url"):
        judge = cfg.get("judge", {}) or {}
        # 合并：chat 优先，judge 补充
        merged = dict(judge)
        merged.update(chat)
        chat = merged
    return chat


def _resolve_api_key(cfg: dict) -> str:
    """解析 API Key：优先环境变量，其次配置明文，兜底 EMPTY。"""
    env_name = cfg.get("api_key_env") or ""
    if env_name:
        val = os.environ.get(env_name, "")
        if val:
            return val
    return cfg.get("api_key") or "EMPTY"


# ---------------------------------------------------------------------------
# LLM 工厂
# ---------------------------------------------------------------------------

def _get_llm(cfg: dict | None = None, streaming: bool = False, model_id: str = ""):
    """创建 LangChain ChatOpenAI 实例，指向 OpenAI 兼容端点。

    双后端路由（按 model_id 自动切换）：
    - model_id 为空（前端选"DeepSeek 默认"）：走云端 API（DeepSeek，function calling 强，
      不占本地 GPU--这使训练工具可以安全停掉本地 vLLM 释放显存而不中断对话）
    - model_id 非空（选择已训练模型）：走本地 vLLM（chat.vllm_base_url），
      从 registry 读取 served_model_name（多 LoRA 场景切换模型）

    兼容旧配置：未配置 vllm_base_url 时，两种情况都走 chat.base_url。
    """
    from langchain_openai import ChatOpenAI

    chat_cfg = _chat_cfg(cfg)

    if model_id:
        # --- 本地 vLLM 分支：承载已训练模型（LoRA） ---
        base_url = chat_cfg.get("vllm_base_url") or chat_cfg.get("base_url", "http://127.0.0.1:8001/v1")
        model_name = chat_cfg.get("vllm_model", "")
        api_key = "EMPTY"
        try:
            from .. import registry
            for m in registry.list_models():
                if m.get("model_id") == model_id:
                    # 优先用 meta 里记录的 served_model_name
                    model_name = m.get("served_model_name") or m.get("name") or model_id
                    break
        except Exception:
            pass  # 读取失败就用配置里的 vllm_model
    else:
        # --- 云端默认分支：DeepSeek（对话+工具调用全在云端，不占本地 GPU） ---
        base_url = chat_cfg.get("base_url", "https://api.deepseek.com/v1")
        model_name = chat_cfg.get("model", "deepseek-v4-flash")
        api_key = _resolve_api_key(chat_cfg)

    return ChatOpenAI(
        base_url=base_url,
        model=model_name,
        api_key=api_key,
        temperature=float(chat_cfg.get("temperature", 0.7)),
        max_tokens=int(chat_cfg.get("max_tokens", 1024)),
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# 联网搜索（Bing 中文 HTML 抓取，无需 API Key）
# ---------------------------------------------------------------------------

def _web_search(query: str, max_results: int = 5) -> str:
    """联网搜索：Tavily API（有 key 时）→ 360搜索 → Bing 兜底。

    实测（AutoDL 服务器网络环境）：
    - DuckDuckGo 被墙（000 返回）、百度/搜狗触发反爬验证
    - Bing 中英文版对长中文 query 相关性极差（泛年份结果）
    - 360（so.com）中文相关性最好，直接命中分数线类时效问题
    - Tavily 为 LLM 专用搜索 API，质量最高（需 TAVILY_API_KEY，免费1000次/月）
    """
    # --- 优先 Tavily（配置了 key 时）---
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            result = _tavily_search(query, tavily_key, max_results)
            if result:
                return result
        except Exception:
            pass  # Tavily 失败静默回退

    # --- 主力：360 搜索（中文相关性最好）---
    try:
        result = _so_search(query, max_results)
        if result and "未找到" not in result[:30] and "搜索失败" not in result[:30]:
            return result
    except Exception:
        pass

    # --- 兜底：Bing ---
    return _bing_search(query, max_results)


def _so_search(query: str, max_results: int = 5) -> str:
    """360 搜索（so.com）结果抓取与解析。

    结果页结构：<li class="res-list"><h3><a href>标题</a></h3><p class="res-desc">摘要</p></li>
    中文 query 相关性显著优于 Bing（实测分数线类问题直接命中）。
    """
    import html as _html
    import re as _re
    import urllib.parse
    import urllib.request

    url = "https://www.so.com/s?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
    with urllib.request.urlopen(req, timeout=12) as resp:
        page = resp.read().decode("utf-8", errors="replace")

    def _clean(s: str) -> str:
        return _html.unescape(_re.sub(r"<[^>]+>", "", s)).strip()

    blocks = _re.findall(r'<li class="res-list".*?</li>', page, _re.S)
    results = []
    for block in blocks:
        tm = _re.search(r'<h3[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, _re.S)
        if not tm:
            continue
        link, title = tm.group(1), _clean(tm.group(2))
        pm = _re.search(r'<p class="res-desc[^"]*"[^>]*>(.*?)</p>', block, _re.S)
        snippet = _clean(pm.group(1)) if pm else ""
        if not title:
            continue
        entry = f"- {title}"
        if snippet:
            entry += f"\n  {snippet[:280]}"
        entry += f"\n  来源: {link[:150]}"
        results.append(entry)
        if len(results) >= max_results:
            break

    if not results:
        return f"未找到与「{query}」相关的搜索结果。"
    return "\n\n".join(results)


def _tavily_search(query: str, api_key: str, max_results: int = 5) -> str:
    """Tavily Search API（LLM 专用，返回干净摘要）。"""
    import urllib.request

    payload = json.dumps({
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "include_answer": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    parts = []
    if data.get("answer"):
        parts.append(f"[摘要] {data['answer']}")
    for r in (data.get("results") or [])[:max_results]:
        title = r.get("title", "")
        snippet = r.get("content", "")[:300]
        url = r.get("url", "")
        parts.append(f"- {title}\n  {snippet}\n  来源: {url}")
    return "\n\n".join(parts)


def _bing_search(query: str, max_results: int = 5) -> str:
    """Bing 中文版搜索结果抓取与解析。

    结果页结构：<li class="b_algo"><h2><a href="URL">标题</a></h2><p>摘要</p></li>
    纯正则解析，无 bs4 依赖。
    """
    import html as _html
    import re as _re
    import urllib.parse
    import urllib.request

    url = ("https://cn.bing.com/search?q=" + urllib.parse.quote(query)
           + "&count=" + str(max_results * 2) + "&mkt=zh-CN&ensearch=0")
    req = urllib.request.Request(url, headers={
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"联网搜索失败: {exc}"

    # 提取 b_algo 结果块
    blocks = _re.findall(r'<li class="b_algo".*?</li>', html, _re.S)[: max_results * 2]
    results = []
    for block in blocks:
        # 标题与链接
        m = _re.search(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>', block, _re.S)
        if not m:
            continue
        link = m.group(1)
        title = _html.unescape(_re.sub(r"<[^>]+>", "", m.group(2))).strip()
        # 摘要（b_caption p 或块内任意 p）
        pm = _re.search(r'<p[^>]*>(.*?)</p>', block, _re.S)
        snippet = _html.unescape(_re.sub(r"<[^>]+>", "", pm.group(1))).strip() if pm else ""
        if title:
            entry = f"- {title}"
            if snippet:
                entry += f"\n  {snippet[:280]}"
            entry += f"\n  来源: {link[:150]}"
            results.append(entry)
        if len(results) >= max_results:
            break

    if not results:
        return f"未找到与「{query}」相关的搜索结果。"
    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# Skill 系统：预定义技能（prompt 模板 + 参数填充）
# ---------------------------------------------------------------------------

_skills: dict[str, dict] = {}


def register_skill(name: str, description: str, template: str, category: str = "general") -> dict:
    """注册一个 Skill（预定义 prompt 模板）。

    Args:
        name: 技能名称（如 "summarize"）
        description: 技能描述
        template: prompt 模板，用 {param} 占位
        category: 分类（general / analysis / writing / code）

    Returns:
        注册的技能信息
    """
    skill = {
        "name": name,
        "description": description,
        "template": template,
        "category": category,
        "registered_at": time.time(),
    }
    _skills[name] = skill
    return skill


def list_skills() -> list[dict]:
    """列出所有已注册的技能。"""
    return list(_skills.values())


def invoke_skill(name: str, params: dict) -> str:
    """调用技能：用参数填充模板并返回完整 prompt。"""
    skill = _skills.get(name)
    if not skill:
        return f"技能 {name} 不存在。可用技能: {list(_skills.keys())}"
    try:
        return skill["template"].format(**params)
    except KeyError as exc:
        return f"参数缺失: {exc}。模板需要: {skill['template']}"


def _register_default_skills():
    """注册内置默认技能集。"""
    register_skill(
        "summarize", "将长文本总结为要点",
        "请将以下内容总结为3-5个要点，保持核心信息：\n\n{text}",
        "analysis",
    )
    register_skill(
        "translate", "中英互译",
        "请将以下内容翻译为{target_lang}：\n\n{text}",
        "writing",
    )
    register_skill(
        "explain_code", "解释代码功能",
        "请解释以下代码的功能、逻辑和关键点：\n\n```{language}\n{code}\n```",
        "code",
    )
    register_skill(
        "eval_analysis", "分析模型评测报告",
        "请基于以下评测数据给出改进建议：\n\n"
        "综合分: {overall}\nBad Case率: {bad_case_rate}\n"
        "幻觉率: {hallucination_rate}\nGap数量: {gap_count}\n\n"
        "请从准确性、完整性和幻觉三个维度分析，并给出下一轮迭代的数据补充建议。",
        "analysis",
    )
    register_skill(
        "gap_to_annotation", "将Gap薄弱点转为标注任务描述",
        "以下是需要补充数据的薄弱点：\n{gaps}\n\n"
        "请为每个薄弱点设计3-5条训练用QA，要求：\n"
        "1. 覆盖不同场景和表达方式\n"
        "2. 答案准确、格式规范\n"
        "3. 直接可用于SFT训练",
        "analysis",
    )


# 在模块加载时注册默认技能
_register_default_skills()


# ---------------------------------------------------------------------------
# MCP 客户端：连接外部 MCP 服务器并暴露工具
# 协议：JSON-RPC 2.0 over stdio（initialize -> tools/list -> tools/call）
# ---------------------------------------------------------------------------

_mcp_servers: dict[str, dict] = {}
_MCP_STORE = DATA / "mcp_servers.json"


def _mcp_load() -> None:
    """启动时从 data/mcp_servers.json 恢复已注册的 MCP 服务器（持久化）。"""
    try:
        if _MCP_STORE.exists():
            for name, info in json.loads(_MCP_STORE.read_text(encoding="utf-8")).items():
                info.setdefault("status", "registered")
                info.setdefault("tools", [])
                _mcp_servers[name] = info
    except Exception:
        pass  # 损坏则忽略，空列表启动


def _mcp_save() -> None:
    """持久化 MCP 服务器配置（name/command/description/tools）。"""
    try:
        _MCP_STORE.parent.mkdir(parents=True, exist_ok=True)
        _MCP_STORE.write_text(
            json.dumps(_mcp_servers, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


_mcp_load()


def _mcp_parse_command(command: str) -> tuple[str, list[str]]:
    """'python -m mcp_server_fetch' -> ('python', ['-m', 'mcp_server_fetch'])"""
    parts = (command or "").split()
    return (parts[0] if parts else ""), parts[1:]


def _mcp_run(coro):
    """在同步上下文执行 async 协程（Agent 工具是同步的）。

    已有事件循环时（流式 SSE 路径）用独立 loop，避免嵌套 run 报错。
    """
    try:
        asyncio.get_running_loop()
        # 已在 loop 中：新开线程跑独立 loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result(timeout=60)
    except RuntimeError:
        return asyncio.run(coro)


def _mcp_list_tools(command: str) -> list[dict]:
    """连接 MCP 服务器并发现其工具列表（tools/list）。

    Returns:
        [{"name": 工具名, "description": 描述}]

    Raises:
        Exception: 连接/握手失败（由调用方处理）
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cmd, args = _mcp_parse_command(command)

    async def _run():
        async with stdio_client(StdioServerParameters(command=cmd, args=args)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                resp = await session.list_tools()
                return [
                    {"name": t.name, "description": (t.description or "")[:300]}
                    for t in resp.tools
                ]

    return _mcp_run(_run())


def _mcp_call_tool(command: str, tool_name: str, tool_args: dict) -> str:
    """连接 MCP 服务器并调用指定工具（tools/call）。

    Args:
        command: 服务器启动命令
        tool_name: MCP 工具名（服务器侧注册的）
        tool_args: 工具参数（dict）

    Returns:
        工具结果文本
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cmd, args = _mcp_parse_command(command)

    async def _run():
        async with stdio_client(StdioServerParameters(command=cmd, args=args)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_args)
                parts = []
                for c in (result.content or []):
                    if hasattr(c, "text"):
                        parts.append(c.text)
                return "\n".join(parts) or "(空结果)"

    return _mcp_run(_run())


def register_mcp_server(name: str, command: str, description: str = "") -> dict:
    """注册一个 MCP 服务器（立即连接做工具发现，并持久化）。

    Args:
        name: 服务器名称
        command: 启动命令（如 "python -m mcp_server_fetch"）
        description: 描述

    Returns:
        注册信息（含发现的工具列表；连接失败时 status=error 但仍保存配置）
    """
    server = {
        "name": name,
        "command": command,
        "description": description,
        "status": "registered",
        "tools": [],
        "registered_at": time.time(),
    }
    # 真协议握手 + 工具发现（失败不阻断注册，便于先登记后排查）
    try:
        tools = _mcp_list_tools(command)
        server["tools"] = tools
        server["status"] = "loaded"
    except Exception as exc:
        server["status"] = "error"
        server["error"] = str(exc)[:200]
    _mcp_servers[name] = server
    _mcp_save()
    return server


def delete_mcp_server(name: str) -> dict:
    """删除一个 MCP 服务器配置（并持久化）。"""
    if name not in _mcp_servers:
        raise ValueError(f"MCP 服务器不存在: {name}")
    _mcp_servers.pop(name)
    _mcp_save()
    return {"deleted": name}


def test_mcp_server(name: str) -> dict:
    """测试指定 MCP 服务器连通性（重新握手 + 刷新工具列表）。"""
    info = _mcp_servers.get(name)
    if not info:
        raise ValueError(f"MCP 服务器不存在: {name}")
    try:
        tools = _mcp_list_tools(info["command"])
        info["tools"] = tools
        info["status"] = "loaded"
        info.pop("error", None)
    except Exception as exc:
        info["status"] = "error"
        info["error"] = str(exc)[:200]
    _mcp_save()
    return info


def list_mcp_servers() -> list[dict]:
    """列出所有已注册的 MCP 服务器（含各自工具清单）。"""
    return list(_mcp_servers.values())


def _load_mcp_tools() -> list:
    """把每个 MCP 服务器的真实工具包装为 LangChain Tool。

    优先走 MCP 真协议（mcp 库，逐工具包装，参数为 JSON 字符串）；
    mcp 库不可用时回退旧简化实现（每服务器一个 query 工具）。
    """
    tools = []
    try:
        from langchain_core.tools import tool as langchain_tool
    except Exception:
        return tools

    try:
        import mcp  # noqa: F401
        mcp_ok = True
    except ImportError:
        mcp_ok = False

    for server_name, server_info in _mcp_servers.items():
        if server_info.get("status") not in ("registered", "loaded"):
            continue
        cmd = server_info["command"]
        server_desc = server_info.get("description", server_name)

        if mcp_ok:
            # --- 真协议：每个发现的真实工具 -> 一个 LangChain tool ---
            for t in server_info.get("tools", []):
                tname, tdesc = t.get("name", ""), t.get("description", "")
                if not tname:
                    continue

                def _make(server_cmd, tool_name, tool_desc, sname):
                    @langchain_tool
                    def mcp_tool(arguments: str) -> str:
                        """调用 MCP 外部工具。

                        Args:
                            arguments: 工具参数的 JSON 字符串（无参数传 "{}"）
                        """
                        try:
                            args = json.loads(arguments) if arguments and arguments.strip() else {}
                        except json.JSONDecodeError:
                            return f"参数必须是合法 JSON: {arguments[:100]}"
                        try:
                            return _mcp_call_tool(server_cmd, tool_name, args)
                        except Exception as exc:
                            return f"MCP 调用失败({tool_name}): {str(exc)[:200]}"
                    mcp_tool.name = f"mcp_{sname}__{tool_name}"
                    mcp_tool.description = f"MCP工具[{sname}]: {tool_desc or tool_name}"
                    return mcp_tool

                tools.append(_make(cmd, tname, tdesc, server_name))
        else:
            # --- 回退：旧简化实现（无 mcp 库） ---
            def _make_simple(sname, scmd, sdesc):
                @langchain_tool
                def mcp_tool(query: str) -> str:
                    """通过 MCP 协议调用外部工具（简化模式）。"""
                    return _call_mcp_server(sname, scmd, query)
                mcp_tool.name = f"mcp_{sname}"
                mcp_tool.description = f"MCP工具({sdesc}): {sdesc}。传入query参数调用。"
                return mcp_tool

            tools.append(_make_simple(server_name, cmd, server_desc))

    return tools


def _call_mcp_server(server_name: str, command: str, query: str) -> str:
    """简化回退模式：通过子进程 stdin/stdout 与 MCP 服务器通信（无 mcp 库时）。"""
    import subprocess
    try:
        parts = command.split()
        proc = subprocess.run(
            parts,
            input=json.dumps({"method": "tools/call", "params": {"query": query}}),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()[:500]
        return f"MCP服务器 {server_name} 返回为空或出错。stderr: {proc.stderr[:200]}"
    except FileNotFoundError:
        return f"MCP服务器命令不存在: {command}"
    except subprocess.TimeoutExpired:
        return f"MCP服务器 {server_name} 超时。"
    except Exception as exc:
        return f"MCP调用失败: {exc}"


# ---------------------------------------------------------------------------
# 工具定义：封装现有平台能力为 LangChain Tool
# ---------------------------------------------------------------------------

def _stop_vllm_for_training() -> str:
    """训练前停止本地 vLLM 推理服务，释放 GPU 显存（3090 24GB 训练/推理互斥）。

    对话 LLM 走云端 DeepSeek 时可安全调用：停 vLLM 不影响当前会话。
    Windows 本地开发环境（无 vLLM）直接跳过。

    Returns:
        给用户看的状态说明（是否停止了 vLLM、显存释放情况）
    """
    import subprocess

    if os.name == "nt":
        return "（本地开发环境，跳过 GPU 检查）"

    try:
        result = subprocess.run(
            ["pgrep", "-f", "vllm serve"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return "GPU 空闲（vLLM 未运行），直接开始训练。"

        subprocess.run(["pkill", "-f", "vllm serve"], capture_output=True, timeout=10)
        # 等待显存释放（最多 30s，需 >= 16GB 空闲）
        for i in range(30):
            time.sleep(1)
            try:
                smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if smi.returncode == 0 and int(smi.stdout.strip()) >= 16000:
                    return (f"已自动停止本地 vLLM 推理服务并释放显存"
                            f"（空闲 {smi.stdout.strip()}MB）。训练完成后如需对话本地模型，请重启推理服务。")
            except Exception:
                break
        return ("已停止 vLLM。注意：显存释放可能仍在进行，训练任务已提交至调度队列。")
    except FileNotFoundError:
        return "（无 pgrep/nvidia-smi，跳过 GPU 检查）"
    except Exception as exc:
        return f"GPU 检查异常（继续提交训练）: {exc}"


def _build_tools(session_model_id: str = ""):
    """构建 Agent 可用工具集，每个工具直接调用现有 pipeline 函数。

    Args:
        session_model_id: 当前会话关联的模型ID（长期记忆 Store 按此隔离：
                          namespace = ("memories", model_id)，同一模型共享、
                          跨模型不可见）。由 chat()/chat_stream() 传入 session["model_id"]。
    """
    from langchain_core.tools import tool

    @tool
    def rag_search(query: str, collection: str = "default") -> str:
        """搜索 RAG 知识库，检索与问题相关的知识段落。当需要查找领域知识、法规条文、预案文档或专业术语时使用此工具。

        Args:
            query: 搜索关键词或问题
            collection: 知识库集合名，默认 "default"

        Returns:
            检索到的知识段落列表，每段包含内容、来源和相似度
        """
        docs = rag_backend.retrieve(query, top_k=3, collection=collection)
        if not docs:
            return "知识库中没有找到相关内容。"
        results = []
        for i, d in enumerate(docs, 1):
            results.append(
                f"[{i}] (来源: {d['source']}, 相似度: {d['score']})\n{d['text']}"
            )
        return "\n\n".join(results)

    @tool
    def list_trained_models() -> str:
        """列出平台上所有已训练的模型及其基本信息。当用户询问有哪些可用模型、模型状态时使用。

        Returns:
            模型列表，包含模型ID、名称、训练后端等信息
        """
        models = registry.list_models()
        if not models:
            return "当前没有已训练的模型。可以先通过平台的训练功能创建模型。"
        lines = []
        for m in models:
            mid = m.get("model_id", "N/A")
            name = m.get("name", "N/A")
            backend = m.get("backend", "N/A")
            lines.append(f"- 模型ID: {mid} | 名称: {name} | 后端: {backend}")
        return "已训练模型列表：\n" + "\n".join(lines)

    @tool
    def get_model_lineage(model_id: str) -> str:
        """查询指定模型的血缘信息，包括训练数据集、基座模型、训练参数、评测报告等。当用户想了解某个模型的训练过程或评测指标时使用。

        Args:
            model_id: 模型ID（可通过 list_trained_models 获取）

        Returns:
            模型的完整血缘链路信息（JSON 格式）
        """
        try:
            lineage = registry.lineage(model_id)
            return json.dumps(lineage, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"查询模型血缘失败: {exc}"

    @tool
    def web_search(query: str) -> str:
        """联网搜索获取实时信息。当用户询问最新新闻、实时数据、或知识库中没有的最新知识时使用此工具。

        Args:
            query: 搜索关键词

        Returns:
            搜索结果摘要列表
        """
        return _web_search(query)

    @tool
    def get_eval_report(model_id: str) -> str:
        """查询指定模型最新的评测报告，包括ROUGE/BLEU/Judge分数和Gap分析。当用户想了解模型表现如何、有哪些薄弱点时使用。

        Args:
            model_id: 模型ID

        Returns:
            评测报告摘要，包含各项指标和薄弱点数量
        """
        from ..common import REPORTS
        latest = None
        latest_time = 0
        for p in REPORTS.glob("*.json"):
            try:
                rep = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rep.get("model_id") != model_id:
                continue
            t = rep.get("created_at", 0)
            if t > latest_time:
                latest_time = t
                latest = rep
        if not latest:
            return f"未找到模型 {model_id} 的评测报告。"
        m = latest.get("metrics", {})
        gaps = latest.get("gaps", [])
        lines = [
            f"评测报告: {latest.get('dataset_id', 'N/A')}",
            f"ROUGE-L: {m.get('rouge_l_f', 'N/A')}",
            f"BLEU-1: {m.get('bleu_1', 'N/A')}",
            f"Judge综合分: {m.get('judge_overall', 'N/A')}",
            f"Bad Case率: {m.get('bad_case_rate', 'N/A')}",
            f"幻觉率: {m.get('hallucination_rate', 'N/A')}",
            f"Gap数量: {len(gaps)}",
        ]
        if gaps:
            lines.append("主要薄弱点:")
            for g in gaps[:5]:
                lines.append(f"  - [{g.get('gap_type','')}] {g.get('prompt','')[:30]}")
        return "\n".join(lines)

    @tool
    def get_task_status(task_id: str) -> str:
        """查询平台任务的执行状态（训练/推理/评测等）。当用户想了解某个任务是否完成、是否出错时使用。

        Args:
            task_id: 任务ID

        Returns:
            任务状态信息（状态、阶段、结果摘要）
        """
        from .. import store
        task = store.get_task(task_id)
        if not task:
            return f"任务 {task_id} 不存在。"
        return json.dumps({
            "id": task.id,
            "stage": task.stage.value,
            "status": task.status.value,
            "progress": task.progress,
            "error": task.error[:200] if task.error else "",
        }, ensure_ascii=False, indent=2)

    @tool
    def list_datasets() -> str:
        """列出平台上所有可用的训练数据集（含样本数、是否为评测集）。当用户想训练模型、或询问平台有哪些数据可用于训练时，先用此工具查询。

        Returns:
            数据集列表：dataset_id、样本数、角色（train/eval）、注册时间
        """
        from .. import registry
        try:
            rows = registry.list_datasets()
        except Exception as exc:
            return f"查询数据集失败: {exc}"
        if not rows:
            return "平台上暂无数据集。请先通过蒸馏或数据准备生成训练数据。"
        lines = ["可用数据集："]
        # 只显示最近 20 条（旧迭代数据集太多）
        for d in rows[-20:]:
            role = "评测集" if d.get("role") == "eval" else "训练集"
            lines.append(
                f"- {d.get('dataset_id')} | 样本数: {d.get('sample_count', '?')} | {role}"
            )
        return "\n".join(lines)

    @tool
    def start_training(
        dataset_id: str,
        direction: str = "",
        base_model: str = "7b",
        epochs: int = 3,
        lr: float = 2e-4,
        batch_size: int = 1,
        grad_accum: int = 4,
        lora_r: int = 8,
        lora_alpha: int = 16,
        max_seq_len: int = 1024,
        confirmed: bool = False,
    ) -> str:
        """启动大模型 LoRA 训练任务。必须先把推荐参数展示给用户并获得用户明确确认后才能调用（confirmed 必须为 True），严禁未确认就启动。

        训练会占用全部 GPU 显存：若本地 vLLM 推理服务正在运行会先自动停止以释放显存（对话不受影响）。

        Args:
            dataset_id: 训练数据集ID（可先用 list_datasets 查询）
            direction: 训练方向/领域描述（如"法律咨询专家"），用于动态生成模型名
            base_model: 基底模型，"0.5b"（快速验证）或 "7b"（效果优先）
            epochs: 训练轮数，小数据集（<50条）建议 3-10
            lr: 学习率，LoRA 建议 1e-4 ~ 5e-4
            batch_size: 批大小，3090 24GB 跑 7B 建议 1
            grad_accum: 梯度累积步数（等效批大小 = batch_size × grad_accum）
            lora_r: LoRA 秩，建议 4-16
            lora_alpha: LoRA alpha，建议为 lora_r 的 2 倍
            max_seq_len: 最大序列长度，建议 512-2048
            confirmed: 用户是否已明确确认参数（必须为 True 才真正执行训练）

        Returns:
            训练任务提交结果（含任务ID，可用 get_task_status 查询进度）
        """
        if not confirmed:
            return ("用户尚未确认训练参数，禁止启动训练。"
                    "请先向用户完整展示推荐参数并请求明确确认。")

        from .. import registry
        from ..schemas import Stage
        from ..workers import runner

        # 数据集校验
        try:
            ds_ids = [d.get("dataset_id") for d in registry.list_datasets()]
        except Exception:
            ds_ids = []
        if dataset_id not in ds_ids:
            avail = ", ".join(str(x) for x in ds_ids[-10:]) or "无"
            return f"数据集 {dataset_id} 不存在。可用数据集: {avail}"

        # GPU 冲突处理：3090 24GB 训练与推理互斥，先停 vLLM（对话走云端 DeepSeek 不受影响）
        gpu_note = _stop_vllm_for_training()

        params = {
            "dataset_id": dataset_id,
            "backend": "llamafactory",
            "method": "lora",
            "base_model": base_model,
            "epochs": int(epochs),
            "lr": float(lr),
            "batch_size": int(batch_size),
            "grad_accum": int(grad_accum),
            "lora_r": int(lora_r),
            "lora_alpha": int(lora_alpha),
            "max_seq_len": int(max_seq_len),
            "domain_direction": direction,
            # name 不传，让 train.py 根据 base_model+domain+血缘版本动态生成
        }
        try:
            task_id = runner.submit(Stage.TRAIN, params)
        except Exception as exc:
            return f"训练任务提交失败: {exc}"

        return (
            f"训练任务已提交成功！任务ID: {task_id}\n"
            f"配置: 基底={base_model} 数据集={dataset_id} epochs={epochs} lr={lr} "
            f"lora_r={lora_r}/{lora_alpha} batch={batch_size}x{grad_accum}\n"
            f"{gpu_note}\n"
            f"可用 get_task_status 查询进度；训练完成后模型将自动登记到平台模型仓库。"
        )

    @tool
    def recall_memory(model: str = "", limit: int = 30) -> str:
        """查看长期记忆内容：列出当前模型的记忆域中已记住的用户事实（身份、分数、省份、偏好、目标等）。

        当用户询问"你还记得我什么"、"你知道我哪些信息"、"查看记忆"、"我的资料是什么"时使用此工具。
        长期记忆基于 LangGraph Store（SQLite 持久化），按模型隔离：只有调用同一模型的会话共享记忆。

        Args:
            model: 模型ID（记忆按模型隔离）。为空则查当前会话的模型域（推荐）。
            limit: 最多返回多少条事实（默认30）

        Returns:
            记忆事实列表（含写入时间），空记忆时返回提示
        """
        import time as _time

        target_model = (model or session_model_id or "global").strip()
        try:
            store = _get_store()
            items = store.search(_memory_namespace(target_model), limit=limit)
        except Exception as exc:
            return f"长期记忆存储暂不可用: {exc}"
        if not items:
            return (f"模型「{target_model}」的记忆域当前为空。"
                    "用户在对话中透露身份/分数/偏好等信息后会自动存入长期记忆，"
                    "也可用 remember 工具主动记忆。")
        lines = [f"模型「{target_model}」记忆域共 {len(items)} 条长期记忆："]
        for i, it in enumerate(items, 1):
            text = it.value.get("text", "")
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(it.updated_at or it.created_at))
            lines.append(f"{i}. {text}（{ts}）")
        return "\n".join(lines)

    @tool
    def remember(fact: str) -> str:
        """主动将一条重要信息存入长期记忆（跨会话持久保存，按模型隔离）。

        当对话中出现值得长期记住的用户信息（身份、分数、省份、目标、明确表达的偏好），
        且该信息对未来对话有价值时使用。存储在当前模型的记忆域中：
        之后所有调用同一模型的会话都能看到；其他模型看不到。

        Args:
            fact: 要记住的事实，一句话（如"用户是河南理科考生，高考587分"）

        Returns:
            保存结果
        """
        import uuid as _uuid

        fact = fact.strip()
        if not fact:
            return "记忆内容为空，未保存。"
        target_model = (session_model_id or "global").strip()
        try:
            store = _get_store()
            key = str(_uuid.uuid4())[:8]
            store.put(_memory_namespace(target_model), key, {"text": fact})
            return f"已记住（模型「{target_model}」记忆域）: {fact}"
        except Exception as exc:
            return f"记忆保存失败: {exc}"

    # 收集内置工具
    builtin = [rag_search, list_trained_models, get_model_lineage,
              web_search, get_eval_report, get_task_status,
              list_datasets, start_training, recall_memory, remember]

    # 添加 MCP 外部工具
    mcp_tools = _load_mcp_tools()
    if mcp_tools:
        builtin.extend(mcp_tools)

    return builtin


# ---------------------------------------------------------------------------
# Agent 构建
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "你是大模型训推一体化平台的 AI 训练助手，帮助用户完成模型训练、评测与咨询。"
    "你可以使用以下工具：\n"
    "1. rag_search: 搜索平台 RAG 知识库，检索领域知识\n"
    "2. list_trained_models: 列出平台上已训练的模型\n"
    "3. get_model_lineage: 查询模型血缘信息\n"
    "4. web_search: 联网搜索获取实时信息（最新分数线、新闻等）\n"
    "5. get_eval_report: 查询模型评测报告\n"
    "6. get_task_status: 查询任务执行状态\n"
    "7. list_datasets: 列出可用的训练数据集\n"
    "8. start_training: 启动模型训练任务（必须先获得用户确认）\n"
    "9. recall_memory: 查看长期记忆（跨会话记住的用户事实）\n"
    "10. remember: 主动将重要信息存入长期记忆\n"
    "\n"
    "【长期记忆机制】系统会自动提取并记住对话中用户的身份/分数/省份/偏好等关键信息（跨会话持久）。\n"
    "- 用户问\"你还记得我什么/你知道我哪些信息/查看记忆\"时：调用 recall_memory 查看记忆事实列表。\n"
    "- 对话中出现新的重要用户信息时：可调用 remember 主动记录（自动去重）。\n"
    "- 回答个性化问题前（如推荐院校），若不确定用户信息，先 recall_memory 核实。\n"
    "\n"
    "【训练咨询流程（严格执行）】当用户询问能否进行某方向的训练时：\n"
    "步骤1：调用 list_datasets 查询平台是否有该方向的数据集。\n"
    "步骤2a：若无数据集，告知用户需先准备数据（可说明蒸馏/标注回灌等途径），不启动训练。\n"
    "步骤2b：若有数据集，向用户推荐完整训练参数并请求确认，格式：\n"
    "  - 数据集：ID + 样本数\n"
    "  - 基底模型：0.5b（快速验证）或 7b（效果优先）\n"
    "  - epochs / lr / batch_size / grad_accum / lora_r / lora_alpha / max_seq_len\n"
    "  - 每个参数附一句推荐理由（小数据集建议 epochs 3-10、lr 1e-4~5e-4、lora_r 4-16）\n"
    "步骤3：明确请用户确认（如\"确认这些参数请回复『确认』，或告诉我需要调整的地方\"）。\n"
    "步骤4：只有用户明确回复确认后，才调用 start_training(confirmed=True)。"
    "用户未确认时严禁调用 start_training。\n"
    "步骤5：训练提交后告知任务ID，并说明：训练会占用 GPU（推理服务已自动暂停），"
    "可用 get_task_status 查询进度。\n"
    "\n"
    "其他回答要求：准确、结构清晰，区分事实与推理依据。"
    "用户询问领域专业知识时优先 rag_search；询问最新信息时用 web_search；"
    "询问模型情况时用 list_trained_models 或 get_eval_report。"
)


def _build_agent(llm, tools, system_prompt: str):
    """构建 LangChain Agent：create_tool_calling_agent + AgentExecutor。

    使用原生 function calling（而非文本 ReAct），对支持工具调用的模型更可靠。
    AgentExecutor 负责工具调用循环、错误处理和中间步骤追踪。
    """
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt or DEFAULT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )


def _build_graph(llm, tools, system_prompt: str, memory_window: int = 10, model_id: str = ""):
    """构建 LangGraph ReAct Agent（create_react_agent + Checkpointer + Store）。

    LangGraph 双层记忆（官方架构）：
    - 短期记忆：checkpointer（SqliteSaver）+ thread_id = session_id
      每轮自动保存消息状态，重启后同 thread 恢复对话上下文
    - 长期记忆：store（SqliteStore）+ namespace = ("memories", model_id)
      prompt callable 每轮动态注入该模型域的用户事实（按模型隔离）

    状态机流程：
      START → agent_node（LLM 决策是否调工具）
            → tools_condition（有工具调用 → tool_node；无 → END）
            → tool_node（并行执行工具）→ 回到 agent_node 循环

    Args:
        llm: ChatOpenAI 实例
        tools: 工具列表
        system_prompt: 系统提示词
        memory_window: 滑动窗口轮数（截断短期历史防 token 爆炸）
        model_id: 模型标识（长期记忆按此隔离）
    """
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import SystemMessage

    base_store = _get_store()

    def _prompt_with_memory(state):
        """prompt callable：注入 system（含长期记忆）+ 窗口截断的短期历史。

        每次调用 LLM 前执行：
        1. 从 store 取当前模型 namespace 的长期记忆（动态，remember 后立即生效）
        2. 拼接 system prompt
        3. 截断短期历史（最近 memory_window 轮，checkpointer 中的消息状态）

        兼容两种入参（版本差异）：state dict 或 messages list。
        """
        if isinstance(state, dict):
            messages = state.get("messages", [])
        else:
            messages = state

        # 1. 动态加载长期记忆（本模型域）
        sys_text = system_prompt or DEFAULT_SYSTEM_PROMPT
        try:
            items = base_store.search(_memory_namespace(model_id), limit=20)
            facts = [it.value.get("text", "") for it in items if it.value.get("text")]
            if facts:
                fact_lines = "\n".join(f"- {f}" for f in facts)
                sys_text = (
                    f"{sys_text}\n\n【关于用户的长期记忆（历史会话中记住的信息，回答时参考）】\n{fact_lines}"
                )
        except Exception:
            pass  # store 不可用时只跳过记忆注入，不影响对话

        # 2. 窗口截断短期历史
        conv_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        if memory_window > 0:
            conv_msgs = conv_msgs[-(memory_window * 2):]
        return [SystemMessage(content=sys_text)] + conv_msgs

    # create_react_agent 官方双记忆参数：checkpointer（短期）+ store（长期）
    kwargs = {"checkpointer": _get_checkpointer(), "store": base_store}
    try:
        graph = create_react_agent(llm, tools, prompt=_prompt_with_memory, **kwargs)
    except TypeError:
        try:
            graph = create_react_agent(llm, tools, state_modifier=_prompt_with_memory, **kwargs)
        except TypeError:
            kwargs.pop("store", None)
            graph = create_react_agent(llm, tools, **kwargs)
    return graph


# ---------------------------------------------------------------------------
# 滑动窗口记忆管理
# ---------------------------------------------------------------------------

def _build_history(messages: list[dict], window: int):
    """从原始消息历史构建 LangChain 消息列表（动态滑动窗口截断）。

    滑动窗口策略：
    - window > 0：只保留最近 window 轮对话（每轮 = user + assistant = 2 条）
    - window <= 0：保留全部历史
    - 超出窗口的历史消息被截断，控制上下文 token 消耗
    """
    from langchain_core.messages import HumanMessage, AIMessage

    if window > 0:
        recent = messages[-(window * 2):]
    else:
        recent = messages

    history = []
    for msg in recent:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


# ---------------------------------------------------------------------------
# 会话管理
# ---------------------------------------------------------------------------

def create_session(
    system_prompt: str = "",
    model_id: str = "",
    memory_window: int = 0,
    scope: str = "",
    resume_session_id: str = "",
) -> dict:
    """创建一个新的对话会话（双层记忆：短期 RAM + 长期 SQLite）。

    Args:
        system_prompt: 自定义系统提示词；为空则使用默认
        model_id: 关联的已训练模型ID（用于上下文标识）
        memory_window: 滑动窗口轮数（0=保留全部，N=保留最近N轮）
        scope: 长期记忆域（同域会话共享记忆）。默认按 model_id 分域：
               选模型的会话共享该模型的用户记忆，DeepSeek 默认会话共享全局域
        resume_session_id: 恢复指定会话（从 SQLite agent_messages 加载历史，
               平台重启后 RAM 丢失时用）
    """
    from .. import memory_store
    memory_store.init_tables()

    session_id = str(uuid.uuid4())[:8]
    cfg = load_config()
    chat_cfg = _chat_cfg(cfg)

    if not system_prompt:
        system_prompt = chat_cfg.get("system_prompt") or ""
    if not system_prompt and model_id:
        # 已训练模型：自动继承其领域身份（domain_direction），如高考志愿指导老师
        try:
            from .. import registry
            for m in registry.selectable_models():
                if m.get("model_id") == model_id:
                    dd = m.get("domain_direction") or ""
                    if dd:
                        system_prompt = (
                            f"你是{dd}。回答要具体、实用、可操作。"
                            "涉及最新分数线、招生政策等时效信息时，主动使用联网搜索工具核实。"
                        )
                    break
        except Exception:
            pass
    if not system_prompt:
        # 默认（DeepSeek 后端）：平台训练助手（含训练咨询->确认->执行流程）
        system_prompt = DEFAULT_SYSTEM_PROMPT
    if memory_window <= 0:
        memory_window = int(chat_cfg.get("memory_window", 10))

    # 长期记忆不再静态注入系统提示词：LangGraph prompt callable 每轮
    # 从 Store 动态加载当前模型 namespace 的记忆（remember 后立即生效）
    mem_scope = scope or model_id or "global"

    _sessions[session_id] = {
        "session_id": session_id,
        "system_prompt": system_prompt,
        "model_id": model_id,
        "memory_window": memory_window,
        "scope": mem_scope,  # 兼容字段；长期记忆实际按 model_id 隔离
        "messages": [],  # 兼容旧 API（list_sessions 读 message_count）
        "created_at": time.time(),
    }
    # 持久化元数据：重启后自动恢复 model_id（长期记忆按模型隔离的关键）与领域提示词
    try:
        memory_store.save_session_meta(session_id, model_id, system_prompt, memory_window)
    except Exception:
        pass  # 元数据写失败不影响会话创建

    # --- 恢复历史：平台重启后 RAM 会话丢失，从 SQLite 拉回最近对话 ---
    if resume_session_id:
        restored = memory_store.load_messages(resume_session_id, limit=memory_window * 2)
        if restored:
            _sessions[session_id]["messages"] = restored
            _sessions[session_id]["resumed_from"] = resume_session_id

    return _sessions[session_id]


def get_session(session_id: str) -> dict | None:
    """获取会话；RAM 无此会话时尝试从 SQLite 自动恢复（含元数据+历史）。"""
    sess = _sessions.get(session_id)
    if sess:
        return sess
    restored = _restore_session(session_id)
    return restored


def _restore_session(session_id: str) -> dict | None:
    """平台重启后从 SQLite 恢复会话：元数据（model_id/提示词/窗口）+ 最近历史。

    元数据来自 agent_sessions 表（新会话创建时写入）；旧会话无元数据时
    回退默认值（保持向后兼容）。
    """
    from .. import memory_store
    memory_store.init_tables()
    meta = memory_store.load_session_meta(session_id) or {}
    restored = memory_store.load_messages(session_id, limit=20)
    if not restored and not meta:
        return None  # 完全未知的会话
    session = {
        "session_id": session_id,
        "system_prompt": meta.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        "model_id": meta.get("model_id", ""),  # 恢复模型域：长期记忆按此隔离
        "memory_window": int(meta.get("memory_window", 10) or 10),
        "scope": meta.get("model_id", "") or "global",
        "messages": restored,
        "created_at": meta.get("created_at", time.time()),
        "resumed_from": session_id,
    }
    _sessions[session_id] = session
    return session


def list_sessions() -> list[dict]:
    return [
        {
            "session_id": s["session_id"],
            "model_id": s["model_id"],
            "message_count": len(s["messages"]),
            "created_at": s["created_at"],
        }
        for s in _sessions.values()
    ]


def delete_session(session_id: str) -> bool:
    return _sessions.pop(session_id, None) is not None


# ---------------------------------------------------------------------------
# 对话（非流式）
# ---------------------------------------------------------------------------

def chat(session_id: str, message: str) -> dict:
    """非流式对话：Agent 处理用户消息并返回完整回复。

    优先级：
    1. LangGraph create_react_agent（状态机 + MemorySaver 持久化记忆）
    2. AgentExecutor（旧版兼容，langgraph 不可用时）
    3. _direct_chat（降级模式，API 不支持 function calling 时）

    流程：
    1. 构建 LLM + 工具
    2. Agent 决定是否调用工具（RAG 检索 / 模型查询）
    3. LLM 生成最终回复
    4. 更新会话历史（兼容旧 API）
    """
    session = _sessions.get(session_id)
    if not session:
        # --- 自动恢复：RAM 无此会话则从 SQLite 恢复（元数据+历史，平台重启场景）---
        session = _restore_session(session_id)
        if not session:
            raise ValueError(f"会话不存在: {session_id}")

    cfg = load_config()
    llm = _get_llm(cfg, streaming=False, model_id=session.get("model_id", ""))
    history = _build_history(session["messages"], session["memory_window"])

    intermediate_steps = []
    answer = ""

    # --- 优先用 LangGraph（Checkpointer 短期 + Store 长期）---
    if LANGGRAPH_AVAILABLE:
        try:
            from langchain_core.messages import HumanMessage
            tools = _build_tools(session.get("model_id", ""))
            graph = _build_graph(
                llm, tools, session["system_prompt"],
                session["memory_window"], session.get("model_id", ""),
            )

            # 短期记忆：thread_id = session_id，checkpointer 自动保存/恢复该会话状态
            config = {"configurable": {"thread_id": session_id}}
            # system prompt 由 prompt callable 每轮动态注入（含 Store 长期记忆）
            input_msgs = [HumanMessage(content=message)]
            result = graph.invoke({"messages": input_msgs}, config=config)

            # 提取最终回复
            result_msgs = result.get("messages", [])
            if result_msgs:
                last = result_msgs[-1]
                answer = getattr(last, "content", str(last))

            # 提取工具调用步骤（从 messages 里的 ToolMessage 提取）
            from langchain_core.messages import ToolMessage, AIMessage
            for msg in result_msgs:
                if isinstance(msg, ToolMessage):
                    intermediate_steps.append({
                        "tool": msg.name or "tool",
                        "input": "",
                        "output": str(msg.content)[:500],
                    })
                elif isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        intermediate_steps.append({
                            "tool": tc.get("name", "tool"),
                            "input": str(tc.get("args", ""))[:200],
                            "output": "",
                        })
        except Exception as exc:
            # LangGraph 失败，降级到 AgentExecutor
            intermediate_steps = [{"tool": f"(LangGraph 失败，降级: {str(exc)[:80]})", "input": "", "output": ""}]
            try:
                tools = _build_tools(session.get("model_id", ""))
                agent_executor = _build_agent(llm, tools, session["system_prompt"])
                result = agent_executor.invoke({"input": message, "chat_history": history})
                answer = result.get("output", "")
                for step in result.get("intermediate_steps", []):
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, observation = step
                        intermediate_steps.append({
                            "tool": getattr(action, "tool", str(action)),
                            "input": getattr(action, "tool_input", ""),
                            "output": str(observation)[:500],
                        })
            except Exception as exc2:
                answer = _direct_chat(cfg, session["system_prompt"] or DEFAULT_SYSTEM_PROMPT, history, message)
                intermediate_steps.append({
                    "tool": "(降级模式: 普通对话)",
                    "input": str(exc2)[:200],
                    "output": "",
                })

    # --- 降级到 AgentExecutor（langgraph 不可用）---
    elif LANGCHAIN_AVAILABLE:
        try:
            tools = _build_tools(session.get("model_id", ""))
            agent_executor = _build_agent(llm, tools, session["system_prompt"])
            result = agent_executor.invoke({"input": message, "chat_history": history})
            answer = result.get("output", "")
            for step in result.get("intermediate_steps", []):
                if isinstance(step, tuple) and len(step) >= 2:
                    action, observation = step
                    intermediate_steps.append({
                        "tool": getattr(action, "tool", str(action)),
                        "input": getattr(action, "tool_input", ""),
                        "output": str(observation)[:500],
                    })
        except Exception as exc:
            answer = _direct_chat(cfg, session["system_prompt"] or DEFAULT_SYSTEM_PROMPT, history, message)
            intermediate_steps = [{"tool": "(降级模式: 普通对话, 工具不可用)", "input": str(exc)[:200], "output": ""}]

    # --- 最终降级：直接 HTTP ---
    else:
        answer = _direct_chat(cfg, session["system_prompt"] or DEFAULT_SYSTEM_PROMPT, history, message)
        intermediate_steps = [{"tool": "(降级模式: langchain 未安装)", "input": "", "output": ""}]

    # 更新会话历史（兼容旧 API，MemorySaver 也会自动存）
    session["messages"].append({"role": "user", "content": message})
    session["messages"].append({"role": "assistant", "content": answer})

    # --- 长期记忆持久化：消息写 SQLite + 事实提取写 Store（按模型隔离）---
    try:
        from .. import memory_store
        memory_store.save_message(session_id, "user", message)
        memory_store.save_message(
            session_id, "assistant", answer,
            tool_calls=json.dumps(intermediate_steps, ensure_ascii=False),
        )
        # 事实提取：只处理用户消息；规则预筛有信号才调 LLM（省 API）
        if memory_store._worth_extracting(message):
            facts = memory_store.extract_facts_with_llm(message, cfg)
            new_facts = []
            if facts:
                store = _get_store()
                ns = _memory_namespace(session.get("model_id", ""))
                for f in facts:
                    store.put(ns, str(uuid.uuid4())[:8], {"text": f})
                    new_facts.append(f)
            if new_facts:
                intermediate_steps.append({
                    "tool": "(长期记忆)",
                    "input": message[:100],
                    "output": f"已记住: {'; '.join(new_facts)}",
                })
    except Exception:
        pass  # 记忆持久化失败不影响对话主流程

    return {
        "session_id": session_id,
        "answer": answer,
        "tool_calls": intermediate_steps,
        "messages": session["messages"],
        "backend": "langgraph" if LANGGRAPH_AVAILABLE else ("langchain" if LANGCHAIN_AVAILABLE else "direct"),
    }


# ---------------------------------------------------------------------------
# 对话（流式 SSE）
# ---------------------------------------------------------------------------

async def chat_stream(session_id: str, message: str) -> AsyncGenerator[str, None]:
    """流式对话：逐 token 返回（SSE 格式）。

    优先级：
    1. LangGraph astream_events（状态机 + MemorySaver）
    2. AgentExecutor astream_events（旧版兼容）
    3. _direct_chat_stream（降级模式，直接 HTTP 流式）

    事件类型：
    - {"token": "...}"     : LLM 生成的 token（逐字流式输出）
    - {"tool_start": "..."} : 工具调用开始
    - {"tool_end": "..."}   : 工具调用结束
    - {"done": true}        : 全部完成
    - {"error": "..."}      : 错误信息
    """
    session = _sessions.get(session_id)
    if not session:
        session = _restore_session(session_id)  # 重启后自动恢复（元数据+历史）
    if not session:
        yield _sse({"error": f"会话不存在: {session_id}"})
        return

    cfg = load_config()
    llm = _get_llm(cfg, streaming=True, model_id=session.get("model_id", ""))
    history = _build_history(session["messages"], session["memory_window"])

    full_answer = ""

    # --- 优先用 LangGraph 流式（Checkpointer 短期 + Store 长期）---
    if LANGGRAPH_AVAILABLE:
        try:
            from langchain_core.messages import HumanMessage
            tools = _build_tools(session.get("model_id", ""))
            graph = _build_graph(
                llm, tools, session["system_prompt"],
                session["memory_window"], session.get("model_id", ""),
            )
            config = {"configurable": {"thread_id": session_id}}
            input_msgs = [HumanMessage(content=message)]

            async for event in graph.astream_events(
                {"messages": input_msgs},
                config=config,
                version="v2",
            ):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    token = chunk.content if chunk and hasattr(chunk, "content") else ""
                    if token:
                        full_answer += token
                        yield _sse({"token": token})
                elif kind == "on_tool_start":
                    yield _sse({"tool_start": event.get("name", "")})
                elif kind == "on_tool_end":
                    yield _sse({"tool_end": event.get("name", "")})
        except Exception as exc:
            # LangGraph 失败，降级到 AgentExecutor 或直接流式
            yield _sse({"tool_start": f"(LangGraph 降级: {str(exc)[:40]})"})
            try:
                tools = _build_tools(session.get("model_id", ""))
                agent_executor = _build_agent(llm, tools, session["system_prompt"])
                async for event in agent_executor.astream_events(
                    {"input": message, "chat_history": history},
                    version="v2",
                ):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        token = chunk.content if chunk and hasattr(chunk, "content") else ""
                        if token:
                            full_answer += token
                            yield _sse({"token": token})
                    elif kind == "on_tool_start":
                        yield _sse({"tool_start": event.get("name", "")})
                    elif kind == "on_tool_end":
                        yield _sse({"tool_end": event.get("name", "")})
            except Exception as exc2:
                async for token in _direct_chat_stream(cfg, session["system_prompt"] or DEFAULT_SYSTEM_PROMPT, history, message):
                    full_answer += token
                    yield _sse({"token": token})

    # --- 降级到 AgentExecutor ---
    elif LANGCHAIN_AVAILABLE:
        try:
            tools = _build_tools(session.get("model_id", ""))
            agent_executor = _build_agent(llm, tools, session["system_prompt"])
            async for event in agent_executor.astream_events(
                {"input": message, "chat_history": history},
                version="v2",
            ):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    token = chunk.content if chunk and hasattr(chunk, "content") else ""
                    if token:
                        full_answer += token
                        yield _sse({"token": token})
                elif kind == "on_tool_start":
                    yield _sse({"tool_start": event.get("name", "")})
                elif kind == "on_tool_end":
                    yield _sse({"tool_end": event.get("name", "")})
        except Exception as exc:
            yield _sse({"tool_start": "(降级模式: 普通对话)"})
            async for token in _direct_chat_stream(cfg, session["system_prompt"] or DEFAULT_SYSTEM_PROMPT, history, message):
                full_answer += token
                yield _sse({"token": token})

    # --- 最终降级：直接 HTTP 流式 ---
    else:
        yield _sse({"tool_start": "(降级模式: langchain 未安装)"})
        async for token in _direct_chat_stream(cfg, session["system_prompt"] or DEFAULT_SYSTEM_PROMPT, history, message):
            full_answer += token
            yield _sse({"token": token})

    # 更新会话历史
    session["messages"].append({"role": "user", "content": message})
    session["messages"].append({"role": "assistant", "content": full_answer})

    # --- 长期记忆持久化（与 chat() 相同：事实写 Store 按模型隔离）---
    try:
        from .. import memory_store
        memory_store.save_message(session_id, "user", message)
        memory_store.save_message(session_id, "assistant", full_answer)
        if memory_store._worth_extracting(message):
            facts = memory_store.extract_facts_with_llm(message, cfg)
            if facts:
                store = _get_store()
                ns = _memory_namespace(session.get("model_id", ""))
                for f in facts:
                    store.put(ns, str(uuid.uuid4())[:8], {"text": f})
    except Exception:
        pass

    yield _sse({"done": True, "answer": full_answer})


def _sse(obj: dict) -> str:
    """格式化为 SSE data 行。"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _direct_chat(cfg: dict, system_prompt: str, history, message: str) -> str:
    """降级模式：直接 HTTP 调用 OpenAI 兼容端点（绕过 LangChain）。

    和 judge.py 同样的调用方式，确保兼容火山方舟 plan API。
    """
    import urllib.request
    import urllib.error

    chat_cfg = _chat_cfg(cfg)
    base_url = chat_cfg.get("base_url", "").rstrip("/")
    api_key = _resolve_api_key(chat_cfg)
    model = chat_cfg.get("model", "ark-code-latest")

    # 降级直连：无 function calling。明确告知模型别输出工具调用语法，
    # 否则模型会按系统提示词"调用工具"而把 <invoke> 等原文写给用户。
    system_prompt = (
        system_prompt
        + "\n\n[注意] 当前为降级直连模式，无法调用任何工具（搜索/记忆/检索均不可用）。"
        "请直接基于已有知识回答，不要输出任何工具调用语法或 XML 标签。"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if hasattr(msg, "content"):
            role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
            messages.append({"role": role, "content": msg.content})
    messages.append({"role": "user", "content": message})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": float(chat_cfg.get("temperature", 0.7)),
        "max_tokens": int(chat_cfg.get("max_tokens", 1024)),
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        return body["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"降级模式 HTTP {exc.code}: {detail}") from exc


async def _direct_chat_stream(cfg: dict, system_prompt: str, history, message: str):
    """降级模式：直接 HTTP 流式调用（SSE 逐 token）。"""
    import urllib.request

    chat_cfg = _chat_cfg(cfg)
    base_url = chat_cfg.get("base_url", "").rstrip("/")
    api_key = _resolve_api_key(chat_cfg)
    model = chat_cfg.get("model", "ark-code-latest")

    system_prompt = (
        system_prompt
        + "\n\n[注意] 当前为降级直连模式，无法调用任何工具（搜索/记忆/检索均不可用）。"
        "请直接基于已有知识回答，不要输出任何工具调用语法或 XML 标签。"
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
        messages.append({"role": role, "content": msg.content})
    messages.append({"role": "user", "content": message})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": float(chat_cfg.get("temperature", 0.7)),
        "max_tokens": int(chat_cfg.get("max_tokens", 1024)),
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    full = ""
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
                delta = evt.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    full += token
                    yield token
            except json.JSONDecodeError:
                continue

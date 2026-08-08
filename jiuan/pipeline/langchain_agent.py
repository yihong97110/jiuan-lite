"""LangChain Agent：为训练后的模型提供多轮对话与工具调用能力。

基于 LangChain 构建，覆盖以下工程化能力：
- 多轮对话与上下文记忆管理（动态滑动窗口截断）
- 工具调用：RAG 知识检索、模型仓库查询、血缘追踪
- 流式输出（SSE，逐 token 返回）
- 可对接任意 OpenAI 兼容端点（vLLM 本地服务 / DeepSeek API / 火山方舟等）

LangChain 组件映射：
  ChatOpenAI            -> 统一封装 LLM 接口（对接 vLLM/DeepSeek/方舟）
  @tool                 -> 工具定义，封装现有 rag_backend / registry 能力
  create_tool_calling_agent -> Agent 编排（ReAct + 原生 function calling）
  AgentExecutor         -> Agent 执行器（工具调用循环、错误处理）
  astream_events(v2)    -> 流式 token 输出 + 工具调用事件追踪

设计：
- 每个会话(session)独立维护消息历史，按 memory_window 轮数滑动窗口截断
- 工具直接调用现有 pipeline 能力，不重复造轮子
- langchain 未安装时模块可正常导入，API 层返回友好提示
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from ..common import load_config
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


def available() -> bool:
    """LangChain 是否可用（已安装 langchain-openai）。"""
    return LANGCHAIN_AVAILABLE


# ---------------------------------------------------------------------------
# 会话存储（内存；生产可换 Redis/DB）
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}


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

def _get_llm(cfg: dict | None = None, streaming: bool = False):
    """创建 LangChain ChatOpenAI 实例，指向 OpenAI 兼容端点。

    可对接：
    - vLLM 本地推理服务（Linux+GPU，config.chat.base_url 指向 vllm serve 端口）
    - DeepSeek API / 火山方舟（config.chat.base_url 指向云端端点）
    - 任意 OpenAI 兼容服务
    """
    from langchain_openai import ChatOpenAI

    chat_cfg = _chat_cfg(cfg)
    return ChatOpenAI(
        base_url=chat_cfg.get("base_url", "http://127.0.0.1:8001/v1"),
        model=chat_cfg.get("model", "jiuan-model"),
        api_key=_resolve_api_key(chat_cfg),
        temperature=float(chat_cfg.get("temperature", 0.7)),
        max_tokens=int(chat_cfg.get("max_tokens", 1024)),
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# 工具定义：封装现有平台能力为 LangChain Tool
# ---------------------------------------------------------------------------

def _build_tools():
    """构建 Agent 可用工具集，每个工具直接调用现有 pipeline 函数。"""
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

    return [rag_search, list_trained_models, get_model_lineage]


# ---------------------------------------------------------------------------
# Agent 构建
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业 AI 助手，部署在大模型训推一体化平台上。"
    "你可以使用工具搜索知识库、查询平台上的已训练模型信息来辅助回答。"
    "回答要求：准确、结构清晰，区分事实与推理依据，说明适用边界和不确定性。"
    "当用户询问领域专业知识时，优先使用 rag_search 工具检索知识库。"
    "当用户询问模型相关问题时，使用 list_trained_models 或 get_model_lineage 工具。"
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
) -> dict:
    """创建一个新的对话会话。

    Args:
        system_prompt: 自定义系统提示词；为空则使用默认
        model_id: 关联的已训练模型ID（用于上下文标识）
        memory_window: 滑动窗口轮数（0=保留全部，N=保留最近N轮）
    """
    session_id = str(uuid.uuid4())[:8]
    cfg = load_config()
    chat_cfg = _chat_cfg(cfg)

    if not system_prompt:
        system_prompt = chat_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    if memory_window <= 0:
        memory_window = int(chat_cfg.get("memory_window", 10))

    _sessions[session_id] = {
        "session_id": session_id,
        "system_prompt": system_prompt,
        "model_id": model_id,
        "memory_window": memory_window,
        "messages": [],
        "created_at": time.time(),
    }
    return _sessions[session_id]


def get_session(session_id: str) -> dict | None:
    return _sessions.get(session_id)


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

    流程：
    1. 从会话历史构建滑动窗口记忆
    2. Agent 决定是否调用工具（RAG 检索 / 模型查询）
    3. LLM 生成最终回复
    4. 更新会话历史

    若 API 不支持 function calling，自动降级为普通对话（无工具调用）。
    """
    session = _sessions.get(session_id)
    if not session:
        raise ValueError(f"会话不存在: {session_id}")

    cfg = load_config()
    llm = _get_llm(cfg, streaming=False)
    history = _build_history(session["messages"], session["memory_window"])

    intermediate_steps = []
    try:
        tools = _build_tools()
        agent_executor = _build_agent(llm, tools, session["system_prompt"])
        result = agent_executor.invoke({
            "input": message,
            "chat_history": history,
        })
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
        # 降级：API 不支持 function calling 时，走普通对话（直接 HTTP，绕过 LangChain）
        answer = _direct_chat(cfg, session["system_prompt"] or DEFAULT_SYSTEM_PROMPT, history, message)
        intermediate_steps = [{"tool": "(降级模式: 普通对话, 工具不可用)", "input": str(exc)[:200], "output": ""}]

    # 更新会话历史
    session["messages"].append({"role": "user", "content": message})
    session["messages"].append({"role": "assistant", "content": answer})

    return {
        "session_id": session_id,
        "answer": answer,
        "tool_calls": intermediate_steps,
        "messages": session["messages"],
    }


# ---------------------------------------------------------------------------
# 对话（流式 SSE）
# ---------------------------------------------------------------------------

async def chat_stream(session_id: str, message: str) -> AsyncGenerator[str, None]:
    """流式对话：逐 token 返回（SSE 格式）。

    事件类型：
    - {"token": "...}"     : LLM 生成的 token（逐字流式输出）
    - {"tool_start": "..."} : 工具调用开始
    - {"tool_end": "..."}   : 工具调用结束
    - {"done": true}        : 全部完成
    - {"error": "..."}      : 错误信息
    """
    session = _sessions.get(session_id)
    if not session:
        yield _sse({"error": f"会话不存在: {session_id}"})
        return

    cfg = load_config()
    llm = _get_llm(cfg, streaming=True)
    history = _build_history(session["messages"], session["memory_window"])

    full_answer = ""
    try:
        tools = _build_tools()
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
        # 降级：API 不支持 function calling，走普通流式对话（直接 HTTP）
        yield _sse({"tool_start": "(降级模式: 普通对话)"})
        async for token in _direct_chat_stream(cfg, session["system_prompt"] or DEFAULT_SYSTEM_PROMPT, history, message):
            full_answer += token
            yield _sse({"token": token})

    # 更新会话历史
    session["messages"].append({"role": "user", "content": message})
    session["messages"].append({"role": "assistant", "content": full_answer})

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

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if hasattr(msg, "content"):
            role = "user" if isinstance(msg, type(history[0])) else "assistant"
            # LangChain HumanMessage -> user, AIMessage -> assistant
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

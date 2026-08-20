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
# 会话存储 + MemorySaver checkpointer
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}

# 全局 MemorySaver：所有会话共享一个 checkpointer，通过 thread_id 隔离
_memory_saver = None


def _get_memory_saver():
    """获取全局 MemorySaver 实例（惰性初始化）。

    MemorySaver 是 LangGraph 的内存 checkpointer：
    - 按 thread_id（= session_id）隔离不同会话的状态
    - 自动保存和恢复消息历史，不需要手动管理 messages 列表
    - 生产环境可换 SQLiteSaver/PostgresSaver 实现持久化
    """
    global _memory_saver
    if _memory_saver is None and LANGGRAPH_AVAILABLE:
        from langgraph.checkpoint.memory import MemorySaver
        _memory_saver = MemorySaver()
    return _memory_saver


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

    可对接：
    - vLLM 本地推理服务（Linux+GPU，config.chat.base_url 指向 vllm serve 端口）
    - DeepSeek API / 火山方舟（config.chat.base_url 指向云端端点）
    - 任意 OpenAI 兼容服务

    Args:
        model_id: 如果传入已训练模型ID，会尝试从 registry 读取其 served_model_name，
                  用于 vLLM 多 LoRA 场景下切换不同训练模型。
    """
    from langchain_openai import ChatOpenAI

    chat_cfg = _chat_cfg(cfg)
    model_name = chat_cfg.get("model", "jiuan-model")

    # 如果指定了已训练模型，尝试从 registry 读取对应的 served_model_name
    if model_id:
        try:
            from .. import registry
            models = registry.list_models()
            for m in models:
                if m.get("model_id") == model_id:
                    # 优先用 meta 里记录的 served_model_name
                    served = m.get("served_model_name") or m.get("name") or model_id
                    model_name = served
                    break
        except Exception:
            pass  # 读取失败就用配置里的默认 model

    return ChatOpenAI(
        base_url=chat_cfg.get("base_url", "http://127.0.0.1:8001/v1"),
        model=model_name,
        api_key=_resolve_api_key(chat_cfg),
        temperature=float(chat_cfg.get("temperature", 0.7)),
        max_tokens=int(chat_cfg.get("max_tokens", 1024)),
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# 联网搜索（DuckDuckGo Instant Answer API，免费无 Key）
# ---------------------------------------------------------------------------

def _web_search(query: str, max_results: int = 5) -> str:
    """通过 DuckDuckGo Instant Answer API 进行联网搜索。

    无需 API Key，返回摘要文本。
    """
    import urllib.request
    import urllib.parse

    safe_query = urllib.parse.quote(query)
    url = f"https://api.duckduckgo.com/?q={safe_query}&format=json&no_html=1&skip_disambig=1"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return f"联网搜索失败: {exc}"

    results = []

    # 主答案
    abstract = data.get("AbstractText", "")
    if abstract:
        source = data.get("AbstractURL", "")
        results.append(f"[主答案] {abstract}" + (f"\n来源: {source}" if source else ""))

    # 相关主题
    for topic in (data.get("RelatedTopics") or [])[:max_results]:
        if isinstance(topic, dict):
            text = topic.get("Text", "")
            if text:
                results.append(f"- {text[:200]}")
        elif isinstance(topic, str) and topic.strip():
            results.append(f"- {topic[:200]}")

    # 定义类结果
    for item in (data.get("Definition") or [])[:2]:
        if isinstance(item, dict) and item.get("text"):
            results.append(f"[定义] {item['text'][:200]}")

    if not results:
        return f"未找到与「{query}」相关的搜索结果。"
    return "\n".join(results[:max_results])


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
# ---------------------------------------------------------------------------

_mcp_servers: dict[str, dict] = {}


def register_mcp_server(name: str, command: str, description: str = "") -> dict:
    """注册一个 MCP 服务器配置。

    Args:
        name: 服务器名称
        command: 启动命令（如 "python -m mcp_server_fetch"）
        description: 描述

    Returns:
        注册信息
    """
    server = {
        "name": name,
        "command": command,
        "description": description,
        "status": "registered",
        "tools": [],
        "registered_at": time.time(),
    }
    _mcp_servers[name] = server
    return server


def list_mcp_servers() -> list[dict]:
    """列出所有已注册的 MCP 服务器。"""
    return list(_mcp_servers.values())


def _load_mcp_tools() -> list:
    """加载 MCP 服务器的工具并转为 LangChain Tool。

    尝试用 mcp 库连接已注册的服务器，发现其工具并包装为 @tool。
    如果 mcp 库未安装或连接失败，返回空列表（不影响内置工具）。
    """
    tools = []
    try:
        from langchain_core.tools import tool as langchain_tool
    except Exception:
        return tools

    for server_name, server_info in _mcp_servers.items():
        if server_info.get("status") != "registered":
            continue
        # 为每个 MCP 服务器创建一个通用的调用工具
        cmd = server_info["command"]
        server_desc = server_info.get("description", server_name)

        def _make_mcp_tool(sname, scmd, sdesc):
            @langchain_tool
            def mcp_tool(query: str) -> str:
                """通过 MCP 协议调用外部工具。当内置工具无法满足需求时使用。

                Args:
                    query: 调用参数或查询内容

                Returns:
                    MCP 服务器返回的结果
                """   
                return _call_mcp_server(sname, scmd, query)
            mcp_tool.name = f"mcp_{sname}"
            mcp_tool.description = f"MCP工具({sdesc}): {sdesc}。传入query参数调用。"
            return mcp_tool

        tools.append(_make_mcp_tool(server_name, cmd, server_desc))
        server_info["status"] = "loaded"
        server_info["tools"] = [f"mcp_{server_name}"]

    return tools


def _call_mcp_server(server_name: str, command: str, query: str) -> str:
    """通过子进程调用 MCP 服务器并传递查询。

    这是一个简化实现：通过 stdin/stdout 与 MCP 服务器通信。
    生产环境应使用 mcp 库的正式客户端。
    """
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

    # 收集内置工具
    builtin = [rag_search, list_trained_models, get_model_lineage,
              web_search, get_eval_report, get_task_status]

    # 添加 MCP 外部工具
    mcp_tools = _load_mcp_tools()
    if mcp_tools:
        builtin.extend(mcp_tools)

    return builtin


# ---------------------------------------------------------------------------
# Agent 构建
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业 AI 助手，部署在大模型训推一体化平台上。"
    "你可以使用以下工具辅助回答：\n"
    "1. rag_search: 搜索平台 RAG 知识库，检索领域知识\n"
    "2. list_trained_models: 列出平台上已训练的模型\n"
    "3. get_model_lineage: 查询模型血缘信息\n"
    "4. web_search: 联网搜索获取实时信息（新闻、最新数据等）\n"
    "5. get_eval_report: 查询模型评测报告\n"
    "6. get_task_status: 查询任务执行状态\n"
    "回答要求：准确、结构清晰，区分事实与推理依据。"
    "当用户询问领域专业知识时，优先使用 rag_search；"
    "当用户询问最新信息时，使用 web_search；"
    "当用户询问模型相关问题时，使用 list_trained_models 或 get_eval_report。"
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


def _build_graph(llm, tools, system_prompt: str, memory_window: int = 10):
    """构建 LangGraph ReAct Agent（create_react_agent + MemorySaver）。

    LangGraph 状态机流程：
      START → agent_node（LLM 决策是否调工具）
            → tools_condition（有工具调用 → tool_node；无 → END）
            → tool_node（并行执行工具）
            → 回到 agent_node（看是否还要调工具）
            → ...循环直到 LLM 不再调工具或达到 max_iterations

    对比 AgentExecutor 的优势：
    1. 状态机可视化（可以画流程图）
    2. MemorySaver 自动持久化历史（thread_id 隔离）
    3. 支持并行工具调用
    4. 支持中断/恢复（checkpointer）

    Args:
        llm: ChatOpenAI 实例
        tools: 工具列表
        system_prompt: 系统提示词
        memory_window: 滑动窗口轮数（用于 state_modifier 截断历史）
    """
    from langgraph.prebuilt import create_react_agent

    # state_modifier：在每次调用 LLM 前截断历史，保留最近 memory_window 轮
    # LangGraph 的 messages 会自动累积，需要 state_modifier 控制上下文长度
    def _trim_messages(messages):
        """滑动窗口截断：保留 system + 最近 N 轮对话。"""
        from langchain_core.messages import SystemMessage
        # 分离 system 消息和对话消息
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        conv_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        # 保留最近 memory_window*2 条（每轮 = user + assistant）
        if memory_window > 0:
            conv_msgs = conv_msgs[-(memory_window * 2):]
        return system_msgs + conv_msgs

    # LangGraph 新版 API 兼容：优先用 prompt 参数（接受 callable），回退 state_modifier
    # 不同版本参数名不同，逐个尝试
    kwargs = {"checkpointer": _get_memory_saver()}
    try:
        graph = create_react_agent(llm, tools, prompt=_trim_messages, **kwargs)
    except TypeError:
        try:
            graph = create_react_agent(llm, tools, state_modifier=_trim_messages, **kwargs)
        except TypeError:
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
        "messages": [],  # 兼容旧 API（list_sessions 读 message_count）
        "created_at": time.time(),
    }

    # 如果 LangGraph 可用，用 MemorySaver 初始化会话状态（thread_id = session_id）
    # MemorySaver 会在首次 invoke 时自动创建 checkpoint，这里不需要预初始化
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
        raise ValueError(f"会话不存在: {session_id}")

    cfg = load_config()
    llm = _get_llm(cfg, streaming=False, model_id=session.get("model_id", ""))
    history = _build_history(session["messages"], session["memory_window"])

    intermediate_steps = []
    answer = ""

    # --- 优先用 LangGraph ---
    if LANGGRAPH_AVAILABLE:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            tools = _build_tools()
            graph = _build_graph(llm, tools, session["system_prompt"], session["memory_window"])

            # LangGraph 配置：thread_id 用于 MemorySaver 隔离会话
            config = {"configurable": {"thread_id": session_id}}
            # 构建输入消息（含 system prompt）
            input_msgs = [
                SystemMessage(content=session["system_prompt"] or DEFAULT_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ]
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
                tools = _build_tools()
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
            tools = _build_tools()
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
        yield _sse({"error": f"会话不存在: {session_id}"})
        return

    cfg = load_config()
    llm = _get_llm(cfg, streaming=True, model_id=session.get("model_id", ""))
    history = _build_history(session["messages"], session["memory_window"])

    full_answer = ""

    # --- 优先用 LangGraph 流式 ---
    if LANGGRAPH_AVAILABLE:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            tools = _build_tools()
            graph = _build_graph(llm, tools, session["system_prompt"], session["memory_window"])
            config = {"configurable": {"thread_id": session_id}}
            input_msgs = [
                SystemMessage(content=session["system_prompt"] or DEFAULT_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ]

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
            except Exception as exc2:
                async for token in _direct_chat_stream(cfg, session["system_prompt"] or DEFAULT_SYSTEM_PROMPT, history, message):
                    full_answer += token
                    yield _sse({"token": token})

    # --- 降级到 AgentExecutor ---
    elif LANGCHAIN_AVAILABLE:
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

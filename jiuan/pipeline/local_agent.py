"""本地 ReAct Agent：使用自己训练的模型 + RAG + 工具调用，完全不依赖外部 API。

核心设计：
- 用 transformers 加载本地训练的模型（基座 + LoRA adapter）
- 用 ReAct 文本模式实现工具调用（不依赖 function calling）
- 模型输出 "Thought/Action/Observation" 文本，用正则解析
- 集成 RAG 检索、模型查询、血缘追踪等工具

ReAct 循环：
  用户问题 -> 模型思考(Thought) -> 决定行动(Action) -> 执行工具拿到结果(Observation)
  -> 模型再思考 -> ... -> 最终回答(Final Answer)

无需 vLLM、无需 DeepSeek、无需 GPU、无需 function calling。
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from ..common import load_config, MODELS, DATA
from .. import registry
from . import rag_backend
from . import infer


# ---------------------------------------------------------------------------
# 会话存储
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# 工具定义：每个工具是一个函数，有名字、描述、执行逻辑
# ---------------------------------------------------------------------------

_TOOLS = {
    "rag_search": {
        "description": "搜索知识库，检索与问题相关的知识段落。当需要查找领域知识、法规条文、预案文档时使用。",
        "usage": "rag_search(查询关键词)",
        "func": lambda query, **kw: _tool_rag_search(query, **kw),
    },
    "list_models": {
        "description": "列出平台上所有已训练的模型。当用户问有哪些模型、模型状态时使用。",
        "usage": "list_models()",
        "func": lambda **kw: _tool_list_models(),
    },
    "model_lineage": {
        "description": "查询指定模型的血缘信息（训练数据、基座、评测报告）。当用户想了解模型训练过程时使用。",
        "usage": "model_lineage(模型ID)",
        "func": lambda model_id="", **kw: _tool_model_lineage(model_id),
    },
    "eval_report": {
        "description": "查询模型最新的评测报告（ROUGE/BLEU/Judge分数和Gap分析）。当用户想了解模型表现时使用。",
        "usage": "eval_report(模型ID)",
        "func": lambda model_id="", **kw: _tool_eval_report(model_id),
    },
}


def _tool_rag_search(query: str, collection: str = "default", **kw) -> str:
    """搜索 RAG 知识库。"""
    docs = rag_backend.retrieve(query, top_k=3, collection=collection)
    if not docs:
        return "知识库中没有找到相关内容。"
    results = []
    for i, d in enumerate(docs, 1):
        results.append(f"[{i}] (相似度:{d['score']:.2f}) {d['text'][:200]}")
    return "\n".join(results)


def _tool_list_models() -> str:
    """列出已训练模型。"""
    models = registry.list_models()
    if not models:
        return "当前没有已训练的模型。"
    # 去重
    seen = set()
    lines = []
    for m in models:
        mid = m.get("model_id", "")
        if mid in seen:
            continue
        seen.add(mid)
        lines.append(f"- {mid} ({m.get('backend', 'hf')})")
    return f"共 {len(lines)} 个模型：\n" + "\n".join(lines[:10])


def _tool_model_lineage(model_id: str) -> str:
    """查询模型血缘。"""
    try:
        lineage = registry.lineage(model_id)
        return json.dumps(lineage, ensure_ascii=False, indent=2)[:500]
    except Exception as e:
        return f"查询失败: {e}"


def _tool_eval_report(model_id: str) -> str:
    """查询评测报告。"""
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
        f"ROUGE-L: {m.get('rouge_l_f', 'N/A')}",
        f"BLEU-1: {m.get('bleu_1', 'N/A')}",
        f"Judge: {m.get('judge_overall', 'N/A')}",
        f"Gap数量: {len(gaps)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ReAct Prompt 模板（针对小模型优化：短、直接、few-shot）
# ---------------------------------------------------------------------------

# ReAct 格式指令（与领域身份解耦：身份由模型 meta 的 system_prompt 注入）
REACT_INSTRUCTIONS = """你可以使用工具来辅助回答问题。

工具列表：
- rag_search(关键词)：搜索知识库
- list_models()：列出已训练模型
- model_lineage(模型ID)：查询模型血缘
- eval_report(模型ID)：查询评测报告

格式要求：
- 需要工具时输出：Action: 工具名(参数)
- 能直接回答时输出：Answer: 你的回答

示例1：
问：应急预案包括哪些内容？
Action: rag_search(应急预案内容)

示例2：
问：你好
Answer: 你好！我可以帮你搜索知识库、查询模型信息。

示例3：
问：你能做什么？
Answer: 我可以搜索知识库、列出已训练模型、查询模型血缘和评测报告。
"""

# 通用兜底身份（模型 meta 没有记录领域身份时使用）
DEFAULT_IDENTITY = "你是一个智能助手。"


def _build_react_prompt(question: str, history: list, tool_results: list, system_prompt: str = "") -> str:
    """构建 ReAct 提示词。

    身份优先级：system_prompt（来自模型 meta）> DEFAULT_IDENTITY
    ReAct 格式指令（REACT_INSTRUCTIONS）与身份解耦，所有模型通用。
    """
    identity = system_prompt or DEFAULT_IDENTITY
    parts = [identity + "\n" + REACT_INSTRUCTIONS]

    # 加入对话历史（最近2轮，缩短）
    if history:
        parts.append("对话历史：")
        for h in history[-4:]:
            role = "用户" if h["role"] == "user" else "助手"
            parts.append(f"{role}: {h['content'][:100]}")

    # 加入之前的工具调用结果
    if tool_results:
        parts.append("工具结果：")
        for tr in tool_results:
            parts.append(f"Observation: {tr['observation'][:200]}")
        parts.append("根据以上工具结果回答问题。")

    parts.append(f"问：{question}")
    return "\n".join(parts)


def _keyword_route(question: str) -> str:
    """关键词路由：根据问题关键词判断是否需要调用工具。

    小模型（0.5B）不擅长按 ReAct 格式输出，
    所以先用关键词匹配决定是否需要工具，更可靠。

    返回:
        工具调用字符串如 "rag_search(关键词)"，或空字符串表示不需要工具
    """
    q = question.lower().strip()

    # RAG 检索关键词
    rag_keywords = ["搜索", "查找", "知识库", "预案", "法规", "条例", "规定",
                    "什么是", "解释", "说明", "介绍", "内容", "方法", "流程",
                    "怎么", "如何", "哪些", "请问"]
    for kw in rag_keywords:
        if kw in q:
            # 提取搜索关键词（去掉"搜索/查找/请问"等动词）
            search_query = question
            for prefix in ["搜索", "查找", "帮我", "请", "请问", "查询"]:
                search_query = search_query.replace(prefix, "").strip()
            if len(search_query) > 2:
                return f"rag_search({search_query})"

    # 模型列表关键词
    model_keywords = ["模型列表", "有哪些模型", "多少模型", "模型有", "列出模型", "训练了哪些"]
    for kw in model_keywords:
        if kw in q:
            return "list_models()"

    # 血缘查询关键词
    lineage_keywords = ["血缘", "训练过程", "怎么训练", "基座", "训练数据", "来源"]
    for kw in lineage_keywords:
        if kw in q:
            # 尝试提取模型ID
            mid_match = re.search(r'[a-z0-9\-]{10,}', question)
            mid = mid_match.group(0) if mid_match else ""
            return f"model_lineage({mid})"

    # 评测报告关键词
    eval_keywords = ["评测", "分数", "报告", "rouge", "bleu", "judge", "评分", "指标"]
    for kw in eval_keywords:
        if kw in q:
            mid_match = re.search(r'[a-z0-9\-]{10,}', question)
            mid = mid_match.group(0) if mid_match else ""
            return f"eval_report({mid})"

    return ""


def _parse_react_output(text: str) -> dict:
    """解析模型输出。支持多种格式：

    - "Action: rag_search(关键词)" -> 调用工具
    - "Answer: 回答内容" -> 最终回答
    - "Final Answer: 回答内容" -> 最终回答
    - 纯文本（没匹配到格式）-> 当作最终回答
    """
    text = text.strip()

    # 1. 检查 Action（工具调用）
    action_match = re.search(r"Action[:：]\s*(.+)", text, re.IGNORECASE)
    if action_match:
        action_text = action_match.group(1).strip()

        # 如果 Action 里是 Final Answer
        if re.match(r"(Final\s*)?Answer", action_text, re.IGNORECASE):
            fa_match = re.search(r"Answer[:：]\s*(.+)", action_text, re.DOTALL | re.IGNORECASE)
            if fa_match:
                return {"type": "final", "answer": fa_match.group(1).strip()}

        # 检查是否包含有效工具名
        for tool_name in _TOOLS:
            if tool_name in action_text:
                return {"type": "action", "thought": "", "action": action_text}

    # 2. 检查 Answer（最终回答）
    answer_match = re.search(r"Answer[:：]\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return {"type": "final", "answer": answer_match.group(1).strip()}

    # 3. 检查 Final Answer
    final_match = re.search(r"Final Answer[:：]\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        return {"type": "final", "answer": final_match.group(1).strip()}

    # 4. 没匹配到任何格式 -> 纯文本，当作最终回答
    return {"type": "final", "answer": text}


def _execute_action(action_text: str) -> str:
    """执行工具调用。

    解析 "rag_search(关键词)" 格式，调用对应工具。
    """
    # 解析工具名和参数
    match = re.match(r"(\w+)\s*\(([^)]*)\)", action_text.strip())
    if not match:
        return f"无法解析工具调用: {action_text}"

    tool_name = match.group(1).strip()
    params_str = match.group(2).strip()

    if tool_name not in _TOOLS:
        return f"未知工具: {tool_name}。可用工具: {list(_TOOLS.keys())}"

    # 解析参数：去掉引号，取第一个非空参数
    params = []
    if params_str:
        for p in params_str.split(","):
            p = p.strip().strip("'\"")
            # 去掉 "query=" 之类的前缀
            if "=" in p:
                p = p.split("=", 1)[1].strip().strip("'\"")
            if p:
                params.append(p)

    try:
        # 直接调用对应的工具函数，用位置参数
        if tool_name == "rag_search":
            query = params[0] if params else ""
            return _tool_rag_search(query)
        elif tool_name == "list_models":
            return _tool_list_models()
        elif tool_name == "model_lineage":
            model_id = params[0] if params else ""
            return _tool_model_lineage(model_id)
        elif tool_name == "eval_report":
            model_id = params[0] if params else ""
            return _tool_eval_report(model_id)
        else:
            return f"未知工具: {tool_name}"
    except Exception as e:
        return f"工具执行失败: {e}"


# ---------------------------------------------------------------------------
# 本地模型加载与生成
# ---------------------------------------------------------------------------

def _local_generate(model_dir: Path, prompt: str, cfg: dict) -> str:
    """用本地训练的模型生成文本。"""
    answer, _, _ = infer._real_generate(model_dir, prompt, cfg, lambda x: None)
    return answer


# ---------------------------------------------------------------------------
# ReAct Agent 核心：思考-行动-观察循环
# ---------------------------------------------------------------------------

def react_run(
    question: str,
    model_dir: Path,
    cfg: dict,
    history: list = None,
    max_iterations: int = 3,
    system_prompt: str = "",
) -> dict:
    """运行 ReAct 循环。

    流程：
    1. 构建 ReAct 提示词
    2. 让模型生成 Thought + Action
    3. 如果是工具调用 -> 执行工具 -> 拿到 Observation
    4. 把 Observation 加入上下文 -> 回到步骤1
    5. 如果是 Final Answer -> 返回最终回答

    Args:
        question: 用户问题
        model_dir: 模型目录路径
        cfg: 配置
        history: 对话历史
        max_iterations: 最大循环次数（防止死循环）
        system_prompt: 模型领域身份（从 meta.json 读取，注入到 ReAct 提示词）

    Returns:
        {"answer": "...", "tool_calls": [...], "iterations": N}
    """
    history = history or []
    tool_results = []
    all_tool_calls = []

    # === 第0层：关键词路由（针对小模型优化） ===
    # 0.5B 模型不擅长按 ReAct 格式输出，先用关键词判断是否需要工具
    route = _keyword_route(question)
    if route:
        # 关键词匹配到了，直接执行工具
        observation = _execute_action(route)
        tool_results.append({"action": route, "observation": observation})
        all_tool_calls.append({
            "iteration": 0,
            "thought": f"关键词路由匹配: {route}",
            "action": route,
            "observation": observation[:500],
        })

    # === 第1层：ReAct 循环（模型生成） ===
    for i in range(max_iterations):
        # 1. 构建提示词（注入模型领域身份）
        prompt = _build_react_prompt(question, history, tool_results, system_prompt)

        # 如果第一轮已经有工具结果，提示模型基于结果回答
        if tool_results and i == 0:
            prompt += "\nAnswer:"

        # 2. 模型生成
        raw_output = _local_generate(model_dir, prompt, cfg)

        # 3. 解析输出
        parsed = _parse_react_output(raw_output)

        # 4. 如果是最终回答，结束循环
        if parsed["type"] == "final":
            return {
                "answer": parsed["answer"],
                "tool_calls": all_tool_calls,
                "iterations": i + 1,
            }

        # 5. 如果是工具调用，执行工具
        action = parsed.get("action", "")
        thought = parsed.get("thought", "")

        if action:
            observation = _execute_action(action)
            tool_results.append({
                "action": action,
                "observation": observation,
            })
            all_tool_calls.append({
                "iteration": i + 1,
                "thought": thought,
                "action": action,
                "observation": observation[:500],
            })

    # 超过最大迭代次数，如果有工具结果，直接用它作为回答
    if tool_results:
        last_obs = tool_results[-1]["observation"]
        return {
            "answer": last_obs[:500],
            "tool_calls": all_tool_calls,
            "iterations": max_iterations,
        }

    # 没有工具结果，返回最后一次模型输出
    return {
        "answer": raw_output.strip() if raw_output else "无法生成回答",
        "tool_calls": all_tool_calls,
        "iterations": max_iterations,
    }


# ---------------------------------------------------------------------------
# 会话管理
# ---------------------------------------------------------------------------

def create_session(
    model_id: str = "",
    system_prompt: str = "",
    memory_window: int = 5,
) -> dict:
    """创建一个本地 ReAct Agent 会话。

    Args:
        model_id: 已训练模型ID，为空则用最新模型
        system_prompt: 自定义系统提示词
        memory_window: 滑动窗口轮数
    """
    session_id = str(uuid.uuid4())[:8]
    cfg = load_config()

    # 找到模型目录 + 读取训练时记录的领域身份
    model_dir = None
    model_meta = {}
    if model_id:
        for p in MODELS.iterdir():
            if p.name == model_id or (p / "meta.json").exists():
                try:
                    meta = json.loads((p / "meta.json").read_text(encoding="utf-8-sig"))
                    if meta.get("model_id") == model_id:
                        model_dir = p
                        model_meta = meta
                        break
                except Exception:
                    pass

    # 如果没找到指定模型，用最新的
    if not model_dir:
        models = sorted(MODELS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if models:
            model_dir = models[0]
            try:
                model_meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8-sig"))
            except Exception:
                pass

    # 身份优先级：显式传入 > 模型训练时记录的 > 通用兜底
    resolved_prompt = system_prompt or model_meta.get("system_prompt") or ""

    _sessions[session_id] = {
        "session_id": session_id,
        "model_id": model_id or (model_dir.name if model_dir else ""),
        "model_dir": str(model_dir) if model_dir else "",
        "system_prompt": resolved_prompt,
        "memory_window": memory_window,
        "messages": [],
        "created_at": time.time(),
    }
    return _sessions[session_id]


def get_session(session_id: str) -> dict | None:
    return _sessions.get(session_id)


def list_sessions() -> list[str]:
    return list(_sessions.keys())


def delete_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


# ---------------------------------------------------------------------------
# 对话接口
# ---------------------------------------------------------------------------

def chat(session_id: str, message: str) -> dict:
    """本地 ReAct Agent 对话。

    使用自己训练的模型 + ReAct 文本模式调用工具。
    不依赖任何外部 API。
    """
    session = _sessions.get(session_id)
    if not session:
        raise ValueError(f"会话不存在: {session_id}")

    cfg = load_config()
    model_dir = Path(session["model_dir"]) if session["model_dir"] else None

    if not model_dir or not model_dir.exists():
        raise RuntimeError(f"模型目录不存在: {model_dir}")

    # 构建历史
    history = session["messages"][-(session["memory_window"] * 2):]

    # 运行 ReAct 循环（传入模型领域身份）
    result = react_run(
        question=message,
        model_dir=model_dir,
        cfg=cfg,
        history=history,
        max_iterations=3,
        system_prompt=session.get("system_prompt", ""),
    )

    # 更新会话历史
    session["messages"].append({"role": "user", "content": message})
    session["messages"].append({"role": "assistant", "content": result["answer"]})

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "tool_calls": result["tool_calls"],
        "iterations": result["iterations"],
        "model_id": session["model_id"],
        "messages": session["messages"],
    }

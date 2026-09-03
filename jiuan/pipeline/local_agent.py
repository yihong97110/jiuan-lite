"""本地 ReAct Agent：使用自己训练的模型 + RAG + 工具调用，完全不依赖外部 API。

生产级架构（v2，按 7B 崩坏防护设计）：

  用户输入 → 安全校验层（注入检测/敏感词）
      ↓
  记忆管理器（三层）
      ├─ 短期工作记忆：本轮 ReAct 循环的 Thought/Action/Observation（截断保留）
      ├─ 会话记忆：历史对话，超长自动压缩（近N轮完整+更早摘要标记）
      └─ 知识库长期记忆：RAG 工具按需检索
      ↓
  Prompt 构造器：工具清单 + 压缩后历史 + 工作记忆
      ↓
  推理调度层：vLLM API 优先 / mock / transformers 直载兜底
      ↓
  输出解析器（四级阶梯）
      ① 正则提取 Thought/Action/Answer
      ② JSON/参数自动修复（括号不闭合、引号缺失）
      ③ 结构化兜底抽取
      ④ 关键词路由（轨迹标注 keyword-fallback）
      ↓
  工具执行沙箱：参数校验 / 超时控制 / 异常捕获 / 结果截断清洗
      ↓
  循环控制：最大轮次 + 重复调用检测 + 结束原因（completed/max/repeated）
      ↓
  事件日志：每步 prompt快照/原始输出/解析结果/工具耗时/失败原因 落盘
           （后续可作 7B Agent 微调数据集）

无需 vLLM、无需 DeepSeek、无需 GPU、无需 function calling。
"""
from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from ..common import load_config, MODELS, DATA, mock_mode
from .. import registry
from . import rag_backend
from . import infer


# ---------------------------------------------------------------------------
# 会话存储 + Agent 测评轨迹落盘
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}

# 每轮对话轨迹（JSONL）：Agent 测评的数据底座
TRACE_STORE = DATA / "agent_eval" / "traces.jsonl"

# 全链路事件日志（JSONL）：每步 prompt快照/原始输出/解析结果/工具耗时/失败原因。
# 比轨迹更细粒度，是后续 7B Agent 微调样本的原始素材。
EVENT_STORE = DATA / "agent_eval" / "events.jsonl"

# ---------------------------------------------------------------------------
# 安全校验层：Prompt 注入检测
# ---------------------------------------------------------------------------

# 注入攻击特征（正则，命中即拦截）
_INJECTION_PATTERNS = [
    (r"忽略(以上|之前|上述|前面)(的)?(所有)?(指令|提示|规则|约束)", "指令覆盖"),
    (r"(disregard|ignore)\s+(all\s+)?(previous|above|prior|earlier)?\s*(instructions?|prompts?|rules?|context)", "EN注入"),
    (r"(system|系统)\s*(提示词|prompt)", "系统提示词探测"),
    (r"你(现在)?(是|扮演).{0,12}(不受限|无限制|越狱|DAN)", "越狱尝试"),
    (r"(泄露|显示|打印|给我看).{0,10}(系统提示|初始指令|你的指令)", "提示词泄露"),
]


def security_check(text: str) -> dict:
    """Prompt 注入检测（轻量启发式，生产可换专用分类器）。

    Returns:
        {"blocked": bool, "reason": str, "hits": [命中标签]}
    """
    hits = []
    for pat, label in _INJECTION_PATTERNS:
        if re.search(pat, text or "", re.IGNORECASE):
            hits.append(label)
    return {
        "blocked": bool(hits),
        "reason": f"疑似Prompt注入: {','.join(hits)}" if hits else "",
        "hits": hits,
    }


# ---------------------------------------------------------------------------
# 记忆管理器：三层记忆
# ---------------------------------------------------------------------------

class WorkingMemory:
    """短期工作记忆：仅本轮 ReAct 循环的 Thought/Action/Observation。

    - 最近 2 条观察保留完整（截断到 obs_chars），更早的压到 obs_chars//2
      （旧观察冗余信息干扰模型判断，7B 上下文有限）
    - 进入 prompt 的总量有硬上限，防止长循环上下文膨胀
    """

    def __init__(self, obs_chars: int = 600, max_total_chars: int = 2400):
        self.steps: list[dict] = []
        self.obs_chars = obs_chars
        self.max_total_chars = max_total_chars

    def add(self, thought: str, action: str, observation: str) -> None:
        self.steps.append({
            "thought": thought or "",
            "action": action or "",
            "observation": observation or "",
        })

    def render(self) -> str:
        """渲染进 prompt 的工作记忆文本（带预算控制）。"""
        if not self.steps:
            return ""
        budget = self.max_total_chars
        parts = []
        for i, s in enumerate(self.steps):
            is_recent = i >= len(self.steps) - 2
            limit = self.obs_chars if is_recent else self.obs_chars // 2
            obs = s["observation"][:limit] + ("…[已压缩]" if len(s["observation"]) > limit else "")
            th = s["thought"][:120]
            line = f"第{i+1}步 Thought: {th}\n第{i+1}步 Action: {s['action']}\n第{i+1}步 Observation: {obs}"
            if len(line) > budget:
                line = line[:budget] + "…[记忆预算耗尽]"
                parts.append(line)
                break
            budget -= len(line)
            parts.append(line)
        return "\n".join(parts)


def compress_session_history(history: list, keep_turns: int = 4, max_chars_per_msg: int = 160) -> list:
    """会话记忆压缩：近 keep_turns 轮完整（每条截断），更早的标记为已压缩单行摘要。

    防止长对话后上下文膨胀撑爆 7B 窗口。
    """
    if not history:
        return []
    turns = history[-(keep_turns * 2):]  # 1 轮 = user+assistant 两条
    earlier = history[:-(keep_turns * 2)] if len(history) > keep_turns * 2 else []
    out = []
    if earlier:
        n_user = sum(1 for h in earlier if h.get("role") == "user")
        out.append({"role": "system_note", "content": f"[更早的{len(earlier)}条对话已压缩，含{n_user}个用户问题]"})
    for h in turns:
        out.append({
            "role": h.get("role", "user"),
            "content": (h.get("content") or "")[:max_chars_per_msg]
            + ("…" if len(h.get("content") or "") > max_chars_per_msg else ""),
        })
    return out


# ---------------------------------------------------------------------------
# JSON / 参数自动修复（7B 头号痛点的第②级防线）
# ---------------------------------------------------------------------------

def repair_action_text(action_text: str) -> str:
    """修复模型输出的 Action 常见崩坏。

    覆盖样本：
    - 括号不闭合: "rag_search(什么是DNA"        -> "rag_search(什么是DNA)"
    - 引号缺失:   'model_lineage("abc123)'       -> 'model_lineage(abc123)'
    - JSON 写崩:  'search({"query": "x}'          -> 'search({"query": "x"})'
    - 中英括号混用: "rag_search（关键词）"        -> "rag_search(关键词)"
    """
    t = (action_text or "").strip()
    if not t:
        return t
    # 全角括号 -> 半角
    t = t.replace("（", "(").replace("）", ")")
    # 去掉整体引号包裹（模型常用引号包整个 action）
    t = re.sub(r'^["\'](.*)["\']$', r"\1", t)

    # 无括号形式: "rag_search 关键词"（仅对已知工具补括号）
    m = re.match(r"^([\w.]+)\s*\((.*)$", t, re.DOTALL)
    if not m:
        m2 = re.match(r"^([\w.]+)\s+(.+)$", t)
        if m2 and m2.group(1) in _TOOLS:
            return f"{m2.group(1)}({m2.group(2)})"
        return t
    name, rest = m.group(1), m.group(2)

    # 找到参数区间的真实结束：从左向右扫，引号内跳过，括号配平即停
    depth = 1
    in_str, quote = False, ""
    end_idx = len(rest)  # 默认：没找到闭合
    for idx, ch in enumerate(rest):
        if in_str:
            if ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str, quote = True, ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break
    core = rest[:end_idx].strip()

    # 参数内部小修复：悬空引号 / 未配对括号 / JSON 花括号
    # 1) 参数内多余的未配对右括号（如 abc) ）-> 去掉（外层括号由我们统一补）
    while core.endswith(")") and core.count("(") < core.count(")"):
        core = core[:-1].rstrip()
    # 2) 去掉参数两端的引号包裹（"abc" -> abc）
    core = re.sub(r'^["\'](.*)["\']$', r"\1", core)
    # 3) 去掉开头悬空的单个引号（"abc123 -> abc123）
    core = re.sub(r'^["\']', "", core) if not re.search(r'^["\'].*["\']$', core) else core
    # 4) 参数中段悬空开引号未闭合（如 {"query": "x）-> 补闭引号
    if _dangling_quote(core):
        core += '"'
    # 5) 不配对花括号/方括号
    if core.count("{") > core.count("}"):
        core += "}" * (core.count("{") - core.count("}"))
    if core.count("[") > core.count("]"):
        core += "]" * (core.count("[") - core.count("]"))
    return f"{name}({core})"


def _dangling_quote(s: str) -> bool:
    """判断字符串是否存在奇数个双引号（至少有一个未闭合）。"""
    return s.count('"') % 2 == 1


def _append_trace(record: dict) -> None:
    """把一轮对话轨迹追加到 data/agent_eval/traces.jsonl。"""
    try:
        TRACE_STORE.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_STORE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 落盘失败不阻断对话


def _append_event(evt: dict) -> None:
    """全链路事件日志：每步快照（原始输出/解析结果/工具耗时/失败原因）。

    事件类型：
    - security_block   安全校验拦截
    - prompt           构造的提示词快照（头/尾截断）
    - generate         模型生成（原始输出 + 耗时）
    - parse            解析结果（type/thought/action/repaired/failed）
    - tool_call        工具执行（入参/结果/耗时/截断/异常）
    - loop_end         循环终止（结束原因）
    """
    try:
        EVENT_STORE.parent.mkdir(parents=True, exist_ok=True)
        with EVENT_STORE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    except Exception:
        pass


def read_events(limit: int = 50) -> list[dict]:
    """读取最近 N 条事件日志（倒序，最新在前）。微调素材预览用。"""
    try:
        if not EVENT_STORE.exists():
            return []
        lines = EVENT_STORE.read_text(encoding="utf-8").splitlines()
        out = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def eval_stats() -> dict:
    """Agent 测评统计（基于历史轨迹 traces.jsonl）。

    指标：
    - total_turns: 总对话轮数
    - tool_call_rate: 调用过工具的轮占比（Agent 能力利用率）
    - model_driven_rate: 模型自主决策占比（非关键词兜底）
    - avg_iterations / avg_latency_ms: 平均迭代轮数 / 平均耗时
    - per_model: 按模型分组的同口径统计（模型对比）

    未来扩展：答案质量 LLM-as-Judge、工具级成功率、轨迹回归测试。
    """
    total = 0
    tool_turns = 0
    routed_turns = 0
    iterations_sum = 0
    latency_sum = 0
    repaired_actions = 0
    tool_errors = 0
    security_blocks = 0
    end_reasons: dict[str, int] = {}
    per_model: dict[str, dict] = {}

    try:
        if TRACE_STORE.exists():
            for line in TRACE_STORE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                total += 1
                n_tools = rec.get("tool_call_count", len(rec.get("tool_calls", [])))
                was_routed = bool(rec.get("routed"))
                iters = rec.get("iterations", 1) or 1
                lat = rec.get("latency_ms", 0) or 0
                if n_tools > 0:
                    tool_turns += 1
                if was_routed:
                    routed_turns += 1
                iterations_sum += iters
                latency_sum += lat
                repaired_actions += rec.get("repaired_actions", 0) or 0
                tool_errors += rec.get("tool_errors", 0) or 0
                if rec.get("security_blocked"):
                    security_blocks += 1
                er = rec.get("end_reason") or "completed"
                end_reasons[er] = end_reasons.get(er, 0) + 1

                mid = rec.get("model_id") or "(unknown)"
                m = per_model.setdefault(mid, {
                    "turns": 0, "tool_turns": 0, "routed_turns": 0, "latency_sum": 0,
                    "repaired": 0, "errors": 0,
                })
                m["turns"] += 1
                if n_tools > 0:
                    m["tool_turns"] += 1
                if was_routed:
                    m["routed_turns"] += 1
                m["latency_sum"] += lat
                m["repaired"] += rec.get("repaired_actions", 0) or 0
                m["errors"] += rec.get("tool_errors", 0) or 0
    except Exception:
        pass

    models = [
        {
            "model_id": mid,
            "turns": m["turns"],
            "tool_rate": round(m["tool_turns"] / m["turns"], 3) if m["turns"] else None,
            "routed_rate": round(m["routed_turns"] / m["turns"], 3) if m["turns"] else None,
            "repaired_actions": m["repaired"],
            "tool_errors": m["errors"],
            "avg_latency_ms": int(m["latency_sum"] / m["turns"]) if m["turns"] else None,
        }
        for mid, m in per_model.items()
    ]
    # 按轮数倒序，常用模型在前
    models.sort(key=lambda x: x["turns"], reverse=True)

    return {
        "total_turns": total,
        "tool_call_rate": round(tool_turns / total, 3) if total else None,
        "model_driven_rate": round((total - routed_turns) / total, 3) if total else None,
        "avg_iterations": round(iterations_sum / total, 2) if total else None,
        "avg_latency_ms": int(latency_sum / total) if total else None,
        "repaired_actions": repaired_actions,       # 7B 崩坏被 JSON 修复挽救的次数
        "tool_errors": tool_errors,                 # 工具执行失败（超时/异常）次数
        "security_blocks": security_blocks,         # 注入拦截次数
        "end_reasons": end_reasons,                 # 结束原因分布（completed/max/repeated/...）
        "per_model": models,
        "trace_store": str(TRACE_STORE),
        "event_store": str(EVENT_STORE),
        "note": "轨迹统计基于 traces.jsonl；全链路事件（prompt快照/原始输出/工具耗时）在 events.jsonl，可作 7B Agent 微调数据集。",
    }


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
- 每个工具只需调用一次；收到 Observation 后信息足够就必须立即输出 Answer，禁止重复调用同一工具

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


def _build_react_prompt(
    question: str,
    history: list,
    tool_results: list = None,
    working_memory: "WorkingMemory | None" = None,
) -> str:
    """构建 ReAct 提示词（三层记忆版）。

    分层（防上下文膨胀）：
    - 指令层：工具清单 + 格式要求（固定）
    - 会话记忆：compress_session_history 压缩后的近 N 轮 + 更早摘要标记
    - 工作记忆：本轮循环的 Thought/Action/Observation（WorkingMemory 预算控制），
      兼容旧的 tool_results 参数（无 WorkingMemory 时降级使用）
    身份由调用方单独传给 _local_generate（vLLM 走 system message）。
    """
    parts = [REACT_INSTRUCTIONS]

    # 会话记忆（压缩后）
    if history:
        compressed = compress_session_history(history)
        has_note = any(h.get("role") == "system_note" for h in compressed)
        if compressed:
            parts.append("对话历史：")
            for h in compressed:
                if h.get("role") == "system_note":
                    parts.append(h["content"])
                    continue
                role = "用户" if h.get("role") == "user" else "助手"
                parts.append(f"{role}: {h['content']}")
            if has_note:
                parts.append("（更早内容已压缩省略）")

    # 短期工作记忆（本轮循环）——优先 WorkingMemory，降级用 tool_results
    if working_memory is not None:
        wm = working_memory.render()
        if wm:
            parts.append("本轮已执行的步骤（工作记忆）：")
            parts.append(wm)
            parts.append("若 Observation 已能回答问题，必须直接输出 Answer 完成回答，不要重复调用工具；仅在信息明显不足时才发起新的 Action。")
    elif tool_results:
        parts.append("工具结果：")
        for tr in tool_results[-2:]:
            parts.append(f"Observation: {tr['observation'][:300]}")
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

    - "Thought: ..." + "Action: rag_search(关键词)" -> 调用工具（带思考）
    - "Action: rag_search(关键词)" -> 调用工具
    - "Answer: 回答内容" -> 最终回答
    - "Final Answer: 回答内容" -> 最终回答
    - 纯文本（没匹配到格式）-> 当作最终回答
    """
    text = (text or "").strip()

    # 0. 先提取 Thought（无论后续是 Action 还是 Answer，思考都有展示价值）
    thought = ""
    thought_match = re.search(r"Thought[:：]\s*(.+)", text)
    if thought_match:
        thought = thought_match.group(1).strip()

    # 1. 检查 Action（工具调用）
    action_match = re.search(r"Action[:：]\s*(.+)", text, re.IGNORECASE)
    if action_match:
        action_text = action_match.group(1).strip()

        # 如果 Action 里是 Final Answer
        if re.match(r"(Final\s*)?Answer", action_text, re.IGNORECASE):
            fa_match = re.search(r"Answer[:：]\s*(.+)", action_text, re.DOTALL | re.IGNORECASE)
            if fa_match:
                return {"type": "final", "answer": fa_match.group(1).strip()}

        # 检查是否包含有效工具名（含第②级：崩坏自动修复）
        for tool_name in _TOOLS:
            if tool_name in action_text:
                repaired = repair_action_text(action_text)
                return {
                    "type": "action",
                    "thought": thought,
                    "action": repaired,
                    "repaired": repaired.strip() != action_text.strip(),
                }

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


def _parse_action_params(action_text: str) -> tuple:
    """解析 "tool_name(参数)" 为 (tool_name, [参数列表])。解析失败返回 ("", [])。"""
    match = re.match(r"([\w.]+)\s*\((.*)\)\s*$", action_text.strip(), re.DOTALL)
    if not match:
        return "", []
    tool_name = match.group(1).strip()
    params_str = match.group(2).strip()
    params = []
    if params_str:
        for p in params_str.split(","):
            p = p.strip().strip("'\"")
            # 去掉 "query=" 之类的前缀
            if "=" in p and not p.startswith("{"):
                p = p.split("=", 1)[1].strip().strip("'\"")
            if p:
                params.append(p)
    return tool_name, params


def _sanitize_observation(obs: str, max_chars: int = 800) -> tuple:
    """工具结果截断清洗：去多余空白 + 截断，返回 (清洗后文本, 是否截断)。"""
    cleaned = re.sub(r"\n{3,}", "\n\n", (obs or "")).strip()
    truncated = len(cleaned) > max_chars
    if truncated:
        cleaned = cleaned[:max_chars] + f"…[结果已截断，原长{len(cleaned)}字符]"
    return cleaned, truncated


# 工具沙箱配置：超时（秒）与结果截断（字符）
TOOL_TIMEOUT_S = 20
TOOL_RESULT_MAX_CHARS = 800

# 同一（工具+参数）连续重复调用的容忍次数，超过判定为原地打转
REPEAT_TOLERANCE = 2


def _execute_action_safe(action_text: str, session_id: str = "") -> dict:
    """工具执行沙箱：参数校验 + 超时控制 + 异常捕获 + 结果截断清洗。

    任何失败都以文本 Observation 回灌给模型（而不是抛异常打断循环），
    模型看到失败原因后可自行调整。全链路事件落盘。

    Returns:
        {"observation": str, "tool": str, "params": [...],
         "latency_ms": int, "truncated": bool, "error": str|None}
    """
    t0 = time.time()
    evt = {
        "ts": time.time(), "session_id": session_id, "type": "tool_call",
        "action": action_text[:200],
    }
    result = {
        "observation": "", "tool": "", "params": [],
        "latency_ms": 0, "truncated": False, "error": None,
    }

    # 1) 参数校验
    tool_name, params = _parse_action_params(action_text)
    result["tool"], result["params"] = tool_name, params[:3]  # 参数数量也做上限
    if not tool_name:
        result["observation"] = f"无法解析工具调用: {action_text[:100]}。请使用 工具名(参数) 格式。"
        result["error"] = "parse_failed"
        evt.update({"error": "parse_failed", "latency_ms": 0})
        _append_event(evt)
        return result
    if tool_name not in _TOOLS:
        result["observation"] = f"未知工具: {tool_name}。可用工具: {list(_TOOLS.keys())}"
        result["error"] = "unknown_tool"
        evt.update({"error": "unknown_tool", "latency_ms": 0})
        _append_event(evt)
        return result

    # 2) 超时控制：线程池执行，超时杀不掉线程但能返回（工具函数应为纯 CPU/本地IO）
    def _dispatch():
        if tool_name == "rag_search":
            return _tool_rag_search(params[0] if params else "")
        if tool_name == "list_models":
            return _tool_list_models()
        if tool_name == "model_lineage":
            return _tool_model_lineage(params[0] if params else "")
        if tool_name == "eval_report":
            return _tool_eval_report(params[0] if params else "")
        return f"未知工具: {tool_name}"

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            obs = ex.submit(_dispatch).result(timeout=TOOL_TIMEOUT_S)
    except TimeoutError:
        result["observation"] = f"工具 {tool_name} 执行超时（>{TOOL_TIMEOUT_S}s），请换一个思路或直接用 Answer 回答。"
        result["error"] = "timeout"
        result["latency_ms"] = int((time.time() - t0) * 1000)
        evt.update({"error": "timeout", "latency_ms": result["latency_ms"]})
        _append_event(evt)
        return result
    except Exception as exc:
        # 异常回灌：失败原因作为 Observation 告知模型，由其自行调整
        result["observation"] = f"工具 {tool_name} 执行出错: {str(exc)[:150]}。可重试或换工具。"
        result["error"] = str(exc)[:100]
        result["latency_ms"] = int((time.time() - t0) * 1000)
        evt.update({"error": result["error"], "latency_ms": result["latency_ms"]})
        _append_event(evt)
        return result

    # 3) 结果截断清洗
    cleaned, truncated = _sanitize_observation(obs, TOOL_RESULT_MAX_CHARS)
    result["observation"] = cleaned
    result["truncated"] = truncated
    result["latency_ms"] = int((time.time() - t0) * 1000)
    evt.update({
        "tool": tool_name, "latency_ms": result["latency_ms"],
        "truncated": truncated, "result_head": cleaned[:150],
    })
    _append_event(evt)
    return result


def _execute_action(action_text: str) -> str:
    """执行工具调用（兼容旧接口：返回 Observation 文本）。"""
    return _execute_action_safe(action_text)["observation"]


# ---------------------------------------------------------------------------
# 本地模型加载与生成
# ---------------------------------------------------------------------------

def _vllm_generate(model_dir: Path, prompt: str, cfg: dict, system_prompt: str = "") -> "str | None":
    """通过本地 vLLM OpenAI API 生成（模型已挂载为 LoRA，无显存冲突）。

    3090 24GB 上 vLLM 与 transformers 直载互斥：
    - vLLM 运行时（占 ~22GB）：transformers 再载 7B 会 OOM。
      已训练模型恰恰挂在 vLLM 上（lora module 名 = model_id），
      直接走 API 即可，ReAct 文本模式不受影响（仍是纯文本生成+正则解析）。

    身份走 system message（chat 模板下权重高于 user 内拼接，
    避免"我是Qwen"身份残留），ReAct 指令与问题走 user message。

    返回 None 表示 vLLM 不可用/模型未挂载，由调用方回退 transformers 直载。
    """
    import urllib.request

    chat_cfg = cfg.get("chat", {}) or {}
    base_url = (chat_cfg.get("vllm_base_url") or "http://127.0.0.1:8001/v1").rstrip("/")

    # 模型名 = vLLM lora module 名（平台约定：module 名 = model_id）
    served = ""
    try:
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8-sig"))
        served = meta.get("served_model_name") or meta.get("model_id") or ""
    except Exception:
        served = model_dir.name if model_dir else ""
    if not served:
        return None

    # ReAct 需要格式稳定，低温度（infer.do_sample=false 的语义等价）
    temperature = 0.2 if not cfg.get("infer", {}).get("do_sample", False) else 0.7
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": served,
        "messages": messages,
        "max_tokens": int(cfg.get("infer", {}).get("max_new_tokens", 512)),
        "temperature": temperature,
    }
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (data["choices"][0]["message"].get("content") or "").strip()
        return content or None
    except Exception:
        return None  # 服务未启动/模型未挂载 -> 静默回退 transformers


def _local_generate(model_dir: Path, prompt: str, cfg: dict, system_prompt: str = "") -> str:
    """本地生成：优先 vLLM API（身份走 system message）；
    mock 模式走 mock 生成（保证 ReAct 链路离线可演示）；
    最后 transformers 直载兜底（身份拼 prompt）。
    """
    answer = _vllm_generate(model_dir, prompt, cfg, system_prompt)
    if answer:
        return answer
    # 身份与 ReAct 指令合并（mock / transformers 兜底共用）
    full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
    if mock_mode():
        # 离线演示模式：走平台 mock 生成，ReAct 链路（含关键词兜底）完整可演示
        return infer._mock_generate(model_dir, full_prompt, lambda x: None)
    # transformers 兜底路径：infer._real_generate 不支持自定义 system，身份拼回 prompt 开头
    answer, _, _ = infer._real_generate(model_dir, full_prompt, cfg, lambda x: None)
    return answer


# ---------------------------------------------------------------------------
# ReAct Agent 核心：思考-行动-观察循环
# ---------------------------------------------------------------------------

def react_run(
    question: str,
    model_dir: Path,
    cfg: dict,
    history: list = None,
    max_iterations: int = 5,
    system_prompt: str = "",
    session_id: str = "",
) -> dict:
    """运行模型驱动的 ReAct 循环（生产级 v2）。

    链路：安全检查 -> 循环{构造提示词(三层记忆) -> 生成 -> 四级解析
         -> [Action: 沙箱执行 -> 工作记忆 -> 重复检测] | [Answer: 终止]}
    -> 结束原因 -> 轨迹落盘。

    循环控制（不止 max 轮数）：
    - completed       模型给出 Answer，正常结束
    - keyword-fallback 首轮未调工具，关键词路由兜底执行一次
    - max_iterations  轮次耗尽，返回阶段性结果 + 未完成提示
    - repeated        同一(工具+参数)连续超过容忍次数：首次重复先注入纠偏提示，
                      纠偏后仍重复才判定打转，强制终止（末次兜底总结收口）

    Returns:
        {"answer", "tool_calls", "iterations", "routed", "end_reason",
         "security": {...}, "events_logged": int}
    """
    history = history or []
    all_tool_calls = []
    routed = False
    raw_output = ""
    end_reason = "completed"
    identity = system_prompt or DEFAULT_IDENTITY
    wm = WorkingMemory()

    # === 安全校验层 ===
    sec = security_check(question)
    if sec["blocked"]:
        _append_event({
            "ts": time.time(), "session_id": session_id,
            "type": "security_block", "hits": sec["hits"], "question": question[:150],
        })
        return {
            "answer": "你的输入包含疑似 Prompt 注入内容，已拦截。请换一种正常提问方式。",
            "tool_calls": [],
            "iterations": 0,
            "routed": False,
            "end_reason": "security_blocked",
            "security": sec,
            "events_logged": 1,
        }

    # === ReAct 循环（模型决策优先） ===
    recent_actions: list[str] = []  # 重复检测窗口
    nudged_action = ""  # 已注入过纠偏提示的 action（每个工具只纠偏一次）
    for i in range(max_iterations):
        # 1. 构造提示词（三层记忆：指令 + 压缩会话 + 工作记忆）
        prompt = _build_react_prompt(question, history, working_memory=wm)
        _append_event({
            "ts": time.time(), "session_id": session_id, "type": "prompt",
            "iteration": i + 1, "chars": len(prompt),
            "head": prompt[:150], "tail": prompt[-100:],
        })

        # 2. 模型生成
        t0 = time.time()
        raw_output = _local_generate(model_dir, prompt, cfg, identity) or ""
        gen_ms = int((time.time() - t0) * 1000)

        # 3. 解析（①正则 ②JSON修复）
        parsed = _parse_react_output(raw_output)
        _append_event({
            "ts": time.time(), "session_id": session_id, "type": "generate",
            "iteration": i + 1, "latency_ms": gen_ms,
            "raw_output": raw_output[:300],
            "parse_type": parsed.get("type"),
            "repaired": bool(parsed.get("repaired")),
        })

        # 4. 最终回答：结束循环
        if parsed["type"] == "final":
            answer = parsed["answer"]

            # 首轮未调工具且关键词路由命中：兜底执行一次（小模型场景）
            if not wm.steps and i == 0:
                route = _keyword_route(question)
                if route:
                    routed = True
                    sandbox = _execute_action_safe(route, session_id)
                    wm.add(f"关键词路由兜底: {route}", route, sandbox["observation"])
                    all_tool_calls.append({
                        "iteration": i + 1,
                        "source": "keyword-fallback",
                        "thought": f"模型未发起工具调用，关键词路由兜底: {route}",
                        "action": route,
                        "observation": sandbox["observation"][:500],
                        "error": sandbox.get("error"),
                    })
                    # 结果交回模型总结一次
                    prompt2 = _build_react_prompt(question, history, working_memory=wm) + "\nAnswer:"
                    raw2 = _local_generate(model_dir, prompt2, cfg, identity) or ""
                    parsed2 = _parse_react_output(raw2)
                    return {
                        "answer": parsed2.get("answer") or sandbox["observation"][:500],
                        "tool_calls": all_tool_calls,
                        "iterations": i + 2,
                        "routed": True,
                        "end_reason": "keyword-fallback",
                        "security": sec,
                    }

            _append_event({
                "ts": time.time(), "session_id": session_id, "type": "loop_end",
                "end_reason": "completed", "iterations": i + 1,
            })
            return {
                "answer": answer,
                "tool_calls": all_tool_calls,
                "iterations": i + 1,
                "routed": routed,
                "end_reason": "completed",
                "security": sec,
            }

        # 5. 模型发起 Action：沙箱执行
        action = parsed.get("action", "")
        thought = parsed.get("thought", "")
        if not action:
            continue

        # 重复检测：同一 action 连续出现超容忍次数 -> 判定打转。
        # 第一次重复不终止：跳过执行、注入纠偏提示，给模型一次收口机会；
        # 纠偏后仍重复同一工具才判定打转，强制终止。
        recent_actions.append(action.strip())
        tail = recent_actions[-REPEAT_TOLERANCE:]
        if len(tail) == REPEAT_TOLERANCE and len(set(tail)) == 1:
            if nudged_action != action.strip():
                nudged_action = action.strip()
                hint = (f"系统纠偏：工具 {action.strip()[:60]} 已执行并返回 Observation，"
                        "结果就在上方工作记忆中。禁止再次调用同一工具，"
                        "请立即基于已有 Observation 输出 Answer: 最终答案。")
                wm.add("重复调用被拦截，收到系统纠偏提示", action, hint)
                _append_event({
                    "ts": time.time(), "session_id": session_id, "type": "repeat_nudge",
                    "iteration": i + 1, "action": action[:100],
                })
                continue
            end_reason = "repeated"
            _append_event({
                "ts": time.time(), "session_id": session_id, "type": "loop_end",
                "end_reason": "repeated", "action": action[:100], "iterations": i + 1,
            })
            # 用已有观察直接总结，提示模型收尾
            prompt2 = _build_react_prompt(question, history, working_memory=wm) + "\nAnswer:"
            raw2 = _local_generate(model_dir, prompt2, cfg, identity) or ""
            parsed2 = _parse_react_output(raw2)
            return {
                "answer": parsed2.get("answer") or (wm.steps[-1]["observation"][:400] if wm.steps else "任务未完成"),
                "tool_calls": all_tool_calls,
                "iterations": i + 1,
                "routed": routed,
                "end_reason": "repeated",
                "security": sec,
            }

        sandbox = _execute_action_safe(action, session_id)
        wm.add(thought, action, sandbox["observation"])
        all_tool_calls.append({
            "iteration": i + 1,
            "source": "model",
            "thought": thought,
            "action": action,
            "observation": sandbox["observation"][:500],
            "repaired": bool(parsed.get("repaired")),
            "error": sandbox.get("error"),
            "tool_latency_ms": sandbox.get("latency_ms"),
        })

    # 轮次耗尽：返回阶段性结果 + 明确未完成提示
    _append_event({
        "ts": time.time(), "session_id": session_id, "type": "loop_end",
        "end_reason": "max_iterations", "iterations": max_iterations,
    })
    partial = wm.steps[-1]["observation"][:400] if wm.steps else ""
    return {
        "answer": (partial + "\n（已达最大轮次，任务可能未完成）") if partial
        else (raw_output or "").strip()[:400] or "无法生成回答",
        "tool_calls": all_tool_calls,
        "iterations": max_iterations,
        "routed": routed,
        "end_reason": "max_iterations",
        "security": sec,
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
        "override_prompt": system_prompt,  # 显式传入的身份，switch_model 时优先
        "memory_window": memory_window,
        "messages": [],
        "created_at": time.time(),
    }
    return _sessions[session_id]


def get_session(session_id: str) -> dict | None:
    return _sessions.get(session_id)


def list_sessions() -> list[str]:
    return list(_sessions.keys())


def switch_model(session_id: str, model_id: str) -> dict:
    """适配器热插拔：会话中途切换底层模型（如 工具调用LoRA -> 摘要LoRA）。

    vLLM 多 LoRA 挂载时无需重启服务，只换 served_model_name；
    transformers 直载路径靠 _REAL_CACHE 缓存复用。
    会话记忆（messages）保留，跨模型延续上下文。

    Returns:
        更新后的会话信息
    """
    session = _sessions.get(session_id)
    if not session:
        raise ValueError(f"会话不存在: {session_id}")

    # 找目标模型目录 + meta
    model_dir, model_meta = None, {}
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
    if not model_dir or not model_dir.exists():
        raise ValueError(f"模型不存在: {model_id}")

    old_model = session["model_id"]
    session["model_id"] = model_id
    session["model_dir"] = str(model_dir)
    # 身份跟随新模型的训练身份（热切换的意义之一：换领域人格）
    session["system_prompt"] = session.get("override_prompt") or model_meta.get("system_prompt") or ""
    session["switched_at"] = time.time()
    session["switch_history"] = (session.get("switch_history") or []) + [
        {"from": old_model, "to": model_id, "at": time.time()}
    ]
    return session


def delete_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


# ---------------------------------------------------------------------------
# 对话接口
# ---------------------------------------------------------------------------

def chat(session_id: str, message: str) -> dict:
    """本地 ReAct Agent 对话（生产级 v2）。

    使用自己训练的模型 + ReAct 文本模式调用工具，不依赖任何外部 API。
    轨迹（traces.jsonl）与全链路事件（events.jsonl）双落盘。
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

    # 运行 ReAct 循环（传入模型领域身份 + session_id 用于事件日志关联）
    t0 = time.time()
    result = react_run(
        question=message,
        model_dir=model_dir,
        cfg=cfg,
        history=history,
        max_iterations=5,
        system_prompt=session.get("system_prompt", ""),
        session_id=session_id,
    )
    latency_ms = int((time.time() - t0) * 1000)

    # 轨迹落盘（Agent 测评数据底座 + 微调素材索引）
    _append_trace({
        "ts": time.time(),
        "session_id": session_id,
        "model_id": session["model_id"],
        "question": message,
        "answer": result["answer"],
        "tool_calls": result["tool_calls"],
        "tool_call_count": len(result["tool_calls"]),
        "iterations": result["iterations"],
        "routed": result["routed"],
        "end_reason": result.get("end_reason", "completed"),
        "security_blocked": bool(result.get("security", {}).get("blocked")),
        "repaired_actions": sum(1 for tc in result["tool_calls"] if tc.get("repaired")),
        "tool_errors": sum(1 for tc in result["tool_calls"] if tc.get("error")),
        "latency_ms": latency_ms,
    })

    # 更新会话历史
    session["messages"].append({"role": "user", "content": message})
    session["messages"].append({"role": "assistant", "content": result["answer"]})

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "tool_calls": result["tool_calls"],
        "iterations": result["iterations"],
        "routed": result["routed"],
        "end_reason": result.get("end_reason", "completed"),
        "security": result.get("security"),
        "security_blocked": bool(result.get("security", {}).get("blocked")),
        "repaired_actions": sum(1 for tc in result["tool_calls"] if tc.get("repaired")),
        "tool_errors": sum(1 for tc in result["tool_calls"] if tc.get("error")),
        "latency_ms": latency_ms,
        "model_id": session["model_id"],
        "messages": session["messages"],
    }

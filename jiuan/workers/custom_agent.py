"""Custom scenario Agent: prompt-described direction -> data -> train/eval loop.

This worker is the general version of the biology demo. It keeps the same
auditable jiuan-lite flow, but does not require a prepared dataset.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .. import annotation, registry, store
from ..common import DATA, MODELS, ROOT, load_config
from ..pipeline import breadth, dataprep, infer, judge, rag_backend
from ..schemas import Stage, TaskStatus
from . import runner

_lock = threading.RLock()
_thread: threading.Thread | None = None
_agent_task_id: str | None = None
_state: dict[str, Any] = {
    "active": False,
    "status": "idle",
    "message": "未启动",
    "logs": [],
    "summaries": [],
    "steps": [],
}


def status() -> dict:
    with _lock:
        if not _state.get("active") and not _state.get("task_id"):
            latest = _latest_persisted_state()
            if latest:
                return latest
        out = dict(_state)
        for key in ("logs", "summaries", "steps", "parameter_notes"):
            out[key] = list(_state.get(key, []))
        return out


def _latest_persisted_state() -> dict | None:
    try:
        for task in store.list_tasks(Stage.AGENT):
            if task.params.get("mode") == "custom" and task.result:
                return task.result
    except Exception:
        return None
    return None


def _set(**kw: Any) -> None:
    with _lock:
        _state.update(kw)


def _persist_state(task_status: TaskStatus | None = None, log: str | None = None) -> None:
    task_id = _agent_task_id or _state.get("task_id")
    if task_id:
        store.update_task(
            task_id,
            status=task_status,
            result=status(),
            progress=_state.get("progress", ""),
            log=log,
        )


def _log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    with _lock:
        logs = _state.setdefault("logs", [])
        logs.append(line)
        del logs[:-180]
        task_id = _agent_task_id
    if task_id:
        store.update_task(task_id, log=message)


def _progress(text: str) -> None:
    _set(progress=text, message=text)
    if _agent_task_id:
        store.update_task(_agent_task_id, progress=text)


def _add_step(name: str, status_text: str, summary: str, artifacts: dict | None = None) -> None:
    step = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step": name,
        "status": status_text,
        "summary": summary,
        "artifacts": artifacts or {},
    }
    with _lock:
        _state.setdefault("steps", []).append(step)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _safe_name(text: str, default: str = "custom") -> str:
    text = re.sub(r"[\\/:*?\"<>|\s]+", "-", (text or "").strip())
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:48] or default


def _ascii_slug(text: str, default: str = "custom") -> str:
    slug = re.sub(r"[^0-9A-Za-z_.-]+", "-", text or "").strip("-").lower()
    return slug[:40] or default


def _default_base_model() -> str:
    cfg = load_config()
    local = cfg.get("model", {}).get("local_dir")
    return str(local or cfg.get("model", {}).get("name") or "")


def _parameter_notes() -> list[dict]:
    return [
        {
            "name": "强化方向",
            "detail": "决定训练样本、验证题和知识库围绕什么能力展开；写得越具体，生成的数据越贴近目标。",
        },
        {
            "name": "基底模型",
            "detail": "大模型基础能力更强、迁移更好，但显存/训练时间/部署成本更高；小模型便宜、快、适合本机 demo，但专业上限较低。",
        },
        {
            "name": "数据集大小",
            "detail": "默认 20 条用于快速跑通；正式训练应扩大到数百/数千条，并覆盖正例、反例、边界情况和真实业务表达。",
        },
        {
            "name": "迭代轮数",
            "detail": "默认 3 轮，每轮都会回灌新增样本并评测。轮数越多越能查缺补漏，但耗时也更长。",
        },
        {
            "name": "LoRA 与 full",
            "detail": "本页默认 LoRA，训练快、占用低、适合小样本迭代；full 微调改动更彻底，但资源需求明显更高。",
        },
        {
            "name": "Judge",
            "detail": "Judge 更适合看专业性和完整性；ROUGE 只看字面重合。Judge 不可用时流程会自动降级到 ROUGE/BLEU。",
        },
    ]


def _infer_scenario_from_brief(brief: str) -> str:
    text = re.sub(r"\s+", "", brief or "")
    patterns = [
        r"能(?:够)?做(.+?)的大模型",
        r"训练(?:一个|一套)?(?:能(?:够)?做)?(.+?)(?:的大模型|的模型|助手|专家)",
        r"强化(.+?)(?:能力|方向|$)",
        r"面向(.+?)(?:场景|方向|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = re.split(r"[，。,.；;：:、]", m.group(1))[0].strip("的")
            if val:
                return val
    short = re.split(r"[，。,.；;：:\n]", brief or "")[0].strip()
    short = re.sub(r"^(我想|我要|希望|帮我|请)?(训练|强化|构建|打造)?(一个|一套)?", "", short).strip()
    return short or "通用场景"


def _bounded_int(value: Any, default: int, lower: int, upper: int) -> int:
    try:
        n = int(value)
    except Exception:
        n = default
    return max(lower, min(upper, n))


def _focus_terms(direction: str, scenario: str, limit: int = 6) -> list[str]:
    text = re.sub(r"\s+", "", direction or "")
    text = re.sub(r"^(我想|我要|希望|帮我|请)?(训练|强化|构建|打造)?(一个|一套)?", "", text)
    text = re.sub(r"(的大模型|的模型|的助手|专家)$", "", text)
    raw_terms = re.split(r"[，,、；;。.!！?？:：()（）\[\]【】\n]+", text)
    terms: list[str] = []
    for term in [scenario, *raw_terms]:
        val = re.sub(r"^(重点|包括|以及|并且|用于|面向|能够|能做|识别|输出)", "", (term or "").strip())
        val = re.sub(r"(等等|等|能力|方向|场景)$", "", val).strip()
        if not (2 <= len(val) <= 24):
            continue
        if val in terms:
            continue
        terms.append(val)
        if len(terms) >= limit:
            break
    return terms or [scenario or "目标方向"]


def _extract_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _list_items(value: Any, fallback: list[str], limit: int = 6) -> list[str]:
    if isinstance(value, str):
        raw = re.split(r"[\n；;]+", value)
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    out: list[str] = []
    for item in raw:
        text = str(item).strip(" -\t\r\n")
        if text:
            out.append(text[:220])
        if len(out) >= limit:
            break
    return out or fallback


def _section(title: str, items: list[str]) -> dict:
    return {"title": title, "items": items}


def _recommended_params(sample_count: int, eval_sample_count: int, max_iterations: int, extra: dict | None = None) -> dict:
    rec = {
        "sample_count": _bounded_int(sample_count, 20, 6, 200),
        "eval_sample_count": _bounded_int(eval_sample_count, 8, 3, 80),
        "max_iterations": _bounded_int(max_iterations, 3, 1, 8),
    }
    extra = extra or {}
    for key in ("sample_count", "eval_sample_count", "max_iterations", "train_max_seq_len", "eval_max_samples", "breadth_max_samples"):
        if key in extra:
            if key == "sample_count":
                rec[key] = _bounded_int(extra[key], rec[key], 6, 200)
            elif key == "eval_sample_count":
                rec[key] = _bounded_int(extra[key], rec[key], 3, 80)
            elif key == "max_iterations":
                rec[key] = _bounded_int(extra[key], rec[key], 1, 8)
            elif key == "train_max_seq_len":
                rec[key] = _bounded_int(extra[key], 512, 128, 2048)
            else:
                rec[key] = _bounded_int(extra[key], rec.get("eval_sample_count", 8), 1, 80)
    return rec


def _fallback_plan_advice(
    direction: str,
    scenario: str,
    sample_count: int,
    eval_sample_count: int,
    max_iterations: int,
    reason: str = "",
) -> dict:
    terms = _focus_terms(direction, scenario)
    focus = "、".join(terms[:5])
    sections = [
        _section("训练目标", [
            f"把“{scenario}”拆成可训练能力：识别问题、说明依据、给出操作建议，并在信息不足时主动提示边界。",
            f"首批样本重点覆盖：{focus}；每条答案都要求包含判断依据、风险点和常见误区，避免只学到口号式回答。",
        ]),
        _section("数据生成策略", [
            f"生成 {sample_count} 条训练 source，按概念解释、真实案例、反例纠错、边界条件、输出格式五类混合，保证不是单一问答模板。",
            "每轮只回灌一部分扩增样本，训练集与 held-out 验证集分离，用于观察回灌前后是否真的改善而不是背题。",
            "知识库同步写入同一批核心知识点，RAG 查询和训练样本可互相对照，便于排查模型答错时是知识缺失还是训练不足。",
        ]),
        _section("验证与查缺补漏", [
            f"构造 {eval_sample_count} 条 held-out 验证题，题面会换一种业务表达，避免直接复用训练题。",
            "评测先看 Judge 的专业性、完整性和可执行性，再用 ROUGE/BLEU 做字面辅助；两者不一致时优先看 Judge 评语。",
            "广度分析会让 Judge 补充相邻子主题，下一轮优先补齐低分主题、边界案例和容易误判的问题。",
        ]),
        _section("迭代安排", [
            "第 1 轮先建立目标回答格式和基础知识覆盖。",
            "第 2 轮补充难例、反例和真实表达，检查模型是否能稳住判断依据。",
            "第 3 轮扩大验证集广度，重点看新问题、新说法、新边界下是否仍能回答。",
        ]),
        _section("风险与参数建议", [
            "20 条样本适合本机 demo 跑通，不代表正式效果；生产场景应逐步替换为真实标注数据并扩到数百条以上。",
            "0.5B 基底模型速度快、成本低，适合演示闭环；更大基底模型或 GPU 会提升专业表达上限，但训练和部署成本更高。",
            "Judge 不可用时仍能跑通流程，但建议质量、数据质量和专业评测都会下降，只能作为 smoke 验证。",
        ]),
    ]
    return {
        "source": "fallback",
        "source_label": "规则兜底",
        "source_reason": reason or "Judge 未配置、不可达或返回内容无法解析，已根据方向关键词生成本地方案。",
        "summary": f"这次建议会围绕“{scenario}”展开，不只是生成数据：我会先覆盖 {focus}，再用 held-out 验证和广度分析检查每轮回灌是否真的补上短板。",
        "sections": sections,
        "recommended_params": _recommended_params(sample_count, eval_sample_count, max_iterations),
    }


def _llm_plan_advice(direction: str, scenario: str, sample_count: int, eval_sample_count: int, max_iterations: int) -> dict:
    system = (
        "你是大模型训练方案架构师。请基于用户的强化方向，设计可落地的小模型微调冷启动方案。"
        "必须输出严格 JSON，不要 markdown，不要额外解释。"
    )
    user = f"""
用户强化方向：{direction}
推断场景名：{scenario}
当前默认参数：训练样本 {sample_count} 条，held-out 验证 {eval_sample_count} 条，迭代 {max_iterations} 轮。

请输出 JSON 对象，字段必须包含：
summary: 80 字以内，说明本方案为什么适合该方向；
training_goal: 一句话目标；
data_strategy: 3-5 条，说明训练样本要覆盖哪些子能力、正反例、边界样本和输出格式；
eval_strategy: 3-5 条，说明 held-out、Judge、ROUGE/广度验证如何证明回灌有效；
iteration_strategy: 3-5 条，说明每轮迭代分别补什么；
risks: 3-5 条，说明样本数、基底模型、Judge、真实业务数据不足等风险；
recommended_params: 对象，可包含 sample_count、eval_sample_count、max_iterations、train_max_seq_len、eval_max_samples、breadth_max_samples。

要求：内容必须贴合“{scenario}”，不要写泛泛的“生成训练数据再评测”。
""".strip()
    obj = _extract_json_object(_judge_chat(system, user, max_tokens=2200))
    if not obj:
        raise RuntimeError("Judge 返回内容不是 JSON 对象")
    rec = _recommended_params(sample_count, eval_sample_count, max_iterations, obj.get("recommended_params") or {})
    data_fallback = _fallback_plan_advice(direction, scenario, rec["sample_count"], rec["eval_sample_count"], rec["max_iterations"])
    sections = [
        _section("训练目标", _list_items(obj.get("training_goal"), data_fallback["sections"][0]["items"], 2)),
        _section("数据生成策略", _list_items(obj.get("data_strategy"), data_fallback["sections"][1]["items"], 5)),
        _section("验证与查缺补漏", _list_items(obj.get("eval_strategy"), data_fallback["sections"][2]["items"], 5)),
        _section("迭代安排", _list_items(obj.get("iteration_strategy"), data_fallback["sections"][3]["items"], 5)),
        _section("风险与参数建议", _list_items(obj.get("risks"), data_fallback["sections"][4]["items"], 5)),
    ]
    return {
        "source": "judge",
        "source_label": "Judge 生成",
        "source_reason": "后端已调用配置中的 Judge/LLM，根据 brief 输出结构化 JSON 方案并解析成功。",
        "summary": str(obj.get("summary") or data_fallback["summary"]).strip()[:260],
        "sections": sections,
        "recommended_params": rec,
    }


def _build_plan_advice(direction: str, scenario: str, sample_count: int, eval_sample_count: int, max_iterations: int) -> dict:
    try:
        return _llm_plan_advice(direction, scenario, sample_count, eval_sample_count, max_iterations)
    except Exception as exc:  # noqa: BLE001
        return _fallback_plan_advice(
            direction,
            scenario,
            sample_count,
            eval_sample_count,
            max_iterations,
            reason=f"Judge 方案生成不可用：{str(exc)[:180]}",
        )


def plan(params: dict) -> dict:
    brief = (params.get("brief") or "").strip()
    cfg = load_config()
    base = _default_base_model()
    cleaned = re.sub(r"\s+", " ", brief)
    short = _infer_scenario_from_brief(cleaned)
    scenario = _safe_name(short, "通用场景")
    collection = _ascii_slug(scenario, "custom")
    sample_count = _bounded_int(params.get("sample_count"), 20, 6, 200)
    eval_sample_count = _bounded_int(params.get("eval_sample_count"), 8, 3, 80)
    max_iterations = _bounded_int(params.get("max_iterations"), 3, 1, 8)
    advice = _build_plan_advice(cleaned or scenario, scenario, sample_count, eval_sample_count, max_iterations)
    rec = advice.get("recommended_params") or {}
    sample_count = _bounded_int(rec.get("sample_count"), sample_count, 6, 200)
    eval_sample_count = _bounded_int(rec.get("eval_sample_count"), eval_sample_count, 3, 80)
    max_iterations = _bounded_int(rec.get("max_iterations"), max_iterations, 1, 8)
    eval_max_samples = _bounded_int(rec.get("eval_max_samples"), eval_sample_count, 1, 80)
    breadth_max_samples = _bounded_int(rec.get("breadth_max_samples"), eval_sample_count, 1, 80)
    train_max_seq_len = _bounded_int(rec.get("train_max_seq_len"), 512, 128, 2048)
    result = {
        "direction": cleaned or "面向具体业务场景的专业问答能力",
        "scenario_name": scenario,
        "source_name": f"{scenario}知识",
        "iteration_prefix": f"{collection}-v",
        "dataset_name": f"{scenario}知识",
        "model_name": f"{scenario}专家",
        "base_model_path": base,
        "output_root": "data/custom_agents",
        "sample_count": sample_count,
        "eval_sample_count": eval_sample_count,
        "max_iterations": max_iterations,
        "collection": collection,
        "data_generation_mode": "auto",
        "train_backend": "hf",
        "train_device": "cpu",
        "train_epochs": 1,
        "train_max_seq_len": train_max_seq_len,
        "eval_max_samples": eval_max_samples,
        "breadth_max_samples": breadth_max_samples,
        "use_judge": cfg.get("eval", {}).get("use_judge", "auto"),
        "run_inference_probe": True,
        "parameter_notes": _parameter_notes(),
        "agent_reply": (
            "我会先为该方向生成训练 source、held-out 验证集和知识库，然后按迭代轮数自动回灌、训练、"
            "评测和广度分析。默认参数偏向本机快速跑通，正式训练请提高样本数并使用更强基底模型或 GPU。"
        ),
    }
    result.update({
        "agent_reply": advice.get("summary") or result["agent_reply"],
        "plan_source": advice.get("source", "fallback"),
        "plan_source_label": advice.get("source_label", "规则兜底"),
        "plan_source_reason": advice.get("source_reason", ""),
        "plan_generation_note": (
            "生成过程：先从 brief 提取场景名和关键词；优先调用 Judge 生成结构化 JSON 训练方案；"
            "若 Judge 不可用或 JSON 解析失败，则用本地规则按目标、数据、验证、迭代、风险五部分兜底。"
        ),
        "plan_sections": advice.get("sections", []),
        "recommended_params": {
            **(advice.get("recommended_params") or {}),
            "sample_count": sample_count,
            "eval_sample_count": eval_sample_count,
            "max_iterations": max_iterations,
            "train_max_seq_len": train_max_seq_len,
            "eval_max_samples": eval_max_samples,
            "breadth_max_samples": breadth_max_samples,
        },
    })
    return result


def _output_dir(params: dict, run_id: str) -> Path:
    root = Path(params.get("output_root") or "data/custom_agents")
    root = root if root.is_absolute() else ROOT / root
    try:
        root.resolve().relative_to(ROOT.resolve())
    except Exception:
        root = DATA / "custom_agents"
    path = root / f"{_safe_name(params.get('scenario_name') or params.get('direction'))}-{run_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_json_array(text: str) -> list[dict] | None:
    text = (text or "").strip()
    candidates = [text]
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        candidates.insert(0, m.group(0))
    for raw in candidates:
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    return None


def _judge_chat(system: str, user: str, max_tokens: int = 4096) -> str:
    cfg = load_config()
    jc = cfg.get("judge", {}) or {}
    base_url = (jc.get("base_url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("未配置 judge.base_url")
    api_key = judge._resolve_key(jc) or "EMPTY"  # noqa: SLF001 - internal helper shared with judge client.
    if jc.get("require_key", True) and not api_key:
        raise RuntimeError("未配置 Judge API Key")
    payload = {
        "model": jc.get("model", "judge"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=jc.get("timeout", 90)) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Judge 生成接口 HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"连接 Judge 生成接口失败: {exc}") from exc
    return body["choices"][0]["message"]["content"]


def _fallback_rows(direction: str, count: int, system_prompt: str) -> list[dict]:
    cats = ["基础概念", "关键流程", "真实案例", "对比辨析", "风险边界", "质量检查"]
    seeds = [x.strip() for x in re.split(r"[，,。；;、\n]", direction) if x.strip()]
    if not seeds:
        seeds = [direction.strip() or "目标能力"]
    rows = []
    for idx in range(count):
        cat = cats[idx % len(cats)]
        seed = seeds[idx % len(seeds)]
        topic = f"{seed}：{cat}"
        rows.append({
            "id": f"custom-{idx + 1:03d}",
            "category": cat,
            "system_prompt": system_prompt,
            "instruction": (
                f"围绕「{direction}」训练大模型。请专业回答：{topic}。"
                "要求说明定义/目标、关键步骤或机制、一个判断要点和一个常见误区。"
            ),
            "input": "",
            "output": (
                f"{topic}的核心是把业务目标、约束条件和可验证证据分开处理。"
                f"关键步骤包括识别场景、提取关键信息、给出判断依据，并说明不确定性。"
                f"应用时应结合真实输入检查是否满足「{direction}」的质量要求。"
                "常见误区是只给结论、不说明依据，或把泛化建议当成适用于所有场景的规则。"
            ),
        })
    return rows


def _llm_rows(direction: str, count: int, system_prompt: str, log) -> tuple[list[dict], str]:
    system = "你是训练数据设计师，擅长为小模型生成高质量 SFT 指令数据。只输出 JSON。"
    user = (
        f"请为大模型强化方向「{direction}」生成 {count} 条中文 SFT 样本。"
        "只输出 JSON 数组，每项字段必须为 category、instruction、output。"
        "要求：instruction 真实、多样、可用于训练；output 准确、结构清晰、包含判断依据和常见误区；"
        "不要输出 markdown，不要解释。"
    )
    content = _judge_chat(system, user, max_tokens=min(12000, max(4096, count * 420)))
    rows = _extract_json_array(content)
    if not rows:
        raise RuntimeError("Judge 返回内容不是 JSON 数组")
    out = []
    for idx, row in enumerate(rows[:count], 1):
        inst = str(row.get("instruction") or "").strip()
        ans = str(row.get("output") or "").strip()
        if not inst or not ans:
            continue
        out.append({
            "id": f"custom-{idx:03d}",
            "category": str(row.get("category") or "未分类").strip() or "未分类",
            "system_prompt": system_prompt,
            "instruction": inst,
            "input": "",
            "output": ans,
        })
    if len(out) < max(3, count // 2):
        raise RuntimeError(f"Judge 生成有效样本不足：{len(out)}/{count}")
    if len(out) < count:
        out.extend(_fallback_rows(direction, count - len(out), system_prompt))
        log(f"Judge 生成 {len(rows)} 条，其中有效样本不足，已用模板补齐到 {count} 条")
    return out[:count], "judge"


def _build_rows(params: dict, log) -> tuple[list[dict], str]:
    direction = params["direction"].strip()
    count = int(params.get("sample_count") or 20)
    system_prompt = _system_prompt(direction)
    mode = str(params.get("data_generation_mode") or "auto").lower()
    if mode in ("auto", "judge"):
        try:
            rows, source = _llm_rows(direction, count, system_prompt, log)
            log(f"已通过 Judge/LLM 生成 {len(rows)} 条训练样本")
            return rows, source
        except Exception as exc:  # noqa: BLE001
            if mode == "judge":
                raise
            log(f"Judge 生成不可用，降级为模板生成：{str(exc)[:160]}")
    rows = _fallback_rows(direction, count, system_prompt)
    return rows, "template"


def _system_prompt(direction: str) -> str:
    return (
        f"你是「{direction}」方向的专业 AI 助手。回答要准确、结构清晰，必须区分事实、推理依据、"
        "适用边界和风险提示；不确定时说明需要补充的信息。"
        "当用户询问身份时，应明确说明你是本训练项目的 AI 助手，不是真实个人、政府部门或机构代表，"
        "不得虚构任职、隶属关系、权限或亲身经历。"
    )


def _identity_rows(direction: str, system_prompt: str) -> list[dict]:
    identity = (
        f"我是用于「{direction}」任务的 AI 助手，可以提供该方向的分析与建议。"
        "我不是真实个人，也不代表政府部门、企业或其他机构；涉及具体业务结论时仍需结合原始材料和人工复核。"
    )
    return [
        {
            "id": "guardrail-identity-001",
            "category": "身份边界",
            "system_prompt": system_prompt,
            "instruction": "你是谁？",
            "input": "",
            "output": identity,
        },
        {
            "id": "guardrail-identity-002",
            "category": "身份边界",
            "system_prompt": system_prompt,
            "instruction": "你来自哪个单位？你能代表政府或公司作决定吗？",
            "input": "",
            "output": identity,
        },
        {
            "id": "guardrail-scope-003",
            "category": "能力边界",
            "system_prompt": system_prompt,
            "instruction": "请说明你的能力范围，以及回答需要人工复核的情况。",
            "input": "",
            "output": (
                f"我主要协助处理「{direction}」相关任务。输入材料不完整、规则存在歧义、"
                "结论会影响客户权益或需要正式授权时，我会说明不确定性并建议由具备权限的人员复核。"
            ),
        },
    ]


def _validate_training_rows(rows: list[dict], direction: str) -> None:
    text = "\n".join(
        str(row.get(key) or "")
        for row in rows
        for key in ("system_prompt", "instruction", "output")
    )
    foreign_markers = ("中国应急管理部", "来自应急管理部", "应急管理领域的专业助手")
    if "应急" not in direction:
        found = [marker for marker in foreign_markers if marker in text]
        if found:
            raise ValueError(f"训练数据检测到跨领域身份污染：{', '.join(found)}")
    mojibake_markers = ("Ã", "Â", "å®¢", "è´¨", "æ£")
    if any(marker in text for marker in mojibake_markers):
        raise ValueError("训练数据检测到乱码样本，请先在标注页修正后再继续")


def _identity_regression(model_id: str, direction: str, system_prompt: str) -> dict:
    """Run non-blocking identity probes and persist warnings for review."""
    probes = ["你是谁？", "你来自哪个单位？你能代表政府或公司作决定吗？"]
    forbidden = (
        "中国应急管理部",
        "来自应急管理部",
        "应急管理领域的专业助手",
        "来自阿里云",
        "来自OpenAI",
    )
    results = []
    for prompt in probes:
        try:
            result = infer.generate(
                model_id,
                prompt,
                log=_log,
                backend=None,
                params={"system_prompt": system_prompt, "do_sample": False, "max_new_tokens": 160},
            )
            answer = str(result.get("answer") or "").strip()
            reasons = []
            failures = [marker for marker in forbidden if marker in answer and "应急" not in direction]
            if failures:
                reasons.append(f"包含跨领域身份：{', '.join(failures)}")
            if not any(marker in answer for marker in ("不代表", "不是真实个人", "不隶属于")):
                reasons.append("未明确机构边界")
            results.append({
                "prompt": prompt,
                "answer": answer,
                "passed": not reasons,
                "warnings": reasons,
            })
        except Exception as exc:  # noqa: BLE001 - identity checks must never block the training loop.
            results.append({
                "prompt": prompt,
                "answer": "",
                "passed": False,
                "warnings": [f"身份探针执行失败：{str(exc)[:180]}"],
            })

    model_dir = MODELS / model_id
    passed = all(item.get("passed") for item in results)
    warnings = [warning for item in results for warning in item.get("warnings", [])]
    validation = {
        "passed": passed,
        "blocking": False,
        "status": "passed" if passed else "warning",
        "direction": direction,
        "forbidden_markers": list(forbidden),
        "results": results,
        "warnings": warnings,
        "created_at": time.time(),
    }
    (model_dir / "identity_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta_path = model_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["identity_validation"] = "passed" if passed else "warning"
        meta["identity_validation_blocking"] = False
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if warnings:
        _log(f"身份验证告警（不阻断训练）：{'；'.join(warnings)[:320]}")
    return validation


def _topic(row: dict) -> str:
    prompt = row.get("instruction") or ""
    m = re.search(r"[「《](.+?)[」》]", prompt)
    if m:
        return m.group(1)[:40]
    return prompt[:40] or "该知识点"


def _eval_rows(train_rows: list[dict], count: int, direction: str) -> list[dict]:
    rows = []
    for idx, src in enumerate(train_rows[: max(1, count)], 1):
        rows.append({
            "id": f"eval-{idx:03d}",
            "category": src.get("category", "未分类"),
            "system_prompt": src.get("system_prompt") or _system_prompt(direction),
            "instruction": (
                f"held-out 验证题：请换一个真实业务表达，说明「{_topic(src)}」在「{direction}」中的"
                "判断方法、依据和一个容易出错的点。"
            ),
            "input": "",
            "output": src["output"],
        })
    return rows


def _write_kb(rows: list[dict], path: Path, direction: str) -> None:
    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        by_cat.setdefault(row.get("category") or "未分类", []).append(row)
    lines = [
        f"# {direction} 知识库",
        "",
        "本知识库由通用训练 Agent 自动生成，用于配套 RAG 检索和训练过程复现。",
        "",
    ]
    for cat, items in sorted(by_cat.items()):
        lines.append(f"## {cat}")
        for row in items:
            lines.append(f"- 问题：{row['instruction']}")
            lines.append(f"  答案要点：{row['output']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _materials(params: dict, run_id: str, log) -> dict:
    out_dir = _output_dir(params, run_id)
    direction = params["direction"].strip()
    count = int(params.get("sample_count") or 20)
    system_prompt = _system_prompt(direction)
    domain_params = {**params, "sample_count": max(3, count - 3)}
    rows, generation_mode = _build_rows(domain_params, log)
    rows = (_identity_rows(direction, system_prompt) + rows)[:count]
    _validate_training_rows(rows, direction)
    eval_count = min(int(params.get("eval_sample_count") or 8), len(rows))
    eval_rows = _eval_rows(rows, eval_count, direction)
    source_name = _safe_name(params.get("source_name") or "通用知识")
    source_path = out_dir / f"{source_name}.jsonl"
    eval_path = out_dir / f"{source_name}_heldout.jsonl"
    kb_path = out_dir / f"{source_name}_kb.md"
    _write_jsonl(source_path, rows)
    _write_jsonl(eval_path, eval_rows)
    _write_kb(rows, kb_path, direction)
    return {
        "output_dir": out_dir,
        "source_path": source_path,
        "eval_path": eval_path,
        "kb_path": kb_path,
        "train_rows": rows,
        "eval_rows": eval_rows,
        "generation_mode": generation_mode,
    }


def _split_rows(rows: list[dict], parts: int) -> list[list[dict]]:
    parts = max(1, min(parts, len(rows)))
    base = len(rows) // parts
    rem = len(rows) % parts
    out = []
    start = 0
    for i in range(parts):
        size = base + (1 if i < rem else 0)
        out.append(rows[start:start + size])
        start += size
    return out


def _create_annotation_task(rows: list[dict], task_id: str, iteration_name: str, params: dict) -> dict:
    ann_dir = DATA / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    path = ann_dir / f"{task_id}.jsonl"
    ann_rows = []
    for row in rows:
        prompt = row["instruction"].strip()
        if row.get("input"):
            prompt += "\n" + row["input"].strip()
        ann_rows.append({
            "prompt": prompt,
            "reference": row["output"],
            "annotation": row["output"],
            "status": "annotated",
            "gap_type": f"{params['direction']} / 自动训练扩增",
            "iteration": iteration_name,
            "category": row.get("category", "未分类"),
            "source": params.get("source_name") or "通用知识",
            "system_prompt": row.get("system_prompt") or _system_prompt(params["direction"]),
        })
    _write_jsonl(path, ann_rows)
    return {"task_id": task_id, "path": path, "count": len(ann_rows)}


def _iteration_records(chunks: list[list[dict]], start_iteration: int = 1) -> list[dict]:
    return [{"iteration": start_iteration + i, "rows": rows} for i, rows in enumerate(chunks)]


def _manual_review_enabled(params: dict, iteration: int) -> bool:
    if iteration <= 1 and bool(params.get("manual_initial_review", False)):
        return True
    return iteration > 1 and bool(params.get("manual_iteration_review", False))


def _pause_for_annotation(ctx: dict, ann_task: dict, iteration_name: str, iteration: int) -> str:
    pending = {
        "iteration": iteration,
        "iteration_name": iteration_name,
        "annotation_task": ann_task["task_id"],
        "annotation_file": _rel(ann_task["path"]),
        "count": ann_task["count"],
    }
    ctx["pending_annotation"] = pending
    _set(
        active=False,
        status="waiting_manual",
        progress="waiting_manual",
        current_phase=f"{iteration_name}：等待人工审核标注",
        message=(
            f"{iteration_name} 已生成候选回灌标注任务 {ann_task['task_id']}。"
            "请到 ⑦ 标注·回灌 加载该任务，修改并保存 annotation 后，回到 ⑨ 点击继续运行。"
        ),
        waiting_for={
            "type": "annotation_review",
            "iteration": iteration,
            "iteration_name": iteration_name,
            "annotation_task": ann_task["task_id"],
            "annotation_file": _rel(ann_task["path"]),
            "count": ann_task["count"],
            "next_action": "请在⑦标注页保存后回到⑨点击继续运行",
        },
        current_annotation_task=ann_task["task_id"],
        resume_context=ctx,
    )
    _add_step(
        f"{iteration_name} 人工审核暂停",
        "waiting",
        f"等待用户审核 {ann_task['count']} 条候选回灌样本，保存后继续 commit/训练/评测",
        {"annotation_task": ann_task["task_id"], "file": _rel(ann_task["path"])},
    )
    _persist_state(TaskStatus.RUNNING, log=f"{iteration_name} 等待人工审核标注")
    return "paused"


def _run_after_annotation(task_id: str, ctx: dict, ann_task_id: str, iteration_name: str, iteration: int) -> None:
    params = ctx["params"]
    direction = params["direction"].strip()
    poll = float(params.get("poll_interval") or 5.0)
    train_backend = params.get("train_backend") or "hf"
    train_device = params.get("train_device") or "cpu"
    train_epochs = int(params.get("train_epochs") or 1)
    train_max_seq_len = int(params.get("train_max_seq_len") or 512)
    eval_max_samples = int(params.get("eval_max_samples") or 8)
    breadth_max_samples = int(params.get("breadth_max_samples") or 8)
    use_judge = params.get("use_judge") or "auto"
    system_prompt = params.get("system_prompt") or _system_prompt(direction)
    parent_dataset = ctx.get("parent_dataset")
    eval_dataset_id = ctx["eval_dataset_id"]
    eval_path = ctx["eval_path"]
    eval_rows = ctx.get("eval_rows") or []

    _progress(f"{iteration_name}：标注回灌生成训练数据集")
    dataset_name = f"{iteration_name}-{params.get('dataset_name') or '领域知识'}"
    ds = annotation.commit(ann_task_id, dataset_name, parent_dataset, hard_weight=1, log=_log)
    parent_dataset = ds["dataset_id"]
    ctx["parent_dataset"] = parent_dataset
    ctx["pending_annotation"] = None
    _set(parent_dataset=parent_dataset, final_dataset_id=parent_dataset, resume_context=ctx)
    _add_step(
        f"{iteration_name} 标注回灌",
        "succeeded",
        f"回灌 {ds['annotated']} 条标注，生成训练数据集 {parent_dataset}，累计 {ds['dataset_count']} 条",
        {"dataset_id": parent_dataset, "parent_dataset": ds.get("parent_dataset"), "added_count": ds.get("added_count")},
    )

    _set(current_phase=f"{iteration_name}：训练模型")
    _progress(f"{iteration_name}：提交训练任务")
    train_params = {
        "dataset_id": parent_dataset,
        "name": f"{iteration_name}-{params.get('model_name') or '领域专家'}",
        "backend": train_backend,
        "method": "lora",
        "device": train_device,
        "epochs": train_epochs,
        "max_seq_len": train_max_seq_len,
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "base_model_path": params.get("base_model_path") or None,
        "system_prompt": system_prompt,
        "domain_direction": direction,
        "knowledge_collection": params.get("collection") or "custom",
        "project_id": ctx.get("run_id") or "",
        "init_model_id": ctx.get("parent_model_id") or None,
    }
    train_task_id = runner.submit(Stage.TRAIN, train_params)
    model = _wait_task(train_task_id, f"{iteration_name} 训练", poll)
    model_id = model["model_id"]
    identity_validation = _identity_regression(model_id, direction, system_prompt)
    source_model_id = ctx.get("parent_model_id")
    ctx["parent_model_id"] = model_id
    _set(last_model_id=model_id, final_model_id=model_id)
    _add_step(
        f"{iteration_name} 模型训练",
        "succeeded",
        (
            f"训练完成，模型 {model_id}，后端 {model.get('backend')}"
            + ("；身份验证存在告警但不阻断后续流程" if not identity_validation.get("passed") else "；身份验证通过")
        ),
        {
            "train_task_id": train_task_id,
            "model_id": model_id,
            "train_loss": model.get("train_loss"),
            "identity_validation": identity_validation,
            "parent_model_id": source_model_id,
        },
    )
    if not identity_validation.get("passed"):
        _add_step(
            f"{iteration_name} 身份验证告警",
            "warning",
            "身份探针未完全通过，已记录结果；训练流程继续进入推理、评测和广度分析",
            {"model_id": model_id, "warnings": identity_validation.get("warnings", [])},
        )

    infer_task_id = None
    if bool(params.get("run_inference_probe", True)):
        _set(current_phase=f"{iteration_name}：推理抽检")
        probe = params.get("infer_probe") or (eval_rows[0]["instruction"] if eval_rows else f"请说明{direction}的关键判断点")
        infer_task_id = runner.submit(
            Stage.INFER,
            {
                "model_id": model_id,
                "prompt": probe,
                "backend": None,
                "system_prompt": system_prompt,
                "max_new_tokens": 160,
                "do_sample": False,
            },
        )
        infer_result = _wait_task(infer_task_id, f"{iteration_name} 推理", poll)
        _add_step(
            f"{iteration_name} 推理抽检",
            "succeeded",
            f"完成一次方向相关推理抽检，回答长度 {len(infer_result.get('answer', ''))}",
            {"infer_task_id": infer_task_id, "prompt": probe, "answer_preview": (infer_result.get("answer") or "")[:160]},
        )

    _set(current_phase=f"{iteration_name}：固定评测")
    eval_task_id = runner.submit(
        Stage.EVAL,
        {
            "model_id": model_id,
            "dataset_id": eval_dataset_id,
            "split": "valid",
            "use_judge": use_judge,
            "max_samples": eval_max_samples,
            "system_prompt": system_prompt,
        },
    )
    ev = _wait_task(eval_task_id, f"{iteration_name} 评测", poll)
    _add_step(
        f"{iteration_name} 固定评测",
        "succeeded",
        f"评测完成，ROUGE-L={ev.get('metrics', {}).get('rouge_l_f')}，gap={ev.get('gap_count')}",
        {"eval_task_id": eval_task_id, "report_id": ev.get("report_id"), "metrics": ev.get("metrics", {})},
    )

    _set(current_phase=f"{iteration_name}：广度分析")
    br = breadth.run(
        {
            "model_id": model_id,
            "source": eval_path,
            "max_samples": breadth_max_samples,
            "use_judge": use_judge,
            "system_prompt": system_prompt,
            "backend": None,
        },
        _log,
        progress=lambda p: _progress(f"{iteration_name} 广度分析 {p}"),
    )
    _add_step(
        f"{iteration_name} 广度分析",
        "succeeded",
        f"按类别验证 {br['samples']} 条，薄弱类别 {len(br.get('issues', []))} 个",
        {"breadth_report_id": br["report_id"], "issues": [x["category"] for x in br.get("issues", [])]},
    )

    summary = {
        "iteration": iteration,
        "annotation_task": ann_task_id,
        "dataset_id": parent_dataset,
        "model_id": model_id,
        "parent_model_id": source_model_id,
        "metrics": ev.get("metrics", {}),
        "gap_count": ev.get("gap_count", len(ev.get("gaps") or [])),
        "train_task_id": train_task_id,
        "infer_task_id": infer_task_id,
        "eval_task_id": eval_task_id,
        "breadth": {
            "report_id": br.get("report_id"),
            "issues": br.get("issues", []),
            "category_scores": br.get("category_scores", []),
        },
    }
    with _lock:
        _state.setdefault("summaries", []).append(summary)
    ctx["next_iteration"] = iteration + 1
    _set(resume_context=ctx)
    paths = _save_steps(ctx["run_id"])
    _set(step_summary_json=paths.get("json"), step_summary_md=paths.get("markdown"))
    _persist_state(TaskStatus.RUNNING)


def _run_iteration_loop(task_id: str, ctx: dict) -> str:
    params = ctx["params"]
    iterations = ctx.get("iterations") or _iteration_records(ctx.get("chunks") or [], int(ctx.get("next_iteration") or 1))
    ctx["iterations"] = iterations
    prefix = params.get("iteration_prefix") or "custom-v"
    scenario_slug = _safe_name(params.get("scenario_name") or params.get("direction") or "custom")
    total = len(iterations)

    while True:
        pending = ctx.get("pending_annotation")
        if pending:
            iteration = int(pending["iteration"])
            iteration_name = pending["iteration_name"]
            ann_task_id = pending["annotation_task"]
        else:
            next_iteration = int(ctx.get("next_iteration") or (iterations[0]["iteration"] if iterations else 1))
            item = next((x for x in iterations if int(x["iteration"]) == next_iteration), None)
            if not item:
                return "done"
            iteration = int(item["iteration"])
            iteration_name = f"{prefix}{iteration}"
            rows = item["rows"]
            _set(iteration=iteration, current_phase=f"{iteration_name}：标注回灌")
            _progress(f"{iteration_name}：生成候选回灌标注任务（本批 {len(rows)} 条，计划 {total} 批）")
            ann_task_id = f"{_ascii_slug(scenario_slug)}-{iteration_name}-ann-{ctx['run_id']}"
            ann_task = _create_annotation_task(rows, ann_task_id, iteration_name, params)
            _add_step(
                f"{iteration_name} 标注任务",
                "succeeded",
                f"已生成并预填 {ann_task['count']} 条候选回灌样本，可自动回灌，也可人工修改后再继续",
                {"annotation_task": ann_task_id, "file": _rel(ann_task["path"])},
            )
            if _manual_review_enabled(params, iteration):
                return _pause_for_annotation(ctx, ann_task, iteration_name, iteration)

        _run_after_annotation(task_id, ctx, ann_task_id, iteration_name, iteration)


def _wait_task(task_id: str, label: str, poll_interval: float) -> dict:
    last = ""
    while True:
        task = store.get_task(task_id)
        if not task:
            raise RuntimeError(f"{label} 子任务不存在: {task_id}")
        msg = f"{label}: {task.status.value}" + (f" {task.progress}" if task.progress else "")
        _progress(msg)
        if msg != last:
            _log(f"{label} 子任务 {task_id} -> {task.status.value}" + (f" ({task.progress})" if task.progress else ""))
            last = msg
        if task.status == TaskStatus.SUCCEEDED:
            return task.result
        if task.status == TaskStatus.FAILED:
            raise RuntimeError(f"{label} 子任务失败 {task_id}: {task.error[:800]}")
        time.sleep(poll_interval)


def _save_steps(run_id: str) -> dict:
    step_dir = DATA / "agent_steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    snap = status()
    json_path = step_dir / f"custom-{run_id}.json"
    md_path = step_dir / f"custom-{run_id}.md"
    json_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# 通用训练 Agent 自动步骤 {run_id}",
        "",
        f"- 状态：{snap.get('status')}",
        f"- 消息：{snap.get('message')}",
        f"- 强化方向：{snap.get('direction', '')}",
        f"- 最终模型：{snap.get('final_model_id', '')}",
        f"- 最终数据集：{snap.get('final_dataset_id', '')}",
        f"- 知识库 collection：{snap.get('kb_collection', '')}",
        "",
        "## 参数注意事项",
    ]
    for note in snap.get("parameter_notes", []):
        lines.append(f"- {note.get('name')}：{note.get('detail')}")
    lines.extend(["", "## 自动步骤"])
    for i, step in enumerate(snap.get("steps", []), 1):
        lines.append(f"{i}. {step.get('step')} [{step.get('status')}]：{step.get('summary')}")
        for k, v in (step.get("artifacts") or {}).items():
            lines.append(f"   - {k}: {v}")
    lines.extend(["", "## 迭代摘要"])
    for item in snap.get("summaries", []):
        metrics = item.get("metrics") or {}
        lines.append(
            f"- iter {item.get('iteration')}: dataset={item.get('dataset_id')}, "
            f"model={item.get('model_id')}, rouge={metrics.get('rouge_l_f')}, "
            f"judge={metrics.get('judge_avg')}, gap={item.get('gap_count')}"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": _rel(json_path), "markdown": _rel(md_path)}


def start(params: dict) -> dict:
    global _thread, _agent_task_id
    with _lock:
        if _thread and _thread.is_alive():
            return status()
        params = {"mode": "custom", **params}
        task = store.create_task(Stage.AGENT, params)
        _agent_task_id = task.id
        _state.clear()
        _state.update({
            "active": True,
            "status": "running",
            "message": "通用训练 Agent 启动中",
            "progress": "starting",
            "task_id": task.id,
            "params": params,
            "direction": params.get("direction", ""),
            "started_at": time.time(),
            "logs": [],
            "summaries": [],
            "steps": [],
            "parameter_notes": _parameter_notes(),
            "kb_collection": params.get("collection") or "custom",
        })
        store.update_task(task.id, status=TaskStatus.RUNNING, log="通用训练 Agent 自动流程启动")
        _thread = threading.Thread(target=_run, args=(task.id, params), name="jiuan-custom-agent", daemon=True)
        _thread.start()
        return status()


def _load_state_for_task(task_id: str | None = None) -> str | None:
    global _agent_task_id
    if task_id is None and _state.get("task_id"):
        _agent_task_id = str(_state["task_id"])
        return _agent_task_id
    task = None
    if task_id:
        task = store.get_task(task_id)
    else:
        for item in store.list_tasks(Stage.AGENT):
            if item.params.get("mode") == "custom" and item.result:
                task = item
                break
    if not task:
        return None
    if task.result:
        with _lock:
            _state.clear()
            _state.update(task.result)
            _state.setdefault("task_id", task.id)
    _agent_task_id = task.id
    return task.id


def continue_run(req: dict | None = None) -> dict:
    global _thread
    req = req or {}
    with _lock:
        if _thread and _thread.is_alive():
            return status()
        task_id = _load_state_for_task(req.get("task_id"))
        if not task_id:
            return {"active": False, "status": "idle", "message": "没有可继续的通用 Agent 任务"}
        if _state.get("status") != "waiting_manual":
            _set(message="当前任务不在人工审核暂停状态，无需继续运行")
            return status()
        ctx = _state.get("resume_context") or {}
        if not ctx.get("pending_annotation"):
            _set(message="暂停状态缺少待回灌标注任务，无法继续")
            return status()
        _set(active=True, status="running", message="继续运行：读取已保存标注并回灌训练", progress="resuming", waiting_for=None)
        _persist_state(TaskStatus.RUNNING, log="用户确认标注审核完成，继续运行")
        _thread = threading.Thread(target=_resume_worker, args=(task_id, ctx), name="jiuan-custom-agent-resume", daemon=True)
        _thread.start()
        return status()


def _resume_worker(task_id: str, ctx: dict) -> None:
    global _agent_task_id
    _agent_task_id = task_id
    run_id = ctx.get("run_id") or time.strftime("%Y%m%d-%H%M%S")
    try:
        result = _run_iteration_loop(task_id, ctx)
        if result == "paused":
            return
        total_done = len(_state.get("summaries") or [])
        _finish("succeeded", f"通用训练 Agent 已继续完成，累计 {total_done} 轮，最终模型 {status().get('final_model_id')}", run_id)
    except Exception as exc:  # noqa: BLE001
        _add_step("继续运行失败", "failed", str(exc))
        _set(active=False, status="failed", message=str(exc), finished_at=time.time(), progress="done")
        paths = _save_steps(run_id)
        _set(step_summary_json=paths.get("json"), step_summary_md=paths.get("markdown"))
        store.update_task(task_id, status=TaskStatus.FAILED, error=str(exc), result=status(), log=f"通用 Agent 继续运行失败: {exc}")


def _latest_breadth_issue_text() -> str:
    issues: list[str] = []
    for item in reversed(_state.get("summaries") or []):
        for issue in ((item.get("breadth") or {}).get("issues") or []):
            if isinstance(issue, dict):
                val = issue.get("category") or issue.get("issue") or issue.get("prompt")
            else:
                val = str(issue)
            if val and val not in issues:
                issues.append(str(val))
            if len(issues) >= 6:
                break
        if len(issues) >= 6:
            break
    return "、".join(issues)


def select_model(model_id: str) -> dict:
    """Switch the singleton UI/Agent state to a model's complete project context."""
    global _agent_task_id
    with _lock:
        if _thread and _thread.is_alive():
            raise ValueError("当前 Agent 正在运行，完成或停止后才能切换模型项目")
        profile = registry.model_resume_profile(str(model_id or "").strip())
        task_id = _load_state_for_task(None)
        params = dict(profile)
        params.pop("summaries", None)
        params.pop("project_steps", None)
        preserved_task_id = task_id or _state.get("task_id")
        _state.clear()
        _state.update({
            "active": False,
            "status": "succeeded",
            "message": f"已切换到 {model_id} 所属项目，可从下一轮继续迭代",
            "progress": "resume_model_selected",
            "current_phase": f"{profile['iteration_prefix']}{(registry.dataset_version(profile['dataset_id']) or 0) + 1}：等待追加迭代",
            "task_id": preserved_task_id,
            "params": params,
            "direction": profile["direction"],
            "summaries": list(profile.get("summaries") or []),
            "steps": list(profile.get("project_steps") or []),
            "logs": [],
            "parameter_notes": _parameter_notes(),
            "final_model_id": model_id,
            "last_model_id": model_id,
            "final_dataset_id": profile["dataset_id"],
            "parent_dataset": profile["dataset_id"],
            "eval_dataset_id": profile["eval_dataset_id"],
            "breadth_source": profile["breadth_source"],
            "kb_collection": profile["kb_collection"],
            "source": profile.get("source") or "",
            "output_dir": profile.get("output_root") or "data/custom_agents",
            "base_model_path": profile.get("base_model_path") or "",
            "selected_resume_profile": {k: v for k, v in profile.items() if k not in ("summaries", "project_steps")},
        })
        _add_step(
            "切换 Agent 项目上下文",
            "succeeded",
            f"已选择 {model_id}，同步切换训练数据、held-out、广度源和知识库",
            {
                "dataset_id": profile["dataset_id"],
                "eval_dataset_id": profile["eval_dataset_id"],
                "breadth_source": profile["breadth_source"],
                "kb_collection": profile["kb_collection"],
            },
        )
        if preserved_task_id:
            _agent_task_id = str(preserved_task_id)
            _persist_state(log=f"切换模型项目上下文：{model_id}")
        return status()


def extend(req: dict | None = None) -> dict:
    global _thread
    req = req or {}
    with _lock:
        if _thread and _thread.is_alive():
            return status()
        task_id = _load_state_for_task(req.get("task_id"))
        if not task_id:
            return {"active": False, "status": "idle", "message": "没有可追加迭代的通用 Agent 任务"}
        if _state.get("status") == "waiting_manual":
            _set(message="当前已有人工审核暂停任务，请先保存标注并点击继续运行")
            return status()
        source_model_id = str(req.get("source_model_id") or "").strip()
        if source_model_id and not registry.model_dataset(source_model_id):
            _set(message=f"所选模型不存在训练数据集血缘，不能作为恢复点：{source_model_id}")
            return status()
        if not source_model_id and not _state.get("final_dataset_id"):
            _set(message="当前任务还没有可继承的最终数据集，暂不能追加迭代")
            return status()
        extra = max(1, min(8, int(req.get("extra_iterations") or 1)))
        _set(active=True, status="running", message=f"追加 {extra} 轮迭代：生成候选回灌样本", progress="extending")
        _persist_state(TaskStatus.RUNNING, log=f"用户请求追加 {extra} 轮迭代")
        _thread = threading.Thread(target=_extend_worker, args=(task_id, req), name="jiuan-custom-agent-extend", daemon=True)
        _thread.start()
        return status()


def _extend_worker(task_id: str, req: dict) -> None:
    global _agent_task_id
    _agent_task_id = task_id
    run_id = time.strftime("%Y%m%d-%H%M%S")
    try:
        extra = max(1, min(8, int(req.get("extra_iterations") or 1)))
        source_model_id = str(req.get("source_model_id") or _state.get("final_model_id") or "").strip()
        profile = registry.model_resume_profile(source_model_id) if source_model_id else None
        params = dict(profile or _state.get("params") or {})
        params["manual_iteration_review"] = bool(req.get("manual_iteration_review", False))
        parent_dataset = profile.get("dataset_id") if profile else _state.get("final_dataset_id")
        if source_model_id and not parent_dataset:
            raise RuntimeError(f"所选模型没有已登记的训练数据集，无法恢复：{source_model_id}")
        if source_model_id and not (MODELS / source_model_id / "meta.json").exists():
            raise RuntimeError(f"所选模型产物不存在，无法恢复：{source_model_id}")
        if profile:
            selected_summaries = list(profile.get("summaries") or [])
            _set(
                params=params,
                direction=params.get("direction", ""),
                summaries=selected_summaries,
                final_model_id=source_model_id,
                last_model_id=source_model_id,
                final_dataset_id=parent_dataset,
                parent_dataset=parent_dataset,
                eval_dataset_id=profile["eval_dataset_id"],
                breadth_source=profile["breadth_source"],
                kb_collection=profile["kb_collection"],
                source=profile.get("source") or "",
                output_dir=profile.get("output_root") or "data/custom_agents",
                selected_resume_profile={k: v for k, v in profile.items() if k != "summaries"},
            )
            _add_step(
                "切换 Agent 项目上下文",
                "succeeded",
                f"已切换到 {source_model_id} 所属项目：{params.get('direction')}，旧任务方向不再参与续训",
                {
                    "model_id": source_model_id,
                    "dataset_id": parent_dataset,
                    "eval_dataset_id": profile["eval_dataset_id"],
                    "breadth_source": profile["breadth_source"],
                    "kb_collection": profile["kb_collection"],
                },
            )
        base_sample_count = int(params.get("sample_count") or 20)
        base_iterations = max(1, int(params.get("max_iterations") or 3))
        per_iter = max(6, base_sample_count // base_iterations)
        total_new = min(200, per_iter * extra)
        issue_text = _latest_breadth_issue_text()
        original_direction = params.get("direction") or "通用语言能力"
        generation_params = dict(params)
        generation_params["sample_count"] = total_new
        if issue_text:
            generation_params["direction"] = f"{original_direction}。继续迭代重点补齐这些薄弱方向：{issue_text}"
        _progress(f"追加迭代：生成 {total_new} 条候选回灌样本")
        domain_count = max(3, total_new - 3)
        generation_params["sample_count"] = domain_count
        rows, generation_mode = _build_rows(generation_params, _log)
        system_prompt = params.get("system_prompt") or _system_prompt(original_direction)
        for row in rows:
            row["system_prompt"] = system_prompt
        rows = (_identity_rows(original_direction, system_prompt) + rows)[:total_new]
        _validate_training_rows(rows, original_direction)

        out_dir = Path(params.get("output_root") or _state.get("output_dir") or "data/custom_agents")
        out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        source_name = _safe_name(params.get("source_name") or "通用知识")
        ext_source = out_dir / f"{source_name}_extend_{run_id}.jsonl"
        _write_jsonl(ext_source, rows)
        _add_step(
            "选择续训恢复点",
            "succeeded",
            f"从模型 {source_model_id or '未指定模型'} 继续，自动继承训练数据集 {parent_dataset}",
            {"source_model_id": source_model_id, "parent_dataset": parent_dataset},
        )
        _add_step(
            "追加迭代样本生成",
            "succeeded",
            f"生成 {len(rows)} 条候选回灌样本，来源 {generation_mode}；重点补齐：{issue_text or '用户指定方向'}",
            {"source": _rel(ext_source), "generation_mode": generation_mode},
        )

        summaries = _state.get("summaries") or []
        start_iteration = max([int(x.get("iteration") or 0) for x in summaries] or [0]) + 1
        chunks = _split_rows(rows, extra)
        old_ctx = {} if profile else (_state.get("resume_context") or {})
        eval_dataset_id = profile.get("eval_dataset_id") if profile else (_state.get("eval_dataset_id") or old_ctx.get("eval_dataset_id"))
        eval_path = profile.get("breadth_source") if profile else (_state.get("breadth_source") or old_ctx.get("eval_path"))
        if not eval_dataset_id or not eval_path:
            raise RuntimeError("缺少 held-out 验证集，无法追加迭代")
        ctx = {
            "params": params,
            "run_id": run_id,
            "eval_dataset_id": eval_dataset_id,
            "eval_path": eval_path,
            "eval_rows": old_ctx.get("eval_rows") or [],
            "parent_dataset": parent_dataset,
            "parent_model_id": source_model_id or None,
            "iterations": _iteration_records(chunks, start_iteration),
            "next_iteration": start_iteration,
            "pending_annotation": None,
        }
        _set(
            params=params,
            data_generation_mode=generation_mode,
            extension_source=_rel(ext_source),
            resume_from_model_id=source_model_id or None,
            resume_from_dataset_id=parent_dataset,
            resume_context=ctx,
        )
        result = _run_iteration_loop(task_id, ctx)
        if result == "paused":
            return
        _finish("succeeded", f"通用训练 Agent 追加 {extra} 轮完成，最终模型 {status().get('final_model_id')}", run_id)
    except Exception as exc:  # noqa: BLE001
        _add_step("追加迭代失败", "failed", str(exc))
        _set(active=False, status="failed", message=str(exc), finished_at=time.time(), progress="done")
        paths = _save_steps(run_id)
        _set(step_summary_json=paths.get("json"), step_summary_md=paths.get("markdown"))
        store.update_task(task_id, status=TaskStatus.FAILED, error=str(exc), result=status(), log=f"通用 Agent 追加迭代失败: {exc}")


def _run(task_id: str, params: dict) -> None:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    try:
        direction = params["direction"].strip()
        poll = float(params.get("poll_interval") or 5.0)
        max_iterations = int(params.get("max_iterations") or 3)
        train_backend = params.get("train_backend") or "hf"
        train_device = params.get("train_device") or "cpu"
        train_epochs = int(params.get("train_epochs") or 1)
        train_max_seq_len = int(params.get("train_max_seq_len") or 512)
        eval_max_samples = int(params.get("eval_max_samples") or 8)
        breadth_max_samples = int(params.get("breadth_max_samples") or 8)
        use_judge = params.get("use_judge") or "auto"
        collection = params.get("collection") or "custom"
        system_prompt = params.get("system_prompt") or _system_prompt(direction)

        _progress("① 标数据阶段：生成 source、held-out 验证集和知识库")
        mat = _materials(params, run_id, _log)
        _set(
            output_dir=_rel(mat["output_dir"]),
            source=_rel(mat["source_path"]),
            breadth_source=_rel(mat["eval_path"]),
            kb_source=_rel(mat["kb_path"]),
            data_generation_mode=mat["generation_mode"],
            base_model_path=params.get("base_model_path") or _default_base_model(),
        )
        _add_step(
            "生成通用场景材料",
            "succeeded",
            f"为「{direction}」生成 {len(mat['train_rows'])} 条训练样本、{len(mat['eval_rows'])} 条 held-out 验证题和配套知识库",
            {
                "source": _rel(mat["source_path"]),
                "heldout": _rel(mat["eval_path"]),
                "kb": _rel(mat["kb_path"]),
                "generation_mode": mat["generation_mode"],
            },
        )

        _progress("① 标数据阶段：注册 held-out 验证集")
        eval_dp = dataprep.run(
            {"source": _rel(mat["eval_path"]), "name": f"{params.get('iteration_prefix') or 'custom-v'}heldout", "valid_ratio": 0.9, "seed": 20260717},
            _log,
        )
        registry.mark_eval_dataset(eval_dp["dataset_id"])
        _set(eval_dataset_id=eval_dp["dataset_id"])
        _add_step(
            "生成固定 held-out 验证集",
            "succeeded",
            f"创建验证集 {eval_dp['dataset_id']}，并标记 role=eval 防止训练泄漏",
            {"dataset_id": eval_dp["dataset_id"], "valid_count": eval_dp["valid_count"]},
        )

        _progress("⑥ 知识库阶段：写入 RAG collection")
        kb = rag_backend.ingest_document(_rel(mat["kb_path"]), chunk_size=700, collection=collection, log=_log)
        _add_step(
            "补全并入库知识库",
            "succeeded",
            f"知识库 collection={collection} 入库 {kb['new_chunks']} 块，总块数 {kb['total_chunks']}",
            {"collection": collection, "kb_source": kb["source"], "total_chunks": kb["total_chunks"]},
        )

        chunks = _split_rows(mat["train_rows"], max_iterations)
        ctx = {
            "params": params,
            "run_id": run_id,
            "eval_dataset_id": eval_dp["dataset_id"],
            "eval_path": _rel(mat["eval_path"]),
            "eval_rows": mat["eval_rows"],
            "parent_dataset": None,
            "parent_model_id": None,
            "iterations": _iteration_records(chunks, 1),
            "next_iteration": 1,
            "pending_annotation": None,
        }
        _set(resume_context=ctx)
        result = _run_iteration_loop(task_id, ctx)
        if result == "paused":
            return
        _finish("succeeded", f"通用训练 Agent 完成 {len(chunks)} 轮，最终模型 {status().get('final_model_id')}", run_id)
        return

        parent_dataset = None
        chunks = _split_rows(mat["train_rows"], max_iterations)
        prefix = params.get("iteration_prefix") or "custom-v"
        scenario_slug = _safe_name(params.get("scenario_name") or direction)
        for idx, rows in enumerate(chunks, 1):
            iteration_name = f"{prefix}{idx}"
            _set(iteration=idx, current_phase=f"第 {idx} 轮：标注回灌")
            _progress(f"第 {idx} 轮/共 {len(chunks)} 轮：自动生成预标注任务")
            ann_task_id = f"{_ascii_slug(scenario_slug)}-{iteration_name}-ann-{run_id}"
            ann_task = _create_annotation_task(rows, ann_task_id, iteration_name, params)
            _add_step(
                f"{iteration_name} 标注任务",
                "succeeded",
                f"自动生成并预填 {ann_task['count']} 条标注，供 Agent 回灌训练",
                {"annotation_task": ann_task_id, "file": _rel(ann_task["path"])},
            )

            _progress(f"第 {idx} 轮：标注回灌生成训练数据集")
            dataset_name = f"{iteration_name}-{params.get('dataset_name') or '领域知识'}"
            ds = annotation.commit(ann_task_id, dataset_name, parent_dataset, hard_weight=1, log=_log)
            parent_dataset = ds["dataset_id"]
            _set(parent_dataset=parent_dataset, final_dataset_id=parent_dataset)
            _add_step(
                f"{iteration_name} 标注回灌",
                "succeeded",
                f"回灌 {ds['annotated']} 条标注，生成训练数据集 {parent_dataset}，累计 {ds['dataset_count']} 条",
                {"dataset_id": parent_dataset, "parent_dataset": ds.get("parent_dataset"), "added_count": ds.get("added_count")},
            )

            _set(current_phase=f"第 {idx} 轮：训练模型")
            _progress(f"第 {idx} 轮：提交训练任务")
            train_params = {
                "dataset_id": parent_dataset,
                "name": f"{iteration_name}-{params.get('model_name') or '领域专家'}",
                "backend": train_backend,
                "method": "lora",
                "device": train_device,
                "epochs": train_epochs,
                "max_seq_len": train_max_seq_len,
                "lora_r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.05,
                "base_model_path": params.get("base_model_path") or None,
            }
            train_task_id = runner.submit(Stage.TRAIN, train_params)
            model = _wait_task(train_task_id, f"{iteration_name} 训练", poll)
            model_id = model["model_id"]
            _set(last_model_id=model_id, final_model_id=model_id)
            _add_step(
                f"{iteration_name} 模型训练",
                "succeeded",
                f"训练完成，模型 {model_id}，后端 {model.get('backend')}",
                {"train_task_id": train_task_id, "model_id": model_id, "train_loss": model.get("train_loss")},
            )

            infer_task_id = None
            if bool(params.get("run_inference_probe", True)):
                _set(current_phase=f"第 {idx} 轮：推理抽检")
                probe = params.get("infer_probe") or (mat["eval_rows"][0]["instruction"] if mat["eval_rows"] else f"请说明{direction}的关键判断点")
                infer_task_id = runner.submit(
                    Stage.INFER,
                    {
                        "model_id": model_id,
                        "prompt": probe,
                        "backend": None,
                        "system_prompt": system_prompt,
                        "max_new_tokens": 160,
                        "do_sample": False,
                    },
                )
                infer_result = _wait_task(infer_task_id, f"{iteration_name} 推理", poll)
                _add_step(
                    f"{iteration_name} 推理抽检",
                    "succeeded",
                    f"完成一次方向相关推理抽检，回答长度 {len(infer_result.get('answer', ''))}",
                    {"infer_task_id": infer_task_id, "prompt": probe, "answer_preview": (infer_result.get("answer") or "")[:160]},
                )

            _set(current_phase=f"第 {idx} 轮：固定评测")
            eval_task_id = runner.submit(
                Stage.EVAL,
                {
                    "model_id": model_id,
                    "dataset_id": eval_dp["dataset_id"],
                    "split": "valid",
                    "use_judge": use_judge,
                    "max_samples": eval_max_samples,
                    "system_prompt": system_prompt,
                },
            )
            ev = _wait_task(eval_task_id, f"{iteration_name} 评测", poll)
            _add_step(
                f"{iteration_name} 固定评测",
                "succeeded",
                f"评测完成，ROUGE-L={ev.get('metrics', {}).get('rouge_l_f')}，gap={ev.get('gap_count')}",
                {"eval_task_id": eval_task_id, "report_id": ev.get("report_id"), "metrics": ev.get("metrics", {})},
            )

            _set(current_phase=f"第 {idx} 轮：广度分析")
            br = breadth.run(
                {
                    "model_id": model_id,
                    "source": _rel(mat["eval_path"]),
                    "max_samples": breadth_max_samples,
                    "use_judge": use_judge,
                    "system_prompt": system_prompt,
                    "backend": None,
                },
                _log,
                progress=lambda p: _progress(f"{iteration_name} 广度分析 {p}"),
            )
            _add_step(
                f"{iteration_name} 广度分析",
                "succeeded",
                f"按类别验证 {br['samples']} 条，薄弱类别 {len(br.get('issues', []))} 个",
                {"breadth_report_id": br["report_id"], "issues": [x["category"] for x in br.get("issues", [])]},
            )

            summary = {
                "iteration": idx,
                "annotation_task": ann_task_id,
                "dataset_id": parent_dataset,
                "model_id": model_id,
                "metrics": ev.get("metrics", {}),
                "gap_count": ev.get("gap_count", len(ev.get("gaps") or [])),
                "train_task_id": train_task_id,
                "infer_task_id": infer_task_id,
                "eval_task_id": eval_task_id,
                "breadth": {
                    "report_id": br.get("report_id"),
                    "issues": br.get("issues", []),
                    "category_scores": br.get("category_scores", []),
                },
            }
            with _lock:
                _state.setdefault("summaries", []).append(summary)
            paths = _save_steps(run_id)
            _set(step_summary_json=paths.get("json"), step_summary_md=paths.get("markdown"))

        _finish("succeeded", f"通用训练 Agent 完成 {len(chunks)} 轮，最终模型 {status().get('final_model_id')}", run_id)
    except Exception as exc:  # noqa: BLE001
        _add_step("异常终止", "failed", str(exc))
        _set(active=False, status="failed", message=str(exc), finished_at=time.time(), progress="done")
        paths = _save_steps(run_id)
        _set(step_summary_json=paths.get("json"), step_summary_md=paths.get("markdown"))
        store.update_task(task_id, status=TaskStatus.FAILED, error=str(exc), result=status(), log=f"通用训练 Agent 失败: {exc}")


def _finish(status_text: str, message: str, run_id: str | None = None) -> None:
    _set(active=False, status=status_text, message=message, progress="done", current_phase="完成", finished_at=time.time())
    if run_id:
        paths = _save_steps(run_id)
        _set(step_summary_json=paths.get("json"), step_summary_md=paths.get("markdown"))
    _log(message)
    if _agent_task_id:
        store.update_task(_agent_task_id, status=TaskStatus.SUCCEEDED, progress="done", result=status(), log="通用训练 Agent 自动流程结束")

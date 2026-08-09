"""评：在验证集上跑推理并算指标 + 生成报告（对标久安「模型评测系统」）。

指标改进：
- 采用字符级 n-gram 的 ROUGE-L(F) 与 BLEU-1/2，比原始"空格分词 token-F1"更贴近中文文本评测。
- mock 模式因返回记忆答案，指标不具参考性，报告中标注 reliable=false / mode=mock。
- 真实评测严格在 valid 切分上进行，不在训练集上"作弊"。
后续 P3 可替换为 OpenCompass / LLM-as-judge。
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from ..common import DATASETS, REPORTS, load_config, mock_mode
from .. import registry
from . import infer, judge


def _chars(text: str) -> list[str]:
    return [c for c in text.strip() if not c.isspace()]


def _ngrams(seq: list[str], n: int) -> Counter:
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)) if len(seq) >= n else Counter()


def _bleu(pred: list[str], ref: list[str], n: int) -> float:
    p, r = _ngrams(pred, n), _ngrams(ref, n)
    if not p:
        return 0.0
    overlap = sum((p & r).values())
    return overlap / max(1, sum(p.values()))


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _rouge_l_f(pred: list[str], ref: list[str]) -> float:
    lcs = _lcs(pred, ref)
    if lcs == 0:
        return 0.0
    prec = lcs / len(pred)
    rec = lcs / len(ref)
    return round(2 * prec * rec / (prec + rec), 4)


def _score(pred_text: str, ref_text: str) -> dict:
    pred, ref = _chars(pred_text), _chars(ref_text)
    return {
        "rouge_l_f": _rouge_l_f(pred, ref),
        "bleu_1": round(_bleu(pred, ref, 1), 4),
        "bleu_2": round(_bleu(pred, ref, 2), 4),
    }


def _resolve_judge(cfg: dict, params: dict) -> tuple[bool, bool, str]:
    """Return (should_attempt, explicitly_required, initial_status)."""
    want = params.get("use_judge")
    if want is None:
        want = cfg.get("eval", {}).get("use_judge", "auto")
    want = str(want).lower()
    if want in ("false", "0", "no"):
        return False, False, "disabled"
    available = judge.available(cfg)
    required = want in ("true", "1", "yes")
    if required and not available:
        raise RuntimeError("已明确开启 LLM-as-Judge，但 Judge 配置、密钥或服务当前不可用")
    return available, required, "pending" if available else "unavailable"


def identify_gaps(details: list[dict], use_judge: bool, top_k: int = 10) -> list[dict]:
    """从评测明细中找出低分样本，生成数据扩增建议(闭环引擎)。

    判定（新版四维度0-100分）：
    - hallucination=1：幻觉，模型编造了无法支撑的断言
    - accuracy < 50：事实错误，知识缺失
    - completeness < 50：关键信息遗漏
    - relevance < 50：答非所问
    - overall < 60：综合薄弱
    - rouge_l_f < 0.15（judge未启用时）：格式偏差
    返回按严重度排序的薄弱点列表，带扩增建议。
    """
    gaps: list[dict] = []
    for d in details:
        rouge = d.get("rouge_l_f", 0.0)
        gap_type = None
        severity = 0.0
        suggestion = ""

        if use_judge:
            halluc = d.get("hallucination", 0)
            acc = d.get("accuracy", 100)
            comp = d.get("completeness", 100)
            rel = d.get("relevance", 100)
            overall = d.get("overall", 100)

            if halluc == 1:
                gap_type = "幻觉"
                severity = 4.0
                suggestion = f"针对「{d.get('prompt','')[:20]}」类问题纠正幻觉内容，补充3-5条事实准确的QA"
            elif acc < 50:
                gap_type = "知识缺失/事实错误"
                severity = (100 - acc) / 25 + (1 - rouge)
                suggestion = f"针对「{d.get('prompt','')[:20]}」类问题补充3-5条同领域不同场景QA，加强知识覆盖"
            elif comp < 50:
                gap_type = "完整性不足"
                severity = (100 - comp) / 25
                suggestion = f"针对「{d.get('prompt','')[:20]}」类问题补充更完整的答案范例，覆盖关键要点"
            elif rel < 50:
                gap_type = "答非所问"
                severity = (100 - rel) / 25
                suggestion = f"针对「{d.get('prompt','')[:20]}」类问题补充切题的QA，训练模型聚焦问题"
            elif overall < 60:
                gap_type = "综合薄弱"
                severity = (60 - overall) / 15
                suggestion = f"针对「{d.get('prompt','')[:20]}」类问题补充多角度QA，全面提升回答质量"
            elif rouge < 0.15:
                gap_type = "格式偏差/覆盖不足"
                severity = 1 - rouge
                suggestion = f"针对「{d.get('prompt','')[:20]}」类问题补充更多回答格式范例"
        else:
            # 未启用judge时，仅用rouge判定
            if rouge < 0.15:
                gap_type = "格式偏差/覆盖不足"
                severity = 1 - rouge
                suggestion = f"针对「{d.get('prompt','')[:20]}」类问题补充更多回答格式范例"

        if not gap_type:
            continue

        gaps.append({
            "prompt": d.get("prompt", ""),
            "reference": d.get("reference", ""),
            "prediction": d.get("prediction", ""),
            "rouge_l_f": rouge,
            "judge_score": d.get("overall"),
            "accuracy": d.get("accuracy"),
            "completeness": d.get("completeness"),
            "relevance": d.get("relevance"),
            "hallucination": d.get("hallucination"),
            "gap_type": gap_type,
            "severity": round(severity, 4),
            "suggestion": suggestion,
        })
    gaps.sort(key=lambda g: g["severity"], reverse=True)
    return gaps[:top_k]


def collect_all_gaps() -> list[dict]:
    """聚合所有评测报告的 gap 建议，全部保留，按时间从新到旧排序并标注所属迭代。

    每条 gap 附加：
    - iteration：该报告对应模型所用数据集在血缘链中的版本号(v1/v2/…)，无血缘则为 None；
    - model_id / dataset_id / report_id / created_at。
    返回按 (created_at 降序) 排列的扁平 gap 列表（从后到前）。
    """
    reports = []
    for p in REPORTS.glob("*.json"):
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not rep.get("gaps"):
            continue
        reports.append((p.stem, rep))
    # 新→旧
    reports.sort(key=lambda x: x[1].get("created_at", 0), reverse=True)

    out: list[dict] = []
    for report_id, rep in reports:
        model_id = rep.get("model_id")
        eval_dataset_id = rep.get("dataset_id")
        # 迭代号以"模型训练所用数据集"的血缘版本为准；评测集(固定 held-out)不代表迭代
        train_dataset_id = registry.model_dataset(model_id) if model_id else None
        version = registry.dataset_version(train_dataset_id) if train_dataset_id else None
        iteration = f"v{version}" if version else "-"
        for g in rep["gaps"]:
            out.append({
                **g,
                "iteration": iteration,
                "iteration_num": version,
                "model_id": model_id,
                "train_dataset_id": train_dataset_id,
                "eval_dataset_id": eval_dataset_id,
                "report_id": report_id,
                "created_at": rep.get("created_at"),
            })
    return out


def _calc_delta(model_id: str, current_metrics: dict) -> dict | None:
    """和同模型的上一份评测报告对比，计算指标变化。

    读取 data/reports/ 下所有评测报告，找到同 model_id 的、时间早于当前的最新一份，
    对比 metrics 中每个指标的变化值。
    """
    if not model_id or model_id in ("base", "", None):
        return None
    prev_report = None
    prev_time = 0
    for p in REPORTS.glob("*.json"):
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rep.get("model_id") != model_id:
            continue
        t = rep.get("created_at", 0)
        if t > prev_time and t < current_metrics.get("_eval_time", time.time()):
            prev_time = t
            prev_report = rep
    if not prev_report or not prev_report.get("metrics"):
        return None
    prev_metrics = prev_report["metrics"]
    delta = {}
    for k, v in current_metrics.items():
        if k.startswith("_"):
            continue
        if k in prev_metrics and isinstance(v, (int, float)):
            delta[k] = round(v - prev_metrics[k], 4)
    return delta if delta else None


def _eval_one(
    cfg,
    model_id,
    dataset_id,
    split,
    use_judge,
    is_mock,
    max_samples,
    log,
    progress=None,
    label="",
    system_prompt=None,
    attach_eval=True,
    judge_required=False,
    judge_initial_status="disabled",
):
    """评测单个模型，写报告文件，返回 (summary, report_id)。"""
    path = DATASETS / dataset_id / f"{split}.jsonl"
    used_split = split
    if not path.exists():
        log(f"未找到 {split}.jsonl，回退使用 train.jsonl（指标会虚高）")
        path = DATASETS / dataset_id / "train.jsonl"
        used_split = "train"

    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows = rows[:max_samples]
    log(f"评测 {len(rows)} 条样本（split={used_split}，模型 {model_id}，mode={'mock' if is_mock else 'real'}）")

    users = [next(m["content"] for m in r["messages"] if m["role"] == "user") for r in rows]
    refs = [next(m["content"] for m in r["messages"] if m["role"] == "assistant") for r in rows]

    # 批量推理：模型仅加载一次，避免逐条 _load_real 开销
    def _infer_progress(i, total):
        if progress:
            progress(f"{label}infer {i}/{total}")

    gen_params = {"system_prompt": system_prompt} if system_prompt else None
    preds = infer.generate_batch(model_id, users, log=lambda _m: None, params=gen_params, on_item=_infer_progress)

    details, agg, total_tokens = [], {"rouge_l_f": [], "bleu_1": [], "bleu_2": []}, 0
    judge_dim_scores: dict[str, list[float]] = {"accuracy": [], "completeness": [], "relevance": [], "overall": []}
    judge_halluc_count = 0
    judge_bad_count = 0
    judge_scored = 0
    for i, (user, ref, res) in enumerate(zip(users, refs, preds), 1):
        sc = _score(res["answer"], ref)
        for k in agg:
            agg[k].append(sc[k])
        total_tokens += res["usage"]["total_tokens"]
        detail = {"prompt": user, "reference": ref, "prediction": res["answer"], **sc}
        details.append(detail)
        if progress:
            progress(f"{label}score {i}/{len(rows)}")
        log(f"[{model_id}][{i}/{len(rows)}] rougeL={sc['rouge_l_f']} bleu1={sc['bleu_1']}")

    judge_status = judge_initial_status
    judge_error = None
    if use_judge:
        if progress:
            progress(f"{label}judge batch 0/{len(rows)}")
        try:
            scored = judge.score_batch(cfg, details, log=log)
            for detail, jr in zip(details, scored):
                detail["judge_score"] = jr["overall"]
                detail["judge_reason"] = jr["reason"]
                detail["accuracy"] = jr["accuracy"]
                detail["completeness"] = jr["completeness"]
                detail["relevance"] = jr["relevance"]
                detail["hallucination"] = jr["hallucination"]
                detail["overall"] = jr["overall"]
                for dim in judge_dim_scores:
                    judge_dim_scores[dim].append(jr[dim])
                judge_halluc_count += jr["hallucination"]
                if jr["overall"] < 60:
                    judge_bad_count += 1
                judge_scored += 1
            judge_status = "succeeded"
            if progress:
                progress(f"{label}judge batch {len(rows)}/{len(rows)}")
        except Exception as exc:
            judge_status = "failed"
            judge_error = str(exc)[:300]
            for detail in details:
                detail["judge_error"] = judge_error
            log(f"LLM-as-Judge 批量评分失败：{judge_error}")
            if judge_required:
                raise RuntimeError(f"LLM-as-Judge 已开启但评分失败：{judge_error}") from exc

    def _avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    metrics = {k: _avg(v) for k, v in agg.items()}
    judge_summary = None
    if use_judge and judge_scored > 0:
        dim_avgs = {f"judge_{k}": _avg(v) for k, v in judge_dim_scores.items()}
        bad_case_rate = round(judge_bad_count / judge_scored, 4)
        hallucination_rate = round(judge_halluc_count / judge_scored, 4)
        judge_summary = {
            "avg_score": dim_avgs["judge_overall"],
            "dim_scores": dim_avgs,
            "bad_case_rate": bad_case_rate,
            "hallucination_rate": hallucination_rate,
            "scored_samples": judge_scored,
        }
        metrics.update(dim_avgs)
        metrics["bad_case_rate"] = bad_case_rate
        metrics["hallucination_rate"] = hallucination_rate

    # --- Delta 对比：和同模型的上一份评测报告对比 ---
    delta = _calc_delta(model_id, metrics)

    reliable = (not is_mock) and used_split != "train"
    gaps = identify_gaps(details, use_judge)
    if gaps:
        log(f"[{model_id}] 发现 {len(gaps)} 个薄弱点，最严重：{gaps[0]['gap_type']}「{gaps[0]['prompt'][:16]}」")
    report = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "split": used_split,
        "mode": "mock" if is_mock else "real",
        "reliable": reliable,
        "judge": judge_summary,
        "judge_enabled": use_judge,
        "judge_status": judge_status,
        "judge_error": judge_error,
        "judge_scored_samples": judge_scored,
        "samples": len(rows),
        "metrics": metrics,
        "delta": delta,
        "gaps": gaps,
        "total_tokens": total_tokens,
        "created_at": time.time(),
        "details": details,
    }
    report_id = f"eval-{model_id}-{time.strftime('%Y%m%d-%H%M%S')}"
    (REPORTS / f"{report_id}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if attach_eval and model_id not in ("base", "", None):
        registry.attach_eval(model_id, report_id, metrics)
    note = "" if reliable else "（mock/train，仅演示不具参考性）"
    log(f"[{model_id}] 评测完成 {metrics} {note}，报告 {report_id}.json")
    summary = {
        "model_id": model_id,
        "report_id": report_id,
        "split": used_split,
        "metrics": metrics,
        "delta": delta,
        "judge": judge_summary,
        "judge_enabled": use_judge,
        "judge_status": judge_status,
        "judge_error": judge_error,
        "judge_scored_samples": judge_scored,
        "reliable": reliable,
        "report_path": str(REPORTS / f"{report_id}.json"),
        "samples": len(rows),
        "gaps": gaps,
        "gap_count": len(gaps),
    }
    return summary


def _build_comparison(target: dict, base: dict) -> dict:
    """并列对比目标模型与基线，计算各指标 delta 与提升百分比。"""
    metric_keys = ["rouge_l_f", "bleu_1", "bleu_2", "judge_avg"]
    rows = []
    for k in metric_keys:
        tv = target["metrics"].get(k)
        bv = base["metrics"].get(k)
        if tv is None and bv is None:
            continue
        tv = tv or 0.0
        bv = bv or 0.0
        delta = round(tv - bv, 4)
        pct = round((delta / bv * 100), 1) if bv else None
        rows.append({"metric": k, "baseline": bv, "target": tv, "delta": delta, "delta_pct": pct})
    return {
        "target_model": target["model_id"],
        "baseline_model": base["model_id"],
        "metrics": rows,
    }


def run(params: dict, log: Callable[[str], None], progress: "Callable[[str], None] | None" = None) -> dict:
    """评测。支持两种模式：
    - 单模型：只传 model_id。
    - 对比：额外传 baseline（如 "base" 或另一个 model_id），同一 valid 集并列评测并给出 delta。
    """
    cfg = load_config()
    model_id = params.get("model_id", "base")
    dataset_id = params["dataset_id"]
    split = params.get("split", "valid")
    max_samples = cfg["eval"]["max_samples"]
    if params.get("max_samples") is not None:
        max_samples = int(params["max_samples"])
        log(f"评测样本上限覆盖为 {max_samples}")
    is_mock = mock_mode()
    use_judge, judge_required, judge_initial_status = _resolve_judge(cfg, params)
    if use_judge:
        log("已启用 LLM-as-Judge 评测（裁判服务可达）")

    baseline = params.get("baseline")
    system_prompt = params.get("system_prompt") or None
    attach_eval = bool(params.get("attach_eval", True))
    tgt_label = "target: " if baseline else ""
    target = _eval_one(
        cfg,
        model_id,
        dataset_id,
        split,
        use_judge,
        is_mock,
        max_samples,
        log,
        progress=progress,
        label=tgt_label,
        system_prompt=system_prompt,
        attach_eval=attach_eval,
        judge_required=judge_required,
        judge_initial_status=judge_initial_status,
    )

    if not baseline or str(baseline) == str(model_id):
        # 单模型评测：保持与旧版一致的返回结构
        return target

    log(f"对比基线模型：{baseline}")
    base = _eval_one(
        cfg,
        baseline,
        dataset_id,
        split,
        use_judge,
        is_mock,
        max_samples,
        log,
        progress=progress,
        label="baseline: ",
        system_prompt=system_prompt,
        attach_eval=attach_eval,
        judge_required=judge_required,
        judge_initial_status=judge_initial_status,
    )
    comparison = _build_comparison(target, base)
    for m in comparison["metrics"]:
        pct = f"（{m['delta_pct']:+.1f}%）" if m["delta_pct"] is not None else ""
        log(f"对比 {m['metric']}: 基线 {m['baseline']} → 目标 {m['target']}  Δ{m['delta']:+}{pct}")

    return {
        **target,
        "comparison": comparison,
        "baseline_report_id": base["report_id"],
        "baseline_metrics": base["metrics"],
    }

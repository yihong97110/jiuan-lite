"""自动回滚判定 + 逐 prompt 分类 + 智能构造回滚数据集。

基于评测报告的指标对比，自动判定子版本是否需要回滚，
并逐 prompt 分析哪些样本的补数据有效（improved）、哪些引发退步（regressed），
据此构造新数据集：父版本全量 + 验证有效的子集样本。

判定逻辑：
  - bad_case_rate 上升 > 20 个百分点 -> critical rollback
  - judge overall 下降 > 5 分 -> major rollback
  - bad_case 微升 + BLEU 下降 -> minor rollback
  - 全部改善或持平 -> promote

分制归一化：
  v1 报告用 5 分制（judge_score 1-5），v2 用 100 分制（overall 0-100）。
  统一归一化到 100 分制做对比：score_100 = score_5 * 20。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common import ROOT
from . import evaluate

# ---------------------------------------------------------------------------
# 分制归一化
# ---------------------------------------------------------------------------

_SCALE_5 = 5.0
_SCALE_100 = 100.0


def _is_5_scale(report: dict) -> bool:
    """检测报告是否使用 5 分制（v1 旧报告）。"""
    avg = report.get("judge", {}).get("avg_score", 0)
    if avg and avg <= 5.0:
        return True
    # 检查 details 里的 judge_score
    details = report.get("details", [])
    if details:
        scores = [d.get("judge_score", 0) for d in details if d.get("judge_score") is not None]
        if scores and max(scores) <= 5.0:
            return True
    return False


def _norm_score(score: float, is_5_scale: bool) -> float:
    """归一化到 100 分制。"""
    if is_5_scale:
        return score * 20.0
    return score


def _norm_bad_case_rate(report: dict, is_5_scale: bool) -> float:
    """提取或计算 bad_case_rate（judge < 60 归一化后）。"""
    # v2+ 报告直接有 bad_case_rate
    rate = report.get("judge", {}).get("bad_case_rate")
    if rate is not None:
        return rate
    # v1 没有，从 details 算
    details = report.get("details", [])
    if not details:
        return 0.0
    bad = 0
    for d in details:
        js = d.get("judge_score")
        if js is None:
            continue
        if _norm_score(js, is_5_scale) < 60:
            bad += 1
    return bad / len(details) if details else 0.0


def _norm_judge_avg(report: dict, is_5_scale: bool) -> float:
    """提取归一化后的 judge 平均分。"""
    # 优先用 judge.avg_score
    avg = report.get("judge", {}).get("avg_score")
    if avg is not None:
        return _norm_score(avg, is_5_scale)
    # 兜底用 metrics.judge_avg
    avg = report.get("metrics", {}).get("judge_avg")
    if avg is not None:
        return _norm_score(avg, is_5_scale)
    return 0.0


# ---------------------------------------------------------------------------
# 指标对比判定
# ---------------------------------------------------------------------------

def should_rollback(parent_report: dict, child_report: dict) -> dict:
    """对比父子报告，自动判定是否回滚。

    Returns:
        {"action": "rollback"|"promote"|"hold",
         "severity": "critical"|"major"|"minor"|"none"|"ambiguous",
         "reason": str,
         "deltas": {bad_case_rate, judge_avg, bleu_1, rouge_l_f}}
    """
    p5 = _is_5_scale(parent_report)
    c5 = _is_5_scale(child_report)

    p_bad = _norm_bad_case_rate(parent_report, p5)
    c_bad = _norm_bad_case_rate(child_report, c5)
    d_bad = c_bad - p_bad

    p_judge = _norm_judge_avg(parent_report, p5)
    c_judge = _norm_judge_avg(child_report, c5)
    d_judge = c_judge - p_judge

    p_bleu = parent_report.get("metrics", {}).get("bleu_1", 0)
    c_bleu = child_report.get("metrics", {}).get("bleu_1", 0)
    d_bleu = c_bleu - p_bleu

    p_rouge = parent_report.get("metrics", {}).get("rouge_l_f", 0)
    c_rouge = child_report.get("metrics", {}).get("rouge_l_f", 0)
    d_rouge = c_rouge - p_rouge

    deltas = {
        "bad_case_rate": round(d_bad, 4),
        "judge_avg": round(d_judge, 2),
        "bleu_1": round(d_bleu, 4),
        "rouge_l_f": round(d_rouge, 4),
        "parent_bad_case_rate": round(p_bad, 4),
        "child_bad_case_rate": round(c_bad, 4),
        "parent_judge_avg": round(p_judge, 2),
        "child_judge_avg": round(c_judge, 2),
    }

    # 判定规则（按严重度排序）
    if d_bad > 0.2:
        return {"action": "rollback", "severity": "critical",
                "reason": f"bad_case_rate 上升 {d_bad:.1%}（{p_bad:.1%}->{c_bad:.1%}），疑似灾难性遗忘",
                "deltas": deltas}

    if d_judge < -5:
        return {"action": "rollback", "severity": "major",
                "reason": f"judge 均分下降 {-d_judge:.1f} 分（{p_judge:.1f}->{c_judge:.1f}）",
                "deltas": deltas}

    if d_bad > 0.05 and d_bleu < 0:
        return {"action": "rollback", "severity": "minor",
                "reason": f"bad_case 微升 {d_bad:.1%} 且 BLEU 下降 {d_bleu:.4f}",
                "deltas": deltas}

    if d_bad <= 0 and d_judge >= 0:
        return {"action": "promote", "severity": "none",
                "reason": f"指标全部改善或持平（judge {p_judge:.1f}->{c_judge:.1f}，bad_case {p_bad:.1%}->{c_bad:.1%}）",
                "deltas": deltas}

    return {"action": "hold", "severity": "ambiguous",
            "reason": f"指标混合：judge {d_judge:+.1f}，bad_case {d_bad:+.1%}，bleu {d_bleu:+.4f}，需人工评审",
            "deltas": deltas}


# ---------------------------------------------------------------------------
# 逐 prompt 分类
# ---------------------------------------------------------------------------

def analyze_prompt_deltas(parent_report: dict, child_report: dict) -> dict:
    """逐题对比，把样本分四类：improved / regressed / stable_good / stable_bad。

    阈值（归一化到 100 分制后）：
      - >= 70 视为好，< 60 视为差
      - parent 差 -> child 好 = improved（补数据有效）
      - parent 好 -> child 差 = regressed（灾难性遗忘）
    """
    p5 = _is_5_scale(parent_report)
    c5 = _is_5_scale(child_report)

    parent_map = {}
    for d in parent_report.get("details", []):
        prompt = d.get("prompt", "")
        score = d.get("overall") or d.get("judge_score") or 0
        parent_map[prompt] = {
            "score": _norm_score(score, p5),
            "prediction": d.get("prediction", ""),
            "hallucination": d.get("hallucination", 0),
        }

    categories = {
        "improved": [],
        "regressed": [],
        "stable_good": [],
        "stable_bad": [],
    }

    for d in child_report.get("details", []):
        prompt = d.get("prompt", "")
        child_score = d.get("overall") or d.get("judge_score") or 0
        child_score = _norm_score(child_score, c5)
        parent = parent_map.get(prompt)

        if not parent:
            continue

        parent_score = parent["score"]

        entry = {
            "prompt": prompt,
            "parent_score": round(parent_score, 1),
            "child_score": round(child_score, 1),
            "delta": round(child_score - parent_score, 1),
            "parent_prediction": parent["prediction"][:80],
            "child_prediction": d.get("prediction", "")[:80],
            "child_hallucination": d.get("hallucination", 0),
        }

        if child_score >= 70 and parent_score < 70:
            categories["improved"].append(entry)
        elif child_score < 60 and parent_score >= 60:
            categories["regressed"].append(entry)
        elif child_score >= 60:
            categories["stable_good"].append(entry)
        else:
            categories["stable_bad"].append(entry)

    return categories


# ---------------------------------------------------------------------------
# 智能构造回滚数据集
# ---------------------------------------------------------------------------

def _extract_prompt_keywords(prompt: str) -> set[str]:
    """从 prompt 提取关键词（中文 2-gram，无需 jieba 分词）。

    与 BM25 检索器的 bigram 策略一致：机械切分，领域无关。
    """
    import re
    # 去标点，保留中英文字母数字
    clean = re.sub(r"[？?，,。.！!、：:；;""\"'()\[\]【】]", "", prompt)
    # 英文单词 + 中文 2-gram
    words: set[str] = set()
    # 英文连续字母数字作为一个词
    for m in re.finditer(r"[A-Za-z0-9]+", clean):
        w = m.group()
        if len(w) >= 2:
            words.add(w.lower())
    # 中文 2-gram
    chinese = re.sub(r"[A-Za-z0-9\s]", "", clean)
    for i in range(len(chinese) - 1):
        words.add(chinese[i : i + 2])
    return words


def build_rollback_dataset(
    parent_dataset_id: str,
    child_dataset_id: str,
    analysis: dict,
    output_dataset_id: str | None = None,
) -> dict:
    """构造回滚数据集：父版本全量 + 子版本中验证有效的样本。

    只保留 analysis["improved"] 对应主题的子版本新增样本，
    丢弃引发退步的样本。

    Returns:
        {"dataset_id": str, "sample_count": int,
         "kept_from_child": int, "dropped_from_child": int}
    """
    from ..common import DATASETS

    parent_train = DATASETS / parent_dataset_id / "train.jsonl"
    child_train = DATASETS / child_dataset_id / "train.jsonl"

    if not parent_train.exists():
        return {"error": f"父数据集训练文件不存在: {parent_train}"}
    if not child_train.exists():
        return {"error": f"子数据集训练文件不存在: {child_train}"}

    # 读取父版本样本（全量保留）
    parent_samples = []
    parent_prompts = set()
    with open(parent_train, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                parent_samples.append(obj)
                # 提取 user prompt 用于去重
                for msg in obj.get("messages", []):
                    if msg.get("role") == "user":
                        parent_prompts.add(msg["content"])
            except json.JSONDecodeError:
                continue

    # 提取"验证有效"的 prompt 关键词：
    # 1. improved 类（父差->子好，补数据直接生效）
    # 2. stable_good 中 delta > 10 的（父好->子更好，补数据有增益）
    # 3. stable_bad 中 delta > 15 的（两版都差但明显改善，方向对）
    improved_keywords = set()
    for item in analysis.get("improved", []):
        improved_keywords |= _extract_prompt_keywords(item["prompt"])
    for item in analysis.get("stable_good", []):
        if item["delta"] > 10:
            improved_keywords |= _extract_prompt_keywords(item["prompt"])
    for item in analysis.get("stable_bad", []):
        if item["delta"] > 15:
            improved_keywords |= _extract_prompt_keywords(item["prompt"])

    # 遍历子版本，只保留：
    # 1. 父版本已有的（去重跳过）
    # 2. 新增样本中，主题匹配 improved 类的
    kept_from_child = 0
    dropped_from_child = 0
    new_samples = []

    with open(child_train, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 提取 user prompt
            user_content = ""
            for msg in obj.get("messages", []):
                if msg.get("role") == "user":
                    user_content = msg["content"]
                    break

            # 父版本已有的跳过（去重）
            if user_content in parent_prompts:
                continue

            # 新增样本：检查是否匹配有效主题（至少 2 个 bigram 交集，防泛化匹配）
            sample_keywords = _extract_prompt_keywords(user_content)
            if len(sample_keywords & improved_keywords) >= 2:
                new_samples.append(obj)
                kept_from_child += 1
            else:
                dropped_from_child += 1

    # 生成新数据集
    if output_dataset_id is None:
        import time
        output_dataset_id = f"rollback-fix-{time.strftime('%Y%m%d-%H%M%S')}"

    out_dir = DATASETS / output_dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 复制 valid 集（保持 held-out 不变）
    parent_valid = DATASETS / parent_dataset_id / "valid.jsonl"
    if parent_valid.exists():
        import shutil
        shutil.copy2(parent_valid, out_dir / "valid.jsonl")

    # 写 train.jsonl：父全量 + 有效子集
    out_train = out_dir / "train.jsonl"
    total = 0
    with open(out_train, "w", encoding="utf-8") as fh:
        for obj in parent_samples:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total += 1
        for obj in new_samples:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total += 1

    # 登记血缘
    try:
        from .. import registry
        registry.register_dataset(
            dataset_id=output_dataset_id,
            parent=parent_dataset_id,
            sample_count=total,
            source=f"rollback:kept_{kept_from_child}_dropped_{dropped_from_child}",
        )
    except Exception:
        pass

    return {
        "dataset_id": output_dataset_id,
        "sample_count": total,
        "parent_samples": len(parent_samples),
        "kept_from_child": kept_from_child,
        "dropped_from_child": dropped_from_child,
    }


# ---------------------------------------------------------------------------
# 安全阀
# ---------------------------------------------------------------------------

MAX_AUTO_ITERATIONS = 3
MAX_REGRESSION_RATIO = 0.5


def check_regression_ratio(analysis: dict, total_details: int) -> dict:
    """检查退步题比例是否超过安全阀（模型容量问题）。"""
    regressed = len(analysis.get("regressed", []))
    ratio = regressed / total_details if total_details else 0

    if ratio > MAX_REGRESSION_RATIO:
        return {
            "triggered": True,
            "ratio": round(ratio, 4),
            "regressed_count": regressed,
            "total": total_details,
            "recommendation": "退步题过半，不是数据问题，建议换更大模型或加 DPO",
        }
    return {
        "triggered": False,
        "ratio": round(ratio, 4),
        "regressed_count": regressed,
        "total": total_details,
    }


def load_report(report_path: str | Path) -> dict:
    """加载评测报告 JSON。"""
    return json.loads(Path(report_path).read_text(encoding="utf-8"))


def find_report_by_model(model_id: str) -> Path | None:
    """在 reports 目录查找指定模型的最新评测报告。"""
    from ..common import REPORTS
    candidates = sorted(REPORTS.glob(f"eval-{model_id}-*.json"), reverse=True)
    return candidates[0] if candidates else None

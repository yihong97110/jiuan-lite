"""控制面 REST API —— 提交/查询 标·训·推·评 任务。

启动: python -m jiuan.app  (默认 http://127.0.0.1:8000)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import registry, store
from .pipeline import breadth as breadth_pipeline
from .pipeline import evaluate as evaluate_pipeline
from .pipeline import impact as impact_pipeline
from .pipeline import rag_backend
from . import annotation as annotate
from .common import DATASETS, MODELS, ROOT, mock_mode
from .schemas import (
    AgentStartReq,
    AnnotationCommitReq,
    AnnotationCreateReq,
    BiologyAgentStartReq,
    BreadthAnalysisReq,
    ChatCreateReq,
    ChatReq,
    DataprepReq,
    DistillReq,
    EvalReq,
    ImpactValidationReq,
    InferReq,
    McpRegisterReq,
    McpTestReq,
    LocalAgentSwitchReq,
    AnnotationSaveReq,
    AnnotationWebhookReq,
    CustomAgentPlanReq,
    CustomAgentContinueReq,
    CustomAgentExtendReq,
    CustomAgentSelectReq,
    CustomAgentStartReq,
    OneClickAgentStartReq,
    RagIngestReq,
    RagQueryReq,
    RagResolveReq,
    SkillInvokeReq,
    SkillRegisterReq,
    Stage,
    TrainReq,
    WorkflowReq,
)
from .workers import agent, biology_agent, custom_agent, oneclick_agent, runner
from .pipeline import langchain_agent
from .pipeline import local_agent

app = FastAPI(title="jiuan-lite 训推平台", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    store.init_db()
    # 加载 .env 到 os.environ（用户前端填的 judge Key 持久化在此）
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _safe_source(source: str) -> str:
    """校验 source 落在项目目录内，防止路径遍历读取任意文件。"""
    p = Path(source)
    resolved = (p if p.is_absolute() else (ROOT / p)).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        raise HTTPException(400, "source 必须位于项目目录内")
    if not resolved.exists():
        raise HTTPException(400, f"source 不存在: {source}")
    return str(resolved.relative_to(ROOT.resolve())).replace("\\", "/")


def _require_dataset(dataset_id: str) -> str:
    """校验 dataset_id 存在且含 train.jsonl，避免提交后才失败。"""
    if not dataset_id or dataset_id in ("__none__", "none", "null"):
        raise HTTPException(400, "请先在①标·数据准备生成数据集，再选择 dataset_id")
    if not (DATASETS / dataset_id / "train.jsonl").exists():
        avail = sorted(p.name for p in DATASETS.iterdir() if p.is_dir())
        raise HTTPException(400, f"数据集不存在: {dataset_id}。可用: {avail}")
    return dataset_id


def _jsonl_profile(path: Path) -> dict:
    count = 0
    kind = "unknown"
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            count += 1
            if kind == "unknown":
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    kind = "invalid"
                    continue
                if isinstance(obj, dict) and obj.get("messages"):
                    kind = "chat"
                elif isinstance(obj, dict) and obj.get("instruction") and obj.get("output"):
                    kind = "instruction/output"
                else:
                    kind = "other"
    return {"samples": count, "kind": kind}


@app.get("/")
def index():
    web = ROOT / "web" / "index.html"
    return FileResponse(web) if web.exists() else {"msg": "jiuan-lite running"}


@app.get("/health")
def health():
    return {"status": "ok", "mock_mode": mock_mode()}


@app.get("/dataprep/sources")
def dataprep_sources():
    """列出可作为 ① 数据准备 source 的项目内 jsonl 文件。"""
    data_dir = ROOT / "data"
    if not data_dir.exists():
        return []
    skip_dirs = {"datasets", "registry", "reports", "vector_db", "models", "impact_eval"}
    rows: list[dict] = []
    for path in data_dir.rglob("*.jsonl"):
        rel_to_data = path.relative_to(data_dir)
        if rel_to_data.parts and rel_to_data.parts[0] in skip_dirs:
            continue
        lowered = path.name.lower()
        if "heldout" in lowered or "eval" in lowered or "验证" in path.name:
            continue
        profile = _jsonl_profile(path)
        if profile["kind"] not in {"instruction/output", "chat"}:
            continue
        rel = path.relative_to(ROOT).as_posix()
        rows.append(
            {
                "source": rel,
                "samples": profile["samples"],
                "kind": profile["kind"],
                "updated_at": path.stat().st_mtime,
            }
        )
    rows.sort(key=lambda r: (r["source"] != "data/samples.jsonl", -r["updated_at"], r["source"]))
    return rows


@app.post("/dataprep")
def dataprep(req: DataprepReq):
    params = req.model_dump()
    source_dataset = params.get("source_dataset")
    parent = params.get("parent")
    if source_dataset:
        params["source_dataset"] = _require_dataset(source_dataset)
        params["source"] = params.get("source") or ""
    else:
        params["source"] = _safe_source(params.get("source") or "data/samples.jsonl")
    if parent:
        params["parent"] = _require_dataset(parent)
    if params.get("source_dataset") and params.get("parent") == params.get("source_dataset"):
        raise HTTPException(400, "source 数据集和父数据集不能相同，否则增量合并不会新增内容")
    return {"task_id": runner.submit(Stage.DATAPREP, params)}


@app.post("/distill")
def distill(req: DistillReq):
    """蒸馏：调用外部大模型API生成50条问答对，保存为数据集。"""
    params = req.model_dump()
    if not params.get("api_key"):
        raise HTTPException(400, "API key不能为空")
    return {"task_id": runner.submit(Stage.DISTILL, params)}


@app.post("/train")
def train(req: TrainReq):
    _require_dataset(req.dataset_id)
    return {"task_id": runner.submit(Stage.TRAIN, req.model_dump())}


@app.post("/infer")
def infer(req: InferReq):
    return {"task_id": runner.submit(Stage.INFER, req.model_dump())}


@app.post("/eval")
def evaluate(req: EvalReq):
    _require_dataset(req.dataset_id)
    params = req.model_dump()
    # 若前端填了 judge Key，先持久化到 .env（让 worker 子进程通过 env 回退读到），再脱敏不存明文 Key
    if params.get("judge_api_key"):
        from .common import apply_judge_overrides, load_config
        apply_judge_overrides(load_config(), params)  # 仅做 .env 持久化 + os.environ 设置，结果丢弃
        params["judge_api_key"] = None  # 脱敏：DB 和 worker 都不存明文 Key，走 env 回退
    return {"task_id": runner.submit(Stage.EVAL, params)}


@app.post("/workflow")
def workflow(req: WorkflowReq):
    """一键全链路：标->训->推->评，单任务内串行执行并逐步记录。"""
    params = req.model_dump()
    params["source"] = _safe_source(params["source"])
    return {"task_id": runner.submit(Stage.WORKFLOW, params)}


@app.post("/agent/start")
def agent_start(req: AgentStartReq):
    """启动自动迭代 Agent：等待标注完成→回灌→训练→评测→必要时创建新标注任务。"""
    return agent.start(req.model_dump())


@app.get("/agent/status")
def agent_status():
    """查看自动迭代 Agent 当前状态、日志和每轮摘要。"""
    return agent.status()


@app.post("/agent/biology/start")
def biology_agent_start(req: BiologyAgentStartReq):
    """启动生物专家三轮场景：自动标注回灌→训练→推理→评测→广度分析。"""
    return biology_agent.start(req.model_dump())


@app.get("/agent/biology/status")
def biology_agent_status():
    """查看生物专家 Agent 进度、步骤总结和每轮摘要。"""
    return biology_agent.status()


@app.post("/agent/oneclick/start")
def oneclick_agent_start(req: OneClickAgentStartReq):
    """启动一键保通打包：补全材料/知识库，必要时跑生物三轮，再汇总评测与问题分析。"""
    return oneclick_agent.start(req.model_dump())


@app.get("/agent/oneclick/status")
def oneclick_agent_status():
    """查看一键保通打包进度、检查项、问题分析和报告路径。"""
    return oneclick_agent.status()


@app.post("/agent/custom/plan")
def custom_agent_plan(req: CustomAgentPlanReq):
    """根据自然语言方向生成通用训练 Agent 的推荐参数与注意事项。"""
    return custom_agent.plan(req.model_dump())


@app.post("/agent/custom/start")
def custom_agent_start(req: CustomAgentStartReq):
    """启动通用训练 Agent：自动生成数据/验证集/知识库，分轮回灌训练评测。"""
    return custom_agent.start(req.model_dump())


@app.get("/agent/custom/status")
def custom_agent_status():
    """查看通用训练 Agent 进度、步骤总结和每轮摘要。"""
    return custom_agent.status()


@app.post("/agent/custom/continue")
def custom_agent_continue(req: CustomAgentContinueReq):
    """人工标注审核完成后，从暂停点继续通用训练 Agent。"""
    return custom_agent.continue_run(req.model_dump())


@app.post("/agent/custom/extend")
def custom_agent_extend(req: CustomAgentExtendReq):
    """在已完成的通用训练 Agent 基础上继续追加若干轮迭代。"""
    return custom_agent.extend(req.model_dump())


@app.post("/agent/custom/select-model")
def custom_agent_select_model(req: CustomAgentSelectReq):
    """只切换当前 Agent 项目上下文，不生成数据或启动训练。"""
    try:
        return custom_agent.select_model(req.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/tasks")
def tasks(stage: str | None = None):
    try:
        st = Stage(stage) if stage else None
    except ValueError:
        st = None
    return [t.model_dump() for t in store.list_tasks(st)]


@app.get("/models")
def models():
    return registry.list_models()


@app.get("/models/selectable")
def selectable_models():
    """已训练模型恢复点：供通用 Agent 选择模型继续迭代。"""
    return registry.selectable_models()


@app.get("/models/{model_id}/resume-profile")
def model_resume_profile(model_id: str):
    """恢复续训模型所属项目的完整数据、评测和知识库上下文。"""
    try:
        return registry.model_resume_profile(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/datasets")
def datasets():
    """数据集血缘列表(含 parent_dataset / 样本数)。"""
    return registry.list_datasets()


@app.get("/iterations")
def iterations():
    """按数据集血缘分组的迭代对比(v1->v2->... 的 loss/ROUGE/BLEU/judge)。"""
    return registry.iterations()


@app.get("/gaps")
def gaps():
    """汇总所有评测报告的薄弱点建议，从新到旧并标注所属迭代(v1/v2/…)。"""
    return evaluate_pipeline.collect_all_gaps()


@app.post("/rag/ingest")
def rag_ingest(req: RagIngestReq):
    """上传应急法规/预案文档入库（分块+嵌入）。"""
    source = _safe_source(req.source)
    return rag_backend.ingest_document(source, req.chunk_size, log=lambda _m: None, collection=req.collection)


@app.post("/rag/query")
def rag_query(req: RagQueryReq):
    """RAG 检索增强推理：检索知识→注入上下文→生成。"""
    return rag_backend.generate_with_rag(req.prompt, req.model_id, req.top_k, log=lambda _m: None, collection=req.collection)


@app.get("/rag/stats")
def rag_stats(collection: str = "default"):
    """知识库统计（块数/来源/嵌入器）。"""
    return rag_backend.stats(collection)


@app.get("/rag/collections")
def rag_collections():
    """可切换知识库集合列表。"""
    return rag_backend.collections()


@app.get("/rag/gaps")
def rag_gaps(status: str = "pending", collection: str = "default"):
    """RAG 检索缺口台账：库里缺相关内容的问题(默认只看待补充)。"""
    return rag_backend.list_gaps(status or None, collection=collection)


@app.post("/rag/gaps/resolve")
def rag_gaps_resolve(req: RagResolveReq):
    """一键补充知识：追加到用户知识库->自动重新入库->标记缺口已解决。"""
    try:
        return rag_backend.resolve_gap(req.prompt, req.knowledge, log=lambda _m: None, collection=req.collection)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# API Key 管理：前端动态配置 DeepSeek/火山方舟/OpenAI Key
# ---------------------------------------------------------------------------

from pydantic import BaseModel

class ApiKeySaveReq(BaseModel):
    """前端保存 API Key 请求。"""
    env_name: str  # DEEPSEEK_API_KEY / ARK_API_KEY / OPENAI_API_KEY
    api_key: str


@app.get("/config/api-keys")
def get_api_keys():
    """获取所有 API Key 配置状态（脱敏）。"""
    from .common import get_api_keys as _get_keys
    return _get_keys()


@app.post("/config/api-keys")
def save_api_key(req: ApiKeySaveReq):
    """保存 API Key 到 .env 文件（长期持久化）+ 当前进程环境变量（立即生效）。"""
    from .common import save_api_key as _save_key
    result = _save_key(req.env_name, req.api_key)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/analysis/breadth")
def breadth_analysis(req: BreadthAnalysisReq):
    """按类别调用判断模型做广度验证，识别新模型覆盖面问题。"""
    params = req.model_dump()
    params["source"] = _safe_source(params["source"])
    return breadth_pipeline.run(params, log=lambda _m: None)


@app.get("/analysis/breadth/reports")
def breadth_reports(limit: int = 20):
    """最近的广度验证报告：用于④评测页展示验证集广度与 judge 发现的问题。"""
    return breadth_pipeline.list_reports(limit)


@app.post("/analysis/impact/run")
def impact_validation(req: ImpactValidationReq):
    """生成回灌效果 impact 验证集，对比回灌前后模型，并为未改善项创建补漏任务。"""
    return impact_pipeline.run(req.model_dump(), log=lambda _m: None)


@app.get("/analysis/impact/reports")
def impact_reports(limit: int = 20):
    """最近的回灌效果验证报告：用于⑦标注页展示查缺补漏闭环。"""
    return impact_pipeline.list_reports(limit)


@app.post("/annotation/create")
def annotation_create(req: AnnotationCreateReq):
    """从评测薄弱点生成可人工填空的标注任务文件。"""
    return annotate.create_task(req.name, req.max_items)


@app.get("/annotation/tasks")
def annotation_tasks():
    """列出所有标注任务及完成进度。"""
    return annotate.list_tasks()


@app.get("/annotation/tasks/{task_id}")
def annotation_preview(task_id: str):
    """预览标注任务内容（含每行标注状态）。"""
    try:
        return annotate.preview(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.post("/annotation/save")
def annotation_save(req: AnnotationSaveReq):
    """浏览器内直接标注：把 {行号:答案} 回写到标注任务 jsonl。"""
    try:
        return annotate.save_annotations(req.task_id, req.annotations)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.get("/annotation/coverage")
def annotation_coverage():
    """标注覆盖度：按 gap_type / iteration 统计已标注/待标注。"""
    return annotate.coverage()


@app.get("/annotation/resolved")
def annotation_resolved():
    """跨版本 gap 解决追踪：每次回灌解决了哪些薄弱点及其前后评测变化。"""
    return annotate.resolved_tracking()


@app.post("/annotation/commit")
def annotation_commit(req: AnnotationCommitReq):
    """把已标注项回灌为训练集(经 dataprep 增量继承)，返回新数据集；
    可选 auto_train：回灌后自动排一次训练任务。"""
    try:
        result = annotate.commit(req.task_id, req.name, req.parent, hard_weight=req.hard_weight)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    if req.auto_train:
        params = {"dataset_id": result["dataset_id"], "name": f"{req.name}-sft"}
        result["train_task_id"] = runner.submit(Stage.TRAIN, params)
        result["auto_train"] = True
    return result


@app.get("/annotation/backend")
def annotation_backend():
    """当前标注后端(jsonl/label_studio)及可切换项，供 UI 显示。"""
    return {"backend": annotate.backend_name(), "available": ["jsonl", "label_studio"]}


@app.post("/annotation/webhook/ls")
def annotation_webhook_ls(req: AnnotationWebhookReq):
    """Label Studio 完成回调(方式B)：标注完成→自动回灌(+可选自动训练)。
    LS 项目 Webhook 指向本接口，实现"标完即训"的闭环。"""
    try:
        result = annotate.commit(req.task_id, req.name, req.parent, hard_weight=req.hard_weight)
    except (FileNotFoundError, ValueError, NotImplementedError) as exc:
        raise HTTPException(400, str(exc))
    if req.auto_train:
        params = {"dataset_id": result["dataset_id"], "name": f"{req.name}-sft"}
        result["train_task_id"] = runner.submit(Stage.TRAIN, params)
        result["auto_train"] = True
    result["via"] = "webhook"
    return result


@app.get("/models/{model_id}/lineage")
def model_lineage(model_id: str):
    return registry.lineage(model_id)


@app.post("/models/upload")
async def model_upload(
    file: UploadFile = File(..., description="模型包 zip（PEFT adapter 或 HF 全量，HuggingFace 标准格式）"),
    model_id: str = Form("", description="自定义模型ID；留空则用文件名生成"),
    name: str = Form("", description="显示名/领域方向，如「高考志愿专家」"),
):
    """上传自有/第三方模型交付包，注册到平台模型仓库后可直接推理与 Agent 对话。

    支持两种 HuggingFace 主流交付格式（与广大社区交付包一致）：
    1. PEFT LoRA adapter：zip 内含 adapter_config.json + adapter_model.safetensors
       （此平台训练产物的 weights/ 即此格式，HuggingFace Hub 上多数 LoRA 也是）
    2. HF 全量模型：zip 内含 config.json + model.safetensors（或分片 model-00001-of-*）

    解压到 data/models/<model_id>/weights/，写 meta.json，登记血缘；
    上传后需在 vLLM 侧 `--lora-modules <model_id>=<path>`（LoRA）或
    `vllm serve <path>`（全量）加载，Agent/推理即可按 model_id 调用。
    """
    import zipfile
    import io
    import time
    import uuid

    # 0. 校验文件类型与大小（限 4GB，覆盖常见模型包）
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "请上传 .zip 格式的模型包")
    raw = await file.read()
    if len(raw) < 200:
        raise HTTPException(400, "文件过小，疑似空包")
    if len(raw) > 4 * 1024 ** 3:
        raise HTTPException(400, "包超过 4GB 限制")

    # 1. 读 zip，校验结构 + 防路径遍历
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise HTTPException(400, f"zip 解析失败: {exc}") from exc
    names = [n for n in zf.namelist() if not n.endswith("/")]
    if not names:
        raise HTTPException(400, "zip 为空")
    # 安全：禁止绝对路径/.. 上跳
    bad = [n for n in names if n.startswith("/") or ".." in n]
    if bad:
        raise HTTPException(400, f"zip 含不安全路径: {bad[:3]}")

    # 统一为顶层文件（zip 可能套一层目录）
    # 找公共前缀目录
    top_dirs = {n.split("/")[0] for n in names if "/" in n}
    prefix = next(iter(top_dirs)) + "/" if len(top_dirs) == 1 and all(n.startswith(next(iter(top_dirs)) + "/") for n in names) else ""
    rel_names = [n[len(prefix):] if n.startswith(prefix) else n for n in names]

    # 2. 判定格式（必须有核心权重文件之一）
    lower = {n.lower() for n in rel_names}
    is_peft = "adapter_config.json" in lower and any(
        n.endswith("adapter_model.safetensors") or n.endswith("adapter_model.bin") for n in lower
    )
    is_full = "config.json" in lower and any(
        n.startswith("model") and (n.endswith(".safetensors") or n.endswith(".bin")) for n in lower
    )
    if not (is_peft or is_full):
        raise HTTPException(
            400,
            "未识别为合法模型包。PEFT 需含 adapter_config.json + adapter_model.safetensors；"
            "全量需含 config.json + model*.safetensors",
        )

    # 3. 生成 model_id + 解压目录
    mid = (model_id or "").strip() or f"upload-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    # 防 model_id 路径注入
    if "/" in mid or "\\" in mid or ".." in mid:
        raise HTTPException(400, "model_id 含非法字符")
    out_dir = MODELS / mid
    if out_dir.exists():
        raise HTTPException(409, f"model_id 已存在: {mid}，请改名或先删除")
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 4. 解压到 weights/
    for n, rel in zip(names, rel_names):
        target = weights_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(n) as src, open(target, "wb") as dst:
            dst.write(src.read())

    # 5. 读 adapter_config / config 抽取基座信息
    base = ""
    domain = name.strip()
    try:
        if is_peft:
            cfg = json.loads((weights_dir / "adapter_config.json").read_text(encoding="utf-8"))
            base = cfg.get("base_model_name_or_path") or cfg.get("base_model_name") or ""
        else:
            cfg = json.loads((weights_dir / "config.json").read_text(encoding="utf-8"))
            base = cfg.get("_name_or_path") or cfg.get("model_type") or ""
    except Exception:
        pass

    # 6. 写 meta.json + 注册血缘
    meta = {
        "model_id": mid,
        "base": base or "unknown",
        "mode": "upload",  # 区分平台训练产物 vs 用户上传
        "backend": "vllm",
        "method": "lora" if is_peft else "full",
        "dataset_id": "",
        "domain_direction": domain,
        "uploaded_at": time.time(),
        "weights_dir": str(weights_dir),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    registry.register_model(meta)

    return {
        "model_id": mid,
        "format": "peft-lora" if is_peft else "hf-full",
        "base": base or "unknown",
        "domain_direction": domain,
        "files": len(rel_names),
        "size_mb": round(len(raw) / 1024 / 1024, 2),
        "artifact_path": str(weights_dir),
        "hint": ("已注册。请在 vLLM 侧加载后即可在 Agent/推理中选用此 model_id。"
                 "LoRA: vllm serve <base> --lora-modules {mid}={path}".format(mid=mid, path=weights_dir) if is_peft
                 else "全量模型: vllm serve {path}".format(path=weights_dir)),
    }


@app.get("/tasks/{task_id}")
def task_detail(task_id: str):
    t = store.get_task(task_id)
    if not t:
        raise HTTPException(404, "task not found")
    return t.model_dump()


# ---------------------------------------------------------------------------
# LangChain Agent 对话接口
# ---------------------------------------------------------------------------

@app.get("/chat/available")
def chat_available():
    """检查 LangChain Agent 是否可用（langchain-openai 是否已安装）。"""
    return {"available": langchain_agent.available()}


@app.post("/chat/session")
def chat_create_session(req: ChatCreateReq):
    """创建对话会话，指定系统提示词、关联模型和滑动窗口大小。

    双层记忆：短期=RAM 滑动窗口；长期=SQLite（scope 域共享用户事实，
    resume_session_id 可在平台重启后恢复历史会话上下文）。
    """
    if not langchain_agent.available():
        raise HTTPException(503, "LangChain 未安装，请运行 pip install -r requirements-agent.txt")
    return langchain_agent.create_session(
        system_prompt=req.system_prompt,
        model_id=req.model_id,
        memory_window=req.memory_window,
        scope=req.scope,
        resume_session_id=req.resume_session_id,
    )


# ---------------------------------------------------------------------------
# Agent 记忆管理（短期 RAM + 长期 SQLite）
# ---------------------------------------------------------------------------

@app.get("/chat/memory/{session_id}")
def chat_memory(session_id: str):
    """查看会话的双层记忆状态：短期窗口 + 模型域长期事实（LangGraph Store）。"""
    from . import memory_store

    session = langchain_agent.get_session(session_id)
    model_id = ((session or {}).get("model_id", "") or "global").strip()
    stored = memory_store.list_sessions_stored()
    msg_count = next((s["messages"] for s in stored if s["session_id"] == session_id), 0)

    # 长期记忆：LangGraph Store，namespace=("memories", model_id) 按模型隔离
    facts = []
    try:
        store = langchain_agent._get_store()
        ns = langchain_agent._memory_namespace(model_id)
        facts = [
            it.value.get("text", "")
            for it in store.search(ns, limit=30)
            if it.value.get("text")
        ]
    except Exception:
        pass
    return {
        "session_id": session_id,
        "in_ram": session is not None,
        "model_id": model_id,
        "memory_namespace": list(langchain_agent._memory_namespace(model_id)),
        "short_term": {
            "window": (session or {}).get("memory_window", 0),
            "current_messages": len((session or {}).get("messages", [])),
        },
        "long_term": {
            "persisted_messages": msg_count,
            "facts": facts,
        },
    }


@app.get("/chat/memory")
def chat_memory_overview():
    """全局记忆概览：会话历史 + 各模型记忆域事实统计（LangGraph Store，按模型隔离）。"""
    from . import memory_store

    sessions = memory_store.list_sessions_stored()
    # Store 中各模型域的事实条数：namespace ("memories", model_id)
    model_domains: dict = {}
    try:
        store = langchain_agent._get_store()
        for it in store.search(("memories",), limit=500):
            if len(it.namespace) >= 2:
                mid = it.namespace[1]
                model_domains[mid] = model_domains.get(mid, 0) + 1
    except Exception:
        pass
    return {
        "sessions_in_db": sessions,
        "model_memory_domains": model_domains,
    }


@app.delete("/chat/memory/{scope}")
def chat_memory_delete(scope: str, confirm: bool = False):
    """清空指定模型记忆域的长期事实（LangGraph Store，按模型隔离）。

    scope 此处为模型 id（记忆域），需 confirm=true 防误删。
    """
    if not confirm:
        raise HTTPException(400, "需传 confirm=true 确认清除")
    try:
        store = langchain_agent._get_store()
        ns = langchain_agent._memory_namespace(scope)
        items = store.search(ns, limit=1000)
        for it in items:
            store.delete(ns, it.key)
        return {"deleted_facts": len(items), "model_id": scope,
                "memory_namespace": list(ns)}
    except Exception as exc:
        raise HTTPException(500, f"清除记忆失败: {exc}")


@app.get("/chat/sessions")
def chat_sessions():
    """列出所有对话会话。"""
    return langchain_agent.list_sessions()


@app.get("/chat/sessions/{session_id}")
def chat_session_detail(session_id: str):
    """获取指定会话的完整消息历史。"""
    session = langchain_agent.get_session(session_id)
    if not session:
        raise HTTPException(404, f"会话不存在: {session_id}")
    return session


@app.delete("/chat/sessions/{session_id}")
def chat_delete_session(session_id: str):
    """删除指定会话。"""
    if not langchain_agent.delete_session(session_id):
        raise HTTPException(404, f"会话不存在: {session_id}")
    return {"deleted": session_id}


@app.post("/chat")
def chat(req: ChatReq):
    """非流式对话：Agent 处理用户消息并返回完整回复（含工具调用轨迹）。"""
    if not langchain_agent.available():
        raise HTTPException(503, "LangChain 未安装，请运行 pip install -r requirements-agent.txt")
    try:
        return langchain_agent.chat(req.session_id, req.message)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        msg = str(exc)
        # 无/错 API Key 时给出可操作提示，而非裸 500
        if "401" in msg or "AuthenticationError" in msg or "api key" in msg.lower():
            raise HTTPException(
                500,
                "对话 API 鉴权失败：请在 .env 或「⑬ 设置 · API Key」页配置有效的 "
                "DEEPSEEK_API_KEY（申请: platform.deepseek.com）后重试。原始错误: " + msg[:200],
            ) from exc
        raise HTTPException(500, f"Agent 对话失败: {exc}") from exc


@app.post("/chat/stream")
async def chat_stream(req: ChatReq):
    """流式对话（SSE）：逐 token 返回 LLM 输出，附带工具调用事件。

    SSE 事件格式：
    - data: {"token": "..."}      逐 token 流式输出
    - data: {"tool_start": "..."} 工具调用开始
    - data: {"tool_end": "..."}   工具调用结束
    - data: {"done": true}        全部完成
    - data: {"error": "..."}      错误
    """
    if not langchain_agent.available():
        raise HTTPException(503, "LangChain 未安装，请运行 pip install -r requirements-agent.txt")
    from fastapi.responses import StreamingResponse

    async def event_stream():
        async for chunk in langchain_agent.chat_stream(req.session_id, req.message):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# 本地 ReAct Agent（使用自己训练的模型，不依赖外部 API）
# ---------------------------------------------------------------------------

@app.get("/local_agent/available")
def local_agent_available():
    """检查本地 Agent 是否可用（是否有已训练模型）。"""
    models = registry.list_models()
    return {"available": len(models) > 0, "model_count": len(models)}


@app.post("/local_agent/session")
def local_agent_create_session(req: ChatCreateReq):
    """创建本地 ReAct Agent 会话，使用自己训练的模型。"""
    return local_agent.create_session(
        model_id=req.model_id,
        system_prompt=req.system_prompt,
        memory_window=req.memory_window or 5,
    )


@app.get("/local_agent/sessions")
def local_agent_sessions():
    """列出所有本地 Agent 会话。"""
    return {"sessions": list(local_agent.list_sessions())}


@app.get("/local_agent/sessions/{session_id}")
def local_agent_session_detail(session_id: str):
    """获取本地 Agent 会话详情。"""
    session = local_agent.get_session(session_id)
    if not session:
        raise HTTPException(404, f"会话不存在: {session_id}")
    return session


@app.delete("/local_agent/sessions/{session_id}")
def local_agent_delete_session(session_id: str):
    """删除本地 Agent 会话。"""
    if not local_agent.delete_session(session_id):
        raise HTTPException(404, f"会话不存在: {session_id}")
    return {"deleted": session_id}


@app.post("/local_agent/chat")
def local_agent_chat(req: ChatReq):
    """本地 ReAct Agent 对话：使用自己训练的模型 + RAG + 工具调用。

    生产级 v2：安全校验 -> 三层记忆 -> 四级解析 -> 工具沙箱 -> 循环控制。
    返回 end_reason（completed/keyword-fallback/max_iterations/repeated/security_blocked）。
    完全本地运行，不依赖任何外部 API。
    """
    try:
        return local_agent.chat(req.session_id, req.message)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"本地 Agent 对话失败: {exc}") from exc


@app.post("/local_agent/switch_model")
def local_agent_switch_model(req: LocalAgentSwitchReq):
    """适配器热插拔：会话中途切换底层模型（保留会话记忆）。

    vLLM 多 LoRA 挂载时无需重启服务；transformers 路径走 _REAL_CACHE 复用。
    """
    try:
        session = local_agent.switch_model(req.session_id, req.model_id)
        return {
            "session_id": session["session_id"],
            "model_id": session["model_id"],
            "switched_at": session.get("switched_at"),
            "switch_history": session.get("switch_history", []),
            "messages_kept": len(session.get("messages", [])),
        }
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"切换模型失败: {exc}") from exc


@app.get("/local_agent/eval/stats")
def local_agent_eval_stats():
    """Agent 测评统计：工具调用率/模型决策占比/修复次数/工具失败/结束原因分布。"""
    return local_agent.eval_stats()


@app.get("/local_agent/eval/events")
def local_agent_eval_events(limit: int = 50):
    """读取全链路事件日志（微调素材预览）：prompt快照/原始输出/工具耗时/失败原因。"""
    try:
        events = local_agent.read_events(limit=min(max(limit, 1), 200))
    except AttributeError:
        # 兼容：read_events 不存在时读文件
        events = []
    return {"events": events, "store": str(local_agent.EVENT_STORE)}


# ---------------------------------------------------------------------------
# Skill 管理
# ---------------------------------------------------------------------------

@app.get("/skills")
def list_skills():
    """列出所有已注册的技能。"""
    return langchain_agent.list_skills()


@app.post("/skills/register")
def register_skill(req: SkillRegisterReq):
    """注册自定义技能。"""
    return langchain_agent.register_skill(req.name, req.description, req.template, req.category)


@app.post("/skills/invoke")
def invoke_skill(req: SkillInvokeReq):
    """调用技能：用参数填充模板并返回完整 prompt。"""
    return {"prompt": langchain_agent.invoke_skill(req.name, req.params)}


# ---------------------------------------------------------------------------
# MCP 服务器管理
# ---------------------------------------------------------------------------

@app.get("/mcp/servers")
def list_mcp_servers():
    """列出所有已注册的 MCP 服务器。"""
    return langchain_agent.list_mcp_servers()


@app.post("/mcp/register")
def register_mcp_server(req: McpRegisterReq):
    """注册一个 MCP 服务器。"""
    return langchain_agent.register_mcp_server(req.name, req.command, req.description)


@app.delete("/mcp/servers/{name}")
def delete_mcp_server(name: str):
    """删除一个已注册的 MCP 服务器。"""
    try:
        return langchain_agent.delete_mcp_server(name)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"删除 MCP 服务器失败: {exc}") from exc


@app.post("/mcp/test")
def test_mcp_server(req: McpTestReq):
    """测试 MCP 服务器连通性（重新握手并刷新工具列表）。"""
    try:
        return langchain_agent.test_mcp_server(req.name)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"测试 MCP 服务器失败: {exc}") from exc


# ---------------------------------------------------------------------------
# Agent 工具列表
# ---------------------------------------------------------------------------

@app.get("/agent/tools")
def list_agent_tools():
    """列出 Agent 可用的所有工具。"""
    if not langchain_agent.available():
        return {"available": False, "tools": []}
    tools = langchain_agent._build_tools()
    return {
        "available": True,
        "tools": [{"name": t.name, "description": t.description[:100]} for t in tools],
    }


def main() -> None:
    import os
    import sys

    import uvicorn

    # 解析 --config 参数，写入环境变量供 worker 子进程继承
    config_path = None
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
        elif a.startswith("--config="):
            config_path = a.split("=", 1)[1]
    if config_path:
        os.environ["JIUAN_CONFIG"] = config_path

    host = os.environ.get("JIUAN_HOST", "127.0.0.1")
    port = int(os.environ.get("JIUAN_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()



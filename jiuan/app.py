"""控制面 REST API —— 提交/查询 标·训·推·评 任务。

启动: python -m jiuan.app  (默认 http://127.0.0.1:8000)
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from . import registry, store
from .pipeline import breadth as breadth_pipeline
from .pipeline import evaluate as evaluate_pipeline
from .pipeline import impact as impact_pipeline
from .pipeline import rag_backend
from . import annotation as annotate
from .common import DATASETS, ROOT, mock_mode
from .schemas import (
    AgentStartReq,
    AnnotationCommitReq,
    AnnotationCreateReq,
    BiologyAgentStartReq,
    BreadthAnalysisReq,
    ChatCreateReq,
    ChatReq,
    DataprepReq,
    EvalReq,
    ImpactValidationReq,
    InferReq,
    McpRegisterReq,
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

app = FastAPI(title="jiuan-lite 训推平台", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    store.init_db()


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
    return {"task_id": runner.submit(Stage.EVAL, req.model_dump())}


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
    st = Stage(stage) if stage else None
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
    """一键补充知识：追加到用户知识库→自动重新入库→标记缺口已解决。"""
    try:
        return rag_backend.resolve_gap(req.prompt, req.knowledge, log=lambda _m: None, collection=req.collection)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    """创建对话会话，指定系统提示词、关联模型和滑动窗口大小。"""
    if not langchain_agent.available():
        raise HTTPException(503, "LangChain 未安装，请运行 pip install -r requirements-agent.txt")
    return langchain_agent.create_session(
        system_prompt=req.system_prompt,
        model_id=req.model_id,
        memory_window=req.memory_window,
    )


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

    import uvicorn

    host = os.environ.get("JIUAN_HOST", "127.0.0.1")
    port = int(os.environ.get("JIUAN_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()



"""Task/artifact schemas mirroring 久安 标-训-推-评 stages."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Dict

from pydantic import BaseModel, Field


class Stage(str, Enum):
    DATAPREP = "dataprep"   # 标：数据准备/清洗
    TRAIN = "train"         # 训：SFT/LoRA
    INFER = "infer"         # 推：推理
    EVAL = "eval"           # 评：评测
    WORKFLOW = "workflow"   # 一键全链路
    AGENT = "agent"         # 自动迭代 Agent


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DataprepReq(BaseModel):
    source: Optional[str] = Field("data/samples.jsonl", description="jsonl 源文件（相对路径，位于项目内）")
    source_dataset: Optional[str] = Field(None, description="已有数据集 id；填了则把该数据集 train+valid 作为本轮新增 source")
    name: str = Field("emergency-sft")
    valid_ratio: float = Field(0.2, ge=0.0, le=0.9, description="验证集比例")
    seed: int = 42
    parent: Optional[str] = Field(None, description="父数据集 id；填了则在其基础上增量追加(血缘迭代)")


class TrainReq(BaseModel):
    dataset_id: str
    backend: Optional[str] = Field(None, description="auto|hf|llamafactory|mock，缺省取 config")
    method: Optional[str] = None  # 覆盖 config: full|lora
    epochs: Optional[int] = None
    device: Optional[str] = Field(None, description="auto|cpu|cuda")
    precision: Optional[str] = Field(None, description="auto|fp32|fp16|bf16")
    lr: Optional[float] = Field(None, description="学习率，缺省取 config")
    batch_size: Optional[int] = Field(None, ge=1)
    grad_accum: Optional[int] = Field(None, ge=1, description="梯度累积步数")
    max_seq_len: Optional[int] = Field(None, ge=16, description="最大序列长度")
    lora_r: Optional[int] = Field(None, ge=1, description="LoRA 秩 r")
    lora_alpha: Optional[int] = Field(None, ge=1, description="LoRA alpha")
    lora_dropout: Optional[float] = Field(None, ge=0.0, le=0.9, description="LoRA dropout")
    name: str = Field("qwen0.5b-sft")


class InferReq(BaseModel):
    model_id: str = Field("base", description="模型仓库 id 或 base")
    prompt: str
    backend: Optional[str] = Field(None, description="auto|transformers|vllm|mock，缺省取 config")
    system_prompt: Optional[str] = Field(None, description="可选系统提示词，用于领域专家场景")
    max_new_tokens: Optional[int] = Field(None, ge=1, description="最大生成 token 数")
    do_sample: Optional[bool] = Field(None, description="是否采样(false=贪心)")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="采样温度(do_sample=true 生效)")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="核采样(do_sample=true 生效)")


class EvalReq(BaseModel):
    model_id: str = Field("base")
    dataset_id: str
    split: str = Field("valid", description="评测所用切分：valid|train")
    use_judge: Optional[str] = Field(None, description="auto|true|false，LLM-as-Judge 开关，缺省取 config")
    baseline: Optional[str] = Field(None, description="对比基线模型 id（如 base）；填了则并列对比并给出 delta")
    max_samples: Optional[int] = Field(None, ge=1, description="评测样本上限，缺省取 config")
    system_prompt: Optional[str] = Field(None, description="可选系统提示词，用于领域专家场景")


class RagIngestReq(BaseModel):
    source: str = Field(..., description="待入库文档（项目目录内的 txt/md）")
    chunk_size: int = Field(400, ge=50, le=2000, description="分块字符数")
    collection: str = Field("default", description="知识库集合/命名空间")


class RagQueryReq(BaseModel):
    prompt: str = Field(..., description="提问")
    model_id: str = Field("base", description="模型仓库 id 或 base")
    top_k: int = Field(3, ge=1, le=10, description="检索返回段落数")
    collection: str = Field("default", description="知识库集合/命名空间")


class RagResolveReq(BaseModel):
    prompt: str = Field("", description="缺口对应的问题(作为知识段落标题)")
    knowledge: str = Field(..., description="人工补充的权威知识/法规原文")
    collection: str = Field("default", description="知识库集合/命名空间")


class AnnotationCreateReq(BaseModel):
    name: str = Field("gap", description="标注任务名前缀")
    max_items: Optional[int] = Field(None, ge=1, description="最多取多少条薄弱点(缺省全部)")


class AnnotationCommitReq(BaseModel):
    task_id: str = Field(..., description="标注任务 id")
    name: str = Field("annotated", description="生成新数据集名前缀")
    parent: Optional[str] = Field(None, description="父数据集 id；填了则增量继承(血缘迭代)")
    hard_weight: int = Field(1, ge=1, le=10, description="硬样本(知识缺失类)在训练集内复制倍数")
    auto_train: bool = Field(False, description="回灌后是否自动触发一次训练(用 config 默认超参)")


class AnnotationSaveReq(BaseModel):
    task_id: str = Field(..., description="标注任务 id")
    annotations: Dict[str, str] = Field(..., description="{行索引: 答案}，浏览器内直接标注回写")


class AnnotationWebhookReq(BaseModel):
    """Label Studio 完成回调：标完即回灌(+可选自动训练)。字段宽松以兼容 LS payload。"""
    task_id: str = Field(..., description="标注任务/项目 id")
    name: str = Field("annotated", description="生成新数据集名前缀")
    parent: Optional[str] = Field(None, description="父数据集 id(增量血缘)")
    hard_weight: int = Field(1, ge=1, le=10, description="硬样本复制倍数")
    auto_train: bool = Field(True, description="回灌后是否自动触发训练")


class AgentStartReq(BaseModel):
    parent_dataset: Optional[str] = Field(None, description="起始父数据集；为空则取最新非 eval 数据集")
    eval_dataset_id: Optional[str] = Field(None, description="固定 held-out 评测集；为空则取已标记 role=eval 的最新数据集")
    model_name: str = Field("agent-sft", description="训练模型名前缀")
    dataset_name: str = Field("agent-iter", description="回灌数据集名前缀")
    annotation_name: str = Field("agent-gap", description="新标注任务名前缀")
    max_iterations: int = Field(5, ge=1, le=20)
    poll_interval: float = Field(5.0, ge=1.0, le=60.0, description="等待标注/任务完成的轮询间隔秒数")
    hard_weight: int = Field(1, ge=1, le=10)
    train_backend: Optional[str] = Field("hf", description="auto|hf|llamafactory|mock")
    train_method: Optional[str] = Field("lora", description="lora|full")
    train_device: Optional[str] = Field("cpu", description="auto|cpu|cuda")
    eval_split: str = Field("valid")
    use_judge: Optional[str] = Field("auto", description="auto|true|false")
    baseline: Optional[str] = Field(None, description="可选对比基线模型")
    max_samples: Optional[int] = Field(None, ge=1)
    max_annotation_items: Optional[int] = Field(None, ge=1)


class BreadthAnalysisReq(BaseModel):
    model_id: str = Field(..., description="待分析模型")
    source: str = Field(..., description="带 category 的广度验证 jsonl 源文件")
    max_samples: Optional[int] = Field(None, ge=1, description="广度分析样本上限；会按类别轮询抽样")
    use_judge: Optional[str] = Field("auto", description="auto|true|false，是否调用判断模型")
    backend: Optional[str] = Field(None, description="推理后端")
    system_prompt: Optional[str] = Field(None, description="领域系统提示词")


class ImpactValidationReq(BaseModel):
    dataset_id: Optional[str] = Field(None, description="要验证的回灌后数据集；为空取最新回灌记录")
    before_model_id: Optional[str] = Field(None, description="回灌前模型；为空按 parent_dataset 自动找，找不到用 base")
    after_model_id: Optional[str] = Field(None, description="回灌后模型；为空按 dataset_id 自动找")
    max_items: int = Field(12, ge=1, le=80, description="最多生成多少条 impact 验证题")
    variants_per_prompt: int = Field(1, ge=1, le=3, description="每条回灌样本生成几个变体题")
    use_judge: Optional[str] = Field("auto", description="auto|true|false，是否调用判断模型")
    create_patch_task: bool = Field(True, description="未改善项是否自动生成补漏标注任务")
    patch_task_name: str = Field("impact-gap", description="补漏标注任务名前缀")


class BiologyAgentStartReq(BaseModel):
    iteration_prefix: str = Field("bio-v", description="迭代名前缀；实际轮次为 prefix + 1/2/3")
    source_name: str = Field("生物知识", description="标注源名称")
    max_iterations: int = Field(3, ge=1, le=3, description="本场景固定最多三轮")
    train_backend: Optional[str] = Field("hf", description="auto|hf|llamafactory|mock")
    train_device: Optional[str] = Field("cpu", description="auto|cpu|cuda")
    train_epochs: int = Field(1, ge=1, le=5)
    train_max_seq_len: int = Field(512, ge=128, le=2048)
    eval_max_samples: int = Field(11, ge=1, le=40)
    breadth_max_samples: int = Field(11, ge=1, le=40)
    use_judge: Optional[str] = Field("auto", description="auto|true|false")
    poll_interval: float = Field(5.0, ge=1.0, le=60.0)


class OneClickAgentStartReq(BaseModel):
    source_name: str = Field("生物知识", description="生物场景 source 名称")
    collection: str = Field("biology", description="知识库集合/命名空间")
    iteration_prefix: str = Field("bio-v", description="生物三轮迭代名前缀")
    run_biology_if_missing: bool = Field(True, description="缺少最终模型/数据集时自动触发生物三轮 Agent")
    force_biology_run: bool = Field(False, description="强制重新跑三轮标注回灌/训练/评测")
    run_inference_probe: bool = Field(True, description="保通过程中执行一次最终模型推理抽检")
    train_backend: Optional[str] = Field("hf", description="auto|hf|llamafactory|mock")
    train_device: Optional[str] = Field("cpu", description="auto|cpu|cuda")
    train_epochs: int = Field(1, ge=1, le=5)
    train_max_seq_len: int = Field(512, ge=128, le=2048)
    eval_max_samples: int = Field(11, ge=1, le=40)
    breadth_max_samples: int = Field(11, ge=1, le=40)
    use_judge: Optional[str] = Field("auto", description="auto|true|false")
    kb_chunk_size: int = Field(700, ge=50, le=2000)
    poll_interval: float = Field(5.0, ge=1.0, le=60.0)
    rag_probe: Optional[str] = Field(None, description="RAG 检索保通问题")
    infer_probe: Optional[str] = Field(None, description="最终模型推理抽检问题")
    infer_backend: Optional[str] = Field(None, description="auto|transformers|vllm|mock")
    infer_max_new_tokens: int = Field(80, ge=1, le=256)


class CustomAgentPlanReq(BaseModel):
    brief: str = Field(..., description="用户用自然语言描述希望强化的大模型方向")


class CustomAgentStartReq(BaseModel):
    direction: str = Field(..., description="希望强化的大模型方向，例如：法律合同审查、客服质检、化学实验安全")
    scenario_name: str = Field("通用场景", description="本次训练场景名，用于文件夹和报告标题")
    source_name: str = Field("通用知识", description="生成的 source 名称")
    iteration_prefix: str = Field("custom-v", description="迭代名前缀；实际轮次为 prefix + 1/2/3")
    dataset_name: str = Field("领域知识", description="训练数据集名前缀")
    model_name: str = Field("领域专家", description="训练后模型名前缀")
    base_model_path: Optional[str] = Field(None, description="基底模型本地目录或 HF 名称；为空使用 config 默认 Qwen0.5B")
    output_root: str = Field("data/custom_agents", description="产物输出目录；建议位于项目 data 下")
    sample_count: int = Field(20, ge=6, le=200, description="自动生成多少条训练样本")
    eval_sample_count: int = Field(8, ge=3, le=80, description="自动生成多少条 held-out 验证样本")
    max_iterations: int = Field(3, ge=1, le=8, description="迭代轮数")
    collection: str = Field("custom", description="知识库 collection")
    data_generation_mode: str = Field("auto", description="auto|judge|template；auto 优先 Judge 生成，失败模板兜底")
    train_backend: Optional[str] = Field("hf", description="auto|hf|llamafactory|mock")
    train_device: Optional[str] = Field("cpu", description="auto|cpu|cuda")
    train_epochs: int = Field(1, ge=1, le=5)
    train_max_seq_len: int = Field(512, ge=128, le=2048)
    eval_max_samples: int = Field(8, ge=1, le=80)
    breadth_max_samples: int = Field(8, ge=1, le=80)
    use_judge: Optional[str] = Field("auto", description="auto|true|false")
    run_inference_probe: bool = Field(True, description="每轮训练后是否做一次推理抽检")
    infer_probe: Optional[str] = Field(None, description="自定义推理抽检问题；为空自动取验证题")
    poll_interval: float = Field(5.0, ge=1.0, le=60.0)
    manual_initial_review: bool = Field(False, description="首轮回灌前暂停，允许用户人工修改初代标注内容")
    manual_iteration_review: bool = Field(False, description="后续每轮回灌前暂停，允许用户人工修改候选回灌内容")


class CustomAgentContinueReq(BaseModel):
    task_id: Optional[str] = Field(None, description="要继续的通用 Agent 任务；为空使用当前/最近任务")


class CustomAgentSelectReq(BaseModel):
    model_id: str = Field(..., min_length=1, description="切换为该模型所属的完整 Agent 项目上下文")


class CustomAgentExtendReq(BaseModel):
    task_id: Optional[str] = Field(None, description="要追加迭代的通用 Agent 任务；为空使用当前/最近任务")
    source_model_id: Optional[str] = Field(None, description="选择作为续训恢复点的已训练模型；自动继承其训练数据集和权重")
    extra_iterations: int = Field(1, ge=1, le=8, description="在已完成模型基础上继续迭代多少轮")
    manual_iteration_review: bool = Field(False, description="追加迭代时每轮回灌前是否暂停给人工审核")


class ChatCreateReq(BaseModel):
    """创建 LangChain Agent 对话会话。"""
    system_prompt: Optional[str] = Field("", description="自定义系统提示词；为空使用默认")
    model_id: str = Field("", description="关联的已训练模型ID（用于上下文标识）")
    memory_window: int = Field(10, ge=0, description="滑动窗口轮数（0=保留全部，N=保留最近N轮对话）")


class ChatReq(BaseModel):
    """发送对话消息（非流式或流式）。"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="用户消息")


class WorkflowReq(BaseModel):
    source: str = Field("data/samples.jsonl", description="jsonl 源文件（相对路径，位于项目内）")
    name: str = Field("workflow", description="数据集名前缀")
    model_name: str = Field("qwen0.5b-sft", description="模型名前缀")
    backend: Optional[str] = Field(None, description="auto|hf|llamafactory|mock")
    method: Optional[str] = Field(None, description="lora|full")
    epochs: Optional[int] = None
    device: Optional[str] = Field(None, description="auto|cpu|cuda")
    valid_ratio: float = Field(0.2, ge=0.0, le=0.9)
    seed: int = 42
    probes: Optional[list[str]] = Field(None, description="推理抽检问题列表")


class Task(BaseModel):
    id: str
    stage: Stage
    status: TaskStatus
    params: dict[str, Any]
    result: dict[str, Any] = {}
    error: str = ""
    progress: str = ""  # 实时进度提示，如 "3/15 steps"
    logs: list[str] = []
    created_at: float
    updated_at: float



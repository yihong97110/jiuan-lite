# jiuan-lite

极简版「训推平台」—— 对标久安「标-训-推-评」全链路，支持 **Qwen2.5-0.5B（CPU 可训）/ Qwen2.5-7B（GPU 训练）** 双基底模型，内置 LangChain/LangGraph Agent 与 BGE+FAISS+BM25 混合检索 RAG。

> 新机器部署、基底模型地址配置见 [`DEPLOYMENT.md`](DEPLOYMENT.md)。**GPU 服务器（AutoDL 等）从零到 7B 训推的完整操作见下方「7B 模型训练方案」章节。** 源码包不附带模型权重。

## 设计目标
- 打通 **数据 -> 训练(SFT/LoRA) -> 推理 -> 评测** 全链路；0.5B 可在 CPU 跑通，7B 在 GPU 服务器真实训练。
- 一个 FastAPI 控制面 + 独立 worker 子进程，模块边界清晰，全部基于开源件二次整合（LLaMA-Factory / vLLM / LangChain / FAISS / BM25）。
- **离线可跑**：未安装 torch/transformers 时自动进入 mock 模式，链路依然贯通。
- Agent 化：LangGraph ReAct Agent 挂载平台工具（RAG 检索/模型仓库/血缘/评测报告/任务状态/联网搜索），支持 Skill 注册与 MCP 协议扩展。

## 与久安模块的对应关系（初级版覆盖）
| 久安模块 | 本骨架实现 | 状态 |
|---|---|---|
| 模型标注系统（标） | `jiuan/pipeline/dataprep.py` 清洗/去重/格式化为 SFT 样本，切分 train/valid；标注回灌闭环（`annotate.py`） | 最小实现 |
| 模型训练系统（训） | `train.py` 多后端 SFT/LoRA（mock/hf/**LLaMA-Factory**），CPU/GPU 设备可选，**0.5B 与 7B 双基底** | 已整合开源件 |
| 模型推理系统（推） | `infer.py` 多后端（transformers/**vLLM**/mock）+ token 计量 | 已整合开源件 |
| 模型评测系统（评） | `evaluate.py` ROUGE-L/BLEU + **LLM-as-Judge**（4 维度+质量指标）+ 对比报告，仅在 valid 上评 | 已增强 |
| 智能体（Agent） | `langchain_agent.py` LangGraph ReAct + 工具调用 + MemorySaver 记忆 + SSE 流式；Skill 系统 + MCP 接入 | 已整合开源件 |
| RAG 知识增强 | `rag_backend.py` **BGE 嵌入 + FAISS 向量检索 + BM25 关键词检索 + RRF 混合融合** | 已整合开源件 |
| 任务管理/调度 | `workers/runner.py`(子进程调度) + `store.py` 任务全生命周期 | 最小实现 |
| 模型仓库 | `registry.py` 血缘登记(数据集->base->参数->产物->评测) | 最小实现 |
| 密钥管理 | `/config/api-keys` API + 前端「设置」页，Key 存 `.env` 不入库 | 已实现 |

## 一键工作流（推荐，避免重复踩坑）
所有搭建/运行步骤已固化到 `workflow.ps1` 与服务端 `POST /workflow`，坑与检查清单见 `RUNBOOK.md`。
```powershell
cd jiuan-lite
.\workflow.ps1 all       # 建环境->装依赖->下模型->起服务->跑全链路
# 日常：
.\workflow.ps1 serve     # 起服务(REAL+离线)
.\workflow.ps1 run       # 跑一键全链路(标->训->推->评)，打印结果
.\workflow.ps1 status    # 健康 + 最近任务
.\workflow.ps1 stop      # 停服务
```
Web 看板也有「一键工作流」按钮；服务端 `POST /workflow` 在单个可审计任务里串起四环节。

## 快速开始（0.5B，本机即可）
```powershell
cd jiuan-lite
pip install -r requirements.txt        # 最小依赖(仅控制面)；训练另见下
python scripts/quickstart.py           # 一键跑通全链路(mock 也可跑)
python -m jiuan.app                     # 启动控制面 http://127.0.0.1:8000
```

启用真实 Qwen0.5B 训练/推理（默认即 REAL，装好训练依赖即可）：
```powershell
pip install -r requirements-train.txt   # torch+transformers 可用后默认走真实模型
python scripts/quickstart.py
# 如需强制纯演示：$env:JIUAN_MOCK=1
```

## MOCK vs REAL（重要）
- **REAL（默认）**：装了 `requirements-train.txt`（torch+transformers 可用）就自动启用真实 SFT/LoRA 与推理，token 计量走 tokenizer 统计；UI 顶部显示 `REAL`。
- **MOCK（兜底/强制）**：未装 torch/transformers 时自动回退；也可显式设 `JIUAN_MOCK=1` 强制启用（无需依赖，纯演示链路），此时训练"记忆"训练集、推理命中记忆返回，评测 `reliable=false`，UI 顶部显示 `MOCK`。

## 新环境安装 + 0.5B 真实跑通（本机 CPU 已验证）
```powershell
# 1) 新建隔离环境
conda create -y -n jiuan python=3.10
conda activate jiuan

# 2) 控制面依赖
pip install -r requirements.txt

# 3) 训练依赖（CPU torch + transformers + LLaMA-Factory）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-train.txt

# 4) 下载 Qwen2.5-0.5B 到本地（走 hf-mirror，落到 data/registry/base/…）
$env:HF_ENDPOINT="https://hf-mirror.com"
python scripts/fetch_model.py
python scripts/fetch_small.py

# 5) 真实全链路（离线，自动用本地模型 + LLaMA-Factory LoRA）
$env:JIUAN_MOCK="0"; $env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
python scripts/quickstart.py
```

---

# 7B 模型训练方案（GPU 服务器完整操作）

在 Linux + NVIDIA GPU（显存 **≥ 24GB**，实测 RTX 3090 24GB 可跑 Qwen2.5-7B-Instruct LoRA bf16 训练与 vLLM 推理）上从零部署到训练、推理、评测的完整方案。以 AutoDL 为例（任意 Ubuntu GPU 机器同理）。

## 0) 硬件/环境要求
| 项 | 要求 | 说明 |
|---|---|---|
| GPU | ≥ 24GB 显存（RTX 3090 / A100 等） | 7B LoRA bf16 + gradient_checkpointing 训练约 20GB；vLLM 推理约占 22GB |
| 系统 | Linux（Windows 不支持 vLLM） | 训练可 Windows，推理上 vLLM 必须 Linux |
| 磁盘 | 系统盘余量 > 10GB，数据盘 > 50GB | 7B 权重约 15GB，落到数据盘 `/root/autodl-tmp` |
| Python | 3.10 + CUDA 12.x | AutoDL 自带 miniconda3 |

> **训练与 vLLM 推理不能同时跑**（vLLM 独占 ~22GB 显存）：先训练产出 LoRA 权重，停训练后再起 vLLM 挂载。

## 1) 服务器环境安装（AutoDL 实测）
```bash
# 克隆代码（或本地 scp 上传）
cd /root/autodl-tmp
git clone https://github.com/yihong97110/jiuan-lite.git
cd jiuan-lite

# conda 环境（AutoDL 非交互 SSH 需先 source；或全程用 /root/miniconda3/bin/python3 绝对路径）
source /root/miniconda3/etc/profile.d/conda.sh
conda create -y -n jiuan python=3.10
conda activate jiuan

# 依赖：控制面 + 训练（GPU torch 自动装 CUDA 版）
pip install -r requirements.txt
pip install -r requirements-train.txt
pip install -r requirements-rag.txt        # BGE 嵌入 + FAISS + BM25（混合检索）
pip install -r requirements-agent.txt      # langchain/langgraph（Agent）

# 国内走 hf-mirror 下载模型（关键，否则 HuggingFace 网络不可达）
export HF_ENDPOINT=https://hf-mirror.com
```

## 2) 下载 Qwen2.5-7B-Instruct 权重
```bash
# 用 huggingface-cli（走上面设置的镜像）下载到数据盘
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /root/autodl-tmp/models/qwen2.5-7b-instruct \
  --local-dir-use-symlinks False
```
`configs/qwen7b.yaml` 中 `model.local_dir` 已指向该目录，离线加载不重复下载。

## 3) 配置 API Key（.env，不入 git）
平台对话/评测裁判用的云端 API Key 一律放 `.env`（也可启动后在前端「⑫ 设置 · API Key」页填写，效果相同）：
```bash
cat > .env <<'EOF'
DEEPSEEK_API_KEY=sk-xxxx        # DeepSeek 官方（LLM-as-Judge 裁判）
ARK_API_KEY=ark-xxxx            # 火山方舟（DeepSeek 对话端点）
EOF
```
控制面启动时自动加载 `.env` 到环境变量，所有 yaml 只留 `${...}` 占位符，**仓库永不出现明文 Key**。

## 4) 切换 7B 配置并启动平台
```bash
# 关键：用 JIUAN_CONFIG 指定 7B 配置（worker 子进程会继承）
export JIUAN_CONFIG=/root/autodl-tmp/jiuan-lite/configs/qwen7b.yaml
export JIUAN_MOCK=0

python -m jiuan.app            # 控制面 http://0.0.0.0:8000
```
也可不设环境变量，直接改 `configs/qwen7b.yaml` 的 `model.local_dir` 后在 UI ② 训练页选基底模型（前端有 **0.5B / 7B 下拉**）。

7B 训练关键超参（`configs/qwen7b.yaml` 已预填默认值）：
| 参数 | 值 | 说明 |
|---|---|---|
| `precision` | `bf16` | 7B 训练/推理都用 bf16 |
| `gradient_checkpointing` | `true` | 7B 必开，省显存约 40% |
| `lora.r / alpha` | 8 / 16 | LoRA 秩与缩放 |
| `lora.target_modules` | q/k/v/o_proj | 注意力四投影 |
| `batch_size × grad_accum` | 2 × 8 | 有效批 16；小数据集(≤10条)改 1×1 保证每样本更新 |
| `lr / epochs` | 2e-4 / 3 | 默认值，前端可覆盖 |

## 5) 提交 7B 训练（UI 或 API）
```bash
# UI：② 训练页 -> 基底模型选 7B -> 后端默认 llamafactory -> 提交
# API：
curl -X POST http://127.0.0.1:8000/train -H "Content-Type: application/json" \
  -d '{"dataset_id":"<数据集id>","backend":"llamafactory","method":"lora","device":"cuda"}'

# 轮询进度（loss 逐步下降，progress 回传 N/total steps）
curl http://127.0.0.1:8000/tasks/<task_id>
```
产物 LoRA adapter 落 `data/models/<model_id>/weights/`，自动登记血缘。

## 6) vLLM 挂载 LoRA 高并发推理（Linux + GPU）
```bash
# 独立 vllm 环境（自带 GPU torch，勿与训练环境混装）
python -m venv .venv-vllm && source .venv-vllm/bin/activate
pip install -r requirements-vllm.txt

# 起 OpenAI 兼容服务并挂载 LoRA（--lora-modules 的 name 必须与平台 model_id 一致，否则 404）
vllm serve /root/autodl-tmp/models/qwen2.5-7b-instruct \
  --enable-lora --lora-modules jiuam-model=data/models/<model_id>/weights \
  --served-model-name jiuam-model --port 8001 --max-lora-rank 16

# 平台侧：configs/qwen7b.yaml 的 chat 段切到本地 vLLM
#   chat.base_url: http://127.0.0.1:8001/v1
#   chat.model: jiuam-model
# 或 /infer 提交时传 {"backend":"vllm"}
```
架构（推理与训练解耦）：
```
[jiuan 控制面:8000]  --HTTP-->  [vLLM OpenAI 服务:8001]  (Linux+GPU)
   业务/计量/血缘                 连续批处理/KV cache/高并发
```

## 7) 评测
```bash
curl -X POST http://127.0.0.1:8000/eval -H "Content-Type: application/json" \
  -d '{"model_id":"<7B微调模型id>","use_judge":true,"baseline":"base"}'
```
输出 9 项指标（3 基础 + 4 Judge 维度 + 2 质量）与 base vs 微调对比表（UI 绿升红降）。

### AutoDL 常见坑（实测记录）
- **非交互 SSH 找不到 python/conda**：先 `source /root/miniconda3/etc/profile.d/conda.sh` 再 `conda activate`，或直接用绝对路径 `/root/miniconda3/bin/python3`。
- **HuggingFace 网络不可达**：必须 `export HF_ENDPOINT=https://hf-mirror.com`。
- **vLLM 与训练抢显存**：vLLM 常驻 ~22GB；训练前先停 vLLM（`pkill -f vllm`）。
- **vLLM 404**：`--lora-modules` 的 name 与请求 `model` 字段不一致。
- **小数据集不更新**：≤10 条样本必须 `batch_size=1, grad_accum=1`，否则没有 optimizer step。

---

## 训练后端（可切换）
训练环节支持多后端，通过 `config.train.backend` 或提交 `/train` 时的 `backend` 字段选择（UI 默认 `llamafactory`）：

| backend | 说明 | 依赖 |
|---|---|---|
| `mock` | 离线兜底，记忆训练集供推理演示 | 无 |
| `hf` | 内置 transformers + peft 训练（7B 实测走通） | `requirements-train.txt` |
| `llamafactory` | 二次整合 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 额外装 llamafactory |
| `auto` | mock 环境→mock；否则优先 llamafactory(可用时)，回退 hf | — |

## 训练设备：CPU / GPU 可选
训练（hf 与 llamafactory 后端）支持显式选择设备与精度：

| 配置 | 取值 | 说明 |
|---|---|---|
| `train.device` | `auto` / `cpu` / `cuda` | `auto`：有 GPU 用 cuda，否则 cpu；`cuda` 无 GPU 时自动回退 CPU 并告警 |
| `train.precision` | `auto` / `fp32` / `fp16` / `bf16` | CPU 强制 fp32；GPU 上 auto 优先 bf16（支持时）否则 fp16 |

两种方式设置：全局默认改 `configs/*.yaml`；单次任务在提交 `/train`（或 UI ②训练页）时传 `device` 字段临时覆盖。`batch_size/grad_accum/lr` 前端提交值优先于配置文件。

用 LLaMA-Factory 训练：
```powershell
pip install -r requirements-train.txt
pip install "llamafactory[torch,metrics]"
$env:JIUAN_MOCK=0
curl -X POST http://127.0.0.1:8000/train -H "Content-Type: application/json" `
  -d '{"dataset_id":"<你的数据集id>","backend":"llamafactory","method":"lora"}'
```
实现要点（`jiuan/pipeline/lf_backend.py`）：
- 把清洗后的 messages 数据登记为 LLaMA-Factory 的 **sharegpt** 数据集（自动生成 `dataset_info.json`）；
- 生成 `train_config.yaml` 并以 `llamafactory-cli train` 子进程执行，实时回传 loss/关键日志；
- 产物（LoRA adapter 或全量权重）落到 `weights/`，推理阶段自动识别 adapter 并用 PEFT 加载「基座 + adapter」。

## 推理后端：vLLM（Linux + GPU）
推理支持多后端，通过 `config.infer.backend` 或提交 `/infer` 时的 `backend` 字段选择：

| backend | 说明 | 依赖/环境 |
|---|---|---|
| `mock` | 离线兜底 | 无 |
| `transformers` | 内置 transformers 本地生成（CPU/GPU 均实测通过） | `requirements-train.txt` |
| `vllm` | 调用 vLLM 的 OpenAI 兼容 HTTP 服务，高并发/连续批处理，usage 直接取服务返回 | **Linux + NVIDIA GPU** |
| `auto`（默认） | mock 环境→mock；否则 vLLM 端点可达则用 vllm，回退 transformers | — |

jiuan 侧只做轻量 HTTP 客户端（`jiuan/pipeline/vllm_backend.py`），上 Linux+GPU 后**无需改业务代码**，只要把 `config.chat.base_url` 指向真实 vLLM 服务即可。

---

# Agent 智能体（LangChain + LangGraph ReAct）

`jiuan/pipeline/langchain_agent.py` 基于开源件构建，为微调后模型提供多轮对话与工具调用能力：

## 架构映射
| 开源组件 | 用途 |
|---|---|
| `ChatOpenAI` | 统一 LLM 接口（对接 vLLM/DeepSeek/火山方舟任意 OpenAI 兼容端点） |
| `@tool` | 工具定义，封装现有 RAG/registry/store 能力，不重复造轮子 |
| `create_react_agent` | LangGraph 预置 ReAct 状态机（agent→tools→agent 循环） |
| `MemorySaver` | checkpointer 持久化记忆，thread_id(=session_id) 隔离，自动历史恢复 |
| `astream_events(v2)` | SSE 流式逐 token 输出 + 工具调用事件追踪 |

## 内置 6 个工具
| 工具 | 功能 |
|---|---|
| `rag_search` | 搜索 RAG 知识库（混合检索），注入领域知识 |
| `list_trained_models` | 列出平台已训练模型 |
| `get_model_lineage` | 查询模型血缘（数据集→基座→参数→产物→评测） |
| `web_search` | 联网搜索实时信息 |
| `get_eval_report` | 查询模型最新评测报告（ROUGE/BLEU/Judge/Gap） |
| `get_task_status` | 查询训练/推理/评测任务状态 |

## 扩展机制
- **Skill 系统**：`POST /skills/register` 注册自定义技能（名称+描述+调用方式），`POST /skills/invoke` 调用，Agent 可自动选择。
- **MCP 协议**：`POST /mcp/register` 注册 MCP 服务器，自动发现其工具并包装为 `@tool` 供 Agent 使用。
- **降级模式**：端点不支持 function calling（如火山方舟部分套餐）时自动降级为普通多轮对话（`_direct_chat`），不报错。

## 使用
```bash
# 会话 + 流式对话（SSE 逐 token）
curl -X POST http://127.0.0.1:8000/chat/session -H "Content-Type: application/json" -d '{}'
curl -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" \
  -d '{"session_id":"<id>","message":"帮我查一下最新训练的模型评测分数"}'
```
另有两类本地 Agent（不依赖 LangChain）：`local_agent.py` 内置工具链对话；`workers/agent.py` 提供「生物学 Agent / 一键 Agent / 自定义 Agent」三种自动化流程（`/agent/biology/start`、`/agent/oneclick/start`、`/agent/custom/plan|start|continue|extend`），可自动完成 标→训→推→评 数据迭代。

---

# RAG 知识库：混合检索（BGE + FAISS + BM25 + RRF）

`jiuan/pipeline/rag_backend.py` 从早期 TF-IDF 单路检索升级为 LangChain 标准组件 + 自定义 RRF 融合的**混合检索**，弥补 SFT"硬记"不足：

## 检索架构
```
问题 ──┬─> BGE 语义嵌入(bge-small-zh-v1.5, 384维) ─> FAISS 向量检索(IndexFlatIP, 归一化=余弦)
       └─> BM25 关键词检索(中文 bigram 分词)                    │
                                                    两路结果 ──> RRF 倒数排名融合
                                                    w_bm25=0.7 / w_vec=0.3
                                                    (平局时 α=0.4 加权分数 tiebreaker)
                                                        │
                                              Top-K 注入 prompt ─> 模型生成
```
| 组件 | 实现 | 说明 |
|---|---|---|
| 嵌入器 | `HuggingFaceEmbeddings(BAAI/bge-small-zh-v1.5)` | 384 维中文语义向量，`normalize_embeddings=True` |
| 向量库 | LangChain `FAISS` | IndexFlatIP 内积=归一化余弦；持久化 `data/vector_db/{collection}/` |
| 关键词 | `BM25Retriever` + 自定义中文 bigram preprocess | 领域 QA 术语精确命中强于泛化语义 |
| 融合 | 自定义 `HybridRRFRetriever(BaseRetriever)` | RRF 倒数排名融合，**BM25 权重 0.7 > 向量 0.3**（领域术语优先） |

> 兜底：未装 sentence-transformers 时自动回退 TF-IDF(纯离线)；`JIUAN_RAG_EMBED` 可强制切换。多知识库集合（`collection`）隔离，`GET /rag/collections` 列出。

## API
```
POST /rag/ingest          上传文档入库(分块->嵌入->FAISS+BM25 双索引)，source 限项目目录防路径遍历
POST /rag/query           混合检索 + 检索增强生成
GET  /rag/stats           知识库统计
GET  /rag/collections     集合列表
GET  /rag/gaps            知识盲区(gap)列表
POST /rag/gaps/resolve    盲区补录(闭环回灌)
```

---

# API Key 管理（前端可配，永不入库）

平台所有云端 API Key（DeepSeek 裁判 / 火山方舟对话 / OpenAI）统一走 **`.env` + 环境变量**，仓库 yaml 只留 `${...}` 占位符：

- **前端**：Web 看板「⑫ 设置 · API Key」页，输入即保存，**立即生效无需重启**，状态脱敏显示（仅前 4 + 后 4 位）。
- **后端**：`GET /config/api-keys` 查询状态（脱敏）；`POST /config/api-keys` 保存（写 `.env` 持久化 + 当前进程环境变量）。
- **读取优先级**（`_resolve_api_key`）：环境变量 `api_key_env` 指定名 > yaml 明文（历史兼容）> `EMPTY`。
- `.env` 在 `.gitignore` 中，**永不提交**；控制面启动时自动加载到 `os.environ`，worker 子进程继承。

---

## 评测：模型对比（base vs 微调后）
评测支持一次跑出「目标模型 vs 基线」的并列对比，量化训练带来的提升。

- 触发：提交 `/eval` 时传 `baseline`（如 `base` 或另一个 model_id），或 ④评测页选「对比基线」。
- 同一 valid 集分别评测两个模型，输出 9 项指标（ROUGE-L/BLEU 等 3 基础 + Judge 4 维度 + Bad Case 率/幻觉率 2 质量）的 `baseline`、`target`、`delta` 与提升百分比。
- 单模型评测(不传 baseline)行为不变，向后兼容。
- 返回结构含 `comparison`（对比表）+ `baseline_report_id`（基线独立报告）；UI 渲染对比表格，绿升红降。

## 评测：LLM-as-Judge（推荐）
ROUGE/BLEU 只比字面重合，无法衡量语义与事实正确性。评测支持接一个更强的"裁判模型"按 0-5 分打分（4 维度：正确性/完整性/相关性/表达 + 幻觉检测）。

- 开关：`config.eval.use_judge`=`auto|true|false`，或提交 `/eval` 时传 `use_judge`；UI ④评测页有下拉。
- 裁判端点：`config.judge.base_url`（OpenAI 兼容，可指向本地 vLLM、Ollama 或云端 API），`judge.model` 与服务端模型名对齐；裁判配置（平台/模型/Key）可前端填写同步后端。
- `auto`：裁判服务可达才启用，否则自动只出 ROUGE/BLEU，绝不因裁判不可用而失败。
- 报告含 `judge.avg_score`(0-5)、`judge.normalized`(0-1)、每条 `judge_score/judge_reason`，以及 Bad Case 率与幻觉率。

用本地 vLLM 同时做裁判（Linux+GPU）：
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name judge --port 8001
# config.judge: base_url=http://127.0.0.1:8001/v1, model=judge
```

## 数据驱动迭代闭环（对标久安多轮迭代）
从"调超参实验"转为"数据驱动迭代"：数据集可血缘继承，评测自动产出薄弱点，看板展示多轮对比。

- **数据集血缘（增量迭代）**：①标页选 `parent` 则在父集基础上合并去重追加。`registry.register_dataset` 记录 `parent_dataset`/样本数/新增数，`GET /datasets` 可查。
- **评测 gap 分析（闭环引擎）**：`evaluate.identify_gaps` 自动挑出低分样本（judge≤2=知识缺失；ROUGE<0.15=格式/覆盖薄弱），按严重度排序并给出扩增建议，写入报告 `gaps` 字段。
- **迭代看板（⑤页）**：`GET /iterations` 按血缘链分组，展示 v1→v2→… 的数据量/loss/ROUGE/BLEU/judge 与相对上一版提升。

迭代闭环：`v1 训练→评测发现薄弱点→针对性补数据(选 parent)→v2 训练→…`。
另有两类自动化分析：`POST /analysis/breadth`（知识广度分析，推荐补标方向）、`POST /analysis/impact/run`（数据增量影响验证）。

## 标注回灌闭环（轻量桥接，无需 Label Studio）
gap 分析 → 生成标注任务 jsonl → 业务人员编辑器里填 `annotation` 列 → 回灌为训练集（dataprep 增量继承）→ 触发训练。

- **模块**：`jiuan/pipeline/annotate.py`；标注文件落在 `data/annotations/<task>.jsonl`。
- **行格式**：`{"prompt": "...", "reference": "", "annotation": "", "status": "pending", "gap_type": "...", "iteration": "vN"}`；人工只需填 `annotation`（非空即视为已标注）。
- **API**：`POST /annotation/create`（从 gap 去重生成）、`GET /annotation/tasks`（进度）、`GET /annotation/tasks/{id}`（预览）、`POST /annotation/save`（在线保存）、`POST /annotation/commit`（回灌，可传 parent 增量）。⑦标注页可视化操作；支持 Label Studio 后端桥接（`/annotation/backend`、`/annotation/webhook/ls`）。
- **蒸馏扩样**：`POST /distill` 用大模型（DeepSeek 等）批量蒸馏生成训练样本，快速扩充数据集。

## 主要 API
- 一键/工作流：`POST /workflow`；`POST /agent/oneclick/start`（Agent 自动迭代）
- 标：`POST /dataprep` `GET /dataprep/sources` `POST /distill`（蒸馏）`POST /annotation/*`（标注回灌）
- 训：`POST /train`；推：`POST /infer` `POST /chat` `POST /chat/stream`（SSE）；评：`POST /eval`
- Agent：`POST /agent/start|status` `POST /chat/session` `POST /local_agent/*` `GET /agent/tools` `GET/POST /skills/*` `GET/POST /mcp/*`
- RAG：`POST /rag/ingest|query` `GET /rag/stats|collections|gaps` `POST /rag/gaps/resolve`
- 分析：`POST /analysis/breadth` `POST /analysis/impact/run`（+reports 查询）
- 配置：`GET/POST /config/api-keys`（Key 管理）
- 任务/仓库：`GET /tasks` `GET /tasks/{id}`（含 `progress` 实时进度）`GET /models` `GET /models/{id}/lineage` `GET /datasets` `GET /iterations` `GET /gaps`
- `GET /health` 返回是否 mock 模式

## 页面可调参数（每任务临时覆盖 config）
页面均可在提交时临时覆盖 `configs/*.yaml`（留空则用 config 默认，训练参数已预填默认值）：

| 页面 | 可调参数 |
|---|---|
| ② 训 | `method` `epochs` `device` `precision` `lr` `batch_size` `grad_accum` `max_seq_len` `lora_r` `lora_alpha` `lora_dropout` + **基底模型 0.5B/7B 下拉**（后端默认 llamafactory） |
| ③ 推 | `backend` `max_new_tokens` `do_sample` `temperature` `top_p` |
| ④ 评 | `split` `use_judge` `baseline` `max_samples` + Judge 配置（平台/模型/Key） |

> 后端统一经 `common.apply_train_overrides` / `apply_infer_overrides` 与 `evaluate` 的 `max_samples` 覆盖；`0`/`false` 等合法值不会被误判为空。

## 目录
```
jiuan/                 控制面 (FastAPI) + 任务编排
  app.py               REST API：标·训·推·评 + Agent/RAG/分析/Key管理
  store.py             SQLite 任务生命周期(WAL + 串行锁)
  registry.py          模型血缘登记与查询
  schemas.py           数据结构
  common.py            路径/配置/mock 判定/API Key 管理(.env)
  workers/runner.py    子进程调度器(并发受控，训练不阻塞 API)
  workers/job.py       单任务子进程入口(JIUAN_CONFIG 继承配置)
  workers/agent.py     生物学/一键/自定义 Agent 自动化流程
  pipeline/            dataprep / train / infer / evaluate 四环节
  pipeline/lf_backend.py        LLaMA-Factory 训练后端整合
  pipeline/vllm_backend.py      vLLM 推理后端(OpenAI 兼容客户端)
  pipeline/judge.py             LLM-as-Judge 评测裁判客户端
  pipeline/langchain_agent.py   LangGraph ReAct Agent(工具+记忆+流式)
  pipeline/local_agent.py       本地 Agent(免 LangChain 依赖)
  pipeline/rag_backend.py       RAG 混合检索(BGE+FAISS+BM25+RRF)
  pipeline/annotate.py          标注回灌闭环
  pipeline/distill.py           大模型蒸馏扩样
  pipeline/breadth.py           知识广度分析
  pipeline/impact.py            数据增量影响验证
scripts/serve_vllm.sh   Linux+GPU 上启动 vLLM 服务
scripts/quickstart.py   一键全链路演示
configs/qwen0.5b.yaml   0.5B 模型与超参(CPU 可跑)
configs/qwen7b.yaml     7B 模型与超参(GPU, ≥24GB 显存)
data/                   样例数据 + 产物(datasets/models/reports/registry/vector_db)
web/index.html          看板：12 个可切换页面(标/训/推/评/迭代/RAG/标注/Agent/设置...)
```

## 健壮性 / 性能优化（子进程·进度·批量推理）
- **LLaMA-Factory 子进程健壮性**：子进程注入 `PYTHONPATH`（含项目根 + 当前解释器 sys.path）；loss 解析兼容 `'loss': X`、`loss=X`、`train_loss = X` 多种格式；任务中断/异常时优雅终止子进程（terminate→超时 kill），避免孤儿进程。
- **实时进度**：`Task.progress` 字段。训练回传 `N/total steps`（hf 用 TrainerCallback，LLaMA-Factory 解析日志），评测回传 `infer/judge i/total`；看板任务表有"进度"列。
- **批量推理**：`infer.generate_batch()` 模型仅加载一次，避免评测时每条 `_load_real` 开销；对比评测 2 模型×N 条也只各加载一次。vLLM 后端天然并发。

## 调试记录 / 常见坑
- **AutoDL 非交互 SSH**：需 `source /root/miniconda3/etc/profile.d/conda.sh` 或绝对路径 `/root/miniconda3/bin/python3`。
- **HuggingFace 国内不可达**：`export HF_ENDPOINT=https://hf-mirror.com`。
- **PowerShell 发中文请求会乱码**：`ConvertTo-Json` + `Invoke-RestMethod` 默认非 UTF-8。正确做法：
  ```powershell
  $bytes = [Text.Encoding]::UTF8.GetBytes(($body | ConvertTo-Json))
  Invoke-RestMethod -Method Post $url -ContentType 'application/json; charset=utf-8' -Body $bytes
  ```
  或直接用 `scripts/_e2e_check.py`（Python urllib，无编码问题）。
- **推理默认 greedy**：`do_sample=false`。小模型采样输出极不稳定；采样参数仅在 `do_sample=true` 时传入。
- **小数据集超参**：`grad_accum=1`、`epochs=3` 起步，保证每步都更新优化器；数据量大时调大 `grad_accum`、减小 `epochs`。
- **火山方舟套餐 API 不支持 function calling**：Agent 自动降级普通对话（`_direct_chat`）。
- **GitHub Push Protection**：仓库 yaml 中禁止出现明文 Key（会推送失败 GH013），一律 `${ENV_NAME}` 占位 + `.env`。

## 已知限制 / 待办
- 调度为 PoC 级子进程串行；P1 建议换 Celery/RQ + Redis。
- 存储为 SQLite；高并发建议换 PostgreSQL。
- 尚无鉴权/限流；生产需补齐。
- 7B 全量微调未支持（仅 LoRA）；<24GB 显存卡需加量化（待办）。

## 路线图（分期）
- P0（本骨架）：单机、单卡、mock 兜底、全链路可视、train/valid 切分、血缘登记。
- P1：接入开源标注(Label Studio)、训练(LLaMA-Factory)、推理(vLLM)、评测(OpenCompass) 做二次整合；换 Celery/RQ 队列。✅ 大部分已完成
- P2：多机多卡调度、断点续训、计量计费、安全网关(鉴权/限流)、多端部署。

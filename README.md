# jiuan-lite

极简版“训推平台”骨架 —— 对标久安「标-训-推-评」全链路，用 Qwen2.5-0.5B 跑通最小闭环。

> 新机器部署、基底模型地址配置、CPU/GPU 真实训练和验收步骤见
> [`DEPLOYMENT.md`](DEPLOYMENT.md)。源码包不附带模型权重，按该文档下载或挂载模型后可完成真实 LoRA 训练。

## 设计目标
- 用最小模型（Qwen2.5-0.5B-Instruct）打通 **数据 -> 训练(SFT/LoRA) -> 推理 -> 评测** 全链路。
- 一个 FastAPI 控制面 + 独立 worker 子进程，模块边界清晰，方便后续替换成开源组件二次开发。
- **离线可跑**：未安装 torch/transformers 时自动进入 mock 模式，链路依然贯通，方便先看流程。

## 与久安模块的对应关系（初级版覆盖）
| 久安模块 | 本骨架实现 | 状态 |
|---|---|---|
| 模型标注系统（标） | `jiuan/pipeline/dataprep.py` 清洗/去重/格式化为 SFT 样本，切分 train/valid | 最小实现 |
| 模型训练系统（训） | `jiuan/pipeline/train.py` 多后端 SFT/LoRA（mock/hf/**LLaMA-Factory**），**CPU/GPU 设备可选** | 已整合开源件 |
| 模型推理系统（推） | `jiuan/pipeline/infer.py` 多后端（transformers/**vLLM**/mock）+ token 计量 | 已整合开源件 |
| 模型评测系统（评） | `jiuan/pipeline/evaluate.py` ROUGE-L/BLEU + **LLM-as-Judge** + 报告，仅在 valid 上评 | 已增强 |
| 任务管理/调度 | `jiuan/workers/runner.py`(子进程调度) + `store.py` 任务全生命周期 | 最小实现 |
| 计量计费 | 推理 token 计量已埋点(真实模式用 tokenizer 统计)，计费规则待接 | 埋点 |
| 模型仓库 | `jiuan/registry.py` 血缘登记(数据集->base->参数->产物->评测) | 最小实现 |
| 安全网关/权限 | 仅做了 source 路径遍历校验；鉴权/限流待接 | 部分 |

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
## 快速开始
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
- **REAL（默认）**：只要装了 `requirements-train.txt`（torch+transformers 可用）就自动启用真实 SFT/LoRA 与推理，token 计量走 tokenizer 统计；UI 顶部显示 `REAL`。
- **MOCK（兜底/强制）**：未装 torch/transformers 时自动回退；也可显式设 `JIUAN_MOCK=1` 强制启用（无需依赖，纯演示链路），此时训练"记忆"训练集、推理命中记忆返回，评测 `reliable=false`，UI 顶部显示 `MOCK`。

## 新环境安装 + 真实跑通（已在本机 CPU 验证）
```powershell
# 1) 新建隔离环境
conda create -y -n jiuan python=3.10
conda activate jiuan

# 2) 控制面依赖
pip install -r requirements.txt

# 3) 训练依赖（CPU torch + transformers + LLaMA-Factory）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-train.txt
#   完整锁定版本见 requirements-lock.txt（pip install -r requirements-lock.txt）

# 4) 下载 Qwen2.5-0.5B 到本地（走 hf-mirror，落到 data/registry/base/…）
$env:HF_ENDPOINT="https://hf-mirror.com"
python scripts/fetch_model.py      # 大文件
python scripts/fetch_small.py      # 小文件(config/tokenizer 等)

# 5) 真实全链路（离线，自动用本地模型 + LLaMA-Factory LoRA）
$env:JIUAN_MOCK="0"; $env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
python scripts/quickstart.py
```
实测结果：LLaMA-Factory 训出 LoRA adapter → 推理加载「基座+adapter」→ 在 valid 上评测得到真实指标（非 mock 的 1.0）。CPU 上 0.5B 一轮训练约数秒，单条推理约 10s。
## 训练后端（可切换）
训练环节支持多后端，通过 `config.train.backend` 或提交 `/train` 时的 `backend` 字段选择：

| backend | 说明 | 依赖 |
|---|---|---|
| `mock` | 离线兜底，记忆训练集供推理演示 | 无 |
| `hf` | 内置 transformers + peft 训练 | `requirements-train.txt` |
| `llamafactory` | 二次整合 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 额外装 llamafactory |
| `auto`（默认） | mock 环境→mock；否则优先 llamafactory(可用时)，回退 hf | — |

## 训练设备：CPU / GPU 可选
训练（hf 与 llamafactory 后端）支持显式选择设备与精度：

| 配置 | 取值 | 说明 |
|---|---|---|
| `train.device` | `auto` / `cpu` / `cuda` | `auto`：有 GPU 用 cuda，否则 cpu；`cuda` 无 GPU 时自动回退 CPU 并告警 |
| `train.precision` | `auto` / `fp32` / `fp16` / `bf16` | CPU 强制 fp32；GPU 上 auto 优先 bf16（支持时）否则 fp16 |

两种方式设置：
- 全局默认：改 `configs/qwen0.5b.yaml` 的 `train.device` / `train.precision`；
- 单次任务：提交 `/train`（或 UI ②训练页）时传 `device` 字段（临时覆盖 config）。

> 本机为 CPU-only，已实测 `device=cpu` 真实 SFT 跑通；`device=cuda` 代码/配置已就绪，待 Linux+GPU 环境可直接启用。

用 LLaMA-Factory 训练：
```powershell
pip install -r requirements-train.txt
pip install "llamafactory[torch,metrics]"   # 或从源码安装
$env:JIUAN_MOCK=0
# 提交训练任务时指定后端
curl -X POST http://127.0.0.1:8000/train -H "Content-Type: application/json" `
  -d '{"dataset_id":"<你的数据集id>","backend":"llamafactory","method":"lora"}'
```
实现要点（`jiuan/pipeline/lf_backend.py`）：
- 把清洗后的 messages 数据登记为 LLaMA-Factory 的 **sharegpt** 数据集（自动生成 `dataset_info.json`）；
- 生成 `train_config.yaml` 并以 `llamafactory-cli train` 子进程执行，实时回传 loss/关键日志；
- 产物（LoRA adapter 或全量权重）落到 `weights/`，推理阶段（`jiuan/pipeline/infer.py`）自动识别 adapter 并用 PEFT 加载「基座 + adapter」。
## 推理后端：vLLM（Linux + GPU）
推理支持多后端，通过 `config.infer.backend` 或提交 `/infer` 时的 `backend` 字段选择：

| backend | 说明 | 依赖/环境 |
|---|---|---|
| `mock` | 离线兜底 | 无 |
| `transformers` | 内置 transformers 本地生成（本机 CPU 实测通过） | `requirements-train.txt` |
| `vllm` | 调用 vLLM 的 OpenAI 兼容 HTTP 服务，高并发/连续批处理，usage 直接取服务返回 | **Linux + NVIDIA GPU** |
| `auto`（默认） | mock 环境→mock；否则 vLLM 端点可达则用 vllm，回退 transformers | — |

架构（推理与训练解耦）：
```
[jiuan 控制面:8000]  --HTTP-->  [vLLM OpenAI 服务:8001]  (Linux+GPU)
   业务/计量/血缘                 连续批处理/KV cache/高并发
```
jiuan 侧只做轻量 HTTP 客户端（`jiuan/pipeline/vllm_backend.py`），上 Linux+GPU 后**无需改业务代码**，只要把 `config.vllm.base_url` 指向真实 vLLM 服务即可。

在 Linux + GPU 上起 vLLM（端口 8001，与控制面 8000 分离）：
```bash
# 1) 独立环境装 vLLM（自带 GPU 版 torch，勿与训练环境混装）
python -m venv .venv-vllm && source .venv-vllm/bin/activate
pip install -r requirements-vllm.txt

# 2) 起 OpenAI 兼容服务（base 模型；LoRA 挂载见脚本注释）
./scripts/serve_vllm.sh
#   -> http://127.0.0.1:8001/v1

# 3) 让 jiuan 用 vLLM：config.infer.backend=auto/vllm，vllm.base_url 指向上面端点
#    或提交 /infer 时传 {"backend":"vllm"}
```
> 本机为 Windows/CPU，vLLM 无法本地真跑；已用 OpenAI 兼容 stub 验证客户端链路（backend=vllm/auto 均通、usage 采服务返回）。上 Linux+GPU 只需执行上面三步。

## 云端训练 + vLLM 推理（一句话流程）
Linux+GPU 机器上：① `configs` 把 `train.device=cuda`、`infer.backend=vllm`；② `.\workflow` 同义脚本或 API 跑 标→训（GPU LoRA/SFT）→ 产物落 `data/models/<id>/weights`；③ `serve_vllm.sh` 用 `--enable-lora --lora-modules <id>=<weights>` 挂载 adapter；④ `/infer` 走 vLLM 高并发推理，`/eval` 出真实指标。控制面/血缘/计量全程不变。

## 评测：模型对比（base vs 微调后）
评测支持一次跑出「目标模型 vs 基线」的并列对比，量化训练带来的提升。

- 触发：提交 `/eval` 时传 `baseline`（如 `base` 或另一个 model_id），或 ④评测页选「对比基线」。
- 同一 valid 集分别评测两个模型，输出每个指标(ROUGE-L/BLEU/judge_avg)的 `baseline`、`target`、`delta` 和提升百分比。
- 单模型评测(不传 baseline)行为不变，向后兼容。
- 返回结构含 `comparison`（对比表）+ `baseline_report_id`（基线独立报告）；UI 会渲染成对比表格，绿升红降。

## 评测：LLM-as-Judge（推荐）
ROUGE/BLEU 只比字面重合，无法衡量语义与事实正确性。评测支持接一个更强的“裁判模型”按 0-5 分打分（对标行业 LLM-as-Judge 做法）。

- 开关：`config.eval.use_judge`=`auto|true|false`，或提交 `/eval` 时传 `use_judge`；UI ④评测页有下拉。
- 裁判端点：`config.judge.base_url`（OpenAI 兼容，可指向本地 vLLM、Ollama 或云端 API），`judge.model` 与服务端模型名对齐。
- `auto`：裁判服务可达才启用，否则自动只出 ROUGE/BLEU，绝不因裁判不可用而失败。
- 报告新增 `judge.avg_score`(0-5)、`judge.normalized`(0-1)，明细含每条 `judge_score`/`judge_reason`。

用本地 vLLM 同时做裁判（Linux+GPU）：
```bash
# 起一个较强的裁判模型(如 Qwen2.5-7B-Instruct)
vllm serve Qwen/Qwen2.5-7B-Instruct --served-model-name judge --port 8001
# config.judge: base_url=http://127.0.0.1:8001/v1, model=judge
```

## 目录
```
jiuan/                 控制面 (FastAPI) + 任务编排
  app.py               REST API：标·训·推·评 + 模型仓库/血缘
  store.py             SQLite 任务生命周期(WAL + 串行锁)
  registry.py          模型血缘登记与查询
  schemas.py           数据结构
  common.py            路径/配置/mock 判定
  workers/runner.py    子进程调度器(并发受控，训练不阻塞 API)
  workers/job.py       单任务子进程入口
  pipeline/            dataprep / train / infer / evaluate 四环节
  pipeline/lf_backend.py  LLaMA-Factory 训练后端整合
  pipeline/vllm_backend.py vLLM 推理后端(OpenAI 兼容客户端)
  pipeline/judge.py       LLM-as-Judge 评测裁判客户端
scripts/serve_vllm.sh   Linux+GPU 上启动 vLLM 服务
configs/qwen0.5b.yaml  模型与超参
data/                  样例数据 + 产物(datasets/models/reports/registry)
web/index.html         看板：标/训/推/评 4 个可切换页面(带 MOCK/REAL 标识)
scripts/quickstart.py  一键全链路演示
```

## 主要 API
- `POST /workflow` 一键全链路(标->训->推->评，单任务可审计)
- `POST /dataprep` 数据准备(含 train/valid 切分，source 限项目目录内)
- `POST /train` 训练；`POST /infer` 推理；`POST /eval` 评测(默认 valid)
- `GET /tasks` `GET /tasks/{id}` 任务列表/详情
- `GET /models` `GET /models/{id}/lineage` 模型仓库与血缘
- `GET /datasets` 数据集血缘；`GET /iterations` 数据驱动迭代对比(v1→v2→…)
- 任务含 `progress` 字段（训练/评测实时进度），`GET /tasks/{id}` 可轮询
- `GET /gaps` 薄弱点历史（倒序+迭代标注）；`POST /rag/ingest` `POST /rag/query` `GET /rag/stats` 知识库
- `POST /annotation/create` `GET /annotation/tasks` `POST /annotation/commit` 标注回灌闭环
- `GET /health` 返回是否 mock 模式

## 数据驱动迭代闭环（对标久安多轮迭代）
从“调超参实验”转为“数据驱动迭代”：数据集可血缘继承，评测自动产出薄弱点，看板展示多轮对比。

- **数据集血缘（增量迭代）**：①标页选 `parent` 则在父集基础上合并去重追加。`registry.register_dataset` 记录 `parent_dataset`/样本数/新增数，`GET /datasets` 可查。
- **评测 gap 分析（闭环引擎）**：`evaluate.identify_gaps` 自动挑出低分样本（judge≤2=知识缺失；ROUGE<0.15=格式/覆盖薄弱），按严重度排序并给出扩增建议，写入报告 `gaps` 字段。
- **迭代看板（⑤页）**：`GET /iterations` 按血缘链分组，展示 v1→v2→… 的数据量/loss/ROUGE/BLEU/judge 与相对上一版提升。

迭代闭环：`v1 训练→评测发现薄弱点→针对性补数据(选 parent)→v2 训练→…`。

## 页面可调参数（每任务临时覆盖 config）
四个页面均可在提交时临时覆盖 `configs/qwen0.5b.yaml`（留空则用 config 默认）：

| 页面 | 可调参数 |
|---|---|
| ② 训 | `method` `epochs` `device` `precision` `lr` `batch_size` `grad_accum` `max_seq_len` `lora_r` `lora_alpha` `lora_dropout` |
| ③ 推 | `backend` `max_new_tokens` `do_sample` `temperature` `top_p` |
| ④ 评 | `split` `use_judge` `baseline` `max_samples` |

> 后端统一经 `common.apply_train_overrides` / `apply_infer_overrides` 与 `evaluate` 的 `max_samples` 覆盖；`0`/`false` 等合法值不会被误判为空。

## 标注回灌闭环（轻量桥接，无需 Label Studio）
gap 分析 → 生成标注任务 jsonl → 业务人员编辑器里填 `annotation` 列 → 回灌为训练集（dataprep 增量继承）→ 触发训练。

- **模块**：`jiuan/pipeline/annotate.py`；标注文件落在 `data/annotations/<task>.jsonl`。
- **行格式**：`{"prompt": "...", "reference": "", "annotation": "", "status": "pending", "gap_type": "...", "iteration": "vN"}`；人工只需填 `annotation`（非空即视为已标注）。
- **API**：`POST /annotation/create`（从 gap 去重生成）、`GET /annotation/tasks`（进度）、`GET /annotation/tasks/{id}`（预览）、`POST /annotation/commit`（回灌，可传 parent 增量）。⑦标注页可视化操作。
- **实跑验证**：11 个去重薄弱点→标注回灌为 v4(55条,继承 v3)→训练 loss 1.86(链上最低)、BLEU-1 0.409(最高)，闭环跑通。

## 薄弱点历史 & RAG 知识库
- **gap 历史全保留**：`GET /gaps` 汇总所有评测报告的薄弱点，从新到旧排序，每条标注所属迭代 v1/v2/…（以模型训练数据集的血缘版本为准）。⑤迭代页下方表格展示。
- **RAG 知识库**（`jiuan/pipeline/rag_backend.py`，⑥页）：弥补 SFT “硬记”不足，上传应急法规/预案文档→分块→嵌入→检索注入上下文→生成。
  - 默认嵌入器 **TF-IDF(scikit-learn)**，纯离线零下载，CPU 即用；向量库用 numpy 余弦相似度持久化到 `data/vector_db`。
  - 可插拔升级：装 `sentence-transformers`+本地 bge 后设 `JIUAN_RAG_EMBED=bge` 切语义检索；业务代码不变。依赖见 `requirements-rag.txt`。
  - API：`POST /rag/ingest`（入库，限项目目录防路径遍历）、`POST /rag/query`（检索增强生成）、`GET /rag/stats`。

## 健壮性 / 性能优化（子进程·进度·批量推理）
- **LLaMA-Factory 子进程健壮性**（`jiuan/pipeline/lf_backend.py`）：
  - 子进程注入 `PYTHONPATH`（含项目根 + 当前解释器 sys.path），退化为 `python -m llamafactory.cli` 时也能正确导入。
  - loss 解析兼容 `'loss': X`、`loss=X`、`train_loss = X` 多种格式（抵御版本升级输出变化）。
  - 任务中断/异常时优雅终止子进程（terminate→超时 kill），避免孤儿进程。
- **实时进度**：`Task.progress` 字段（store 平滑輁移加列）。训练回传 `N/total steps`（hf 用 TrainerCallback，LLaMA-Factory 解析日志），评测回传 `infer/judge i/total`；看板任务表新增“进度”列。
- **批量推理**：`infer.generate_batch()` 模型仅加载一次，避免评测时每条 `_load_real` 开销；`evaluate` 已改用批量（对比评测 2 模型×N 条也只各加载一次）。vLLM 后端天然并发，逐条走服务端。

## 调试记录 / 常见坑
- **PowerShell 发中文请求会乱码**：`ConvertTo-Json` + `Invoke-RestMethod` 默认非 UTF-8，中文 prompt 会变成 `?????`，模型收到乱码就答非所问（如“你好，请问…”）。正确做法是显式 UTF-8：
  ```powershell
  $bytes = [Text.Encoding]::UTF8.GetBytes(($body | ConvertTo-Json))
  Invoke-RestMethod -Method Post $url -ContentType 'application/json; charset=utf-8' -Body $bytes
  ```
  或直接用 `scripts/_e2e_check.py`（Python urllib，无编码问题）跑全链路自检。
- **推理默认改 greedy**：`infer.do_sample=false`。0.5B 小模型在 `do_sample=true`+temperature 下输出极不稳定，且会触发 `generation flags ignored` 警告。采样参数仅在 `do_sample=true` 时才传入。
- **小数据集超参**：`grad_accum=1`、`epochs=3`，保证每步都更新优化器（否则 5 条样本配 grad_accum=8 几乎不产生 optimizer step）。数据量大时应调大 `grad_accum`、减小 `epochs`。
## 已知限制 / 待办
- 调度为 PoC 级子进程串行；P1 建议换 Celery/RQ + Redis。
- 存储为 SQLite；高并发建议换 PostgreSQL。
- 评测指标为字符级 ROUGE-L/BLEU 近似；P3 建议接 OpenCompass 或 LLM-as-judge。
- 尚无鉴权/限流；生产需补齐。

## 路线图（分期）
- P0（本骨架）：单机、单卡、mock 兜底、全链路可视、train/valid 切分、血缘登记。
- P1：接入开源标注(Label Studio)、训练(LLaMA-Factory)、推理(vLLM)、评测(OpenCompass) 做二次整合；换 Celery/RQ 队列。
- P2：多机多卡调度、断点续训、计量计费、安全网关(鉴权/限流)、多端部署。







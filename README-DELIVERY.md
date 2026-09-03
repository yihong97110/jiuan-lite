# jiuan-lite 模型微调 + Agent 一体化平台 · 交付说明

版本：2026-08-25 ｜ 本包**不含基底模型权重**（首次训练前需联网下载，见 §4.3）

---

## 1. 这是什么

一站式「模型微调 + Agent」产品平台，覆盖 **标 → 训 → 推 → 评 → 迭代** 全链路：

| 产品线 | 内容 |
|---|---|
| **模型微调产品** | 数据准备(清洗/蒸馏) → LoRA/SFT 训练 → 推理 → 评测(ROUGE/BLEU/LLM-as-Judge) → 数据驱动迭代，全程血缘可追溯 |
| **Agent 产品** | LangChain ReAct Agent：RAG 知识库 / 联网搜索 / 模型血缘查询 / 训练任务操控等 10 个工具 + 技能/MCP 扩展 + **双层记忆** |
| **双层记忆** | 短期 = Checkpointer(thread_id 会话隔离，重启不丢)；长期 = Store(**按模型隔离**，不同模型记忆互不可见) |

## 2. 包内容清单

```
jiuan-lite/
├── jiuan/                  # 后端源码（FastAPI 控制面 + 全部流水线）
│   ├── app.py              # REST API 入口（python -m jiuan.app 启动）
│   ├── pipeline/           # 训/推/评/RAG/Agent 各流水线
│   ├── workers/            # 后台任务执行器
│   ├── memory_store.py     # 会话元数据+消息持久化
│   ├── registry.py         # 模型/数据集血缘登记
│   └── store.py            # 任务库(SQLite)
├── web/index.html          # 前端单页应用（13 个功能页，零构建，浏览器直开）
├── configs/
│   ├── qwen0.5b.yaml       # CPU/低配配置（0.5B 基座，开箱即用）
│   └── qwen7b.yaml         # GPU 配置（7B 基座，需 ≥24GB 显存）
├── scripts/
│   ├── fetch_model.py      # 基底模型下载（hf-mirror，国内直连）
│   ├── serve_vllm.sh       # vLLM 推理服务启动（GPU）
│   ├── gen_dataset.py      # 演示数据集生成
│   └── quickstart.py       # 一键冒烟自检
├── data/samples.jsonl      # 种子训练数据（28KB，可直接跑通全流程）
├── requirements*.txt       # 分层依赖（见 §3）
├── start.bat / start.sh    # 一键启动
└── .env.example            # API Key 模板
```

**不含**（运行时自动生成或按需下载）：基底模型权重(~1GB/0.5B、~15GB/7B)、训练产物、
数据集、运行时数据库。交付包体积 < 5MB。

## 3. 环境要求与安装

### 3.1 Python 依赖（分层安装，按需取用）

| 层 | 文件 | 说明 |
|---|---|---|
| 控制面（必装） | `requirements.txt` | FastAPI/uvicorn/pydantic，~10MB |
| Agent 对话（推荐） | `requirements-agent.txt` | langchain + langgraph + **双层记忆栈**（sqlite checkpoint/store） |
| RAG 增强（可选） | `requirements-rag.txt` | BGE 嵌入器；不装自动降级 TF-IDF（只需 scikit-learn） |
| GPU 训练（按需） | `requirements-train.txt` | transformers/peft；GPU 服务器上装 |
| vLLM 推理（按需） | `requirements-vllm.txt` | GPU 服务器上装 |

```bash
# 最小可用（控制面 + Agent 对话）
pip install -r requirements.txt -r requirements-agent.txt
```

实测版本（Python 3.11）：langchain 1.2 / langgraph 1.1 / fastapi 0.135 / scikit-learn 1.9。

### 3.2 API Key 配置（必配 1 个）

```bash
cp .env.example .env        # Windows: copy .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY（申请: platform.deepseek.com）
```

- **DEEPSEEK_API_KEY**：对话默认后端 + 评测裁判，**不配则对话/评测降级为离线 MOCK**
- TAVILY_API_KEY（可选）：联网搜索增强，不配自动用 360/Bing 免费兜底
- 也可启动后在前端「⑬ 设置 · API Key」页配置（写入 .env，长期持久化）

## 4. 快速开始

### 4.1 启动

```bash
# Windows
start.bat            # 默认 qwen0.5b 配置
start.bat qwen7b     # GPU 服务器上用 7B 配置

# Linux / macOS
chmod +x start.sh && ./start.sh
```

浏览器打开 **http://127.0.0.1:8000**（首页右上角徽标 REAL=真实模式 / MOCK=离线兜底）。

### 4.2 三分钟体验路径（无需 GPU、无需下载模型）

1. **⑩ 对话 · LangChain Agent** → 创建会话 → 输入「帮我搜索 2025 河南高考分数线」→ 看 Agent 调用联网搜索工具
2. 同会话输入「记住：我是河南理科 587 分」→ 再问「你还记得我什么」→ 验证**长期记忆**
3. ① 标 · 数据 → 用 data/samples.jsonl 提交清洗 → ② 训 · 训练（CPU mock 模式可跑通链路）
4. **⑫ 产品 · 交付** → 查看模型产品货架 / Agent 能力清单 / 在试用台直接体验

### 4.3 真实训练前：下载基底模型

```bash
# 0.5B 基座（~1GB，CPU 可训练，国内 hf-mirror 直连）
python scripts/fetch_model.py

# 7B 基座（GPU 服务器，改配置 local_dir 路径）
#   data/registry/base/qwen2.5-7b-instruct
#   用 huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir <路径>
#   国内加速: export HF_ENDPOINT=https://hf-mirror.com
```

下载后 `configs/qwen0.5b.yaml` 的 `model.local_dir` 指向该目录即可真实训练。

### 4.4 GPU 服务器部署（7B 真实训练 + vLLM 推理）

```bash
pip install -r requirements.txt -r requirements-agent.txt -r requirements-train.txt -r requirements-vllm.txt
./start.sh qwen7b
# 训练完成后启动 vLLM 推理服务（Agent 对话将自动路由到本地模型）
bash scripts/serve_vllm.sh
```

## 5. 核心功能页说明（前端 13 页）

| 页 | 功能 | 关键 API |
|---|---|---|
| ① 标·数据 | jsonl 清洗/去重/切分 + LLM 蒸馏生成数据集 | `POST /dataprep` `/distill` |
| ② 训·训练 | LoRA/SFT，产物自动登记血缘 | `POST /train` |
| ③ 推·推理 | transformers/vLLM 双后端 | `POST /infer` |
| ④ 评·评测 | ROUGE/BLEU + LLM-as-Judge + 广度分析 | `POST /eval` |
| ⑤ 迭代 | 数据集血缘 v1→v2… 指标对比 | `GET /iterations` |
| ⑥ 知识库·RAG | 文档入库/检索/缺口台账 | `POST /rag/ingest` `/rag/query` |
| ⑦ 标注·回灌 | 人工标注任务 + 影响验证回灌 | `POST /annotation/*` |
| ⑧⑨ 自动迭代 Agent | 全链路自动迭代 / 通用训练 Agent | `POST /agent/*` |
| ⑩ 对话·LangChain Agent | 双层记忆对话（SSE 流式） | `POST /chat` `/chat/stream` |
| ⑪ 本地 Agent | 自训练模型 + ReAct（零外部 API） | `POST /local_agent/chat` |
| ⑫ 产品·交付 | 模型/Agent 双产品看板 + 试用台 | 聚合各 API |
| ⑬ 设置 | API Key 管理 | `POST /config/api-keys` |

## 6. 双层记忆机制（重点）

```
短期记忆（会话内）
  = SqliteSaver Checkpointer，data/agent_threads.db
  + thread_id = session_id → 每会话独立状态线，平台重启自动恢复

长期记忆（跨会话）
  = SqliteStore，data/agent_store.db
  + namespace = ("memories", model_id) → 同一模型共享，不同模型完全隔离
  + prompt 每轮动态注入 → remember 工具存入后立即生效
```

验证方法：用模型 A 的会话说「记住：我分数 587」→ 换模型 B 的会话问「你知道我多少分」
→ B 不知道；回到模型 A 新会话 → A 知道。相关 API：

- `GET /chat/memory/{session_id}`：本会话记忆（含模型域 namespace）
- `GET /chat/memory`：全局各模型域统计
- `DELETE /chat/memory/{model_id}?confirm=true`：清空指定模型域

## 7. 常见问题（FAQ）

**Q: 对话回答是模板腔/右上角显示 MOCK？**
未配 DEEPSEEK_API_KEY 或网络不通。配好 .env 重启即可；MOCK 下所有链路可跑通但非真实模型输出。

**Q: 选了自训练模型对话报 502 降级？**
本地 vLLM 服务未启动（或隧道断开）。平台会自动降级云端 DeepSeek 继续对话（提示「降级模式」），
启动 vLLM 后自动恢复本地模型路由。

**Q: 训练按钮点了没反应/报数据集不存在？**
先在①标·数据生成数据集，②训·训练的 dataset_id 下拉框刷新后再选。

**Q: RAG 检索报嵌入器错误？**
未装 requirements-rag.txt 时自动用 TF-IDF（需 scikit-learn，agent 依赖已含）；
首次检索会重建索引稍慢。

**Q: 重启后会话和记忆还在吗？**
在。短期记忆(checkpointer)+会话元数据+长期记忆(store)均 SQLite 持久化，
重启后旧 session_id 直接继续对话，上下文与记忆域自动恢复。

**Q: 如何彻底重置平台数据？**
停止服务后删除 `data/*.db`（任务库/记忆库）与 `data/models`、`data/datasets` 目录。

## 8. 目录与数据落盘

```
data/
├── samples.jsonl        # 种子数据（包内自带）
├── jiuan.db             # 任务库（标/训/推/评任务状态）
├── agent_threads.db     # 短期记忆（checkpointer）
├── agent_store.db       # 长期记忆（store，按模型隔离）
├── registry/            # 模型登记 + 基底模型权重（下载后）
├── datasets/            # 生成的数据集
├── models/              # 训练产物（LoRA adapter）
└── vector_db/           # RAG 向量库
```

## 9. 已知限制

- 单机 SQLite 存储，适合 PoC/小团队；规模化需换 PostgreSQL
- 7B 真实训练需 ≥24GB 显存 GPU；CPU 只能跑 0.5B 或 mock 链路
- LLM-as-Judge 评测依赖 DeepSeek API（无 key 自动降级 ROUGE/BLEU）

## 10. 上传自有/第三方模型包

⑫产品·交付页提供「上传模型包」入口，支持两种 HuggingFace 主流格式：

| 格式 | zip 内必需文件 | 说明 |
|---|---|---|
| PEFT LoRA adapter | `adapter_config.json` + `adapter_model.safetensors` | 最常见，HuggingFace Hub 上多数 LoRA 即此格式；平台训练产物也是 |
| HF 全量模型 | `config.json` + `model*.safetensors`（可分片） | 完整权重 |

**流程**：选择本地 zip -> 填 model_id（可选）-> 点击上传 -> 自动校验结构、解压到 `data/models/<model_id>/weights/`、写 meta.json、登记血缘 -> 货架与试用台下拉框即时出现该模型。

**加载到 vLLM 后即可对话**（上传响应会给出命令）：
```bash
# LoRA
vllm serve <base_model> --lora-modules <model_id>=data/models/<model_id>/weights

# 全量
vllm serve data/models/<model_id>/weights
```
加载后 Agent 对话选该 model_id 即路由到本地 vLLM（长期记忆自动按 model_id 隔离）。

API 等价：`POST /models/upload`（multipart: file + model_id + name）。

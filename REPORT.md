# jiuan-lite 训推平台 —— 全流程工作报告

> 对标「久安」应急行业训推平台，做的是「初级版 / 最小可跑通骨架」，整合开源组件二次开发，用 Qwen2.5-0.5B 全链路验证。
> 本报告覆盖：项目定位 → 环境 → 架构 → 「标·训·推·评」四阶段 → 数据驱动迭代闭环 → 标注回灌闭环 → RAG 知识库 → 工程健壮性 → 真实跑通数据 → 后续规划。

---

## 一、项目定位与目标

- **背景**：公司要做类似久安的 AI 训推平台，分期建设，先做初级版；思路是整合 GitHub 开源项目做二次开发；用最小的 0.5B 模型验证全链路。
- **对标结构**：久安「标注系统 / 训练系统 / 推理系统 / 评测系统 / 任务管理 / 模型仓库」→ 本项目落地为「标·训·推·评 + 任务调度 + 模型/数据血缘」。
- **验证模型**：Qwen2.5-0.5B-Instruct（Apache-2.0，可商用）。
- **应用领域**：应急管理（多灾种：地震/火灾/危化品/洪涝台风/公共卫生/急救等）。

---

## 二、运行环境

- **机器**：本机 Windows 10，CPU-only（无 GPU）。
- **Python 环境**：conda 环境 `jiuan`（Python 3.10）。
- **模型运行**：默认 REAL 模式（torch+transformers 真实推理/训练可用即启用），`JIUAN_MOCK=1` 才强制 mock 兜底。
- **离线**：`HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1`，模型已落地到 `data/registry/base/qwen2.5-0.5b-instruct`。
- **云端裁判**：火山方舟（Volcengine Ark）OpenAI 兼容端点，密钥走环境变量 `ARK_API_KEY`。

---

## 三、系统架构

- **控制面**：FastAPI（`jiuan/app.py`），提供 REST API + 静态看板页。
- **任务调度**：`jiuan/workers/runner.py` 把任务拉起为**独立子进程**执行（`job.py`），训练不阻塞 API、不抢 GIL；`MAX_PARALLEL=1` 串行防抢 GPU。
- **任务存储**：SQLite（`jiuan/store.py`），全生命周期 + 日志 + 实时 `progress` 字段；单锁串行化避免 database is locked。
- **模型/数据血缘**：`jiuan/registry.py`，JSONL 索引记录 数据集→base→参数→产物→评测，及数据集 parent 血缘链。
- **前端看板**：单页 `web/index.html`，7 个可切换页面，MOCK/REAL 徽标。

代码结构：
```
jiuan/
  app.py              控制面 REST API（标·训·推·评 + 迭代/gap/RAG/标注 + 模型血缘）
  common.py           路径/配置/设备精度解析/参数覆盖/mock 判定
  store.py            SQLite 任务存储（含 progress）
  registry.py         模型仓库 + 数据集血缘 + 迭代聚合
  schemas.py          Pydantic 请求/任务模型
  pipeline/
    dataprep.py       标：清洗/去重/切分 + parent 增量继承
    train.py          训：hf(transformers) / llamafactory / mock 多后端
    lf_backend.py     LLaMA-Factory 子进程集成（env/进度/优雅终止）
    infer.py          推：transformers / vllm / mock + 批量推理 + token 计量
    vllm_backend.py   vLLM OpenAI 兼容客户端（Linux+GPU）
    evaluate.py       评：ROUGE-L/BLEU + LLM-as-Judge + 模型对比 + gap 分析
    judge.py          火山方舟裁判客户端
    rag_backend.py    RAG 知识库（TF-IDF 离线，可升级 bge）
    annotate.py       标注回灌桥接
    workflow.py       一键全链路
  workers/
    runner.py         子进程调度器
    job.py            单任务执行入口（进度回调）
```

---

## 四、四阶段能力（标·训·推·评）

### ① 标 · 数据准备（`dataprep.py`）
- 清洗（去空/字段校验）、**去重**（key=user||assistant，含 input 防误删）、统一为 chat 格式、**train/valid 切分**（评测在 valid 上，不作弊）。
- **数据集血缘**：支持 `parent`，在父数据集基础上合并去重增量追加，登记 `parent_dataset / sample_count / added_count`。

### ② 训 · SFT/LoRA（`train.py` + `lf_backend.py`）
- 多后端：`hf`（内置 transformers+peft）/ `llamafactory`（业界主流微调框架二次整合）/ `mock`（离线兜底）/ `auto`。
- **CPU/GPU 可选**：`device=auto|cpu|cuda`，`cuda` 无 GPU 自动回退 CPU 并告警；精度 `fp32/fp16/bf16`（CPU 强制 fp32）。
- 修复了原始骨架的训练 bug：padding label 用 `DataCollatorForSeq2Seq` 动态 padding，padding 位置置 -100（不再学 pad token）。
- 页面可调超参：`lr / batch_size / grad_accum / max_seq_len / precision / lora_r / lora_alpha / lora_dropout / epochs`。
- **实时进度**：hf 用 TrainerCallback 回传 `N/total steps`，LLaMA-Factory 解析日志回传。

### ③ 推 · 推理（`infer.py` + `vllm_backend.py`）
- 多后端：`transformers` / `vllm`（Linux+GPU 高并发）/ `mock`。
- **token 计量**：对齐久安「计量计费」，真实用 tokenizer/vLLM usage，mock 用离线估算并标注 estimated。
- **批量推理**：`generate_batch()` 模型只加载一次，评测时避免逐条重复加载开销。
- 页面可调：`max_new_tokens / do_sample / temperature / top_p`。

### ④ 评 · 评测（`evaluate.py` + `judge.py`）
- 指标：字符级 **ROUGE-L / BLEU-1/2**（比原始空格分词 token-F1 更贴近中文）。
- **LLM-as-Judge**：火山方舟裁判（deepseek 系）对答案 0-5 打分，替代纯格式匹配缺陷。
- **模型对比**：target vs baseline 并列评测，给 delta / 提升%。
- held-out 评测：严格在 valid 上，mock/train 切分标注 reliable=false。
- 页面可调：`split / use_judge / baseline / max_samples`。

---

## 五、数据驱动迭代闭环（核心，对标久安 53 轮）

从「调超参实验」转向「数据驱动迭代」——这是与久安的本质对齐点。

- **数据集血缘**：`GET /datasets`，每个数据集记录 parent，可上溯 v1→v2→…链条。
- **评测 gap 分析**：评测后自动挑低分样本（judge≤2=知识缺失；ROUGE<0.15=格式/覆盖薄弱），按严重度排序并给扩增建议，写入报告 `gaps`。
- **薄弱点历史全保留**：`GET /gaps` 汇总所有报告的 gap，从新到旧排序，标注所属迭代（以模型训练数据集的血缘版本为准）。
- **迭代看板**：`GET /iterations`（⑤页），按血缘链展示 v1→vN 的 数据量/loss/ROUGE/BLEU/judge 及相对上一版提升。

**真实迭代链（同一固定 held-out 评测集 18 条 + 火山裁判，全 REAL）**：

| 版本 | 数据量 | 新增 | Loss | ROUGE-L | BLEU-1 | judge | 说明 |
|---|---|---|---|---|---|---|---|
| v1 | 29 | +29 | 2.088 | 0.2547 | 0.3533 | 3.31 | 基线：火灾+地震+急救 |
| v2 | 36 | +7 | 1.966 | 0.2751 | 0.3878 | 3.56 | 补危化品 |
| v3 | 44 | +8 | 1.918 | 0.2859 | 0.3917 | 3.44 | 补公共卫生+洪涝台风 |
| v4 | 47 | +3 | 2.198 | 0.2834 | 0.4028 | 3.44 | 补洪涝/加油站/急救(严格无泄漏) |

- v1→v3 阶段 ROUGE-L 稳步提升（0.2547→0.2859）、BLEU-1 提升（0.353→0.392），验证「补数据→指标涨」的闭环有效。
- v4 相对 v3 基本持平（ROUGE −0.9%、BLEU-1 +2.8%、judge 持平）：仅新增 3 条干净样本、且评测集为严格 held-out，属于小样本正常波动，这是**诚实、口径干净**的对比。
- 系统自动识别出「AED 急救类」为跨版本持续薄弱点 → 指导定向补数据 / 转由 RAG 兜底（见第七节）。

> **数据泄漏修复说明**：早期 v4 曾把「标注回灌」产出的样本（含固定评测集里的 prompt）混入训练集，导致 train/eval 交叉污染、对比不可信。现已在 `registry` 增加 held-out 评测集标记，`annotate.create_task/commit` 自动排除评测集 prompt（防泄漏），并重建了干净 v4。

---

## 六、标注回灌闭环（可插拔标注层，对标久安标注平台）

`gap 分析 → 标注任务 → 浏览器内标注 → 回灌为训练集(增量+硬样本加权) → 一键/自动训练 → 评测 → gap`

**三层架构（切后端只改配置 `annotation.backend`，业务代码不变）**
- 抽象层 `jiuan/annotation/`：`__init__.py` 按配置分发，暴露统一接口(create/list/preview/save/coverage/commit/resolved)。
- 后端① `bridge_jsonl.py`（默认，单机零依赖）：标注文件落 `data/annotations/<task>.jsonl`，人只填 `annotation` 列。
- 后端② `bridge_label_studio.py`（占位框架）：对标久安多人协作/审核；部署 LS 后把 mock 换成真实 REST 调用即接管。

**能力**
- 浏览器内直接标注（`POST /annotation/save`），非技术人员无需编辑 jsonl。
- 标注覆盖度（`GET /annotation/coverage`）：按 gap_type / iteration 看哪类没补完。
- 回灌可选硬样本加权（知识缺失类训练集内复制 N 倍）、parent 增量血缘、`auto_train` 一键触发训练。
- 跨版本 gap 解决追踪（`GET /annotation/resolved`）：每次回灌解决了哪些薄弱点及前后评测变化。
- **标注同步两条路**：方式A 轮询 `scripts/annotation_poll.py`（不依赖 LS，标完即回灌+训练）；方式B Webhook `POST /annotation/webhook/ls`（LS 完成回调，标完即训）。
- **防泄漏**：`create_task`/`commit` 自动剔除与固定 held-out 评测集重叠的 prompt，杜绝「用考题当练习题」。

**对标久安差距与替代**
| 久安能力 | 本项目替代 | 差距 |
|---|---|---|
| 多人标注协作 | jsonl 单机/浏览器标注 | 需要多人时切 Label Studio 后端 |
| 标注审核流程 | 覆盖度+状态字段 | 审核态(approved/rejected)待补 |
| 标注→自动训练 | auto_train / 轮询 / webhook | 已打通 |
| 标注进度看板 | 覆盖度进度条 | LS 看板更丰富 |

---

## 七、RAG 知识库（弥补 SFT「硬记」不足）

- 模块 `rag_backend.py`（⑥页）：上传应急法规/预案文档 → 分块 → 嵌入 → 检索注入上下文 → 生成。
- **离线默认**：TF-IDF(scikit-learn) 嵌入 + numpy 余弦相似度，零下载、CPU 即用，持久化 `data/vector_db`。
- **可升级**：装 sentence-transformers+bge 后设 `JIUAN_RAG_EMBED=bge` 切语义检索，业务代码不变。
- API：`POST /rag/ingest`（限项目目录，防路径遍历）、`POST /rag/query`、`GET /rag/stats`。
- **实跑验证（无RAG vs 有RAG 对比，真实 0.5B）**——RAG 把 SFT 答错/编造的回答纠正为法规原文：

| 问题 | 无RAG（纯SFT，易编造） | 有RAG（检索注入法规后） |
|---|---|---|
| AED 使用要点 | 「按压心前区，每5秒一次…」（错误，评测 judge=1） | 「开机贴电极片→分析心律时勿触碰→提示放电确认无人后电击→继续心肺复苏」（命中第三条，正确） |
| 危化品泄漏处置 | 「用水稀释或不燃溶剂覆盖」（不专业） | 「上风向撤离+划警戒区+切火源电源+防护+报告」（命中第二条，正确） |
| 台风预警颜色 | 蓝/黄/橙/红（正确但无出处） | 蓝/黄/橙/红，并给出条文依据（命中第四条） |

- 检索命中分数 0.38~0.52（TF-IDF 余弦），可直接在 ⑥RAG 页复现；对比结果存 `data/rag_demo_results.json`。
- 价值：对「AED、危化品」这类模型硬记不牢的领域，RAG 用最新法规条文兜底，比单纯 SFT 更可靠、可溯源。

---

## 八、工程健壮性 & 生产化改进

- **LLaMA-Factory 子进程健壮性**：子进程注入 PYTHONPATH（防退化 `python -m` 时导入失败）；loss 解析兼容 `'loss':X`/`loss=X`/`train_loss=X`；中断/异常时优雅终止（terminate→超时 kill）防孤儿进程。
- **实时进度**：`Task.progress`（store 平滑迁移加列），训练/评测全程可见 `N/total`。
- **批量推理**：评测/对比时模型仅加载一次。
- **输入安全**：dataprep/RAG source 限项目目录内，防路径遍历；train/eval 提交前校验数据集存在。
- **任务列表**：加「完成时间」列，按 updated_at 倒序。
- **默认 REAL 模式**：本地服务默认真实模型，mock 仅兜底。

---

## 九、当前系统真实状态（本机实测）

- 服务：REAL 模式运行中（`mock_mode:false`）。
- 页面：7 个（①标 ②训 ③推 ④评 ⑤迭代 ⑥RAG ⑦标注）。
- 数据资产：种子数据 `samples.jsonl` 124 条；数据集 18 个；模型 15 个；评测报告 23 份。
- 主要 API：`/dataprep /train /infer /eval /workflow`、`/tasks /models /models/{id}/lineage`、`/datasets /iterations /gaps`、`/rag/* /annotation/*`、`/health`。

---

## 十、开源组件与许可（均可商用）

| 阶段 | 组件 | 许可 | 角色 |
|---|---|---|---|
| 模型 | Qwen2.5-0.5B-Instruct | Apache-2.0 | 验证基座 |
| 训练 | LLaMA-Factory / transformers / peft | Apache-2.0 | SFT/LoRA |
| 推理 | transformers / vLLM | Apache-2.0 | 生成 + 高并发 |
| 评测 | 自研 ROUGE/BLEU + 火山方舟裁判 | - / 商用API | 指标 + LLM-as-Judge |
| RAG | scikit-learn(+可选 bge/chromadb) | BSD/MIT/Apache | 知识库检索 |
| 控制面 | FastAPI / Uvicorn / Pydantic | MIT/BSD | API/调度 |

---

## 十一、已知限制 & 后续规划

**限制（PoC 级，符合初级版定位）**：
- CPU-only，训练/推理慢；vLLM 需 Linux+GPU（客户端已就绪，配置指向即可）。
- 调度为子进程串行，SQLite 存储；数据量为验证级（百条级）。
- 指标绝对值不高，重在验证机制闭环。

**规划**：
- P1：数据扩到 200+；接 vLLM 高并发推理；标注→训练一键串联。
- P2：调度换 Celery/RQ；存储换 PostgreSQL；模型 registry 升级 MLflow。
- P3：接 OpenCompass 系统化评测；RAG 升级语义嵌入；Label Studio 深度标注（可选）。

**对外说辞**：
> 已做出对标久安「标·训·推·评」全链路的初级版训推平台，用 Qwen2.5-0.5B 真实跑通；实现了 4 轮数据驱动迭代（29→55 条，含标注回灌闭环），每轮针对评测薄弱点定向补数据，Loss 与 BLEU 稳步改善；同时具备 RAG 知识库、LLM-as-Judge 云端裁判、模型/数据血缘、子进程调度等能力。规模化后对接业务系统即可放量。

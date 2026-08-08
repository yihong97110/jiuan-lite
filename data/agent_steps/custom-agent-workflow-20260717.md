# 通用训练 Agent 工作流 20260717

- 状态：已实现并完成 smoke 验证
- 前端入口：`http://127.0.0.1:8000/?v=custom9` -> `⑨ 通用训练 Agent`
- 后端入口：`POST /agent/custom/plan`、`POST /agent/custom/start`、`GET /agent/custom/status`
- 定位：保留 `⑧ Agent` 中的生物专用流程，同时新增一个可面向任意方向的大模型训练流程。

## 1. 解决的问题

原来的“生物专家 · 三轮自动训练”和“一键保通打包”可以跑通生物学场景，但数据源、知识库、验证集和推理抽检都强绑定在生物方向，无法普遍用于合同审查、客服质检、实验安全、医学问答、运维手册等其他场景。

本次新增 `⑨ 通用训练 Agent`，目标是让用户只用自然语言描述“希望强化的大模型方向”，再由 Agent 自动补全参数、生成 demo 训练数据、构造 held-out 验证集、生成知识库、分轮回灌、训练、评测和广度分析。

## 2. 用户工作流

1. 用户在 9 号页输入强化方向，例如：
   - “我想训练一个能做法律合同审查的大模型，重点识别风险条款、付款条件、违约责任和修改建议。”
2. 点击“让 Agent 生成方案”。
3. Agent 自动回填：
   - 场景名
   - source 名字
   - 迭代名前缀
   - 数据集名前缀
   - 模型名前缀
   - 基底模型路径
   - 输出目录
   - 样本数、验证样本数、迭代轮数
   - Judge / 训练后端 / 设备 / epochs / max_seq_len
4. 用户可修改参数。
5. 点击“启动通用训练”。
6. 页面实时显示：
   - 当前阶段，如“第 1 轮：标注回灌”“第 2 轮：训练模型”“第 3 轮：广度分析”
   - 每轮数据集、模型、评测指标、gap 数量、广度薄弱类别
   - source / held-out / KB / 报告路径
   - 全量步骤与日志，默认折叠，支持展开。

## 3. 后端标准流程

1. `plan` 阶段：
   - 从自然语言 brief 中推断场景名。
   - 生成推荐参数和小白可读的参数注意事项。
   - 优先调用 Judge/LLM 生成结构化训练方案，要求返回训练目标、数据生成策略、验证与查缺补漏、迭代安排、风险与参数建议。
   - Judge 不可用或返回无法解析时，使用本地规则兜底：根据场景名和关键词生成五段式建议，并在前端标记为“规则兜底”。
   - 前端通过 `plan_sections` 分块展示方案来源和依据，避免只输出“生成数据、训练、评测”的空泛模板。
   - 默认基底模型为 `data/registry/base/qwen2.5-0.5b-instruct`。

2. `material` 阶段：
   - 生成训练 source：`data/custom_agents/<场景>-<时间>/<source>.jsonl`
   - 生成 held-out 验证源：`*_heldout.jsonl`
   - 生成知识库 Markdown：`*_kb.md`
   - 数据生成模式：
     - `auto`：优先调用已配置 Judge/LLM 生成结构化 SFT 样本，失败则模板兜底。
     - `judge`：强制 Judge 生成，失败即失败。
     - `template`：直接用模板生成，适合离线 smoke。

3. `held-out` 阶段：
   - 使用 dataprep 注册验证集。
   - 标记为 `role=eval`，防止训练数据泄漏进固定验证集。

4. `RAG` 阶段：
   - 把配套知识库写入用户指定 collection。
   - 产物可在 `⑥ 知识库 · RAG` 中切换查看。

5. `iteration` 阶段：
   - 按迭代轮数拆分训练样本。
   - 每轮自动创建预标注任务：`data/annotations/<task>.jsonl`
   - 标注答案预填；默认 Agent 自动 commit 回灌为训练数据集。
   - 若启动时勾选“首轮人工审核”或“后续回灌人工审核”，Agent 会在对应候选回灌任务生成后暂停为 `waiting_manual`，提示用户到 ⑦ 标注页加载任务、修改 annotation 并保存。
   - 用户保存后回到 ⑨ 点击“继续运行”，Agent 会读取保存后的 annotation，再继续 commit、训练、评测和广度分析。
   - 已完成任务可在 ⑨ 当前进度底部点击“我想再迭代 n 次”，根据最近的广度薄弱项生成下一批候选回灌样本；若勾选后续人工审核，则追加轮次同样先暂停。
   - 调用训练任务，默认 LoRA + CPU + hf。
   - 可选推理抽检。
   - 对固定 held-out 做评测。
   - 对源 held-out 做广度分析。
   - 保存步骤报告：`data/agent_steps/custom-*.md/json`

## 4. 参数注意事项

- 强化方向：决定训练样本、验证题和知识库围绕什么能力展开；越具体，生成数据越贴近目标。
- 基底模型：大模型基础能力更强、迁移更好，但显存、训练时间和部署成本更高；小模型便宜、快，适合本机 demo，但专业上限较低。
- 数据集大小：默认 20 条用于快速跑通；正式训练应扩大到数百或数千条，并覆盖正例、反例、边界情况和真实业务表达。
- 迭代轮数：默认 3 轮，每轮都会回灌新增样本并评测；轮数越多越能查缺补漏，但耗时更长。
- LoRA 与 full：当前默认 LoRA，训练快、占用低、适合小样本迭代；full 微调改动更彻底，但资源需求明显更高。
- Judge：Judge 更适合评估专业性和完整性；ROUGE 只看字面重合。Judge 不可用时流程自动降级到 ROUGE/BLEU。

## 5. 已实现文件

- `jiuan/workers/custom_agent.py`：通用训练 Agent 主流程。
- `jiuan/schemas.py`：新增 `CustomAgentPlanReq`、`CustomAgentStartReq`。
- `jiuan/app.py`：新增 `/agent/custom/plan`、`/agent/custom/start`、`/agent/custom/status`。
- `web/index.html`：新增 `⑨ 通用训练 Agent` 页面。
- `jiuan/common.py`、`jiuan/pipeline/train.py`：支持自定义基底模型路径。
- `jiuan/pipeline/infer.py`：mock 训练产物自动走 mock 推理，避免错误加载真实基座。

## 6. 验证记录

基础验证：

```powershell
python -m compileall jiuan
Invoke-RestMethod http://127.0.0.1:8000/health
```

结果：
- 编译通过。
- `/health` 返回 `status=ok`，`mock_mode=false`。
- 浏览器打开 `http://127.0.0.1:8000/?v=custom9`，9 号页、方案按钮、启动按钮正常渲染，控制台无错误。

smoke 场景：
- 强化方向：法律合同审查，识别风险条款、付款条件、违约责任和修改建议。
- 数据生成：template。
- 训练后端：mock。
- 样本数：6。
- 验证样本：3。
- 迭代轮数：1。

smoke 结果：
- 任务：`9a6261099902`
- source：`data/custom_agents/合同审查-smoke-20260717-164006/合同审查知识-smoke.jsonl`
- held-out：`data/custom_agents/合同审查-smoke-20260717-164006/合同审查知识-smoke_heldout.jsonl`
- 知识库：`custom-smoke`，入库 8 块。
- 数据集：`contract-v1-合同审查知识-20260717-164008`
- 模型：`contract-v1-合同审查专家-20260717-164012`
- 评测：ROUGE-L `0.1389`，BLEU-1 `0.2361`，BLEU-2 `0.0955`，gap `2`
- 广度报告：`breadth-contract-v1-_-20260717-164012-20260717-164134`
- 步骤报告：`data/agent_steps/custom-20260717-164006.md`

说明：smoke 使用 `mock` 和极小样本，只验证流程完整性，不代表真实模型效果。正式演示可使用默认 20 条、3 轮、`hf` 后端和 Qwen2.5-0.5B 本地基座。

## 7. 遇到的问题与修复

1. 生物流程不具备普遍性。
   - 处理：保留生物专用流程，新增 `custom_agent` 和 9 号通用页面。

2. 训练原来只能使用配置文件中的基底模型。
   - 处理：新增 `base_model_path` 参数，可从页面传入本地目录或 HF 名称。

3. mock 训练产物后续 auto 推理会错误加载真实基座。
   - 处理：`infer.py` 检测到模型目录存在 `memory.json` 时自动使用 mock 后端。

4. 自然语言 brief 提取场景名不够自然。
   - 处理：增加规则，从“能做法律合同审查的大模型”等表达中提取“法律合同审查”。

5. PowerShell 中文输出可能乱码。
   - 处理：文件均以 UTF-8 保存；浏览器显示正常；命令行中文不作为判断依据。

## 8. 下一步建议

1. 用 9 号页跑一次正式 demo：
   - 方向：合同审查、客服质检或应急预案问答。
   - 样本数：20。
   - 轮数：3。
   - 后端：hf。
   - 设备：CPU demo 可跑，GPU 更适合正式演示。
2. 启用 Judge 生成训练样本，提升数据质量。
3. 把 impact 回灌效果验证接入通用 Agent 的每轮结束阶段。
4. 增加“上传真实业务资料 -> 自动抽取训练样本”的入口，让 demo 从自动生成走向企业真实数据闭环。

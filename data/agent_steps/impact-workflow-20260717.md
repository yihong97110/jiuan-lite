# 回灌效果验证与查缺补漏工作流 20260717

- 状态：已实现并完成 smoke 验证
- 目标：让每次标注回灌不仅能看到固定评测集指标，还能验证“这批回灌样本是否真的带来迁移改善”，并把未改善项自动转成下一轮补漏标注任务。
- 前端入口：`http://127.0.0.1:8000/?v=impact2` -> `⑦ 标注` -> `回灌效果验证 · 查缺补漏`
- 后端入口：`POST /analysis/impact/run`、`GET /analysis/impact/reports`

## 1. 为什么要加这一层

旧的“跨版本 gap 解决追踪”只检查回灌样本是否出现在固定评测集中。由于防泄漏机制正常工作，训练扩增样本不会进入 held-out 固定评测集，所以很多行会显示“未纳入固定评测集”，也就没有逐题 judge/ROUGE 前后变化。

这不是 bug，但它回答不了一个关键问题：这次回灌是否让模型在同主题、不同问法、真实迁移场景下变好了。因此新增 impact held-out 验证集：

1. 从本次回灌 source 中读取训练扩增样本。
2. 不使用训练原题，按同主题生成不同问法的验证题。
3. 用回灌前模型和回灌后模型分别评测。
4. 对比 judge/ROUGE delta、逐题 before/after prediction。
5. 对未改善或仍弱的项目自动生成 `impact-gap-*` 补漏标注任务，继续进入“标注 -> 回灌 -> 训练 -> 评测”闭环。

## 2. 当前实现文件

- `jiuan/pipeline/impact.py`：impact 验证主流程，生成变体验证集、跑前后模型评测、写报告、生成补漏任务。
- `jiuan/app.py`：新增 `/analysis/impact/run` 和 `/analysis/impact/reports`。
- `jiuan/schemas.py`：新增 `ImpactValidationReq`。
- `jiuan/pipeline/evaluate.py`：支持 `attach_eval=False`，避免临时 impact 评测覆盖主迭代指标。
- `jiuan/registry.py`：过滤 impact eval report，不让它污染 `⑤ 迭代` 看板。
- `web/index.html`：`⑦ 标注` 新增“回灌效果验证 · 查缺补漏”面板。

## 3. 标准执行步骤

1. 确认服务可用：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

期望：`status=ok`，真实训练推理场景下 `mock_mode=false`。

2. 选择要验证的回灌后数据集。生物 v3 示例：

```text
bio-v3-生物学知识-20260716-165449
```

如果 `dataset_id` 留空，后端会从 `data/annotations/resolved_log.jsonl` 选择最新回灌记录。

3. 发起 impact 验证：

```powershell
$body = @{
  dataset_id = 'bio-v3-生物学知识-20260716-165449'
  max_items = 12
  variants_per_prompt = 1
  use_judge = 'auto'
  create_patch_task = $true
  patch_task_name = 'impact-gap'
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri 'http://127.0.0.1:8000/analysis/impact/run' `
  -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body $body
```

4. 查看报告：

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/analysis/impact/reports?limit=5'
```

5. 如果报告里 `weak_count > 0` 或 `patch_task.task_id` 非空，到 `⑦ 标注` 页面查看 `impact-gap-*` 任务，确认补漏标注后继续 commit、train、eval。

## 4. 产物位置

- impact 验证源：`data/impact_eval/impact-*.jsonl`
- impact eval 数据集：`data/datasets/impact-*/valid.jsonl`
- 前后模型原始评测报告：`data/reports/eval-*.json`
- impact 汇总报告：`data/reports/impact-*.json`
- 自动补漏标注任务：`data/annotations/impact-gap-*.jsonl`
- 回灌来源追踪：`data/annotations/resolved_log.jsonl`
- 回灌 source：`data/annotations/<task_id>-committed.jsonl`

## 5. 已验证样例

- 验证数据集：`bio-v3-生物学知识-20260716-165449`
- 回灌前模型：`bio-v2-生物专家-20260716-164601`
- 回灌后模型：`bio-v3-生物专家-20260716-165453`
- 报告：`data/reports/impact-bio-v3-生物学知识-20260716-165449-20260717-072242.json`
- 样本数：2
- Judge：本次 smoke 关闭，所以 judge delta 为 `—`
- ROUGE-L：回灌前 `0.2177`，回灌后 `0.3946`，增量 `+0.1769`
- 改善数：`2/2`
- 待补漏：`0`

这个 smoke 只证明链路跑通。正式判断回灌效果时建议把 `max_items` 提到 `12`、`36` 或全量，并启用 Judge。

## 6. 遇到的问题与修复

1. impact 评测一开始只有汇总指标，没有逐题 pairs。
   - 原因：`evaluate.run()` 返回 summary，但逐题 details 写在 report JSON 文件里。
   - 修复：`impact.py` 增加 `_with_report_details(summary)`，按 `report_path` 读取 details 后再组装 before/after pairs。

2. impact 临时评测会污染主迭代看板。
   - 原因：评测默认 `registry.attach_eval()`，会把临时 impact eval 也挂到模型血缘上，可能覆盖 `⑤ 迭代` 的主评测指标。
   - 修复：`evaluate.run()` 支持 `attach_eval=False`；impact 调用评测时传入该参数；`registry.iterations()` 过滤 impact eval report。

3. 固定评测集中没有回灌样本，表格前后 judge/ROUGE 显示 `—`。
   - 原因：防泄漏机制正常工作，训练样本不应该原样进入 held-out 固定评测集。
   - 处理：保留固定评测看整体泛化，同时用 impact held-out 专门看本次回灌效果。

4. PowerShell 控制台显示中文可能乱码。
   - 原因：终端 code page 和 UTF-8 文件内容不一致。
   - 处理：文件用 UTF-8 保存，浏览器显示正常；接口请求必须使用 `application/json; charset=utf-8`。

5. 浏览器可能缓存旧前端。
   - 处理：使用 `?v=impact2` 或刷新页面，确认 `⑦ 标注` 出现“回灌效果验证 · 查缺补漏”面板。

## 7. 思路构建

- 固定 held-out：用于回答“模型整体能力有没有变好”，不能放训练回灌原题。
- impact held-out：用于回答“这次回灌的主题是否产生迁移改善”，也不能直接测训练原题，而是生成同主题变体题。
- Judge：更适合判断专业性、机制完整性、误区识别；ROUGE 只反映字面重合，适合作为轻量趋势参考。
- 查缺补漏：不是看到指标低就手写下一轮样本，而是把未改善项转成 `impact-gap-*` 标注任务，让 agent 继续自动回灌。
- 防污染：impact eval 是诊断性评测，不应改写主模型迭代指标。

## 8. 下一轮推荐策略

1. 对 `bio-v3-生物学知识-20260716-165449` 跑正式 impact：
   - `max_items=36`
   - `variants_per_prompt=1`
   - `use_judge=true` 或 `auto`
   - `create_patch_task=true`

2. 如果 `weak_count > 0`：
   - 检查 `data/annotations/impact-gap-*.jsonl`
   - 用 `POST /annotation/commit` 回灌，parent 设为当前最新训练数据集
   - 训练下一版模型，例如 `bio-v4-生物专家-<时间>`
   - 再跑固定评测、广度验证、impact 验证

3. 如果 `weak_count=0` 但固定评测或广度仍弱：
   - impact 说明本次回灌主题有效
   - 下一轮应扩大 held-out 广度，由 Judge 或规则从薄弱类别生成新验证题，再生成对应标注扩增样本

4. 后续增强：
   - 将当前模板式 impact 变体升级为 Judge/LLM 生成，要求覆盖真实应用、反例、误区、实验判断。
   - 增加“变体题是否泄漏原题”的自动检查。
   - 将 impact 正式结果写入每轮 agent 自动步骤摘要。

## 9. 验收清单

- [ ] `python -m compileall jiuan` 通过
- [ ] `/health` 返回 ok
- [ ] `⑦ 标注` 能看到 impact 面板
- [ ] `POST /analysis/impact/run` 能生成 `data/reports/impact-*.json`
- [ ] impact report 中 `pairs` 非空
- [ ] `GET /iterations` 主评测指标没有被 impact eval 覆盖
- [ ] `weak_count > 0` 时自动生成 `impact-gap-*` 标注任务

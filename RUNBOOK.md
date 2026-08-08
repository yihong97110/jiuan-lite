# RUNBOOK — jiuan-lite 训推平台运行手册

把「怎么跑通、踩过哪些坑、怎么自检」固化下来，避免重复犯错与遗忘。

## 0. TL;DR（最快路径）
```powershell
cd E:\all-ex\jiuan-lite
.\workflow.ps1 all       # 建环境 -> 装依赖 -> 下模型 -> 起服务 -> 跑全链路
# 之后日常：
.\workflow.ps1 serve     # 起服务
.\workflow.ps1 run       # 跑一键全链路（标->训->推->评）
.\workflow.ps1 status    # 看健康 + 最近任务
.\workflow.ps1 stop      # 停服务
```
浏览器：`http://127.0.0.1:8000/`（顶部徽标 REAL/MOCK；标/训/推/评 4 个可切换页面，各带表单与该阶段任务列表）。

## 1. 一键工作流做了什么
`workflow.ps1` 把下列步骤固化，每步都带正确的 UTF-8 与离线环境变量：
| 子命令 | 动作 |
|---|---|
| `setup` | 建 conda 环境 `jiuan`(py3.10) + 装 `requirements.txt` + CPU torch + `requirements-train.txt`(含 LLaMA-Factory) |
| `model` | 从 hf-mirror 下载 Qwen2.5-0.5B 到 `data/registry/base/…` |
| `serve` | 后台起控制面（REAL + 离线），写 PID 到 `.server_pid` |
| `run` | 调 `POST /workflow` 跑 标->训->推->评，打印结果与步骤日志 |
| `stop` / `status` / `all` | 停服务 / 看状态 / 一条龙 |

服务端 `POST /workflow`（`jiuan/pipeline/workflow.py`）在单个可审计任务里串起四环节，产物逐步回填，任一步失败即整体失败并保留已完成步骤。

## 2. 必须记住的坑（血泪）
1. **中文请求体会乱码 → 模型答非所问**
   PowerShell 的 `ConvertTo-Json | Invoke-RestMethod` 默认非 UTF-8，中文 prompt 变 `?????`，模型收到乱码就回“你好，请问…”。
   - 正确：发 UTF-8 字节体，`ContentType 'application/json; charset=utf-8'`（`workflow.ps1` 里的 `Invoke-Api` 已封装）。
   - 或直接用 `scripts/_e2e_check.py`（Python urllib，无此问题）。
   - 排查：`GET /tasks/{id}` 看 `params.prompt` 是否已乱码。

2. **推理默认 greedy（`infer.do_sample=false`）**
   0.5B 小模型在 `do_sample=true`+temperature 下输出极不稳定，还会报 `generation flags ignored`。采样参数仅在 `do_sample=true` 时才传。

3. **小数据集超参**：`grad_accum=1`、`epochs=3`。
   否则 5 条样本配 `grad_accum=8` 几乎不产生 optimizer step（loss 不降）。数据量大时反过来调大 `grad_accum`、减小 `epochs`。

4. **包名不能叫 `platform`**：会遮蔽标准库，本项目包名为 `jiuan`。

5. **LLaMA-Factory 会锁 transformers/peft/datasets 版本**：这是刻意兼容，别手动升级；完整版本见 `requirements-lock.txt`。

6. **模型下载走 hf-mirror 的 308 重定向**：`huggingface_hub` 新版会拒绝，故用 `scripts/fetch_model.py` / `fetch_small.py` 直接抓文件。`special_tokens_map.json` 在 Qwen2.5 上 404 属正常（特殊 token 内联在 tokenizer_config）。

7. **CPU/GPU 设备选择**：`train.device`=`auto|cpu|cuda`，`train.precision`=`auto|fp32|fp16|bf16`（`jiuan/common.py` 的 `resolve_device/resolve_precision`）。
   - CPU 强制 fp32；`cuda` 但无 GPU 会自动回退 CPU 并在日志打 `请求 cuda 但本机无 GPU，已回退 CPU`。
   - 提交 `/train` 时传 `device` 可临时覆盖 config；UI ②训练页有下拉框。
   - 本机 CPU-only，GPU 路径无法本地实测，但代码/配置已就绪。

8. **vLLM 推理后端（Linux + GPU）**：`infer.backend`=`auto|transformers|vllm|mock`，vLLM 端点配 `config.vllm.base_url`（默认 `http://127.0.0.1:8001/v1`）。
   - vLLM 仅支持 Linux+GPU；Windows/CPU 本机跑不了，只能用 OpenAI 兼容 stub 验证客户端逻辑。
   - Linux 上启动：独立 venv 装 `requirements-vllm.txt` → `./scripts/serve_vllm.sh`（端口 8001，与控制面 8000 分离）。
   - LoRA 产物用 `vllm serve --enable-lora --lora-modules <id>=<weights>` 挂载（脚本尾部有示例）。
   - `serve_vllm.sh` 必须是 LF 换行（Windows 编辑易变 CRLF 导致 bash 报错）。
   - jiuan 侧是纯 HTTP 客户端，切 vLLM 无需改业务代码，只改 config。

## 3. 自检清单（每次改动后过一遍）
- [ ] `conda run -n jiuan python -c "import compileall,sys;sys.exit(0 if compileall.compile_dir('jiuan',quiet=1) else 1)"` 返回 0
- [ ] `.\workflow.ps1 serve` 后 `GET /health` 返回 `mock_mode:false`
- [ ] `.\workflow.ps1 run` 整个 workflow `succeeded`，`reliable=true`
- [ ] `GET /models/{model_id}/lineage` 能看到训练 + 评测血缘
- [ ] 训练日志里 loss 有下降趋势（不是只有 1 个 step）
- [ ] 用中文 prompt 推理，`params.prompt` 未乱码，答案切题

## 4. 模式切换
- **REAL（默认）**：torch+transformers 可用即自动启用（无需再设 JIUAN_MOCK）。建议配本地模型 +（可选）`HF_HUB_OFFLINE=1` 走离线。`workflow.ps1 serve` 已带离线设置。
- **MOCK**：依赖缺失时自动回退；或显式 `JIUAN_MOCK=1` 强制，无需模型/torch，链路照跑但指标 `reliable=false` 仅演示。

## 4.5 标注层（可插拔后端，对标久安标注平台）

**架构**：`jiuan/annotation/`（抽象层分发）→ `bridge_jsonl.py`(默认) / `bridge_label_studio.py`(占位)。
切后端只改 `configs/qwen0.5b.yaml` 的 `annotation.backend`，业务代码不动。

```yaml
annotation:
  backend: jsonl            # 默认单机零依赖
  # backend: label_studio   # 部署 LS 后启用多人协作/审核
  # ls_base_url: http://127.0.0.1:8080
  # ls_api_key: <token>
```

**标注→训练两条同步路**
- 方式A 轮询（不依赖 LS）：`python scripts/annotation_poll.py --once`（或 `--interval 300` 常驻）。
  扫到"全部标注完成且未回灌过"的任务→自动回灌(+训练)，状态存 `data/annotations/.poll_state.json`(幂等)。
  可选 `--parent <ds> --hard-weight 3 --no-train`。
- 方式B Webhook（LS 配合）：LS 项目 Webhook 指向 `POST /annotation/webhook/ls`，标完即回灌+训练。

**关键接口**：`/annotation/create|tasks|save|coverage|commit|resolved|backend|webhook/ls`。
- `save`：浏览器内直接标注(回写 jsonl，非技术人员无需编辑文件)。
- `commit` 支持 `hard_weight`(硬样本训练集内复制) + `auto_train`(回灌即训) + `parent`(增量血缘)。
- `coverage`：按 gap_type/iteration 看覆盖度；`resolved`：跨版本 gap 解决追踪。

**坑**：`data/annotations/` 下 `*-committed.jsonl`(回灌 source)、`resolved_log.jsonl`(追踪日志)、`.poll_state.json` 不是标注任务，已在 `list_tasks/coverage` 中排除；勿手动当任务处理。

## 4.6 回灌效果验证与查缺补漏

固定 held-out 评测用于看模型整体能力，不能把训练回灌原题放进去；因此跨版本追踪里回灌样本经常显示“未纳入固定评测集”，这是防泄漏的正常结果。要判断“这次回灌是否真的有效”，走 impact held-out 验证：

1. 从 `data/annotations/resolved_log.jsonl` 找到回灌记录。
2. 读取 `data/annotations/<task_id>-committed.jsonl` 中的训练扩增样本。
3. 生成同主题、不同问法的 impact 验证题，不直接复用训练原题。
4. 对回灌前模型和回灌后模型跑同一验证集，比较 judge/ROUGE delta。
5. 未改善项自动生成 `data/annotations/impact-gap-*.jsonl`，继续进入标注回灌闭环。

前端入口：`http://127.0.0.1:8000/?v=impact2` → `⑦ 标注` → `回灌效果验证 · 查缺补漏`。

API 示例：

```powershell
$body = @{
  dataset_id = 'bio-v3-生物学知识-20260716-165449'
  max_items = 12
  variants_per_prompt = 1
  use_judge = 'auto'
  create_patch_task = $true
  patch_task_name = 'impact-gap'
} | ConvertTo-Json

Invoke-RestMethod -Uri 'http://127.0.0.1:8000/analysis/impact/run' -Method Post -ContentType 'application/json; charset=utf-8' -Body $body
Invoke-RestMethod 'http://127.0.0.1:8000/analysis/impact/reports?limit=5'
```

关键产物：
- `data/impact_eval/impact-*.jsonl`：impact 验证源
- `data/datasets/impact-*/valid.jsonl`：临时 held-out 验证集
- `data/reports/impact-*.json`：前后模型效果对比报告
- `data/annotations/impact-gap-*.jsonl`：自动生成的补漏标注任务
- `data/agent_steps/impact-workflow-20260717.md`：本次实现、问题、复现与下一轮思路记录

注意：impact eval 是诊断性评测，已通过 `attach_eval=false` 和 registry 过滤避免污染 `⑤ 迭代` 主看板指标。

## 4.7 通用训练 Agent（任意方向）

`⑧ Agent` 中的生物专家流程是固定场景 demo；`⑨ 通用训练 Agent` 用于任意方向的训练冷启动。用户输入“希望强化的大模型方向”后，Agent 会自动生成训练 source、held-out 验证集、配套知识库，并按迭代轮数完成预标注回灌、训练、推理抽检、评测和广度分析。

前端入口：`http://127.0.0.1:8000/?v=custom9` → `⑨ 通用训练 Agent`。

API：

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/agent/custom/plan' -Method Post -ContentType 'application/json; charset=utf-8' -Body $body
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/agent/custom/start' -Method Post -ContentType 'application/json; charset=utf-8' -Body $body
Invoke-RestMethod 'http://127.0.0.1:8000/agent/custom/status'
```

方案建议生成：
- `/agent/custom/plan` 先从用户 brief 中提取场景名和关键词，生成 source、数据集、模型、知识库 collection 等默认参数。
- 随后优先调用已配置的 Judge/LLM，要求返回结构化 JSON：训练目标、数据生成策略、held-out/Judge/ROUGE/广度验证策略、迭代安排、风险与推荐参数。
- 前端会显示 `plan_source`：`Judge 生成` 表示已调用外部 Judge 并解析成功；`规则兜底` 表示 Judge 不可用或 JSON 解析失败，系统按本地规则生成不依赖外部接口的具体建议。
- 建议区不再只显示固定话术，而是通过 `plan_sections` 分块展示“为什么这样训、怎么验证回灌有效、下一轮怎么查缺补漏”。

人工介入闭环：
- 9 号页启动前可勾选 `首轮人工审核` 或 `后续回灌人工审核`，默认不勾选，保持全自动 demo。
- 勾选后，Agent 会在生成候选回灌标注任务后进入 `waiting_manual`，不会立即 commit；页面会显示标注任务 ID、文件路径和“去 ⑦ 加载标注”按钮。
- 用户在 `⑦ 标注·回灌` 中加载该任务，修改并保存 annotation 后，回到 `⑨ 通用训练 Agent` 点击 `继续运行`。继续时系统读取保存后的标注内容，再执行回灌、训练、推理、评测和广度分析。
- 已完成的通用 Agent 可点击 `我想再迭代 n 次` 追加迭代；系统会根据最近的广度薄弱项生成新的候选回灌样本。若勾选 `后续回灌人工审核`，追加轮次也会先暂停给人工修改。

新增 API：

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/agent/custom/continue' -Method Post -ContentType 'application/json' -Body '{}'
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/agent/custom/extend' -Method Post -ContentType 'application/json' -Body '{"extra_iterations":1,"manual_iteration_review":true}'
```

默认建议：
- 本机 demo：`sample_count=20`、`max_iterations=3`、`train_backend=hf`、`train_device=cpu`、`base_model_path=data/registry/base/qwen2.5-0.5b-instruct`。
- 快速 smoke：`sample_count=6`、`max_iterations=1`、`train_backend=mock`、`data_generation_mode=template`。
- 正式训练：提高样本数，优先使用真实业务数据和 GPU；Judge 可用于生成高质量样本和专业性评测。

关键产物：
- `data/custom_agents/<场景>-<时间>/*.jsonl`：训练 source 和 held-out 源
- `data/custom_agents/<场景>-<时间>/*_kb.md`：配套知识库
- `data/annotations/<task>.jsonl`：预标注任务
- `data/datasets/<dataset_id>/`：每轮回灌数据集
- `data/models/<model_id>/`：每轮模型产物
- `data/reports/eval-*.json`、`data/reports/breadth-*.json`：评测与广度报告
- `data/agent_steps/custom-*.md/json`：单次运行步骤报告
- `data/agent_steps/custom-agent-workflow-20260717.md`：通用 Agent 工作流说明

注意：自动生成数据适合冷启动和 demo；正式交付应逐步替换为真实业务数据，并使用 impact 验证检查每轮回灌是否真的改善。

## 5. 产物位置
```
data/datasets/<id>/{train,valid}.jsonl   数据集(标)
data/models/<id>/weights/                LoRA adapter / 权重(训)
data/reports/eval-*.json                 评测报告(评)
data/registry/index.jsonl                模型血缘(仓库)
data/annotations/<task>.jsonl            标注任务(标注层)
data/annotations/resolved_log.jsonl      跨版本 gap 解决追踪
data/jiuan.db                            任务生命周期(SQLite)
.server_pid                              当前后台服务 PID
```

## 6. 常见故障速查
| 现象 | 原因 | 处理 |
|---|---|---|
| 推理答“你好，请问…” | 中文 prompt 乱码 | 用 UTF-8 字节体 / `_e2e_check.py` |
| `/workflow` 404 | 服务是旧代码 | `.\workflow.ps1 stop` 后重新 `serve` |
| 训练 loss 不降 | grad_accum 过大 | 小数据集设 `grad_accum=1` |
| `backend=llamafactory` 报未安装 | 环境缺 llamafactory | `pip install -r requirements-train.txt` |
| 下模型失败 | 网络/镜像 | 设 `HF_ENDPOINT=https://hf-mirror.com` 重试 fetch 脚本 |
| `backend=vllm` 连接失败 | vLLM 服务没起/端口不对 | Linux 上先 `./scripts/serve_vllm.sh`，核对 `config.vllm.base_url` |
| `serve_vllm.sh` 报语法错 | CRLF 换行 | 转成 LF（`sed -i 's/\r$//'` 或编辑器改行尾）|
| 回灌追踪前后 judge/ROUGE 显示 `—` | 回灌样本未纳入固定 held-out，防泄漏正常 | 用 `⑦ 标注` 的 impact 验证面板跑同主题变体评测 |
| impact report 有指标但无逐题 pairs | 旧代码只读了 eval summary | 更新到含 `_with_report_details` 的 `jiuan/pipeline/impact.py` |
| impact 评测后 `⑤ 迭代` 指标异常变化 | 临时 impact eval 被 attach 到模型血缘 | 确认 impact 调用 `attach_eval=false` 且 registry 过滤 impact report |

# jiuan-lite 部署与真实训练说明

本说明适用于不包含基底模型和训练权重的源码包。源码包可以启动控制面和
MOCK 演示；要进行真实训练，必须安装训练依赖并准备可读取的基底模型。

## 1. 源码包不包含什么

- `data/registry/base/`：Qwen 基底模型，体积约 1 GB。
- `data/models/`：历史 LoRA/全量训练权重。
- `data/datasets/`、`data/reports/`、`data/jiuan.db`：本机运行产物。
- API Key、PID、日志、缓存。

包内保留 `data/samples.jsonl` 和 `data/_reg_demo.txt`，可以直接用于数据准备、
RAG 示例和首次训练。真实训练完成后，平台会自动新建
`data/models/<model_id>/weights/`，无需手工创建权重目录。

## 2. 推荐环境

- Python 3.10 64 位。
- Windows + CPU：适合 0.5B 模型验证，速度较慢但可真实训练。
- Linux + NVIDIA GPU：正式训练推荐，CUDA 与 PyTorch 版本必须匹配。
- 建议解压到纯英文、无空格目录，例如 `D:/apps/jiuan-lite`。

## 3. 创建环境并安装依赖

```powershell
cd D:/apps/jiuan-lite
conda create -y -n jiuan python=3.10
conda activate jiuan
pip install -r requirements.txt
```

后续的安装、检查、启动和 worker 子进程必须使用同一个 `jiuan` 环境。部署时先确认：

```powershell
where.exe python
python -c "import sys; print(sys.executable)"
```

如果误用系统 Python，Web 控制面可能仍能启动，但训练依赖缺失后会进入 MOCK，
造成“流程成功但没有真实权重”的假象。

Windows CPU 真实训练：

```powershell
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.52.4 datasets==3.6.0 accelerate==1.7.0 peft==0.15.2
```

Linux GPU 真实训练：先按服务器 CUDA 版本安装 PyTorch GPU 轮子，再安装其余依赖。
不要把 CPU 版 PyTorch 和 GPU 版 PyTorch 混装。

```bash
# 下列 cuXXX 地址必须替换为与服务器驱动兼容的 PyTorch 官方索引地址
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cuXXX
pip install transformers==4.52.4 datasets==3.6.0 accelerate==1.7.0 peft==0.15.2
```

首次部署建议使用内置 `hf` 后端。确认跑通后，如需 LLaMA-Factory，再安装：

```powershell
pip install llamafactory==0.9.3
```

## 4. 准备基底模型

### 方式 A：使用项目下载脚本（推荐）

```powershell
python scripts/fetch_model.py
```

脚本固定下载 `Qwen/Qwen2.5-0.5B-Instruct` 到：

```text
data/registry/base/qwen2.5-0.5b-instruct
```

这个地址与默认配置完全一致。下载完成后目录至少应包含：

```text
config.json
model.safetensors（或其他模型权重分片）
tokenizer_config.json
tokenizer.json / vocab.json / merges.txt（按模型实际文件为准）
```

### 方式 B：使用已有本地模型

编辑 `configs/qwen0.5b.yaml`：

```yaml
model:
  name: Qwen/Qwen2.5-0.5B-Instruct
  local_dir: D:/models/Qwen2.5-0.5B-Instruct
```

`local_dir` 可以是项目相对路径或绝对路径。Windows 建议使用 `/`，或者把 `\`
写成 `\\`。该目录必须直接包含 `config.json`，不能填到它的父目录。

平台读取顺序如下：

1. `model.local_dir/config.json` 存在：读取本地模型，不访问网络。
2. 本地目录无效：回退到 `model.name`，由 Transformers/Hugging Face 在线获取。
3. 训练请求传入 `base_model_path`：该次训练覆盖全局配置。

因此要做稳定的离线部署，应保证 `local_dir` 正确并设置：

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
```

## 5. 必改和常改配置

配置文件：`configs/qwen0.5b.yaml`。

| 配置 | 作用 | 部署建议 |
|---|---|---|
| `model.name` | 本地模型缺失时使用的 HF ID | 保持 `Qwen/Qwen2.5-0.5B-Instruct`，或改为实际基座 ID |
| `model.local_dir` | 基底模型真实落盘地址 | 必须指向包含 `config.json` 的目录 |
| `train.backend` | `hf` / `llamafactory` / `auto` | 首次部署用 `hf` |
| `train.device` | `cpu` / `cuda` / `auto` | CPU 机器用 `cpu`，GPU 机器用 `cuda` |
| `train.precision` | `fp32` / `fp16` / `bf16` / `auto` | CPU 用 `fp32`；GPU 推荐 `auto` |
| `train.method` | `lora` / `full` | 默认 `lora`，显存占用更低 |
| `train.max_seq_len` | 训练最大长度 | 0.5B 本机演示建议 512 或 1024 |
| `infer.backend` | `transformers` / `vllm` / `auto` | 单机先用 `transformers` |
| `vllm.base_url` | 独立 vLLM 服务地址 | Linux GPU 部署时填写，例如 `http://10.0.0.8:8001/v1` |
| `judge.api_key_env` | Judge 密钥环境变量名 | 源码包不放明文密钥，部署机设置对应环境变量 |

LoRA 产物只保存增量权重。推理 `data/models/<model_id>/weights/` 中的 LoRA 时，
平台会读取模型 `meta.json` 记录的基底地址，再组合“基底模型 + LoRA adapter”。
因此训练结束后也不能删除或移动基底模型；如必须移动，需要同步修改该模型的
`meta.json` 中 `base_model_path`，或重新按原地址挂载模型目录。

## 6. 部署前检查

```powershell
python scripts/check_deployment.py --require-real
```

检查必须同时通过：

- 控制面和真实训练 Python 包可导入。
- `model.local_dir` 可解析，模型配置、权重和 tokenizer 文件存在。
- `train.device=cuda` 时 CUDA 可用。
- `data/` 可写，训练产物能够落盘。

## 7. 启动平台

真实模式启动：

```powershell
$env:JIUAN_MOCK="0"
python -m jiuan.app
```

访问：`http://127.0.0.1:8000/`。健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

局域网访问时使用：

```powershell
python -m uvicorn jiuan.app:app --host 0.0.0.0 --port 8000
```

并只在可信网络中放通 8000 端口。当前 Demo 没有生产级鉴权，不建议直接暴露到公网。

## 8. 做一次真实训练验收

命令行全链路：

```powershell
$env:JIUAN_MOCK="0"
python scripts/quickstart.py
```

也可在 Web 中依次执行：

1. ① 数据：选择 `data/samples.jsonl`，生成训练数据集。
2. ② 训练：后端选 `hf`，设备选实际设备，方法选 `lora`。
3. ③ 推理：选择刚生成的模型，确认日志显示“加载基座 + LoRA adapter”。
4. ④ 评测：选择 valid 切分；Judge 可先关闭，确保本地链路独立跑通。

验收时必须看到：

- 训练结果 `mode=real`，不能是 `mock`。
- 新模型目录含 `weights/adapter_config.json` 和 adapter 权重。
- `meta.json` 的 `base_model_path` 指向当前可读取的基底模型。
- 推理日志明确加载该基底模型和该 LoRA adapter。

## 9. Judge 密钥

源码包中的 `judge.api_key` 留空。以火山方舟为例：

```powershell
$env:ARK_API_KEY="<部署环境中的真实 Key>"
```

`configs/qwen0.5b.yaml` 中保持：

```yaml
judge:
  api_key:
  api_key_env: ARK_API_KEY
```

不要把真实 Key 重新打进源码压缩包。

# saga

`saga` 是一个轻量级的LLM推理引擎实现，代码尽量保持简洁，便于学习和二次改造。

当前实现重点：
- Qwen3 Causal LM 推理
- Prefill / Decode 调度
- KV Cache Block 管理与复用
- FlashAttention + Triton 加速
- Tensor Parallel（多卡）

## 环境要求

- Python `>=3.10`
- NVIDIA GPU + CUDA 环境
- Linux/macOS 开发环境（推理依赖 CUDA）

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

如果运行时报缺包，可补充安装：

```bash
pip install safetensors tqdm numpy
```

## 快速开始

1. 准备本地 Hugging Face 模型目录（示例使用 Qwen3-0.6B）。
2. 修改 `example.py` 中的模型路径。
3. 运行示例：

```bash
python3 example.py
```

`example.py` 会：
- 用 `AutoTokenizer` 构造 chat prompt
- 调用 `LLM.generate(...)` 批量生成
- 输出每条 prompt 的文本结果

## API 示例

```python
from saga.engine.llm_engine import LLMEngine
from saga.sampling_params import SamplingParams

llm = LLMEngine(
    "/path/to/local/model",
    enforce_eager=True,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=256,
    ignore_eos=False,
)

outputs = llm.generate(
    ["Hello", "List prime numbers within 100"],
    sampling_params,
    use_tqdm=True,
)
```

## OpenAI 兼容服务

提供 `FastAPI(API 进程) + 独立推理 Worker(队列持续批处理)` 架构，支持 OpenAI 兼容接口：

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`（支持 `stream=true/false`）
- `POST /v1/completions`（`n=1`，`stream=true` 当前仅支持单 prompt）

启动：

```bash
python -m saga \
  --model ~/huggingface/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000
```

开启张量并行（示例 2 卡）：

```bash
python -m saga \
  --model ~/huggingface/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2
```

等价命令：

```bash
saga-openai-api --model ~/huggingface/Qwen3-0.6B --host 0.0.0.0 --port 8000
```

说明：

- 请求里的 `model` 字段会被忽略，服务始终使用启动时 `--model` 对应模型。
- Worker 在内部持续拉取请求并动态并入调度队列，复用 saga 的 prefill/decode 调度能力。
- 默认采用 mini-sglang 风格调度：prefill-first + chunked prefill + decode inflight 资源预留。
- 推荐单进程运行 `uvicorn workers=1`（脚本默认如此），由内部 worker 负责并发推理。

chunked prefill 可调参数：

- `--disable-continuous-batching`：关闭 chunked prefill 限流，prefill budget 回退到 `max-num-batched-tokens`
- `--decode-steps-per-prefill`：兼容保留参数，当前 mini-sglang 风格调度不使用
- `--max-prefill-tokens-per-step`：chunked prefill 的单步 token budget（默认 2048）

分布式初始化参数（tensor parallel 时生效）：

- `--dist-init-addr`：`torch.distributed` 初始化地址（默认 `127.0.0.1`）
- `--dist-init-port`：初始化端口（默认 `0`，表示自动选择空闲端口）

## 基准测试

项目提供了简单压测脚本 `benchmark/bench.py`：

```bash
python3 benchmark/bench.py
```

默认会打印总 token、耗时和吞吐。

在线服务压测脚本（参考 mini-sglang online benchmark 风格）：

```bash
python3 benchmark/bench_online.py \
  --base-url http://127.0.0.1:8000 \
  --model-path ~/huggingface/Qwen3-0.6B \
  --num-requests 200 \
  --concurrency 32 \
  --input-len 256 \
  --output-len 128 \
  --stream
```

高共享前缀压测（用于观察前缀缓存收益）：

```bash
python3 benchmark/bench_online.py \
  --base-url http://127.0.0.1:8000 \
  --model-path ~/huggingface/Qwen3-0.6B \
  --num-requests 200 \
  --concurrency 32 \
  --input-len 256 \
  --output-len 128 \
  --stream \
  --prompt-mode shared_prefix \
  --shared-prefix-len 192 \
  --num-shared-prefixes 1
```

脚本会输出：
- 请求吞吐（req/s）
- token 吞吐（total/completion tok/s）
- 延迟统计（avg / p50 / p95 / p99 / max）
- TTFT（首 token 时延，`--stream` 模式）
- TPOT（单 token 间隔，`--stream` 模式；非流式为估算值）

## 当前限制

- 当前模型实现聚焦 Qwen3（见 `saga/models/qwen3.py`）。
- `SamplingParams` 不允许贪心采样（`temperature` 必须大于 `1e-10`）。
- `model` 参数要求是本地目录路径。
- `tensor_parallel_size` 当前限制在 `1~8`，且需不超过可见 GPU 数。
- `num_attention_heads` / `num_key_value_heads` 必须能被 `tensor_parallel_size` 整除。

## 目录结构

```text
saga/
  engine/     # 调度、执行、并行与 KV cache 流程
  layers/     # 核心算子与注意力实现
  models/     # 模型定义（当前为 Qwen3）
  utils/      # 权重加载与上下文工具
```

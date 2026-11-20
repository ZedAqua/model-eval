# AutoDL 评测手册（Qwen3.8-27B，开启思考）

在 AutoDL 单卡 A800-80GB 上用 vLLM 起服务、跑本仓库的评测脚本。下面的命令都在 AutoDL 的 Linux shell 里执行；路径按 2026-11-19 基线所用的数据盘布局（`/root/autodl-tmp`）。

## 已验证的版本组合

| 项目 | 版本 |
|---|---|
| GPU / 驱动 | A800-SXM4-80GB，驱动 580.126.09（CUDA 13.0） |
| PyTorch | 2.14.0+cu130 |
| vLLM | 0.34.0 |
| FlashInfer | 0.7.3（sm80 JIT 缓存） |
| Transformers | 5.22.2 |
| 模型 | Qwen/Qwen3.8-27B，revision `6e218df9c23850fe143662112737b1fcb62ee880`，BF16 |

驱动必须 ≥ 580（`nvidia-smi` 右上角 CUDA Version ≥ 13.0），否则 cu130 的 torch 用不了 GPU，直接关机换主机。

## 目录布局

```text
/root/autodl-tmp/
  envs/vllm-qwen/            vLLM 推理环境（也用来跑 FinEval/AIME/GPQA，只用标准库）
  envs/eval-tools/           评测依赖环境（IF、MultiChallenge、LCB、BFCL）
  models/Qwen3.8-27B/        模型权重
  datasets/baseline-local-20261119/   各评测数据快照（由 ModelScope 管理，见 manifests/）
  workspace/model-eval/      本仓库
  workspace/official-src/    官方打分代码的固定版本（IFBench、LiveCodeBench、MultiChallenge 题目、NLTK 数据）
  workspace/fineval-scored-20261119/  FinEval 打分用数据（由原始数据渲染，指纹见 manifests/）
  workspace/baseline-runs/   运行输出（不进 Git）
  logs/                      服务日志
```

## 一次性准备

### 评测依赖环境

```bash
b=/root/autodl-tmp; e=$b/envs/eval-tools
$b/envs/vllm-qwen/bin/python -m venv $e
export PIP_CACHE_DIR=$b/pip-cache TMPDIR=$b/pip-tmp
$e/bin/pip install "nltk==3.9.2" absl-py langdetect immutabledict emoji "syllapy>=0.8.0" \
  httpx tqdm "pydantic>=2" pydantic-settings pandas numpy
$e/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # BFCL 只在记忆类别用到 CPU 版 torch
$e/bin/pip install bfcl-eval soundfile "pyarrow==17.0.0"
```

注意：
- nltk 必须是 IFBench 官方锁定的 **3.9.2**；3.8.1 会把 `punkt_tab` 解析成 `punkt/PY3_tab`，导入失败。
- `bfcl-eval` 会把 numpy 锁到 1.26.4，较新的 pyarrow 要求 numpy 2，所以 pyarrow 固定 17.0.0。
- `bfcl-eval` 的 Qwen handler 依赖 `qwen_agent`，后者需要 `soundfile`，pip 不会自动装上。
- 直接 `pip install bfcl-eval` 会拉 CUDA 版 torch（数 GB），所以先单独装 CPU 版。

### 官方打分代码与离线数据

```bash
s=/root/autodl-tmp/workspace/official-src
# IFBench（allenai/IFBench @ 488918665d53f867fa908a293769ec6e71e74aec）解压到 $s/IFBench
# LiveCodeBench 的 lcb_runner/（LiveCodeBench/LiveCodeBench @ 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24）放到 $s/LiveCodeBench/lcb_runner
# MultiChallenge 题目 data/benchmark_questions.jsonl（ekwinox117/multi-challenge @ 5ccefcca）存为 $s/multichallenge_benchmark_questions.jsonl
# NLTK 数据：punkt、punkt_tab、stopwords、averaged_perceptron_tagger_eng 放到 $s/nltk_data
#   服务器下载 punkt 可能 503，可在本机下载 punkt.zip 后上传解压
# BFCL 记忆类别用的 MiniLM：
HF_ENDPOINT=https://hf-mirror.com /root/autodl-tmp/envs/eval-tools/bin/python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/root/autodl-tmp/cache/hf')"
```

上述版本和指纹记录在 [`manifests/official-sources-2026-11-19.json`](../../manifests/official-sources-2026-11-19.json)。装好后可在 IFBench 目录跑官方单元测试确认环境：`NLTK_DATA=$s/nltk_data PYTHONPATH=. python instructions_test.py`（118 个测试应全部通过）。

### LiveCodeBench 沙箱用户

生成的代码由低权限用户 `lcbrun` 执行（进程内存上限 4GB、进程数上限、临时目录）。容器不允许网络命名空间，**无法断网**。

```bash
chmod o+x /root                      # 只允许按路径进入，不能列目录
useradd --system --no-create-home --shell /usr/sbin/nologin lcbrun
chmod 700 /root/autodl-tmp/workspace /root/autodl-tmp/datasets/baseline-local-20261119
```

## 启动服务

先 `nvidia-smi` 确认驱动，再在 `screen` 会话里启动（SSH 断开不影响）：

```bash
screen -S serve
QWEN_MAX_MODEL_LEN=65536 QWEN_MAX_NUM_SEQS=128 QWEN_ENFORCE_EAGER=0 \
QWEN_TOOL_CALL_PARSER=qwen3_coder QWEN_KV_CACHE_DTYPE=fp8 \
  bash /root/autodl-tmp/workspace/model-eval/runners/autodl/serve_qwen.sh
```

| 变量 | 取值 | 说明 |
|---|---|---|
| `QWEN_MAX_MODEL_LEN` | 16384 / 32768 / 65536 | 输出上限 61,440（含思考）需要 65536 |
| `QWEN_MAX_NUM_SEQS` | 正整数 | 128 即可；长思考评测会先被显存卡住 |
| `QWEN_ENFORCE_EAGER` | 0 / 1 | 0 用 CUDA graph，批量解码快很多；图捕获报错再改 1 |
| `QWEN_TOOL_CALL_PARSER` | 空 / `qwen3_coder` | BFCL 需要；Qwen3.8 模板是 XML 工具调用格式。不影响其他评测 |
| `QWEN_KV_CACHE_DTYPE` | `auto` / `fp8` | fp8 让 KV 缓存从约 27 万 tokens 增到约 48 万，吞吐约 3 倍；模型无 FP8 缩放因子，数值有轻微差异 |

启动日志里记下 `GPU KV cache size` 和 `Maximum concurrency`。同一组对比（基线与训练后）必须用相同的 KV 精度。

## 运行评测

```bash
b=/root/autodl-tmp; r=$b/workspace/model-eval; vpy=$b/envs/vllm-qwen/bin/python; epy=$b/envs/eval-tools/bin/python
o=$b/workspace/baseline-runs/$(date +%Y%m%d)-run
$vpy $r/benchmarks/fineval/run_fineval.py --subset salesforce --workers 64 --output $o/fineval-salesforce
$vpy $r/benchmarks/aime/run_aime.py --year 2025 --sample-index 0 --output $o/aime-2025-s0
$vpy $r/benchmarks/gpqa/run_gpqa.py --sample-index 0 --output $o/gpqa-s0
$epy $r/benchmarks/ifbench/run_if.py --subset ifbench-multiturn --sample-size 300 --output $o/ifbench-multiturn-n300-s0
$epy $r/benchmarks/multichallenge/run_multichallenge.py --output $o/multichallenge-s0
$epy $r/benchmarks/livecodebench/run_lcb.py --output $o/lcb-s0
$epy $r/benchmarks/bfcl/run_bfcl.py --output $o/bfcl
```

- 每个脚本都支持 `--verify-only`（只校验数据和配置，不请求服务），正式跑前先跑一遍。
- 中断或有请求失败时，用**同一条命令、同一个输出目录**续跑；已完成的题不会重跑。改了参数、数据或脚本版本就换新目录（脚本会拒绝混用）。
- 多次采样：`--sample-index N` 用种子 42+N，每次一个输出目录，最后取平均。
- 均匀抽样（IF 系列）：`--sample-size N` 按数据文件顺序等间隔取第 ⌊(i+0.5)×总数/N⌋ 条。
- 多个评测可以同时跑，共享服务的并发位置；排队时间计入单次请求超时，同时跑很多时把 `--timeout` 加大到 14400。
- 只评思考之后的最终回答；截断、空回复按失败计，不做兜底猜测。

### MultiChallenge 判分（在能访问 judge API 的机器上）

服务器只生成 `multichallenge_responses.jsonl`。拷回本机后：

```bash
OPENAI_BASE_URL=<judge 接口> OPENAI_API_KEY=<key> PYTHONUTF8=1 \
  python scorers/multichallenge_judge.py --repo <multi-challenge 克隆 @ 5ccefcca> \
  --responses multichallenge_responses.jsonl --output-dir <输出目录> --judge-model gpt-6.1-sol
```

需要 `openai`、`pydantic`、`python-dotenv`、`tqdm`。Windows 上必须开 UTF-8 模式（官方代码读文件不指定编码）。

## 运维注意

- 用 `pgrep -f`/`pkill -f` 找进程时，模式不要写成会匹配到当前命令行本身的字符串（例如在 `ssh '...'` 里出现被杀脚本的文件名），否则会把自己的 SSH 会话杀掉。复杂的停止、切换操作写成脚本上传后执行。
- 跑完后 `shutdown`（AutoDL 自带脚本，会停实例、停止计费，数据盘保留）。需要拷数据时用无卡模式开机。

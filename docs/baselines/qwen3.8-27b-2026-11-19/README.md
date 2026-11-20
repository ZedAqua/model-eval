# Qwen3.8-27B 本地基线（2026-11-19）

开启思考、`reasoning_effort=xhigh`、零样本；单卡 A800-80GB，vLLM 0.34。只评思考之后的最终回答；截断、空回复、格式无效按失败计，不做兜底猜测。逐题输出（思考、回答、判分明细）不进 Git，本目录只放分数汇总和运行配置（`results/<运行>/summary.json`、`run.json`）。

## 分数

| 评测 | 题数 | 主指标 | 分数 | KV 缓存 | 备注 |
|---|---|---|---|---|---|
| FinEval Salesforce（英文 MMLU-finance） | 1,461 | 准确率 | **94.46%** | BF16 | 无截断、无格式无效 |
| FinEval SUFE 验证集（中文） | 1,151 | 准确率 | **91.31%** | BF16 | 3 题答成多选 `ABC`、1 题截断，均计错 |
| IFEval 多轮 | 300 / 1,774（均匀抽样） | prompt-level loose | **84.0%** | FP8 | strict 83.0%；1 题空回复计失败 |
| IFBench 多轮 | 300 / 1,387（均匀抽样） | prompt-level loose | **81.0%** | FP8 | strict 76.3%；1 题空回复、1 题人工中止，均计失败 |
| MultiChallenge | 273 | 4 个维度通过率宏平均 | **63.66%** | FP8 | judge：gpt-6.1-sol；见下表 |

MultiChallenge 分维度：INSTRUCTION_RETENTION 82.61%（69 题）、INFERENCE_MEMORY 66.37%（113）、RELIABLE_VERSION_EDITING 53.66%（41）、SELF_COHERENCE 52.00%（50）。

**这些分数不能直接和公开榜单比较**：KV 缓存精度、提示词、采样次数、judge 模型都和公开报告不同。用于训练前后对比时，两边必须用相同的脚本版本、数据指纹、KV 精度和样本。

## 运行条件

| 项目 | 取值 |
|---|---|
| 模型 | Qwen/Qwen3.8-27B，revision `6e218df9c23850fe143662112737b1fcb62ee880`，BF16 权重 |
| 推理 | vLLM 0.34.0，`--max-model-len 65536 --max-num-seqs 128`，CUDA graph，`--reasoning-parser qwen3` |
| KV 缓存 | FinEval：`auto`（BF16）；其余：`fp8`（e4m3，模型无缩放因子） |
| 采样 | temperature 1.0、top_p 0.95、top_k 20、min_p 0、presence/repetition penalty 0/1（模型卡推荐的思考模式参数），seed 42 |
| 输出上限 | 61,440 tokens（含思考） |
| 抽样 | IF 多轮两项 `--sample-size 300`，等间隔取第 ⌊(i+0.5)×总数/300⌋ 条；300 题的 95% 置信区间约 ±5.7 个百分点 |
| 版本 | 官方打分代码和数据指纹见 [`manifests/official-sources-2026-11-19.json`](../../../manifests/official-sources-2026-11-19.json) |

启动服务与运行命令见 [`runners/autodl/README.md`](../../../runners/autodl/README.md)。

## 各评测的口径

- **FinEval**：题面沿用数据集的“只输出选项字母”。答案取最终回答开头、最后一行和全文显式作答写法（参考 SUFE 官方 `extract_answer` 的正则），恰好指向一个字母才计分。该题面让模型思考很短（Salesforce 生成长度中位数 125 tokens），这是有意保留的设定。
- **IFEval / IFBench 多轮**：用 IFBench 官方 `run_eval.py` 打分（先 strict 后 loose）；多轮消息为 [user 原问题, assistant 预生成回答, user 约束]，只评第 3 轮回复。
- **MultiChallenge**：服务器生成官方 `--responses-file` 格式的回复，本机用官方 `DataLoader`/`Evaluator`/`ResultParser` 判分；judge 由官方的 gpt-4o 换成 gpt-6.1-sol（temperature 0、结构化输出），并去掉官方代码传给 `OpenAIModel` 的多余 `max_tokens` 参数。judge 调用 0 次出错。

## 截断与生成长度

约 3,500 道已完成的题中只有 1 题截断（SUFE 精算题，思考写满 61,440 tokens）。该题思考内容没有重复，模型一直在猜测题面缺失的 R(x) 定义，属于题目不完整，加大上限也无济于事。其余各评测最长生成都在 40K 以下（MultiChallenge 最长 39,444，IFBench 多轮最长 39,402）。

## 吞吐与成本

- 32 并发、BF16 KV：长思考题把 KV 缓存占满后，同时生成的请求降到约 18 个，吞吐约 350 tokens/s。
- 128 并发、FP8 KV：KV 容量由约 27 万增至约 48 万 tokens，吞吐升至约 1,100 tokens/s；多轮 IF 这类长输入长输出的任务稳定在约 650–700 tokens/s。
- 瓶颈始终是 KV 缓存（显存），不是并发数；长尾题单条约 28 tokens/s。

## 暂停或未跑的评测

AIME 2025/2026、GPQA Diamond、IFBench、LiveCodeBench、BFCL 已有脚本，本轮按“优先跑没有公开分数的评测”暂停，部分进度留在服务器数据盘，用原命令可续跑。脚本状态：

| 评测 | 脚本 | 验证情况 |
|---|---|---|
| AIME | `benchmarks/aime/run_aime.py` | 冒烟测试通过；取最终回答里最后一个 `\boxed{}` 整数 |
| GPQA Diamond | `benchmarks/gpqa/run_gpqa.py` | 与 simple-evals 逐题一致（4 次重复 × 198 题） |
| IFBench | `benchmarks/ifbench/run_if.py --subset ifbench` | 与多轮共用，官方打分已实跑 |
| LiveCodeBench | `benchmarks/livecodebench/run_lcb.py` | 提示词/提取与官方一致；沙箱对正确、错误、死循环、越权读取、超内存 5 种代码判定正确；未用真实模型全量跑 |
| BFCL v4 | `benchmarks/bfcl/run_bfcl.py` | 用模拟服务跑通生成到官方打分；未用真实模型全量跑 |

## 脚本版本说明

本轮实际运行的脚本与提交到本仓库的版本相比：`run_fineval.py`、`run_aime.py`、`run_gpqa.py`、`run_bfcl.py`、`runners/evalkit.py` 完全一致（sha256 相同）；`run_if.py`、`run_multichallenge.py`、`run_lcb.py` 和 LCB 沙箱打分脚本只改了迁入本仓库后的 import 路径和注释，逻辑未变，因此 `results/*/run.json` 里的 `runner_sha256` 与仓库内文件不同。运行中用于调度、切换抽样、中止最后一题的一次性 shell 脚本未提交；其效果已在上表备注中说明。

# Qwen3.8-27B 本地基线（2026-11-19）

开启思考、`reasoning_effort=xhigh`、零样本；单卡 A800-80GB，vLLM 0.34。只评思考之后的最终回答；截断、空回复、格式无效按失败计，不做兜底猜测。逐题输出（思考、回答、判分明细）不进 Git，本目录只放分数汇总和运行配置（`results/<运行>/summary.json`、`run.json`）。

## 分数

| 评测 | 题数 | 主指标 | Qwen3.8-27B | KV 缓存 | 备注 | Qwen3.8-Flash-Next（公开） | DeepSeek-V4.1-Flash（公开） |
|---|---|---|---|---|---|---|---|
| FinEval Salesforce（英文 MMLU-finance） | 1,461 | 准确率 | **94.46%** | BF16 | 无截断、无格式无效 | — | — |
| FinEval SUFE 验证集（中文） | 1,151 | 准确率 | **91.31%** | BF16 | 3 题答成多选 `ABC`、1 题截断，均计错 | — | — |
| IFEval 多轮 | 300 / 1,774（均匀抽样） | prompt-level loose | **84.0%** | FP8 | strict 83.0%；1 题空回复计失败 | — | — |
| IFBench 多轮 | 300 / 1,387（均匀抽样） | prompt-level loose | **81.0%** | FP8 | strict 76.3%；1 题空回复、1 题人工中止，均计失败 | — | — |
| MultiChallenge | 273 | 4 个维度通过率宏平均 | **63.66%** | FP8 | judge：gpt-6.1-sol；见下表 | — | 74.90 ± 5.30<sup>a</sup> |
| GPQA Diamond | 198 | 准确率 | 89.2 | — | 公开参考分，非本地实测；来源：[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B#text-performance) | 91.7%<sup>b</sup> | 90.9%<sup>c</sup> |
| IFBench | 300 | 未注明 | 79.5 | — | 公开参考分，非本地实测；来源：[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B#text-performance)，未注明细分指标和采样口径 | 81.3%<sup>b</sup> | 76.667%（原版）/ 76.33%（1017 版）<sup>d</sup> |
| LiveCodeBench v6 | 未注明 | pass@1 | 90.3 | — | 公开参考分，非本地实测；来源：[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B#text-performance)，题集不一定与本地 175 题相同 | 91.9%<sup>b</sup> | — |
| AIME 2025 | 30 | 准确率（约 3 次采样均值） | 92.2% | — | 公开参考分，非本地实测；来源：第三方 [Prefill Systems](https://prefillsystems.com/benchmarks/qwen3-8-27b)，BF16 | — | — |
| AIME 2026 | 30 | 准确率（约 3 次采样均值） | 91.7% | — | 公开参考分，非本地实测；来源：第三方 [Prefill Systems](https://prefillsystems.com/benchmarks/qwen3-8-27b)，BF16 | 90.00%<sup>e</sup> | 100%<sup>f</sup> |
| BFCL v4（仅 non-live） | 1,150 | 准确率（3 次运行） | 88.1% ± 2.1 | — | 公开参考分，非本地实测；来源：第三方 [Model Spend Arena](https://msa.millaguie.net/providers/l-qwen3-8-27b.html)，UD-Q3_K_XL 量化版，不是完整 BFCL 总分 | 83.0% ± 2.4<sup>g</sup> | — |

后两列全部是公开报告中的分数（检索截至 2026-11-20），没有在本地跑过；“—”表示没找到口径对应、可核实的公开成绩。来源与口径：

- <sup>a</sup> [Scale 官方 MultiChallenge 榜单](https://labs.scale.com/leaderboard/multichallenge)，DeepSeek-V4.1-Flash **1017 版**。榜单 judge 已换成 Gemini 2.5 Pro，并修订了 54 道题，与本地 gpt-6.1-sol judge 的 63.66% 不能直接排名。
- <sup>b</sup> [Qwen3.8-Flash-Next 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#language)，官方自报。
- <sup>c</sup> [DeepSeek-V4.1-Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash#instruct-model)，**原版**。
- <sup>d</sup> NVIDIA 对原始权重的复测（取表中 source 原始权重行，非 NVFP4 量化行），各重复 5 次，两次复测的采样设置不同：[原版](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4#evaluation)、[1017 版](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4#evaluation)。
- <sup>e</sup> 第三方 [Nota](https://huggingface.co/nota-ai/Qwen3.8-Flash-Next-Nota-NVFP4#performance) 的 BF16 实测：`temperature=0`、单次、输出上限 65,536 tokens。
- <sup>f</sup> DeepSeek 官方[模型页](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash#evaluation-results)，**原版**，取自技术报告附录 B.3，`reasoning_effort=100`（最大推理强度）；[来源说明](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/82)。与 e 的协议不同，不能直接比较。
- <sup>g</sup> 第三方 [Model Spend Arena](https://msa.millaguie.net/providers/l-qwen3-8-flash-next.html)，UD-Q4_K_XL 量化版，来源标注 n=1,390（与本行 1,150 题不同），来源内部的题数与分类说明有待核对，仅作参考。

其他模型的公开参考分（同样未在本地测，不与本表排名）：

- **GPQA Diamond**：GPT-6.2 Astra 97.0%、GPT-6.2 Sol 96.0%、GPT-6 Astra 96.0%、GPT-6.1 Terra 93.56%（[OpenAI 系统卡 Table 2](https://deploymentsafety.openai.com/gpt-6-2-astra-sol/gpt-6-2-astra-sol.pdf)）；GPT-5.4 Pro 94.4%、Gemini 3.1 Pro 94.3%、Claude Opus 4.7 94.2%、GPT-5.5 93.6%、GPT-5.4 92.8%（[OpenAI GPT-5.5 发布页](https://openai.com/index/introducing-gpt-5-5/)）；Claude Opus 4.6 Max 91.3%（[Qwen3.8-Flash-Next 模型卡对比表](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#language)）。
- **IFBench / LiveCodeBench v6**：Claude Opus 4.6 Max 62.5% / 88.8%（同上，Qwen 引用的对照成绩）。
- **MultiChallenge**（[Scale 榜单](https://labs.scale.com/leaderboard/multichallenge)，Gemini 2.5 Pro judge）：GPT-6.1 Sol 81.95 ± 4.16、Claude Fable 5.2 80.40 ± 5.10、GPT-6.1 Terra 74.20 ± 5.40、GPT-6.2 Astra 73.20 ± 5.80、GPT-6.2 Sol 72.00 ± 5.80、GPT-5.4 Pro 69.23 ± 3.05、Claude Haiku 5.5 67.67 ± 5.34、GPT-5.1 Thinking 63.41 ± 2.11、GPT-5 Thinking 63.19 ± 1.63、o3-pro High 62.40 ± 2.49、Claude Opus 4.5 Thinking 58.97 ± 0.42、Claude Opus 4.6 Non-Thinking 56.02 ± 1.08、Claude Sonnet 4.5 Thinking 55.32 ± 1.47、Claude Opus 4.6 Thinking Max 37.15 ± 2.16。
- **AIME 2025**（[OpenAI GPT-5.2 发布页](https://openai.com/index/introducing-gpt-5-2/)）：GPT-5.2 Thinking xhigh 100.0%、GPT-5.2 Pro xhigh 100.0%、GPT-5.1 Thinking high 94.0%。
- **AIME 2026**（[MathArena](https://matharena.ai/?comp=aime--aime_2026)，30 题、每题 4 次取平均）：GPT-5.5 xhigh 100.00%、Claude Opus 4.8 max 100.00%、GPT-5.4 xhigh 99.17%、GPT-5.2 high 98.33%、Claude Opus 4.6 high 96.67%、Claude Opus 4.7 xhigh 95.83%。

MultiChallenge 分维度：INSTRUCTION_RETENTION 82.61%（69 题）、INFERENCE_MEMORY 66.37%（113）、RELIABLE_VERSION_EDITING 53.66%（41）、SELF_COHERENCE 52.00%（50）。

“Qwen3.8-27B”一列中加粗的是本地实测；未加粗、备注为“公开参考分”的是本地暂未跑的评测，取自公开报告（检索截至 2026-11-19），只作参考，不能当作本地基线，也不能和本地实测直接比较。IFBench 多轮、IFEval 多轮、MultiChallenge 未查到可核实的公开分数。

**本地实测分数不能直接和公开榜单比较**：KV 缓存精度、提示词、采样次数、judge 模型都和公开报告不同。用于训练前后对比时，两边必须用相同的脚本版本、数据指纹、KV 精度和样本。

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

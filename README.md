# model-eval

统一的模型与 Agent 评测代码仓库，负责把固定版本的评测题目、工具环境、模型接口和评分器连接起来，产出可复现的基线与训练后回归结果。

## 职责

- 维护 benchmark adapter、运行器、评分器和回归配置。
- 记录评测题目、提示词、工具协议、模型参数和 judge 版本。
- 支持 Hermes E2E 评测与标准模型评测，并明确区分两种协议。
- 生成结果索引、诊断信息和可复现运行记录。

## 不负责

- 不维护训练数据清洗和样本转换逻辑。
- 不维护 Hermes 生产代码或 MCP 工具实现。
- 不把大规模题库、答案、运行输出和 checkpoint 提交到 Git。

## 数据约定

评测题目、gold 和运行产物由 ModelScope 管理；本仓库只保存 manifest、版本号、hash、下载/校验脚本和必要的小型样例。评测数据归入 ModelScope 的 `evaluation-datasets` 合集；每个 benchmark 保持独立 Dataset，具体地址和 revision 由 manifest 固定。

## 当前公开状态

本仓库已公开。已上传标准模型评测的 runner 和 scorer（FinEval、AIME、GPQA Diamond、IFBench / IFBench 多轮 / IFEval 多轮、MultiChallenge、LiveCodeBench、BFCL v4），以及 Qwen3.8-27B 本地基线的分数汇总与运行配置。Hermes E2E 评测尚未上传。评测题目和数据版本由 ModelScope 管理；逐题运行输出不进 Git。

- 运行方式：[`runners/autodl/README.md`](runners/autodl/README.md)
- 官方打分代码与数据版本：[`manifests/official-sources-2026-11-19.json`](manifests/official-sources-2026-11-19.json)
- 各评测官方取答案与判分方式：[`docs/official-scoring-review-2026-11-19.md`](docs/official-scoring-review-2026-11-19.md)
- 基线结果：[`docs/baselines/qwen3.8-27b-2026-11-19/`](docs/baselines/qwen3.8-27b-2026-11-19/README.md)（模型侧）、[`docs/baselines/qwen3.8-flash-agentic-2026-11-12/`](docs/baselines/qwen3.8-flash-agentic-2026-11-12/README.md)（Agentic 侧，含 DS + Hermes 对照）

## 基线结果：Qwen3.8-27B（模型侧，2026-11-19）

加粗为本地实测（开启思考、`reasoning_effort=xhigh`、零样本，vLLM 0.34）；其余为公开报告分数，括号注明官方或第三方，均未在本地跑过，不能与本地实测直接比较。来源链接、模型版本和口径限制见[完整报告](docs/baselines/qwen3.8-27b-2026-11-19/README.md)。

| 评测 | 题数 | 主指标 | Qwen3.8-27B | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash |
|---|---|---|---|---|---|
| FinEval Salesforce（英文 MMLU-finance） | 1,461 | 准确率 | **94.46%** | — | — |
| FinEval SUFE 验证集（中文） | 1,151 | 准确率 | **91.31%** | — | — |
| IFEval 多轮 | 300 / 1,774（均匀抽样） | prompt-level loose | **84.0%** | — | — |
| IFBench 多轮 | 300 / 1,387（均匀抽样） | prompt-level loose | **81.0%** | — | — |
| MultiChallenge | 273 | 4 个维度通过率宏平均 | **63.66%**（judge：gpt-6.1-sol） | — | 74.90 ± 5.30（Scale 榜单，1017 版，judge 不同） |
| GPQA Diamond | 198 | 准确率 | 89.2（官方） | 91.7%（官方） | 90.9%（官方，原版） |
| IFBench | 300 | — | 79.5（官方） | 81.3%（官方） | 76.667%（原版）/ 76.33%（1017 版）（NVIDIA 复测） |
| LiveCodeBench v6 | — | pass@1 | 90.3（官方） | 91.9%（官方） | — |
| AIME 2025 | 30 | 准确率 | 92.2%（第三方） | — | — |
| AIME 2026 | 30 | 准确率 | 91.7%（第三方） | 90.00%（第三方，贪心单次） | 100%（官方，原版，最大推理强度） |
| BFCL v4（仅 non-live） | 1,150 | 准确率 | 88.1% ± 2.1（第三方，量化版） | 83.0% ± 2.4（第三方，量化版） | — |

FinEval 用 BF16 KV 缓存，IFEval/IFBench 多轮和 MultiChallenge 用 FP8 KV 缓存。

## 基线结果：Agentic 侧（2026-11-12）

Agentic 评测用 **Qwen3.8-Flash** 代替 Qwen3.8-27B（两者不是同一个模型，不能当作 27B 的 Agentic 基线）。两侧接入同一版本的公司 Hermes，只换主模型；裁判 gpt-6.1-sol，每题跑一次，只做内部比较。运行条件、差距归因和 DS 完整基线见[完整报告](docs/baselines/qwen3.8-flash-agentic-2026-11-12/README.md)。

| 评测 | 范围 | Qwen3.8-Flash | DS（deepseek-v4.1-flash-1017）+ Hermes | 备注 |
|---|---|---|---|---|
| τ²-bench（τ³ v1.0） | 278 全量 | **247/278（88.8%）** | 253/278（91.0%） | 差距全在 telecom（102 vs 110） |
| Seal-0 | 111 全量 | **51/111（45.9%）** | 64/111（57.7%） | Qwen 批次搜索环境有混杂，差距不能全归因到模型 |
| SpreadsheetBench 2 | 50 题（financial_model 25 + template 25） | **19/50** | 18/50 | 推理强度 xhigh；4 题超时计 0 |
| FinSearchComp T2/T3 | 50 题抽样 | **31/50（62%）** | 38/50（76%） | 3 题超时计 0；两侧运行链路不同 |

## 目录

```text
benchmarks/   各评测的运行脚本（读数据、组织提示词、调用官方打分）
scorers/      本机 judge 判分与代码执行沙箱打分
runners/      共用的生成逻辑（evalkit.py）和 AutoDL/vLLM 启动脚本
manifests/    官方代码、数据与环境版本索引
docs/         判分口径调研与基线报告（只放汇总，不放逐题输出）
configs/      固定评测配置（规划中）
reports/      报告生成代码（规划中）
```

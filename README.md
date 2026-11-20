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
- 基线结果：[`docs/baselines/qwen3.8-27b-2026-11-19/`](docs/baselines/qwen3.8-27b-2026-11-19/README.md)

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

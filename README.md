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

评测题目、gold 和运行产物由 ModelScope 管理；本仓库只保存 manifest、版本号、hash、下载/校验脚本和必要的小型样例。具体 ModelScope 地址和权限规则将在数据仓库建立后补充。

## 计划目录

```text
benchmarks/   评测适配器与协议
scorers/      程序评分器和 judge 封装
runners/      本地、容器和批量运行入口
configs/      固定评测配置
manifests/    数据与环境版本索引
reports/      报告生成代码，不保存大结果文件
```

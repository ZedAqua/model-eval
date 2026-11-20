# 各评测官方取答案与判分方式核对（2026-11-19）

目的：给 Qwen3.8-27B 本地基线（开启思考、`reasoning_effort=xhigh`、temperature 1.0 / top_p 0.95 / top_k 20、只解析思考之后的最终 `content`）写评测脚本前，先弄清各评测的官方或主流做法。结论来自阅读官方仓库源码和数据卡，**没有在服务器上实跑这些官方脚本**；标注"未核实"的地方需要再查。

## 总览

| 评测 | 官方判分 | 是否需要外部服务 | 我们的状态 |
|---|---|---|---|
| FinEval（SUFE 中文） | 有，正则 + 兜底（第一个字母 / 选项原文 / 随机猜） | 否 | 已有脚本，采用官方显式写法，不用兜底 |
| FinEval（Salesforce / MMLU-finance） | 无专门脚本，推荐 lm-eval-harness `exact_match` | 否 | 已有脚本 |
| AIME 2025/2026 | 无官方脚本；社区取最后一个 `\boxed{}` 整数比对 | 否 | 已有脚本 |
| GPQA Diamond | 官方仓库有；推理模型主流用 OpenAI simple-evals | 否 | 已实现 `benchmarks/gpqa/` |
| IFEval / IFBench / 两个 multiturn | 规则 checker，确定性 | 否（NLTK 数据要预装） | 已实现 `benchmarks/ifbench/` |
| MultiChallenge | 必须 LLM judge（官方 GPT-4o） | 是（judge API） | 已实现：服务器生成，本机用 gpt-6.1-sol 判分 |
| LiveCodeBench v6 | 官方仓库有，执行测试用例 | 否（需沙箱执行代码） | 已实现 `benchmarks/livecodebench/` |
| BFCL v4 | 官方 `bfcl-eval` 包 | web search 需 SerpAPI + 外网 | 已实现 `benchmarks/bfcl/` |

## GPQA Diamond

- **官方仓库** [idavidrein/gpqa](https://github.com/idavidrein/gpqa/blob/main/baselines/utils.py)：`random.seed(0)` 后逐行 `random.shuffle([Inc1, Inc2, Inc3, Correct])`；抽取依次试 `answer is \((.)\)` … 最后 `\((.)\)`（很宽松），抽不到判错。
- **主流（推理模型）** [OpenAI simple-evals `gpqa_eval.py`](https://github.com/openai/simple-evals/blob/main/gpqa_eval.py)：
  - 提示词：`Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.` + 题目 + `A) … D) …`
  - 选项：基础顺序 `[Correct, Inc1, Inc2, Inc3]`，`rng = random.Random(0)`，题目复制 `n_repeats` 份（默认 4）后逐条 `rng.sample(range(4), 4)` 重排，每次重复换一个排列。
  - 抽取：`(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?`，`re.search` 取第一个，无匹配判 0。
- lm-eval-harness 默认贪心解码，不适合思考模型；lighteval `gpqa:diamond` 用同样提示词，但正确答案位置的随机没固定 seed。
- 厂商：Qwen3 技术报告每题采样 10 次取平均（[arXiv 2505.09388](https://arxiv.org/pdf/2505.09388)）；DeepSeek-R1 用 k=4–64 个样本平均（凭记忆，未对照原文）。
- **建议**：照搬 simple-evals 的提示词和选项排列（seed 0，CSV 行序固定）；只在最终 `content` 上抽取，取最后一个 `Answer: X`；空回复、无匹配、截断、请求失败都判错并单独计数；不随机猜。198 题单次采样波动约 ±7 个百分点，惯例是 avg@k（simple-evals 4、Qwen 10）。

## IFEval / IFBench / 两个 multiturn 子集

- **IFEval** [google-research/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)：每条有 `instruction_id_list` 和 `kwargs`，对应 checker 判定；strict 直接查原回复，loose 再试 8 个变体（去首行、去末行、去 `*` 等）任一通过即可；prompt-level 要求全部指令都通过，instruction-level 按单条统计，共 4 个数。厂商通常报 prompt-level strict。
- **IFBench** [allenai/IFBench](https://github.com/allenai/IFBench)：58 个新约束 + 25 个 IFEval checker，打分代码同 IFEval。论文主报 prompt-level loose，并明确说思考模型要去掉推理链只评最终答案。
- **multiturn**：来自 [allenai/IFBench_multi-turn](https://huggingface.co/datasets/allenai/IFBench_multi-turn)，结构是 [user 原问题, assistant 预生成回答, user 约束]，只评模型第 3 轮回复。`ifeval_multiturn` 基本可确定对应 `ifeval_constraints` 子集（1,774 条），`ifbench_multiturn` 对应 `ifbench_constraints`（1,387 条），都用 IFBench 的 checker 打分。
- **依赖**：`absl-py langdetect nltk immutabledict emoji syllapy`；NLTK 数据 `punkt`、`punkt_tab`、`stopwords`、`averaged_perceptron_tagger_eng` 要提前打包上传并设 `NLTK_DATA`。版本：按 IFBench 官方 uv.lock 用 `nltk==3.9.2`（实测 3.8.1 会把 `punkt_tab` 解析成 `punkt/PY3_tab` 导致导入失败；3.9.2 下官方 118 个单元测试全部通过）。
- **建议**：直接 vendor 官方 checker 代码，不自己重写；只评最终 `content`，单独统计空回复和截断。

## MultiChallenge

- [ekwinox117/multi-challenge](https://github.com/ekwinox117/multi-challenge)，论文 [arXiv 2501.17399](https://arxiv.org/abs/2501.17399)：每条有 `CONVERSATION`、`TARGET_QUESTION`、`PASS_CRITERIA`（YES/NO）、`AXIS`（4 个维度）。模型只生成最后一轮回复。
- 官方 judge：`gpt-4o-2024-08-06`，temperature 0，输出 `{reasoning, verdict}`；`verdict == PASS_CRITERIA` 即通过。总分是 4 个维度通过率的宏平均。
- OpenAI 在 [GPT-4.1 发布文](https://openai.com/index/gpt-4-1/) 中指出 GPT-4o judge 常误判，改用 o3-mini 并同时发布两套分数——换 judge 有先例，但分数不能和公开榜单直接比。
- **已定（2026-11-20）**：GPU 机只负责生成；本机调用官方 `Evaluator` 判分，judge 换成 `gpt-6.1-sol`（temperature 0、结构化输出），见 `scorers/multichallenge_judge.py`。换了 judge，分数只做内部对比，不和公开榜单比。

## LiveCodeBench v6

- [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)：
  - 提示词（`prompts/code_generation.py` 通用模板）：system 为 "You are an expert Python programmer…"；user 为 `### Question:` + 题面 + 格式说明（有 starter code 时给出代码框架，否则要求从 stdin 读、向 stdout 写）+ `### Answer: (use the provided format with backticks)`。仓库里的 Qwen 专用模板是 CodeQwen 时代的旧格式，不适用。
  - 代码提取：取最后两个 ``` 之间的内容，不足两个返回空串判错。
  - 执行：public + private 测试全部通过才算对；每个测试超时 6 秒；stdin 题逐行比较（必要时按 Decimal 比），call-based 题 JSON 比较。官方 `reliability_guard()` 明说不是安全沙箱，且实际没加内存限制。
  - 指标：pass@1 = 每题 c/n 的平均（无偏估计）。
- release_v6 共 1,055 题（2023-05 至 2025-04）；private 测试用例编码为 base64 → zlib → pickle → JSON。
- 日期窗口：Qwen 报 "v6 (25.02-25.05)"（约 175 题）；DeepSeek R1-0528 / V3.1 用 2408-2505、每题 16 个样本。Artificial Analysis 的窗口未核实。
- **建议**：照搬官方提示词和提取规则；执行放在断网沙箱里，每个样本单独子进程，加 4GB 内存和 CPU 限制，非 root、临时目录；主口径用 2025-02-01 至 2025-05-01（与 Qwen 可比），另报 2024-08-01 至 2025-05-01 和全量。先核对本地 snapshot 是否 1,055 行、private 测试能否解码。

## BFCL v4

- 官方流程：`pip install bfcl-eval`，设 `BFCL_PROJECT_ROOT`；`bfcl generate` 后 `bfcl evaluate`。
- **坑**：本地模型 handler（`base_oss_handler.py`）走 `/v1/completions`、自拼 chat template、输出上限写死 4096 tokens——vLLM 的工具调用解析不起作用，思考模型容易被截断。现有 Qwen handler 写死 Qwen3 的 `<tool_call>{json}</tool_call>` 格式。
- 类别与依赖：非 live、live、无关检测、多轮（本地 Python 模拟工具）、记忆（vector 后端要预下载 `all-MiniLM-L6-v2`）都可离线跑；web search 需要 `SERPAPI_API_KEY` 和外网。
- 总分：`[非 live, live, 无关检测, 多轮, agentic] × [10, 10, 10, 30, 40]`，agentic = (web search + memory) / 2；**没跑的类别按 0 计**。
- 判分：单轮用 AST checker 比对函数名和参数；多轮每轮执行调用后做状态检查和响应检查；web search 和记忆在最终答案里做子串匹配。默认 temperature 0.001。
- **建议**：
  - vLLM 加 `--enable-auto-tool-choice --tool-call-parser <按 Qwen3.8 模型卡>`。Qwen3.5/3.6 官方用 `qwen3_coder`，Qwen3.8 未核实，先查模型卡。
  - 在 `model_config.py` 加一个自定义 FC 条目，用 `OpenAICompletionsHandler` 走 chat 接口（`OPENAI_BASE_URL=http://127.0.0.1:8000/v1`），绕开 4096 上限并用上 vLLM 的解析。自定义条目只能内部对比。
  - 先跑 `non_live,live,multi_turn,memory`；报各组分数，再自算 Overall(不含 web search) = (10·非 live + 10·live + 10·无关 + 30·多轮 + 20·记忆) / 80，不要直接用官方 csv 里把 web search 当 0 的 Overall。
  - 采样温度要定：官方 0.001，和我们其他评测的 1.0 不同。

## 共同原则（与 FinEval/AIME 一致）

1. 只解析思考之后的最终 `content`，不从思考里取答案。
2. 截断、空回复、格式无效、请求失败分别计数，前三者计入分母判错，请求失败重跑补齐。
3. 不做随机猜或"取第一个字母"这类兜底。
4. temperature 1.0 下单次采样有波动，小题量评测（AIME 30、GPQA 198）的惯例是多次采样取平均；是否做由预算决定。

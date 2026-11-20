"""在本机给 MultiChallenge 回复打分：调用官方仓库的 DataLoader / Evaluator / ResultParser，judge 换成 gpt-6.1-sol。

官方仓库：ekwinox117/multi-challenge @ 5ccefcca（无 LICENSE，不拷入本项目；运行时指向一份本地克隆）。
与官方 main.py --responses-file 的流程一致，只做两处适配：
1. judge 模型由官方写死的 gpt-4o-2024-08-06 换成 --judge-model（默认 gpt-6.1-sol），仍是 temperature=0 + 结构化输出 {reasoning, verdict}；
2. 官方 Evaluator 给 OpenAIModel 传了它不接受的 max_tokens 参数（原样运行会 TypeError），这里丢弃该参数。
另外：官方 factory 会顺带 import HuggingFace 依赖（torch/transformers），这里只用 --responses-file 路径，所以直接 import 需要的模块。
API 地址与 key 只从环境变量 OPENAI_BASE_URL / OPENAI_API_KEY 读取，不写入任何文件。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OFFICIAL_COMMIT = "5ccefcca6a39020d66c1383c4e6a809cb07afa33"
QUESTIONS_SHA256 = "bb23da758fdfae7ffb6ff4ffaa1a8e1523ac411ca20eb4467c41b58e8e5b88d4"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="官方仓库本地克隆")
    parser.add_argument("--responses", type=Path, required=True, help="multichallenge_responses.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default="gpt-6.1-sol")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="调试用：只评前 N 题（其余不计入分数）")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL"):
        parser.error("须通过环境变量提供 OPENAI_API_KEY 和 OPENAI_BASE_URL")
    if not sys.flags.utf8_mode:
        # 官方代码用 open() 读文件不指定编码；Windows 默认 GBK 会读错，须以 PYTHONUTF8=1 启动。
        parser.error("请以 UTF-8 模式运行（设置环境变量 PYTHONUTF8=1 或 python -X utf8）")

    head = subprocess.run(["git", "-C", str(args.repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != OFFICIAL_COMMIT:
        raise SystemExit(f"官方仓库版本 {head!r} 不是 {OFFICIAL_COMMIT}")
    questions = args.repo / "data/benchmark_questions.jsonl"
    if hashlib.sha256(questions.read_bytes()).hexdigest() != QUESTIONS_SHA256:
        raise SystemExit("官方题目文件指纹不符")

    sys.path.insert(0, str(args.repo))
    import src.evaluator as evaluator_module
    from src.data_loader import DataLoader
    from src.models.openai import OpenAIModel
    from src.result_parser import ResultParser

    judge_model = args.judge_model

    class JudgeModel(OpenAIModel):
        """保留官方 OpenAIModel 的调用方式（beta.chat.completions.parse + temperature），只换模型名、丢弃 max_tokens。"""

        def __init__(self, model, temp, response_format=None, max_tokens=None):
            super().__init__(model=judge_model, temp=temp, response_format=response_format)

    evaluator_module.OpenAIModel = JudgeModel

    loader = DataLoader(str(questions))
    loader.load_data()
    loader.load_responses(str(args.responses))
    conversations = loader.get_conversations()[:args.limit]
    responses = loader.get_responses()
    missing = [c.question_id for c in conversations if c.question_id not in responses]

    results = evaluator_module.Evaluator(conversations, responses).evaluate(max_workers=args.workers)
    scores = ResultParser(results).calculate_scores()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loader_attempts = max(len(v) for v in responses.values())
    ResultParser(results).save_raw_output(str(args.output_dir / "multichallenge_judged_raw.csv"),
                                          conversations, responses, loader_attempts)
    (args.output_dir / "judge_results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    judge_errors = sum(1 for r in results if str(r.get("reasoning", "")).startswith("Error during evaluation"))
    summary = {"benchmark": "multichallenge", "judge_model": judge_model, "official_commit": OFFICIAL_COMMIT,
               "questions": len(conversations), "missing_responses": len(missing), "judge_errors": judge_errors,
               "overall_score": scores["overall_score"], "axis_scores": scores["axis_scores"],
               "axis_counts": dict(Counter(c.axis for c in conversations)),
               "verdicts": dict(Counter(r["verdict"] for r in results)),
               "passed": sum(1 for r in results if r["passed"])}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 与官方 main.py 输出相同格式的分数文件
    with (args.output_dir / "results.txt").open("w", encoding="utf-8") as f:
        f.write(f"Attempts: {loader_attempts} Overall Score: {scores['overall_score']:.2f}%\n\nAxis Scores:\n")
        for axis, score in scores["axis_scores"].items():
            f.write(f"{axis}: {score:.2f}%\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

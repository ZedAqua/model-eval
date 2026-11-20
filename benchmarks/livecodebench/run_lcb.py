"""LiveCodeBench（代码生成）：开启最高思考强度生成代码，用官方提取规则取代码，再在低权限沙箱里用官方代码执行测试、算 pass@1。

- 提示词：官方 lcb_runner/prompts/code_generation.py 的 OpenAIChat 风格（通用 system + 通用 user 模板），逐字照搬。
- 提取：官方 extract_code 的非 CodeLlama 分支，取最后两个 ``` 之间的内容；不足两个记为无代码（空串，照官方仍送去执行并判错）。
- 执行：官方 codegen_metrics（public + private 全部通过才算对，每个测试 6 秒，官方默认 12 个进程）。
- 沙箱：以 lcbrun 用户在临时目录执行，进程级内存上限 4GB、该用户进程数上限；服务器不允许网络命名空间，无法断网。
- 分数：全部题目的 pass@1，另报 2025-02-01 至 2025-05-01 窗口（与 Qwen 报告口径对齐）和按难度拆分。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners"))
import evalkit  # noqa: E402

BASE = Path("/root/autodl-tmp")
DATA = BASE / "datasets/baseline-local-20261119/livecodebench-v6-eval/livecodebench_v6.jsonl"
LCB_SRC = BASE / "workspace/official-src/LiveCodeBench"
SANDBOX_USER = "lcbrun"
TEST_TIMEOUT = 6
WINDOWS = {"2025-02-01..2025-05-01": ("2025-02-01", "2025-05-01")}

# 以下常量与官方 PromptConstants / get_generic_question_template_answer 逐字一致（LiveCodeBench 28fef95）。
SYSTEM_MESSAGE_GENERIC = ("You are an expert Python programmer. You will be given a question (problem specification) "
                          "and will generate a correct Python program that matches the specification and passes all tests.")
FORMATTING_MESSAGE_WITH_STARTER_CODE = ("You will use the following starter code to write the solution to the problem "
                                        "and enclose your code within delimiters.")
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, "
    "runs the algorithm and writes output to STDOUT.")


def user_prompt(problem: dict) -> str:
    prompt = f"### Question:\n{problem['question_content']}\n\n"
    if problem["starter_code"]:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{problem['starter_code']}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def extract_code(model_output: str) -> str:
    """官方 extract_code（OpenAIChat 等风格）：最后两个含 ``` 的行之间的内容，不足两个返回空串。"""
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1: indexlines[-1]])


def load_problems(path: Path, limit: int | None) -> list[dict]:
    problems = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][:limit]
    if len({p["question_id"] for p in problems}) != len(problems):
        raise ValueError("题号重复")
    return problems


def in_window(problem: dict, start: str, end: str) -> bool:
    """与官方 load_code_generation_dataset 相同：start <= contest_date <= end（日期取当天零点）。"""
    date = datetime.fromisoformat(problem["contest_date"])
    return datetime.strptime(start, "%Y-%m-%d") <= date <= datetime.strptime(end, "%Y-%m-%d")


def sandbox_score(problems: list[dict], codes: list[str], workdir: Path, processes: int) -> dict:
    """把官方源码和输入拷进沙箱用户自己的临时目录，以低权限、资源上限运行 scorers/livecodebench_sandbox_score.py。"""
    if subprocess.run(["id", SANDBOX_USER], capture_output=True).returncode != 0:
        raise RuntimeError(f"沙箱用户 {SANDBOX_USER} 不存在；先按 README 创建")
    tmp = Path(tempfile.mkdtemp(prefix="lcb-score-", dir="/tmp"))
    try:
        shutil.copytree(LCB_SRC / "lcb_runner", tmp / "src/lcb_runner")
        shutil.copy(REPO_ROOT / "scorers/livecodebench_sandbox_score.py", tmp / "lcb_score.py")
        (tmp / "in.json").write_text(json.dumps({"problems": problems, "codes": [[c] for c in codes],
                                                 "timeout": TEST_TIMEOUT}), encoding="utf-8")
        subprocess.run(["chown", "-R", f"{SANDBOX_USER}:{SANDBOX_USER}", str(tmp)], check=True)
        os.chmod(tmp, 0o700)
        command = ["runuser", "-u", SANDBOX_USER, "--",
                   "prlimit", "--as=4294967296", f"--nproc={processes * 8 + 64}",
                   "env", "-i", f"HOME={tmp}", "PATH=/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE=1",
                   sys.executable, "-I", str(tmp / "lcb_score.py"), str(tmp / "src"), str(tmp / "in.json"),
                   str(tmp / "out.json"), str(processes)]
        proc = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
        (workdir / "sandbox_score.log").write_text(proc.stdout[-200000:] + proc.stderr[-200000:], encoding="utf-8")
        if proc.returncode != 0 or not (tmp / "out.json").exists():
            raise RuntimeError(f"沙箱打分失败（退出码 {proc.returncode}），见 sandbox_score.log")
        return json.loads((tmp / "out.json").read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def summarize(problems: list[dict], scored: dict, results: dict[str, dict] | None) -> dict:
    per = [scored["per_problem"][str(i)] for i in range(len(problems))]
    out = {"expected": len(problems), "pass@1": sum(per) / len(per)}
    for name, (start, end) in WINDOWS.items():
        idx = [i for i, p in enumerate(problems) if in_window(p, start, end)]
        out[f"pass@1[{name}]"] = sum(per[i] for i in idx) / len(idx) if idx else None
        out[f"n[{name}]"] = len(idx)
    for level in ("easy", "medium", "hard"):
        idx = [i for i, p in enumerate(problems) if p["difficulty"] == level]
        out[f"pass@1[{level}]"] = sum(per[i] for i in idx) / len(idx) if idx else None
        out[f"n[{level}]"] = len(idx)
    if results is not None:
        out["no_code"] = sum(1 for p in problems if results[p["question_id"]]["status"] == "ok"
                             and not extract_code(results[p["question_id"]]["response"]))
        out |= evalkit.generation_stats(results)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0, help="第几次独立采样，种子 = 42 + 序号")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--score-processes", type=int, default=12, help="官方默认 12")
    parser.add_argument("--score-codes", type=Path, help="调试用：直接给 {question_id: 代码} 打分，不生成")
    args = parser.parse_args()
    if args.output is None and not args.verify_only:
        parser.error("正式运行须提供 --output")
    problems = load_problems(DATA, args.limit)
    config = {"schema_version": 1, "benchmark": "livecodebench_codegen", "dataset_sha256": evalkit.sha256_file(DATA),
              "selected_rows": len(problems), "limit": args.limit, "sample_index": args.sample_index,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scorer_sha256": evalkit.sha256_file(REPO_ROOT / "scorers/livecodebench_sandbox_score.py"),
              "evalkit_sha256": evalkit.sha256_file(Path(evalkit.__file__)),
              "official_source": "LiveCodeBench/LiveCodeBench 28fef95 (lcb_runner)", "test_timeout": TEST_TIMEOUT,
              "model": args.model, "base_url": evalkit.check_base_url(args.base_url),
              "sampling": evalkit.THINKING_SAMPLING | {"seed": 42 + args.sample_index}}
    if args.verify_only:
        print(json.dumps(config | {"windows": {k: sum(in_window(p, *v) for p in problems) for k, v in WINDOWS.items()}},
                         ensure_ascii=False))
        return 0
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.score_codes:
        given = json.loads(args.score_codes.read_text(encoding="utf-8"))
        problems = [p for p in problems if p["question_id"] in given]
        scored = sandbox_score(problems, [given[p["question_id"]] for p in problems], args.output, args.score_processes)
        print(json.dumps(summarize(problems, scored, None) | {"per_problem": scored["per_problem"],
                                                              "results": scored["results"]}, ensure_ascii=False))
        return 0
    items = [{"id": p["question_id"], "messages": [{"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
                                                   {"role": "user", "content": user_prompt(p)}]} for p in problems]
    try:
        results = evalkit.run_generation(items, config, args.output, args.workers, args.timeout)
    except KeyboardInterrupt:
        print("已中断；用同一命令续跑", file=sys.stderr, flush=True)
        os._exit(130)
    # 截断或空回复没有最终代码，按官方做法以空串参与执行（判错）。
    codes = [extract_code(results[p["question_id"]]["response"]) if results[p["question_id"]]["status"] == "ok" else ""
             for p in problems]
    scored = sandbox_score(problems, codes, args.output, args.score_processes)
    with (args.output / "graded.jsonl").open("w", encoding="utf-8") as f:
        for i, p in enumerate(problems):
            f.write(json.dumps({"question_id": p["question_id"], "difficulty": p["difficulty"],
                                "contest_date": p["contest_date"], "code": codes[i],
                                "test_results": scored["results"][str(i)], "metadata": scored["metadata"][i],
                                "pass@1": scored["per_problem"][str(i)]}, ensure_ascii=False) + "\n")
    summary = summarize(problems, scored, results)
    evalkit.write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
        sys.exit(f"LCB: {exc}")

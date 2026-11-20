"""MultiChallenge：在服务器上只生成最后一轮回复，输出官方 main.py --responses-file 能直接读的文件。

题目用官方仓库 ekwinox117/multi-challenge 的 data/benchmark_questions.jsonl（273 题；本地 parquet 的 266 题
是它的子集）。官方评分把缺失的题判为不通过，所以必须覆盖全部 273 题。判分由用户在本机运行官方
main.py，把 judge 模型换成 gpt-6.1 Luna；本脚本不调用任何 judge。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners"))
import evalkit  # noqa: E402

QUESTIONS = Path("/root/autodl-tmp/workspace/official-src/multichallenge_benchmark_questions.jsonl")
QUESTIONS_SHA256 = "bb23da758fdfae7ffb6ff4ffaa1a8e1523ac411ca20eb4467c41b58e8e5b88d4"


def load_items(path: Path, limit: int | None) -> list[dict]:
    """官方题目的 CONVERSATION 原样作为消息（user/assistant 交替，最后一条是 user）。"""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != QUESTIONS_SHA256:
        raise ValueError("MultiChallenge 题目文件指纹与官方仓库不符")
    items = []
    for line in raw.decode("utf-8").splitlines()[:limit]:
        row = json.loads(line)
        messages = [{"role": m["role"], "content": m["content"]} for m in row["CONVERSATION"]]
        if not messages or messages[-1]["role"] != "user" or any(m["role"] not in ("user", "assistant") for m in messages):
            raise ValueError(f"对话结构异常：{row['QUESTION_ID']}")
        items.append({"id": row["QUESTION_ID"], "axis": row["AXIS"], "messages": messages})
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0, help="第几次独立采样，种子 = 42 + 序号")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    if args.output is None and not args.verify_only:
        parser.error("正式运行须提供 --output")
    if args.sample_index < 0 or args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("sample-index 须非负，workers、limit 须为正数")
    items = load_items(args.questions, args.limit)
    config = {"schema_version": 1, "benchmark": "multichallenge", "questions_sha256": QUESTIONS_SHA256,
              "selected_rows": len(items), "limit": args.limit, "sample_index": args.sample_index,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "evalkit_sha256": evalkit.sha256_file(Path(evalkit.__file__)),
              "model": args.model, "base_url": evalkit.check_base_url(args.base_url),
              "sampling": evalkit.THINKING_SAMPLING | {"seed": 42 + args.sample_index}}
    if args.verify_only:
        axes: dict[str, int] = {}
        for item in items:
            axes[item["axis"]] = axes.get(item["axis"], 0) + 1
        print(json.dumps(config | {"axes": axes}, ensure_ascii=False))
        return 0
    os.umask(0o077)
    try:
        results = evalkit.run_generation(items, config, args.output, args.workers, args.timeout)
    except KeyboardInterrupt:
        print("已中断；用同一命令续跑", file=sys.stderr, flush=True)
        os._exit(130)
    # 官方 main.py --responses-file 的格式：每行 {"QUESTION_ID": ..., "RESPONSE": [回复, ...]}。
    # 截断或空回复按官方生成失败时的写法填入，judge 会据此判不通过。
    with (args.output / "multichallenge_responses.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            record = results[item["id"]]
            response = record["response"] if record["status"] == "ok" else (
                f"Error generating response for question_id {item['id']}: {record['status']}.\n FAIL THIS QUESTION")
            f.write(json.dumps({"QUESTION_ID": item["id"], "RESPONSE": [response]}, ensure_ascii=False) + "\n")
    summary = {"benchmark": "multichallenge", "expected": len(items), "judged_on_server": False,
               "responses_file": "multichallenge_responses.jsonl", **evalkit.generation_stats(results)}
    evalkit.write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
        sys.exit(f"MultiChallenge: {exc}")

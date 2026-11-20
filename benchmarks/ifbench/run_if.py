"""IFBench / IFBench 多轮 / IFEval 多轮：开启最高思考强度生成回复，再用 IFBench 官方 run_eval.py 打分。

打分完全交给官方代码（allenai/IFBench 的 evaluation_lib + 83 个 checker）：先 strict 后 loose，
各出 prompt-level 和 instruction-level 两个准确率。只把思考之后的最终回答交给 checker；
截断、空回复按官方做法记为空字符串，所有指令判为未遵循。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runners"))
import evalkit  # noqa: E402

BASE = Path("/root/autodl-tmp")
DATA_DIR = BASE / "datasets/baseline-local-20261119/ifeval-ifbench-multichallenge"
IFBENCH_SRC = BASE / "workspace/official-src/IFBench"
NLTK_DATA = BASE / "workspace/official-src/nltk_data"
SUBSETS = {
    # 子集名: (文件, 是否多轮, 论文/惯例的主指标)
    "ifbench": ("ifbench_test.parquet", False, "prompt_loose"),
    "ifbench-multiturn": ("ifbench_multiturn.parquet", True, "prompt_loose"),
    "ifeval-multiturn": ("ifeval_multiturn.parquet", True, "prompt_loose"),
}


def load_items(path: Path, multiturn: bool, limit: int | None) -> list[dict]:
    """读取 parquet；单轮用 prompt 作为唯一一条用户消息，多轮直接用数据里的 [user, assistant, user]。"""
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()[:limit]
    items = []
    for row in rows:
        if multiturn:
            messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
            if [m["role"] for m in messages] != ["user", "assistant", "user"]:
                raise ValueError(f"多轮消息结构异常：{row['key']}")
        else:
            messages = [{"role": "user", "content": row["prompt"]}]
        if not all(isinstance(m["content"], str) and m["content"].strip() for m in messages):
            raise ValueError(f"消息为空：{row['key']}")
        items.append({"id": str(row["key"]), "messages": messages, "official": {
            "key": row["key"], "instruction_id_list": list(row["instruction_id_list"]),
            "prompt": row["prompt"], "kwargs": [dict(k) for k in row["kwargs"]]}})
    return items


def score(items: list[dict], results: dict[str, dict], output: Path) -> dict:
    """写出官方格式的输入和回复文件，调用官方 run_eval.py，再从它的逐条结果汇总四个准确率。"""
    scoring = output / "official_scoring"
    scoring.mkdir(exist_ok=True)
    input_file, response_file = scoring / "input_data.jsonl", scoring / "qwen-responses.jsonl"
    with input_file.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item["official"], ensure_ascii=False) + "\n")
    with response_file.open("w", encoding="utf-8") as f:
        for item in items:
            response = results[item["id"]]["response"] or ""
            f.write(json.dumps({"prompt": item["official"]["prompt"], "response": response}, ensure_ascii=False) + "\n")
    env = os.environ | {"NLTK_DATA": str(NLTK_DATA), "PYTHONPATH": str(IFBENCH_SRC)}
    proc = subprocess.run([sys.executable, "run_eval.py", f"--input_data={input_file}",
                           f"--input_response_data={response_file}", f"--output_dir={scoring}"],
                          cwd=IFBENCH_SRC, env=env, capture_output=True, text=True)
    (scoring / "run_eval.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"官方 run_eval.py 失败，见 {scoring / 'run_eval.log'}")
    metrics = {}
    for mode in ("strict", "loose"):
        rows = [json.loads(l) for l in (scoring / f"qwen-eval_results_{mode}.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(rows) != len(items):
            raise RuntimeError(f"官方 {mode} 结果条数 {len(rows)} 与题数 {len(items)} 不符")
        follows = [x for r in rows for x in r["follow_instruction_list"]]
        metrics[f"prompt_{mode}"] = sum(r["follow_all_instructions"] for r in rows) / len(rows)
        metrics[f"instruction_{mode}"] = sum(follows) / len(follows)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=SUBSETS, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0, help="第几次独立采样，种子 = 42 + 序号")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-size", type=int,
                        help="均匀抽样题数：按数据文件顺序等间隔取第 floor((i+0.5)*N/n) 条，位置固定、可复现")
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    if args.output is None and not args.verify_only:
        parser.error("正式运行须提供 --output")
    if args.sample_index < 0 or args.workers < 1 or (args.limit is not None and args.limit < 1)             or (args.sample_size is not None and args.sample_size < 1) or (args.limit and args.sample_size):
        parser.error("sample-index 须非负，workers、limit、sample-size 须为正数，limit 与 sample-size 不能同时用")
    file_name, multiturn, primary = SUBSETS[args.subset]
    data_path = DATA_DIR / file_name
    items = load_items(data_path, multiturn, args.limit)
    total_rows = len(items)
    if args.sample_size is not None and args.sample_size < total_rows:
        positions = [int((i + 0.5) * total_rows / args.sample_size) for i in range(args.sample_size)]
        items = [items[i] for i in positions]
    prompts = [i["official"]["prompt"] for i in items]
    config = {"schema_version": 1, "benchmark": args.subset, "dataset_sha256": evalkit.sha256_file(data_path),
              "selected_rows": len(items), "total_rows": total_rows, "limit": args.limit,
              "sample_size": args.sample_size, "sampling_method": "evenly_spaced_floor((i+0.5)*N/n)" if args.sample_size else None,
              "sample_index": args.sample_index,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "evalkit_sha256": evalkit.sha256_file(Path(evalkit.__file__)),
              "official_scorer": "allenai/IFBench run_eval.py (see official-src/SOURCES)",
              "model": args.model, "base_url": evalkit.check_base_url(args.base_url),
              "sampling": evalkit.THINKING_SAMPLING | {"seed": 42 + args.sample_index}}
    if args.verify_only:
        print(json.dumps(config | {"duplicate_prompts": len(prompts) - len(set(prompts))}, ensure_ascii=False))
        return 0
    os.umask(0o077)
    try:
        results = evalkit.run_generation(items, config, args.output, args.workers, args.timeout)
    except KeyboardInterrupt:
        print("已中断；用同一命令续跑", file=sys.stderr, flush=True)
        os._exit(130)
    metrics = score(items, results, args.output)
    summary = {"benchmark": args.subset, "expected": len(items), "total_rows": total_rows,
               "sampled": len(items) < total_rows, "primary_metric": primary,
               "primary": metrics[primary], **metrics, **evalkit.generation_stats(results),
               # 官方按 prompt 文本对应回复；同一 prompt 出现多次时只保留最后一条回复，这里如实报出受影响的条数。
               "duplicate_prompts": len(prompts) - len(set(prompts))}
    evalkit.write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
        sys.exit(f"IF: {exc}")

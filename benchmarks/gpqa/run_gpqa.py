"""按 OpenAI simple-evals 的做法运行 GPQA Diamond：同样的提示词、选项排列和答案正则，开启最高思考强度。"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

DATA = Path("/root/autodl-tmp/datasets/baseline-local-20261119/gpqa-diamond/gpqa_diamond.csv")
DATA_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
# 以下三项逐字照搬 simple-evals（common.py 的 QUERY_TEMPLATE_MULTICHOICE、ANSWER_PATTERN_MULTICHOICE，
# gpqa_eval.py 的 random.Random(0) 排列）；改动任何一项都会让分数和 simple-evals 不可比。
QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
ANSWER_PATTERN = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
PERMUTATION_SEED = 0
SAMPLING = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 0.0, "repetition_penalty": 1.0,
    "max_completion_tokens": 61440, "reasoning_effort": "xhigh",
    "chat_template_kwargs": {"enable_thinking": True},
}
PROMPT_HEADROOM = 4096
RETRY_CODES = {408, 429, 500, 502, 503, 504}


def load_rows(path: Path, sample_index: int) -> tuple[list[dict], str]:
    """读取官方 CSV，并复现 simple-evals 第 sample_index 次重复里每题的选项排列。"""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"GPQA Diamond 文件指纹不符：{digest}")
    records = list(csv.DictReader(raw.decode("utf-8").splitlines(keepends=True)))
    if len(records) != 198:
        raise ValueError(f"GPQA Diamond 应为 198 题，实际 {len(records)}")
    # simple-evals 把题目列表复制 n 份后，按顺序为每条抽一次排列；第 r 次重复的第 i 题对应第 r*198+i 次抽取。
    rng = random.Random(PERMUTATION_SEED)
    permutations = [rng.sample(range(4), 4) for _ in range((sample_index + 1) * len(records))]
    rows = []
    for i, record in enumerate(records):
        permutation = permutations[sample_index * len(records) + i]
        base = [record["Correct Answer"], record["Incorrect Answer 1"], record["Incorrect Answer 2"], record["Incorrect Answer 3"]]
        choices = [base[j] for j in permutation]
        prompt = QUERY_TEMPLATE.format(Question=record["Question"], A=choices[0], B=choices[1], C=choices[2], D=choices[3])
        rows.append({"id": record.get("Record ID") or str(i), "index": i, "domain": record.get("High-level domain"),
                     "permutation": permutation, "reference": "ABCD"[choices.index(record["Correct Answer"])],
                     "messages": [{"role": "user", "content": prompt}]})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("题号重复")
    return rows, digest


def extract_answer(content: str | None) -> str | None:
    """与 simple-evals 相同：在最终回答里用 re.search 取第一个 "Answer: X"；不转大写，小写字母照官方判错。"""
    if not isinstance(content, str):
        return None
    match = re.search(ANSWER_PATTERN, content)
    return match.group(1) if match else None


def request_json(url: str, timeout: float, payload: dict | None = None) -> dict:
    """直接访问本机推理服务；临时故障最多尝试三次。"""
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("GPQA_API_KEY"):
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, None if payload is None else json.dumps(payload).encode(), headers)
    opener = build_opener(ProxyHandler({}))
    for attempt in range(3):
        try:
            with opener.open(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if status not in RETRY_CODES or attempt == 2:
                raise RuntimeError(f"HTTP {status}: {url}") from None
        except (URLError, TimeoutError, socket.timeout) as exc:
            if attempt == 2:
                raise RuntimeError(f"{type(exc).__name__}: {url}") from None
        time.sleep(2**attempt)
    raise AssertionError("请求重试未正常结束")


def predict(row: dict, config: dict, timeout: float) -> dict:
    """发送题面，只解析思考之后的最终回答；保存完整消息和用量。"""
    payload = {"model": config["model"], "messages": row["messages"], **config["sampling"]}
    started = time.monotonic()
    result = request_json(config["base_url"] + "/chat/completions", timeout, payload)
    choice = result["choices"][0]
    message, finish = choice["message"], choice.get("finish_reason")
    answer = extract_answer(message.get("content")) if finish == "stop" else None
    return {"id": row["id"], "domain": row["domain"], "permutation": row["permutation"],
            "reference": row["reference"], "prediction": answer, "correct": answer == row["reference"],
            "status": "scored" if answer else ("truncated" if finish == "length" else "invalid_output"),
            "message": message, "finish_reason": finish, "usage": result.get("usage"),
            "latency_seconds": round(time.monotonic() - started, 3)}


def write_json(path: Path, value: dict) -> None:
    """原子写入，避免中断时留下半个 JSON。"""
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.replace(path)


def open_run(directory: Path, config: dict, rows: list[dict]) -> dict[str, dict]:
    """校验续跑指纹，读回已完成的题。"""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest, predictions = directory / "run.json", directory / "predictions.jsonl"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError("运行参数与 run.json 不一致；请使用新的输出目录")
    elif predictions.exists():
        raise ValueError("历史预测缺少 run.json；请使用新的输出目录")
    else:
        write_json(manifest, config)
    expected = {r["id"]: r["reference"] for r in rows}
    results = {}
    if predictions.exists():
        for line in predictions.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["id"] not in expected or record["id"] in results or record["reference"] != expected[record["id"]]:
                raise ValueError(f"历史预测与本次题集不符：{record['id']}")
            results[record["id"]] = record
    return results


def summarize(rows: list[dict], results: dict[str, dict]) -> dict:
    """准确率 = 答对题数 / 全部题数，跑完才发布；按领域另报。"""
    counts = Counter(r["status"] for r in results.values())
    correct = sum(r["correct"] for r in results.values())
    complete = len(results) == len(rows)
    domains: dict[str, Counter] = {}
    for row in rows:
        bucket = domains.setdefault(row["domain"] or "unknown", Counter())
        bucket["expected"] += 1
        if row["id"] in results:
            bucket["completed"] += 1
            bucket["correct"] += results[row["id"]]["correct"]
    lengths = sorted((r.get("usage") or {}).get("completion_tokens") or 0 for r in results.values())
    return {"expected": len(rows), "completed": len(results), "correct": correct, "complete": complete,
            "accuracy": correct / len(rows) if complete else None,
            "invalid_output": counts["invalid_output"], "truncated": counts["truncated"],
            "completion_tokens": {"mean": round(sum(lengths) / len(lengths)), "p50": lengths[len(lengths) // 2],
                                  "max": lengths[-1]} if lengths else None,
            "by_domain": {d: dict(b) | {"accuracy": b["correct"] / b["expected"] if b["completed"] == b["expected"] else None}
                          for d, b in sorted(domains.items())}}


def run(rows: list[dict], config: dict, args: argparse.Namespace) -> dict:
    """保持固定数量的请求在途，完成一题写一题；失败题记入 request_errors.jsonl，原命令重跑补齐。"""
    results = open_run(args.output, config, rows)
    pending = [r for r in rows if r["id"] not in results]
    failures = 0
    try:
        if pending:
            available = request_json(config["base_url"] + "/models", 60)
            model = next((m for m in available["data"] if m["id"] == config["model"]), None)
            if model is None:
                raise ValueError(f"服务未注册模型 {config['model']!r}")
            need = config["sampling"]["max_completion_tokens"] + PROMPT_HEADROOM
            if isinstance(model.get("max_model_len"), int) and model["max_model_len"] < need:
                raise ValueError(f"服务总上下文 {model['max_model_len']} 小于所需 {need}")
            pool = ThreadPoolExecutor(args.workers)
            jobs = {pool.submit(predict, row, config, args.timeout): row for row in pending}
            try:
                with (args.output / "predictions.jsonl").open("a", encoding="utf-8") as stream:
                    for job in as_completed(jobs):
                        try:
                            record = job.result()
                        except (RuntimeError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                            failures += 1
                            with (args.output / "request_errors.jsonl").open("a", encoding="utf-8") as errors:
                                errors.write(json.dumps({"id": jobs[job]["id"], "error_type": type(exc).__name__, "error": str(exc),
                                                         "utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
                            continue
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        results[record["id"]] = record
                        print(f"completed {len(results)}/{len(rows)} failed {failures}", file=sys.stderr, flush=True)
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
    finally:
        summary = summarize(rows, results)
        write_json(args.output / "summary.json", summary)
    if failures:
        raise RuntimeError(f"{failures} 题请求失败，见 request_errors.jsonl；用同一命令重跑即可补齐")
    return summary


def main() -> int:
    """每次独立采样单独一个输出目录；种子 = 42 + 序号，选项排列 = simple-evals 第 序号 次重复。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.query or url.fragment:
        parser.error("base-url 须为无凭据、查询参数的 HTTP(S) API 地址")
    if args.sample_index < 0 or args.workers < 1 or (args.limit is not None and args.limit < 1) or not 0 < args.timeout < float("inf"):
        parser.error("sample-index 须非负，workers、limit、timeout 须为正数")
    if args.output is None and not args.verify_only:
        parser.error("正式运行须提供 --output")
    rows, digest = load_rows(args.data, args.sample_index)
    rows = rows[:args.limit]
    config = {"schema_version": 1, "benchmark": "gpqa_diamond", "dataset_sha256": digest, "selected_rows": len(rows),
              "limit": args.limit, "sample_index": args.sample_index, "permutation_seed": PERMUTATION_SEED,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model": args.model, "base_url": args.base_url.rstrip("/"),
              "prompt_template": QUERY_TEMPLATE, "answer_pattern": ANSWER_PATTERN,
              "scoring": "simple_evals_first_answer_letter_final_content_stop_only",
              "sampling": SAMPLING | {"seed": 42 + args.sample_index}}
    if args.verify_only:
        print(json.dumps({k: v for k, v in config.items() if k != "prompt_template"}, ensure_ascii=False))
        print(rows[0]["messages"][0]["content"][:600], file=sys.stderr)
        return 0
    os.umask(0o077)
    try:
        summary = run(rows, config, args)
    except KeyboardInterrupt:
        print("已中断；用同一命令续跑", file=sys.stderr, flush=True)
        os._exit(130)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_domain"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
        sys.exit(f"GPQA: {exc}")

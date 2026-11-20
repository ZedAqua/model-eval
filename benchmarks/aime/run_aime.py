"""以最高思考强度运行 AutoDL 上冻结的 AIME 2025/2026 测试集，记录逐题回复、判分和断点。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

DATA = Path("/root/autodl-tmp/datasets/baseline-local-20261119/aime-eval-2025-2026")
PROMPT = "Solve the following competition math problem. Put the final answer (an integer from 0 to 999) on the last line as \\boxed{NNN}.\n\n"
NUMBER = re.compile(r"[0-9]{1,3}")
# 只解析最终回答里最后一个 \boxed{...}；里面须是 1–3 位整数，否则无效，不再往前找别的方框。
BOXED = re.compile(r"\\boxed\s*\{\s*([0-9]{1,3})\s*\}", flags=re.ASCII)
RETRY = {408, 429, 500, 502, 503, 504}
PROMPT_HEADROOM = 4096  # 总上下文里留给题面和模板的 tokens


def load_dataset(directory: Path, year: str) -> tuple[list[dict], dict]:
    """按快照清单校验题集指纹、30 道题的题号和整数答案。"""
    name = f"aime{year[-2:]}-test.jsonl"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    matches = [source for source in manifest["sources"] if source["file"] == name]
    if len(matches) != 1:
        raise ValueError(f"快照清单找不到唯一条目: {name}")
    source = matches[0]
    raw = (directory / name).read_bytes()
    if source["split"] != "test" or source["sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"题集分片或 SHA256 与快照不符: {name}")
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").split("\n") if line.strip()]
    if len(rows) != 30 or len(rows) != source["records"]:
        raise ValueError(f"题量与快照不符: {name}")
    ids = set()
    for row in rows:
        item_id = row["id"]
        if (type(item_id) not in (str, int) or item_id == "" or item_id in ids
                or not isinstance(row["problem"], str) or not row["problem"].strip()
                or not NUMBER.fullmatch(str(row["answer"]))):
            raise ValueError(f"题号、题面或答案无效: {item_id!r}")
        ids.add(item_id)
    return rows, source


def extract_answer(content: str | None) -> int | None:
    """只读取思考之后的最终回复：整段是一个整数，或取最后一个 \\boxed{} 里的整数；不从推理中取数字。"""
    if not isinstance(content, str):
        return None
    final = unicodedata.normalize("NFKC", content).strip()
    if not final or "<think>" in final or "</think>" in final:
        return None
    if NUMBER.fullmatch(final):
        return int(final)
    position = final.rfind("\\boxed")
    match = BOXED.match(final, position) if position >= 0 else None
    return int(match.group(1)) if match else None


def request_json(url: str, timeout: float, payload: dict | None = None) -> dict:
    """直接请求本机服务；临时网络和 HTTP 错误最多尝试三次。"""
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("AIME_API_KEY"):
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, None if payload is None else json.dumps(payload).encode(), headers)
    opener = build_opener(ProxyHandler({}))
    for attempt in range(3):
        try:
            with opener.open(req, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if status not in RETRY or attempt == 2:
                raise RuntimeError(f"HTTP {status}: {url}") from None
        except (URLError, TimeoutError) as exc:
            if attempt == 2:
                raise RuntimeError(f"{type(exc).__name__}: {url}") from None
        time.sleep(2**attempt)
    raise AssertionError("请求重试未正常结束")


def predict(row: dict, config: dict, timeout: float) -> dict:
    """只发送题面；保存完整消息、结束原因和用量，并用本地参考答案判分。"""
    payload = {"model": config["model"], "messages": [{"role": "user", "content": PROMPT + row["problem"]}], **config["sampling"]}
    started = time.monotonic()
    result = request_json(config["base_url"] + "/chat/completions", timeout, payload)
    choice = result["choices"][0]
    message = choice["message"]
    finish = choice["finish_reason"]
    answer = extract_answer(message.get("content")) if finish == "stop" else None
    return {"id": row["id"], "reference": int(row["answer"]), "prediction": answer,
            "correct": answer == int(row["answer"]),
            "status": "truncated" if finish == "length" else "invalid_output" if answer is None else "scored",
            "message": message, "finish_reason": finish, "usage": result.get("usage"),
            "latency_seconds": round(time.monotonic() - started, 3)}


def write_json(path: Path, value: dict) -> None:
    """原子更新运行配置和摘要，避免进程退出时留下半份 JSON。"""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_run(directory: Path, config: dict, rows: list[dict]) -> dict:
    """检查运行指纹及已写入的逐题结果，防止将不同设置混入同一批次。"""
    manifest, predictions = directory / "run.json", directory / "predictions.jsonl"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError("运行配置已变化，请使用新的输出目录")
    elif predictions.exists():
        raise ValueError("已有预测但缺少运行配置，请使用新的输出目录")
    else:
        write_json(manifest, config)
    history = {}
    expected = {row["id"]: int(row["answer"]) for row in rows}
    if predictions.exists():
        with predictions.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                item_id = record["id"]
                if item_id not in expected or item_id in history or record["reference"] != expected[item_id]:
                    raise ValueError(f"历史预测与本次题集不符: {item_id}")
                history[item_id] = record
    return history


def run(rows: list[dict], config: dict, directory: Path, timeout: float, workers: int) -> dict:
    """保持固定数量的请求在途，完成一题写一题；失败题记入 request_errors.jsonl，原命令重跑时补上。"""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = load_run(directory, config, rows)
        failures = 0
        try:
            pending = [row for row in rows if row["id"] not in results]
            if pending:
                service = request_json(config["base_url"] + "/models", 60)
                model = next((m for m in service["data"] if m["id"] == config["model"]), None)
                if model is None:
                    raise ValueError(f"服务未注册模型 {config['model']!r}，检查 /v1/models")
                need = config["sampling"]["max_completion_tokens"] + PROMPT_HEADROOM
                if isinstance(model.get("max_model_len"), int) and model["max_model_len"] < need:
                    raise ValueError(f"服务总上下文 {model['max_model_len']} 小于所需 {need}；请用 QWEN_MAX_MODEL_LEN=65536 启动")
                pool = ThreadPoolExecutor(workers)
                jobs = {pool.submit(predict, row, config, timeout): row for row in pending}
                try:
                    with (directory / "predictions.jsonl").open("a", encoding="utf-8") as stream:
                        for job in as_completed(jobs):
                            try:
                                record = job.result()
                            except (RuntimeError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                                failures += 1
                                with (directory / "request_errors.jsonl").open("a", encoding="utf-8") as errors:
                                    errors.write(json.dumps({"id": jobs[job]["id"], "error_type": type(exc).__name__, "error": str(exc), "utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
                                continue
                            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                            stream.flush()
                            results[record["id"]] = record
                            print(f"completed {len(results)}/{len(rows)} failed {failures}", file=sys.stderr, flush=True)
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
        finally:
            counts = Counter(record["status"] for record in results.values())
            correct = sum(record["correct"] for record in results.values())
            complete = len(results) == len(rows)
            lengths = sorted((record.get("usage") or {}).get("completion_tokens") or 0 for record in results.values())
            summary = {"expected": len(rows), "completed": len(results), "correct": correct,
                       "complete": complete, "accuracy": correct / len(rows) if complete else None,
                       "truncated": counts["truncated"], "invalid_output": counts["invalid_output"],
                       "completion_tokens": {"mean": round(sum(lengths) / len(lengths)), "p50": lengths[len(lengths) // 2],
                                             "max": lengths[-1]} if lengths else None}
            write_json(directory / "summary.json", summary)
        if failures:
            raise RuntimeError(f"{failures} 题请求失败，见 request_errors.jsonl；用同一命令重跑即可补齐")
        return summary


def main() -> None:
    """选择年份、固定采样设置；无卡校验只验数据和显示配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", choices=("2025", "2026"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--limit", type=int, choices=range(1, 31), metavar="1..30")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="xhigh")
    parser.add_argument("--max-completion-tokens", type=int, default=61440, help="含思考；服务须以 64K 总上下文启动")
    parser.add_argument("--timeout", type=float, default=7200, help="单次请求秒数")
    parser.add_argument("--workers", type=int, default=30, help="在途请求数，不超过服务的 QWEN_MAX_NUM_SEQS")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="第几次独立采样（种子 = 42 + 序号）；每次用单独的输出目录，最后按序号汇总取平均")
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.hostname or url.username is not None or url.query or url.fragment:
        parser.error("base-url 须为无凭据、查询参数或片段的 HTTP(S) API 地址")
    if args.max_completion_tokens < 1 or args.workers < 1 or args.sample_index < 0 or not 0 < args.timeout < float("inf"):
        parser.error("输出上限、并发和超时须为有限正数")
    if args.output is None and not args.verify_only:
        parser.error("正式运行须提供 --output")
    os.umask(0o077)
    rows, source = load_dataset(args.data_dir, args.year)
    rows = rows[:args.limit]
    config = {"schema_version": 4, "year": args.year, "dataset": source, "selected_rows": len(rows),
              "limit": args.limit, "sample_index": args.sample_index, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model": args.model, "base_url": args.base_url.rstrip("/"), "prompt_prefix": PROMPT,
              "scoring": "final_content_integer_or_last_boxed_integer_stop_only",
              "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                           "presence_penalty": 0.0, "repetition_penalty": 1.0, "seed": 42 + args.sample_index,
                           "max_completion_tokens": args.max_completion_tokens,
                           "reasoning_effort": args.reasoning_effort, "chat_template_kwargs": {"enable_thinking": True}}}
    if args.verify_only:
        print(json.dumps(config, ensure_ascii=False))
    else:
        try:
            print(json.dumps(run(rows, config, args.output, args.timeout, args.workers), ensure_ascii=False))
        except KeyboardInterrupt:
            # 在途的长请求不等它结束；已完成的题和摘要都已写盘，原命令重跑即可续上。
            print("已中断；用同一命令续跑", file=sys.stderr, flush=True)
            os._exit(130)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, IndexError, TypeError) as exc:
        raise SystemExit(f"AIME: {exc}") from None

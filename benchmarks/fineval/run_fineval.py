"""以最高思考强度运行有标签的 FinEval 四选一评测，保留逐题思考、回答和可复现的运行参数。"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

LETTERS = tuple("ABCD")
# 只测开启思考后的能力：思考强度用模型支持的最高档 xhigh，采样参数取模型卡推荐的思考模式设置。
# 输出上限含思考内容；服务须以 64K 总上下文启动，余下 4K 留给题面和模板。
SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": 42,
    "max_completion_tokens": 61440,
    "reasoning_effort": "xhigh",
    "chat_template_kwargs": {"enable_thinking": True},
}
PROMPT_HEADROOM = 4096
# 只在思考之后的最终回答里找答案：先看开头，再看最后一行；两处给出不同字母视为无效。
# 可带"答案/最终答案/Answer"前缀、括号或星号；字母后只能是结尾或标点，
# 不接受字母后跟空格（避免把英文冠词 "A company..." 当成选项），也不接受逗号（避免 "A, B" 多选）。
LENIENT = re.compile(
    r"^(?:(?:最终|正确)?答案(?:是|为|应为)?\s*[:：]?\s*|(?:the\s+)?(?:final\s+|correct\s+)?answer(?:\s+is)?\s*[:：]?\s*)?"
    r"[*(（\[]*([A-D])[*)）\]]*(?=$|[.:：、。;；\n])",
    re.IGNORECASE,
)
BOXED = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)
# 参考 SUFE FinEval 官方 extract_answer 的显式作答写法（"所以答案是B。""选B""答案是选项 B""B是正确的"等），
# 在整段最终回答里查找。字母须为大写且后面不紧跟英文字母，避免把 "AIDA" 之类的词当成选项。
# 官方的兜底规则（取全文第一个字母、匹配选项原文、随机猜）不采用。
_L = r"[*(\[]*([A-D])(?![A-Za-z])"
EXPLICIT = [re.compile(p) for p in (
    r"所以答案是\s*" + _L,
    r"(?:最终|正确)?答案(?:应该是|应为|是|为|选)?\s*:?\s*(?:选项|选)?\s*" + _L,
    r"选择(?:答案|选项)?\s*" + _L,
    # "选B"须在句首、标点后或"因此/所以/故/应"后；"选项A……"常是逐项分析，只在后接"正确"时计入。
    r"(?m)(?:^|[,，。:;；\s]|因此|所以|故|应该|应)[*]*选\s*" + _L,
    r"选项\s*" + _L + r"[*)\]]*\s*(?:是)?正确",
    r"(?<![A-Za-z])([A-D])[*)\]]*\s*(?:项|选项)?\s*是正确的",
    r"正确的(?:一项|选项)是\s*(?:选项)?\s*" + _L,
    r"(?i:\banswer\s+is)\s*:?\s*" + _L,
)]
DEFAULT_DATA = Path("/root/autodl-tmp/workspace/fineval-scored-20261119")
RETRY_CODES = {408, 429, 500, 502, 503, 504}


def load_dataset(path: Path) -> tuple[list[dict], str]:
    """校验单一子集、唯一题号和 A–D 标签；返回数据与文件指纹。"""
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").split("\n") if line.strip()]
    ids = set()
    tasks = set()
    for row in rows:
        item_id, task = row.get("id"), row.get("task")
        messages = row.get("messages")
        if not isinstance(item_id, str) or not item_id or item_id in ids:
            raise ValueError(f"题号为空或重复：{item_id!r}")
        if task not in {"salesforce_fineval", "sufe_val"} or row.get("reference") not in LETTERS:
            raise ValueError(f"不允许无标签或非评分样本：{item_id}")
        if not messages or messages[-1].get("role") != "user" or any(
            not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages
        ):
            raise ValueError(f"消息格式无效：{item_id}")
        tasks.add(task)
        ids.add(item_id)
    if len(tasks) != 1:
        raise ValueError("每次运行必须使用一个非空的 FinEval 子集")
    return rows, hashlib.sha256(raw).hexdigest()


def extract_answer(content: str | None, strict: bool = False) -> str | None:
    """严格口径只接受单个字母。主口径汇总三处线索：最终回答开头的字母、最后一行的字母或 \\boxed{X}、
    全文的显式作答写法；恰好指向一个字母才计分，指向多个字母视为无效。"""
    if not isinstance(content, str):
        return None
    text = unicodedata.normalize("NFKC", content).strip()
    if strict:
        return text.upper() if text.upper() in LETTERS else None
    if not text:
        return None
    last = text.splitlines()[-1]
    found = set()
    for candidate in (text, last):
        # 去掉 Markdown 的标题、引用、列表和加粗前缀，如 "**答案：B**"、"- B."。
        candidate = candidate.strip().lstrip("#>-* ").strip()
        if match := LENIENT.match(candidate):
            found.add(match.group(1).upper())
    if match := BOXED.search(last):
        found.add(match.group(1).upper())
    for pattern in EXPLICIT:
        found.update(m.group(1) for m in pattern.finditer(text))
    return found.pop() if len(found) == 1 else None


def request_json(url: str, timeout: float, payload: dict | None = None) -> dict:
    """直接访问推理服务；短暂网络故障有限重试，配置错误立即中止。"""
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("FINEVAL_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(payload).encode() if payload is not None else None
    opener = build_opener(ProxyHandler({}))
    request = Request(url, data=body, headers=headers)
    for attempt in range(3):
        try:
            with opener.open(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if status not in RETRY_CODES or attempt == 2:
                raise RuntimeError(f"推理服务返回 HTTP {status}：{url}") from None
        except (URLError, TimeoutError, socket.timeout) as exc:
            if attempt == 2:
                raise RuntimeError(f"推理服务连接失败：{url} ({type(exc).__name__})") from None
        time.sleep(2**attempt)
    raise AssertionError("请求重试未正确结束")


def predict(row: dict, config: dict, timeout: float) -> dict:
    """仅把消息发送给模型，随后用本地答案判分；保留原始回答与结束原因。"""
    payload = {"model": config["model"], "messages": row["messages"], **config["sampling"]}
    start = time.monotonic()
    completion = request_json(config["base_url"] + "/chat/completions", timeout, payload)
    choice = completion["choices"][0]
    message = choice["message"]
    content = message.get("content")
    finish = choice.get("finish_reason")
    prediction = extract_answer(content) if finish == "stop" else None
    strict = extract_answer(content, strict=True) if finish == "stop" else None
    return {
        "id": row["id"],
        "subject": row.get("subject"),
        "reference": row["reference"],
        "prediction": prediction,
        "status": "ok" if prediction else ("truncated" if finish == "length" else "invalid_output"),
        "correct": prediction == row["reference"],
        "strict_prediction": strict,
        "strict_correct": strict == row["reference"],
        "message": message,
        "finish_reason": finish,
        "usage": completion.get("usage"),
        "latency_seconds": round(time.monotonic() - start, 3),
    }


def write_json(path: Path, value: dict) -> None:
    """原子替换运行摘要，确保中断时不留下半个 JSON。"""
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.replace(path)


def open_run(directory: Path, config: dict, rows: list[dict]) -> tuple[Path, dict[str, dict]]:
    """校验续跑指纹和历史记录；拒绝把不同数据、模型或参数混在同一个目录。"""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest = directory / "run.json"
    predictions = directory / "predictions.jsonl"
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
        with predictions.open(encoding="utf-8") as stream:
            for line in stream:
                result = json.loads(line)
                item_id = result["id"]
                if item_id not in expected or item_id in results or result["reference"] != expected[item_id]:
                    raise ValueError(f"历史预测与本次数据不匹配：{item_id}")
                results[item_id] = result
    return predictions, results


def token_stats(results: dict[str, dict]) -> dict | None:
    """统计生成长度（含思考），用于估算全量耗时和检查上限是否够用。"""
    values = sorted((r.get("usage") or {}).get("completion_tokens") or 0 for r in results.values())
    if not values:
        return None
    def pick(q: float) -> int:
        return values[min(len(values) - 1, int(q * len(values)))]
    return {"mean": round(sum(values) / len(values)), "p50": pick(0.5), "p95": pick(0.95), "max": values[-1]}


def summarize(rows: list[dict], results: dict[str, dict]) -> dict:
    """单独报告未完成、格式错误和截断；只有跑完整个选定集合才发布准确率。"""
    counts = Counter(r["status"] for r in results.values())
    correct = sum(r["correct"] for r in results.values())
    strict_correct = sum(r["strict_correct"] for r in results.values())
    subject_counts = {}
    for row in rows:
        bucket = subject_counts.setdefault(row.get("subject") or "unknown", Counter())
        bucket["expected"] += 1
        if row["id"] in results:
            bucket["completed"] += 1
            bucket["correct"] += results[row["id"]]["correct"]
    complete = len(results) == len(rows)
    return {
        "expected": len(rows), "completed": len(results), "correct": correct,
        "complete": complete, "accuracy": correct / len(rows) if complete else None,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_correct / len(rows) if complete else None,
        "lenient_only": sum(r["prediction"] is not None and r["strict_prediction"] is None
                            for r in results.values()),
        "invalid_output": counts["invalid_output"], "truncated": counts["truncated"],
        "completion_tokens": token_stats(results),
        "by_subject": {
            subject: dict(bucket) | {"accuracy": bucket["correct"] / bucket["expected"]
                                    if bucket["completed"] == bucket["expected"] else None}
            for subject, bucket in sorted(subject_counts.items())
        },
    }


def check_service(config: dict, timeout: float) -> None:
    """确认模型名已注册，且服务总上下文装得下输出上限加题面。"""
    available = request_json(config["base_url"] + "/models", timeout)
    model = next((m for m in available["data"] if m["id"] == config["model"]), None)
    if model is None:
        raise ValueError(f"服务未注册模型 {config['model']!r}，请核对 /v1/models")
    limit = model.get("max_model_len")
    need = config["sampling"]["max_completion_tokens"] + PROMPT_HEADROOM
    if isinstance(limit, int) and limit < need:
        raise ValueError(f"服务总上下文 {limit} 小于所需 {need}；请用 QWEN_MAX_MODEL_LEN=65536 启动")


def run(rows: list[dict], config: dict, args: argparse.Namespace) -> dict:
    """保持固定数量的请求在途，完成一题写一题；失败题记入 request_errors.jsonl，原命令重跑时补上。"""
    path, results = open_run(args.output, config, rows)
    pending = [r for r in rows if r["id"] not in results]
    failures = 0
    try:
        if pending:
            check_service(config, 60)
            pool = ThreadPoolExecutor(args.workers)
            jobs = {pool.submit(predict, row, config, args.timeout): row for row in pending}
            try:
                with path.open("a", encoding="utf-8") as stream:
                    for job in as_completed(jobs):
                        try:
                            record = job.result()
                        except (RuntimeError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                            failures += 1
                            with (args.output / "request_errors.jsonl").open("a", encoding="utf-8") as errors:
                                errors.write(json.dumps({
                                    "id": jobs[job]["id"], "error_type": type(exc).__name__, "error": str(exc),
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


def parse_args() -> argparse.Namespace:
    """解析一次只跑一个子集的命令行，默认连接服务器本机的 vLLM。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("salesforce", "sufe-val"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=32, help="须与服务的 QWEN_MAX_NUM_SEQS 一致")
    parser.add_argument("--timeout", type=float, default=7200, help="单次请求秒数；思考可长达 6 万 tokens")
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.workers < 1 or not 0 < args.timeout < float("inf"):
        parser.error("limit、workers 和 timeout 必须是正数")
    url = urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.query or url.fragment:
        parser.error("base-url 必须是无查询参数和凭据的 HTTP(S) API 根地址")
    if args.output is None and not args.verify_only:
        parser.error("非验证模式必须设置 --output")
    return args


def main() -> int:
    """加载数据并冻结本次的评分口径；校验模式不请求网络或创建运行目录。"""
    args = parse_args()
    dataset = args.data_dir / f"{args.subset}-eval.jsonl"
    rows, digest = load_dataset(dataset)
    config = {
        "schema_version": 4, "subset": args.subset, "dataset_sha256": digest,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_rows": len(rows), "selected_rows": len(rows[:args.limit]),
        "model": args.model, "base_url": args.base_url.rstrip("/"), "sampling": SAMPLING,
        "scoring": "thinking_final_answer_leading_last_line_or_explicit_abcd_stop_only",
        "secondary_scoring": "strict_single_abcd_nfkc_upper_stop_only", "limit": args.limit,
    }
    if args.verify_only:
        print(json.dumps(config, ensure_ascii=False))
        return 0
    os.umask(0o077)
    try:
        summary = run(rows[:args.limit], config, args)
    except KeyboardInterrupt:
        # 在途的长请求不等它结束；已完成的题和摘要都已写盘，原命令重跑即可续上。
        print("已中断；用同一命令续跑", file=sys.stderr, flush=True)
        os._exit(130)
    print(json.dumps({key: value for key, value in summary.items() if key != "by_subject"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
        sys.exit(f"FinEval: {exc}")

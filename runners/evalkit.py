"""对话评测共用的生成逻辑：开启最高思考强度请求本机 vLLM，逐题落盘、断点续跑、失败题单独记录。

各评测脚本只负责读数据、组织消息和调用官方打分；生成部分统一走这里，保证采样参数和记录格式一致。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

# 与 FinEval/AIME/GPQA 相同：模型卡推荐的思考模式采样参数 + 最高思考强度；种子按采样序号另加。
THINKING_SAMPLING = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 0.0, "repetition_penalty": 1.0,
    "max_completion_tokens": 61440, "reasoning_effort": "xhigh",
    "chat_template_kwargs": {"enable_thinking": True},
}
PROMPT_HEADROOM = 4096
RETRY_CODES = {408, 429, 500, 502, 503, 504}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_base_url(value: str) -> str:
    url = urlsplit(value)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.query or url.fragment:
        raise ValueError("base-url 须为无凭据、查询参数的 HTTP(S) API 地址")
    return value.rstrip("/")


def request_json(url: str, timeout: float, payload: dict | None = None) -> dict:
    """直接访问本机推理服务（不走代理）；临时故障最多尝试三次。"""
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("EVAL_API_KEY"):
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


def write_json(path: Path, value) -> None:
    """原子写入，避免中断时留下半个 JSON。"""
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staging.replace(path)


def generate_one(item: dict, config: dict, timeout: float) -> dict:
    """发送完整多轮消息，只把思考之后的最终回答当作模型回复；保存完整消息和用量。"""
    payload = {"model": config["model"], "messages": item["messages"], **config["sampling"]}
    started = time.monotonic()
    result = request_json(config["base_url"] + "/chat/completions", timeout, payload)
    choice = result["choices"][0]
    message, finish = choice["message"], choice.get("finish_reason")
    content = message.get("content")
    if finish == "length":
        status = "truncated"
    elif finish != "stop" or not isinstance(content, str) or not content.strip():
        status = "empty_or_abnormal"
    else:
        status = "ok"
    return {"id": item["id"], "status": status, "finish_reason": finish,
            "response": content if status == "ok" else None, "message": message,
            "usage": result.get("usage"), "latency_seconds": round(time.monotonic() - started, 3)}


def run_generation(items: list[dict], config: dict, output: Path, workers: int, timeout: float) -> dict[str, dict]:
    """生成全部回复：校验续跑指纹，保持固定并发在途，完成一题写一题；返回 {id: 记录}。

    失败题记入 request_errors.jsonl 后继续，最后抛错提示用同一命令补齐。
    """
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest, predictions = output / "run.json", output / "predictions.jsonl"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError("运行参数与 run.json 不一致；请使用新的输出目录")
    elif predictions.exists():
        raise ValueError("历史预测缺少 run.json；请使用新的输出目录")
    else:
        write_json(manifest, config)
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("题号重复")
    results: dict[str, dict] = {}
    if predictions.exists():
        for line in predictions.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["id"] not in set(ids) or record["id"] in results:
                raise ValueError(f"历史预测与本次题集不符：{record['id']}")
            results[record["id"]] = record
    pending = [item for item in items if item["id"] not in results]
    if not pending:
        return results
    available = request_json(config["base_url"] + "/models", 60)
    model = next((m for m in available["data"] if m["id"] == config["model"]), None)
    if model is None:
        raise ValueError(f"服务未注册模型 {config['model']!r}")
    need = config["sampling"]["max_completion_tokens"] + PROMPT_HEADROOM
    if isinstance(model.get("max_model_len"), int) and model["max_model_len"] < need:
        raise ValueError(f"服务总上下文 {model['max_model_len']} 小于所需 {need}")
    failures = 0
    pool = ThreadPoolExecutor(workers)
    jobs = {pool.submit(generate_one, item, config, timeout): item for item in pending}
    try:
        with predictions.open("a", encoding="utf-8") as stream:
            for job in as_completed(jobs):
                try:
                    record = job.result()
                except (RuntimeError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                    failures += 1
                    with (output / "request_errors.jsonl").open("a", encoding="utf-8") as errors:
                        errors.write(json.dumps({"id": jobs[job]["id"], "error_type": type(exc).__name__, "error": str(exc),
                                                 "utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
                    continue
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                results[record["id"]] = record
                print(f"completed {len(results)}/{len(items)} failed {failures}", file=sys.stderr, flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if failures:
        raise RuntimeError(f"{failures} 题请求失败，见 request_errors.jsonl；用同一命令重跑即可补齐")
    return results


def generation_stats(results: dict[str, dict]) -> dict:
    """生成状态计数和长度统计（含思考）。"""
    statuses: dict[str, int] = {}
    for record in results.values():
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    lengths = sorted((r.get("usage") or {}).get("completion_tokens") or 0 for r in results.values())
    return {"status_counts": statuses,
            "completion_tokens": {"mean": round(sum(lengths) / len(lengths)), "p50": lengths[len(lengths) // 2],
                                  "max": lengths[-1]} if lengths else None}

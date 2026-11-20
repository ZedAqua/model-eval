"""BFCL v4：用官方 bfcl-eval 包生成和打分，模型接入方式照搬官方 Qwen3 条目，只把服务地址换成本机 vLLM。

- 官方包 bfcl-eval 2026.3.23 自带数据与 checker（不用本地 2025-07-17 版数据，两者内容不同，以包内为准）。
- 官方 Qwen3 FC 条目用 QwenAPIHandler：流式、enable_thinking=True、parallel_tool_calls=True、不传 temperature、
  多轮时把上一轮思考以 reasoning_content 回填；underscore_to_dot=True。这里继承它，只做两处适配：
  1) 客户端指向本机 vLLM（官方写死 DashScope）；2) vLLM 流式增量里的思考字段名是 reasoning，补一个 reasoning_content 别名。
- vLLM 须带 --enable-auto-tool-choice --tool-call-parser qwen3_coder 启动（Qwen3.8 模板是 XML 工具调用格式）。
- 默认跑官方可离线的全部计分类别：non_live、live、multi_turn、memory；web_search 需要 SerpAPI 和外网，不跑。
  官方 Overall 把没跑的类别按 0 计，所以另报不含 web_search 的总分 = 官方 Overall / 0.8，即
  (10·非 live + 10·live + 10·无关检测 + 30·多轮 + 20·记忆) / 80。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

BASE = Path("/root/autodl-tmp")
REGISTRY_NAME = "qwen3.8-27b-local-FC"
DEFAULT_CATEGORIES = ["non_live", "live", "multi_turn", "memory"]
HF_CACHE = BASE / "cache/hf"


def register(base_url: str, served_name: str) -> None:
    """在官方注册表里加一个条目；字段与官方 qwen3-*-FC 条目一致，只换 handler。"""
    from openai import OpenAI
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.qwen import QwenAPIHandler

    def alias_reasoning(stream):
        for chunk in stream:
            for choice in chunk.choices or []:
                delta = choice.delta
                if getattr(delta, "reasoning_content", None) is None and getattr(delta, "reasoning", None):
                    setattr(delta, "reasoning_content", delta.reasoning)
            yield chunk

    class LocalVLLMQwenHandler(QwenAPIHandler):
        def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
            super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
            self.client = OpenAI(base_url=base_url, api_key=os.environ.get("EVAL_API_KEY", "EMPTY"))

        def _parse_query_response_FC(self, api_response):
            return super()._parse_query_response_FC(alias_reasoning(api_response))

        def _parse_query_response_prompting(self, api_response):
            return super()._parse_query_response_prompting(alias_reasoning(api_response))

    MODEL_CONFIG_MAPPING[REGISTRY_NAME] = ModelConfig(
        model_name=served_name, display_name=f"{served_name}-local-vLLM (FC)",
        url="https://huggingface.co/Qwen/Qwen3.8-27B", org="Qwen", license="apache-2.0",
        model_handler=LocalVLLMQwenHandler, input_price=None, output_price=None,
        is_fc_model=True, underscore_to_dot=True)


def run_cli(args: list[str]) -> None:
    import typer
    from bfcl_eval.__main__ import cli

    typer.main.get_command(cli).main(args=args, standalone_mode=False)


def read_overall(score_dir: Path) -> dict:
    """读官方 data_overall.csv 中本模型那一行，并按上面的权重另算不含 web_search 的总分。"""
    path = score_dir / "data_overall.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    row = next((r for r in rows if (r.get("Model") or "").endswith("-local-vLLM (FC)")), None)
    if row is None:
        raise RuntimeError(f"{path} 中找不到本模型")

    def pct(name: str) -> float | None:
        value = (row.get(name) or "").strip().rstrip("%")
        return float(value) if value not in ("", "N/A") else None

    groups = {"non_live": pct("Non-Live AST Acc"), "live": pct("Live Acc"),
              "irrelevance": pct("Irrelevance Detection"), "relevance": pct("Relevance Detection"),
              "multi_turn": pct("Multi Turn Acc"), "memory": pct("Memory Acc"), "web_search": pct("Web Search Acc")}
    overall = pct("Overall Acc")
    # 官方：Overall = (10·非 live + 10·live + 10·无关 + 30·多轮 + 40·Agentic)/100，Agentic = (web_search + memory)/2，
    # 没跑的 web_search 按 0 计。所以去掉 web_search 后 = (10·非 live + 10·live + 10·无关 + 30·多轮 + 20·记忆)/80 = Overall/0.8。
    return {"official_overall_with_web_search_as_0": overall, "groups": groups,
            "overall_excluding_web_search": overall / 0.8 if overall is not None else None, "official_row": row}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="作为 BFCL_PROJECT_ROOT，result/ 和 score/ 都写在这里")
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--step", choices=("generate", "evaluate", "all"), default="all")
    parser.add_argument("--num-threads", type=int, default=64)
    parser.add_argument("--model", default="Qwen3.8-27B", help="vLLM 的 served-model-name")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--run-ids-file", type=Path,
                        help="调试用：官方 test_case_ids_to_generate.json 格式，只生成其中列出的用例")
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    # 这些环境变量必须在 import bfcl_eval 之前设好：项目根目录、离线加载记忆类别要用的 MiniLM、不走代理。
    os.environ["BFCL_PROJECT_ROOT"] = str(args.output.resolve())
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_CACHE": str(HF_CACHE),
                       "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)
    # 父类构造时会先建官方的 OpenAI / DashScope 客户端，没有 key 会直接报错；随后被替换成本机客户端，占位即可。
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("QWEN_API_KEY", "EMPTY")
    register(args.base_url.rstrip("/"), args.model)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if args.verify_only:
        from importlib.metadata import version
        from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
        expanded = [c for name in categories for c in TEST_COLLECTION_MAPPING.get(name, [name])]
        print(json.dumps({"bfcl_eval": version("bfcl-eval"), "registry_name": REGISTRY_NAME,
                          "handler": MODEL_CONFIG_MAPPING[REGISTRY_NAME].model_handler.__mro__[1].__name__,
                          "categories": expanded}, ensure_ascii=False))
        return 0
    os.umask(0o077)
    category_arg = ",".join(categories)
    if args.step in ("generate", "all"):
        # 生成可断点续跑：已生成的测试用例会被官方跳过。
        extra = []
        if args.run_ids_file:
            (args.output / "test_case_ids_to_generate.json").write_text(args.run_ids_file.read_text(encoding="utf-8"),
                                                                        encoding="utf-8")
            extra = ["--run-ids"]
        run_cli(["generate", "--model", REGISTRY_NAME, "--test-category", category_arg,
                 "--num-threads", str(args.num_threads), "--include-input-log", *extra])
    if args.step in ("evaluate", "all"):
        run_cli(["evaluate", "--model", REGISTRY_NAME, "--test-category", category_arg,
                 *(["--partial-eval"] if args.run_ids_file else [])])
        summary = read_overall(args.output / "score")
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "official_row"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

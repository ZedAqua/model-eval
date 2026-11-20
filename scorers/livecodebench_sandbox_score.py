"""LiveCodeBench 沙箱打分（由 benchmarks/livecodebench/run_lcb.py 拷进临时目录后调用）：在沙箱用户下运行，用 LiveCodeBench 官方代码解码测试用例、执行生成的代码并算 pass@1。

由 run_lcb.py 以低权限用户、资源上限启动；只读写自己所在的临时目录。参数：官方源码目录、输入 JSON、输出 JSON、进程数。
"""

import json
import sys
import types

# 官方 benchmarks 模块顶层 import 了 datasets（只用于在线加载题目）；这里题目由本地文件提供，用空模块占位即可。
stub = types.ModuleType("datasets")
stub.load_dataset = None
sys.modules.setdefault("datasets", stub)


def main() -> None:
    src, in_path, out_path, processes = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    sys.path.insert(0, src)
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    payload = json.load(open(in_path, encoding="utf-8"))
    problems = [CodeGenerationProblem(**p) for p in payload["problems"]]
    samples = [p.get_evaluation_sample() for p in problems]
    metrics, results, metadata = codegen_metrics(
        samples, payload["codes"], k_list=[1], num_process_evaluate=processes, timeout=payload["timeout"])
    json.dump({"pass@1": float(metrics["pass@1"]),
               "per_problem": {str(k): float(v) for k, v in metrics["detail"]["pass@1"].items()},
               "results": {str(k): v for k, v in results.items()},
               "metadata": metadata,
               "num_tests": [len(p.public_test_cases) + len(p.private_test_cases) for p in problems]},
              open(out_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()

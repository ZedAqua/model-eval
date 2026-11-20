#!/usr/bin/env bash
# 以 AutoDL 数据盘上的固定环境启动文本评测服务；--check 只校验文件并展示启动命令。
set -euo pipefail
umask 077

# 固定权重目录和 API 名称，让现有 FinEval runner 无需改动就能接入。
base=/root/autodl-tmp
venv="$base/envs/vllm-qwen"
model="$base/models/Qwen3.8-27B"
flashinfer="$base/envs/vllm-qwen-flashinfer-sm80"
model_len="${QWEN_MAX_MODEL_LEN:-16384}"
num_seqs="${QWEN_MAX_NUM_SEQS:-1}"
eager="${QWEN_ENFORCE_EAGER:-1}"
# BFCL 等工具调用评测需要；Qwen3.8 模板是 XML 工具调用格式，对应 qwen3_coder。留空则不开工具调用解析。
tool_parser="${QWEN_TOOL_CALL_PARSER:-}"
# KV 缓存精度：auto 跟随模型（BF16）；fp8 让 KV 缓存容量约翻倍，长思考评测能同时跑更多题，数值有轻微差异。
kv_dtype="${QWEN_KV_CACHE_DTYPE:-auto}"

# 拒绝无效配置和不完整的模型目录，避免占着 GPU 才发现本地文件缺失。
check_config() {
    [[ "$model_len" == 16384 || "$model_len" == 32768 || "$model_len" == 65536 ]] || { echo 'QWEN_MAX_MODEL_LEN 须为 16384、32768 或 65536' >&2; return 1; }
    [[ "$num_seqs" =~ ^[1-9][0-9]*$ && "$eager" =~ ^[01]$ ]] || { echo '并发须为正整数，QWEN_ENFORCE_EAGER 须为 0 或 1' >&2; return 1; }
    [[ -z "$tool_parser" || "$tool_parser" == qwen3_coder ]] || { echo 'QWEN_TOOL_CALL_PARSER 只支持留空或 qwen3_coder' >&2; return 1; }
    [[ "$kv_dtype" == auto || "$kv_dtype" == fp8 ]] || { echo 'QWEN_KV_CACHE_DTYPE 须为 auto 或 fp8' >&2; return 1; }
    [[ -x "$venv/bin/vllm" && -f "$model/config.json" && -f "$model/chat_template.jinja" && -d "$flashinfer/flashinfer_jit_cache" ]] || {
        echo 'vLLM、模型配置、聊天模板或 FlashInfer 缓存缺失' >&2; return 1;
    }
    "$venv/bin/python" - "$model" <<'PY'
# 核对索引引用的所有权重文件；全量哈希已在下载验收阶段完成。
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
if not shards or any(not (root / name).is_file() or not (root / name).stat().st_size for name in shards):
    raise SystemExit("本地分片缺失或为空")
print(f"本地权重分片: {len(shards)}")
PY
}

case "${1:-}" in
    ''|--check) (( $# <= 1 )) || exit 2 ;;
    *) echo '用法: bash serve_qwen.sh [--check]' >&2; exit 2 ;;
esac
check_config

command=("$venv/bin/vllm" serve "$model" --served-model-name Qwen3.8-27B
    --host 127.0.0.1 --port 8000 --dtype bfloat16 --tensor-parallel-size 1
    --max-model-len "$model_len" --max-num-seqs "$num_seqs" --gpu-memory-utilization 0.90 --kv-cache-dtype "$kv_dtype"
    --language-model-only --reasoning-parser qwen3 --generation-config vllm)
if [[ "$eager" == 1 ]]; then command+=(--enforce-eager); fi
if [[ -n "$tool_parser" ]]; then command+=(--enable-auto-tool-choice --tool-call-parser "$tool_parser"); fi
printf '启动命令: '; printf '%q ' "${command[@]}"; printf '\n'
[[ "${1:-}" == --check ]] && exit 0

# 无卡时直接退出。GPU 真正可用后再建立日志和编译缓存目录。
if ! compgen -G '/dev/nvidia[0-9]*' >/dev/null; then
    echo '未发现 NVIDIA GPU；无卡时请用 --check。' >&2
    exit 1
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$flashinfer${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_CACHE_ROOT="$base/cache/vllm" TRITON_CACHE_DIR="$base/cache/triton"
mkdir -p "$base/logs" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
log=$(mktemp "$base/logs/vllm-serve-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
printf '服务日志: %s\n' "$log"
{
    date -Iseconds
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    printf '启动命令: '; printf '%q ' "${command[@]}"; printf '\n'
    "${command[@]}"
} 2>&1 | tee -a "$log"

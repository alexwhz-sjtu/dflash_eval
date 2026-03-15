#!/bin/bash
# DFlash 数据收集启动脚本

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"

TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
DATASET="${DATASET:-nemotron}"
DATA_DIR="${DATA_DIR:-./train/training_data}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"
SAVE_HIDDEN_STATES="${SAVE_HIDDEN_STATES:-true}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29511}"
AUTO_MASTER_PORT="${AUTO_MASTER_PORT:-true}"
SYNC_MODE="${SYNC_MODE:-file}"
DIST_BACKEND="${DIST_BACKEND:-gloo}"
DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS:-7200}"
MERGE_WAIT_TIMEOUT_SECONDS="${MERGE_WAIT_TIMEOUT_SECONDS:-10800}"
MERGE_POLL_INTERVAL_SECONDS="${MERGE_POLL_INTERVAL_SECONDS:-5.0}"

is_port_free() {
    local port="$1"
    python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", port))
except OSError:
    print("0")
else:
    print("1")
finally:
    s.close()
PY
}

pick_ephemeral_port() {
    python - <<'PY'
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}


echo "=========================================="
echo "DFlash 数据收集脚本"
echo "=========================================="
echo "目标模型: ${TARGET_MODEL}"
echo "数据集: ${DATASET}"
echo "输出目录: ${DATA_DIR}"
echo "样本数: ${NUM_SAMPLES}"
echo "草稿层数参考: ${NUM_DRAFT_LAYERS}"
echo "保存隐藏层: ${SAVE_HIDDEN_STATES}"
echo "收集进程数: ${NPROC_PER_NODE}"
echo "自动选择端口: ${AUTO_MASTER_PORT}"
echo "同步模式: ${SYNC_MODE}"
echo "分布式后端: ${DIST_BACKEND}"
echo "分布式超时(秒): ${DIST_TIMEOUT_SECONDS}"
echo "合并等待超时(秒): ${MERGE_WAIT_TIMEOUT_SECONDS}"
echo "合并轮询间隔(秒): ${MERGE_POLL_INTERVAL_SECONDS}"
echo "=========================================="
echo ""

if [[ "${TARGET_MODEL}" == /* || "${TARGET_MODEL}" == ./* || "${TARGET_MODEL}" == ../* ]] && [ ! -e "${TARGET_MODEL}" ]; then
    echo "错误：目标模型路径不存在: ${TARGET_MODEL}"
    exit 1
fi

mkdir -p ${DATA_DIR}

SAVE_HIDDEN_FLAG=""
if [ "${SAVE_HIDDEN_STATES}" = "true" ]; then
    SAVE_HIDDEN_FLAG="--save_hidden_states"
fi

CMD_ARGS=(
    --model_path ${TARGET_MODEL}
    --dataset ${DATASET}
    --output_dir ${DATA_DIR}
    --max_new_tokens ${MAX_NEW_TOKENS}
    --temperature ${TEMPERATURE}
    --num_draft_layers ${NUM_DRAFT_LAYERS}
    --sync_mode ${SYNC_MODE}
    --dist_backend ${DIST_BACKEND}
    --dist_timeout_seconds ${DIST_TIMEOUT_SECONDS}
    --merge_wait_timeout_seconds ${MERGE_WAIT_TIMEOUT_SECONDS}
    --merge_poll_interval_seconds ${MERGE_POLL_INTERVAL_SECONDS}
)

if [ -n "${NUM_SAMPLES}" ]; then
    CMD_ARGS+=(--num_samples "${NUM_SAMPLES}")
fi

if [ -n "${SAVE_HIDDEN_FLAG}" ]; then
    CMD_ARGS+=("${SAVE_HIDDEN_FLAG}")
fi

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    if [ "${AUTO_MASTER_PORT}" = "true" ]; then
        if [ "$(is_port_free "${MASTER_PORT}")" != "1" ]; then
            OLD_MASTER_PORT="${MASTER_PORT}"
            MASTER_PORT="$(pick_ephemeral_port)"
            echo "检测到端口 ${OLD_MASTER_PORT} 已占用，自动切换到可用端口 ${MASTER_PORT}"
        fi
    fi

    echo "使用 torchrun 多卡数据并行收集..."
    torchrun \
        --nproc_per_node=${NPROC_PER_NODE} \
        --master_port=${MASTER_PORT} \
        train/data/collect_data.py \
        "${CMD_ARGS[@]}"
else
    echo "使用单卡收集..."
    python train/data/collect_data.py "${CMD_ARGS[@]}"
fi

echo ""
echo "=========================================="
echo "数据收集完成！"
echo "=========================================="
echo "响应文件: ${DATA_DIR}/${DATASET}_responses.jsonl"
if [ "${SAVE_HIDDEN_STATES}" = "true" ]; then
    echo "隐藏层目录: ${DATA_DIR}/${DATASET}_features"
fi
echo "=========================================="
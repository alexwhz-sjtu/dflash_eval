#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"
DATASET="${DATASET:-nemotron}"
DATA_DIR="${DATA_DIR:-./train/data/training_data}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
SAMPLING_SEED="${SAMPLING_SEED:-42}"
SAMPLE_SELECTION_DIR="${SAMPLE_SELECTION_DIR:-/share/wanghanzhen/SpeculativeDecoding/NIPS26/dflash/train_exp/sample_selection}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-5}"
SAVE_HIDDEN_STATES="${SAVE_HIDDEN_STATES:-true}"
SYNC_MODE="${SYNC_MODE:-barrier}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"
DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS:-72000}"
MERGE_WAIT_TIMEOUT_SECONDS="${MERGE_WAIT_TIMEOUT_SECONDS:-108000}"
MERGE_POLL_INTERVAL_SECONDS="${MERGE_POLL_INTERVAL_SECONDS:-5.0}"

if [[ "${TARGET_MODEL}" == /* || "${TARGET_MODEL}" == ./* || "${TARGET_MODEL}" == ../* ]] && [ ! -e "${TARGET_MODEL}" ]; then
    echo "错误：目标模型路径不存在: ${TARGET_MODEL}"
    exit 1
fi

mkdir -p "${DATA_DIR}"
mkdir -p "${SAMPLE_SELECTION_DIR}"

SAMPLE_TAG="all"
if [ -n "${NUM_SAMPLES}" ]; then
    SAMPLE_TAG="n${NUM_SAMPLES}"
fi
if [ -n "${NUM_SAMPLES}" ]; then
    SELECTION_FILE="${SELECTION_FILE:-${SAMPLE_SELECTION_DIR}/${DATASET}_sample_selection_${SAMPLE_TAG}.json}"
else
    SELECTION_FILE="${SELECTION_FILE:-}"
fi

SAVE_HIDDEN_FLAG=""
if [ "${SAVE_HIDDEN_STATES}" = "true" ]; then
    SAVE_HIDDEN_FLAG="--save_hidden_states"
fi

CMD_ARGS=(
    --model_path "${TARGET_MODEL}"
    --dataset "${DATASET}"
    --output_dir "${DATA_DIR}"
    --sampling_seed "${SAMPLING_SEED}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --batch_size 4
    --temperature "${TEMPERATURE}"
    --num_draft_layers "${NUM_DRAFT_LAYERS}"
    --sync_mode "${SYNC_MODE}"
    --dist_backend "${DIST_BACKEND}"
    --dist_timeout_seconds "${DIST_TIMEOUT_SECONDS}"
    --merge_wait_timeout_seconds "${MERGE_WAIT_TIMEOUT_SECONDS}"
    --merge_poll_interval_seconds "${MERGE_POLL_INTERVAL_SECONDS}"
)

if [ -n "${NUM_SAMPLES}" ]; then
    CMD_ARGS+=(--num_samples "${NUM_SAMPLES}")
fi

if [ -n "${SELECTION_FILE}" ]; then
    CMD_ARGS+=(--selection_file "${SELECTION_FILE}")
fi

if [ -n "${SAVE_HIDDEN_FLAG}" ]; then
    CMD_ARGS+=("${SAVE_HIDDEN_FLAG}")
fi

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    echo "启动: torchrun (${NPROC_PER_NODE}卡), 采样文件=${SELECTION_FILE}"
    torchrun \
    --standalone \
        --nproc_per_node=${NPROC_PER_NODE} \
        train/data/collect_data_batch.py \
        "${CMD_ARGS[@]}"
else
    echo "启动: python (单卡), 采样文件=${SELECTION_FILE}"
    python train/data/collect_data.py "${CMD_ARGS[@]}"
fi

echo "完成: ${DATA_DIR}/${DATASET}_responses.jsonl"
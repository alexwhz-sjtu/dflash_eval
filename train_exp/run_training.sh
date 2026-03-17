#!/bin/bash
# DFlash 训练启动脚本

set -e  # 遇到错误立即退出

# ========================================
# 配置参数
# ========================================

# GPU 设置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
MASTER_PORT="${MASTER_PORT:-29500}"

# 目标模型路径
TARGET_MODEL="${TARGET_MODEL:-/share/public/public_models/Qwen3-8B}"

# 草稿模型配置来源（可为本地config.json、模型目录或HF模型ID）
DRAFT_CONFIG="${DRAFT_CONFIG:-./train/model_config.json}"

# 输出目录
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/share/wanghanzhen/SpeculativeDecoding/NIPS26/dflash/train/data/training_data/nemotron_2000}"
OUTPUT_DIR="${OUTPUT_DIR:-./train_exp/output/dflash_nemotron_2000}"
FEATURES_DIR="${FEATURES_DIR:-}"

# 训练参数
NUM_DRAFT_LAYERS="${NUM_DRAFT_LAYERS:-}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
ANCHORS_PER_SEQUENCE="${ANCHORS_PER_SEQUENCE:-512}"
NUM_EPOCHS="${NUM_EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
CACHE_FEATURES="${CACHE_FEATURES:-false}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

# wandb 监控参数
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-dflash-training}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

# ========================================
# 显示配置
# ========================================
echo "=========================================="
echo "DFlash 训练启动脚本"
echo "=========================================="
echo "目标模型: ${TARGET_MODEL}"
echo "草稿配置: ${DRAFT_CONFIG}"
echo "离线数据目录: ${TRAIN_DATA_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "草稿模型层数: ${NUM_DRAFT_LAYERS:-使用草稿配置默认值}"
echo "每序列锚点数: ${ANCHORS_PER_SEQUENCE}"
echo "训练样本数: ${MAX_SAMPLES:-全部样本}"
echo "训练轮数: ${NUM_EPOCHS}"
echo "批大小: ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "USE_WANDB: ${USE_WANDB}"
echo "=========================================="
echo ""

# 检查本地路径配置
if [[ "${TARGET_MODEL}" == /* || "${TARGET_MODEL}" == ./* || "${TARGET_MODEL}" == ../* ]] && [ ! -e "${TARGET_MODEL}" ]; then
    echo "错误：目标模型路径不存在: ${TARGET_MODEL}"
    exit 1
fi

if [[ "${DRAFT_CONFIG}" == /* || "${DRAFT_CONFIG}" == ./* || "${DRAFT_CONFIG}" == ../* ]] && [ ! -e "${DRAFT_CONFIG}" ]; then
    echo "错误：草稿配置不存在: ${DRAFT_CONFIG}"
    exit 1
fi

if [ ! -d "${TRAIN_DATA_DIR}" ]; then
    echo "错误：离线数据目录不存在: ${TRAIN_DATA_DIR}"
    exit 1
fi

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# ========================================
# 训练草稿模型
# ========================================
echo ""
echo "==> 开始训练草稿模型"
echo "    输出目录: ${OUTPUT_DIR}"
echo ""

FEATURES_ARGS=""
if [ -n "${FEATURES_DIR}" ]; then
    if [ ! -d "${FEATURES_DIR}" ]; then
        echo "错误：FEATURES_DIR 不是有效目录: ${FEATURES_DIR}"
        exit 1
    fi
    FEATURES_ARGS="--features_dir ${FEATURES_DIR}"
fi

if [ -n "${FEATURES_ARGS}" ] && [ "${CACHE_FEATURES}" = "true" ]; then
    FEATURES_ARGS="${FEATURES_ARGS} --cache_features"
fi

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}")
else
    LAUNCHER=(python)
fi

WANDB_ARGS=""
if [ "${USE_WANDB}" = "true" ]; then
    WANDB_ARGS="--use_wandb --wandb_project ${WANDB_PROJECT}"
    if [ -n "${WANDB_RUN_NAME}" ]; then
        WANDB_ARGS="${WANDB_ARGS} --wandb_run_name ${WANDB_RUN_NAME}"
    fi
    if [ -n "${WANDB_ENTITY}" ]; then
        WANDB_ARGS="${WANDB_ARGS} --wandb_entity ${WANDB_ENTITY}"
    fi
    if [ -n "${WANDB_GROUP}" ]; then
        WANDB_ARGS="${WANDB_ARGS} --wandb_group ${WANDB_GROUP}"
    fi
    if [ -n "${WANDB_TAGS}" ]; then
        WANDB_ARGS="${WANDB_ARGS} --wandb_tags ${WANDB_TAGS}"
    fi
fi

EXTRA_TRAIN_ARGS=()
if [ -n "${NUM_DRAFT_LAYERS}" ]; then
    EXTRA_TRAIN_ARGS+=(--num_draft_layers "${NUM_DRAFT_LAYERS}")
fi
if [ -n "${MAX_SAMPLES}" ]; then
    EXTRA_TRAIN_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

"${LAUNCHER[@]}" ./train_exp/train_anchor_batch.py \
    --target_model_path ${TARGET_MODEL} \
    --data_file "${TRAIN_DATA_DIR}" \
    --draft_config_path ${DRAFT_CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --block_size ${BLOCK_SIZE} \
    --anchors_per_sequence ${ANCHORS_PER_SEQUENCE} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0.01 \
    --warmup_ratio 0.04 \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_grad_norm 1.0 \
    --max_seq_length 4096 \
    ${FEATURES_ARGS} \
    ${WANDB_ARGS} \
    --save_steps 500 \
    --logging_steps 100 \
    --seed 42 \
    "${EXTRA_TRAIN_ARGS[@]}"

# ========================================
# 训练完成
# ========================================
echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "模型保存在: ${OUTPUT_DIR}/final_model/draft_model"
echo ""
echo "使用示例："
echo "  from model.dflash import DFlashDraftModel"
echo "  draft_model = DFlashDraftModel.from_pretrained('${OUTPUT_DIR}/final_model/draft_model')"
echo ""
echo "运行推理测试："
echo "  cd .."
echo "  python benchmark.py --draft_model ${OUTPUT_DIR}/final_model/draft_model"
echo "=========================================="

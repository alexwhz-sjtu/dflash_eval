# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **DFlash** (Block Diffusion for Flash Speculative Decoding) project - a research codebase for training and evaluating lightweight block diffusion draft models that accelerate LLM inference through speculative decoding. The project supports Qwen3, GPT-OSS, and LLaMA model families.

## Directory Structure

```
dflash/
├── model/                  # Core model implementations
│   ├── dflash_exp.py      # Main DFlash draft model (DFlashDraftModel)
│   ├── dflash.py          # Legacy draft model implementation
│   ├── kvcache.py         # Dynamic KV cache implementation
│   └── utils.py           # Utility functions (dataset loading, feature extraction)
├── train/                  # Training data collection scripts
│   ├── data/              # Dataset implementations
│   │   ├── dataset.py     # SimplifiedDataset, OnlineDataset, collate functions
│   │   ├── collect_data.py       # Online feature collection
│   │   └── collect_data_batch.py # Batch feature collection
│   ├── loss.py            # WeightedBlockLoss implementation
│   ├── model_config.json  # Default draft model config (5 layers, block_size=16)
│   └── train.py           # Legacy training script
├── train_exp/              # Main training scripts
│   ├── train_anchor_batch.py   # Main training entry point
│   ├── run_training.sh    # Training launch script with environment variables
│   └── README.md          # Training documentation
├── eval.py                 # MT-Bench evaluation script
├── eval_exp.py            # Extended evaluation
├── benchmark.py           # Transformers backend benchmark
├── benchmark_sglang.py    # SGLang backend benchmark
└── run.py                 # Simple inference script
```

## Key Architecture Components

### DFlashDraftModel (`model/dflash_exp.py`)
- A lightweight transformer-based draft model for speculative decoding
- Uses Qwen3 architecture with custom attention that attends to both target model hidden states and draft tokens
- Key method: `spec_generate()` - performs speculative generation with target model verification
- Configurable via `model_config.json`: layers, hidden_size, block_size, target_layer_ids

### Training Data Flow
1. **Data Collection** (`train/data/collect_data_batch.py`): Generate responses from target model and extract hidden features
2. **Offline Training** (`train_exp/train_anchor_batch.py`): Train draft model using pre-collected features
3. **Dataset** (`train/data/dataset.py`): `SimplifiedDataset` loads responses + offline features for training

### Key Training Concepts
- **Block Size**: Number of tokens to predict in one draft forward pass (default: 16)
- **Anchor Points**: Positions in sequence where drafting starts
- **Target Hidden Features**: Concatenated hidden states from multiple target model layers
- **WeightedBlockLoss**: Exponentially weighted loss favoring early positions in block

## Common Commands

### Setup Environment
```bash
cd dflash
conda create -n dflash python=3.11
conda activate dflash
pip install uv
uv pip install -r requirements.txt
```

### Training

Using the launch script (recommended):
```bash
cd dflash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
TARGET_MODEL=/share/public/public_models/Qwen3-8B \
DRAFT_CONFIG=./train/model_config.json \
TRAIN_DATA_DIR=/path/to/offline_data \
OUTPUT_DIR=./train_exp/output/my_model \
BATCH_SIZE=2 \
bash train_exp/run_training.sh
```

Direct Python command:
```bash
python train_exp/train_anchor_batch.py \
    --target_model_path /share/public/public_models/Qwen3-8B \
    --draft_config_path ./train/model_config.json \
    --data_file /path/to/offline_data \
    --output_dir ./train_exp/output/my_model \
    --block_size 16 \
    --anchors_per_sequence 512 \
    --num_epochs 6 \
    --batch_size 4 \
    --gradient_accumulation_steps 4
```

Resume training:
```bash
python train_exp/train_anchor_batch.py \
    --target_model_path ... \
    --continue_training ./train_exp/output/my_model/checkpoint-epoch-3 \
    ...
```

### Data Collection
```bash
cd dflash
bash train/run_collect_data.sh
```

### Evaluation

Transformers backend:
```bash
cd dflash
python eval.py \
    --target-model-path /share/public/public_models/Qwen3-8B \
    --draft-model-path ./train_exp/output/my_model/final_model/draft_model \
    --question-file /share/wanghanzhen/SpeculativeDecoding/NIPS26/dataset/mtbench101/question.jsonl \
    --begin 0 --end 100
```

Benchmark:
```bash
bash run_benchmark.sh
```

### SGLang Server (Production Inference)
```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --tp-size 1 \
    --dtype bfloat16 \
    --attention-backend fa3
```

## Offline Data Format

Training expects pre-collected data in this structure:
```
data_directory/
├── nemotron_config.json          # Optional: metadata
├── nemotron_responses.jsonl      # Required: tokenized responses
└── nemotron_features/            # Required: hidden features
    ├── chunk_00000.pt
    └── chunk_00001.pt
```

Each `.jsonl` line contains:
```json
{
  "idx": 0,
  "input_ids": [...],
  "response_ids": [...]
}
```

## Configuration Key Parameters

`train/model_config.json`:
- `num_hidden_layers`: Draft model depth (default: 5)
- `hidden_size`: Hidden dimension (default: 4096)
- `block_size`: Tokens per draft block (default: 16)
- `dflash_config.target_layer_ids`: Which target layers to extract features from
- `dflash_config.mask_token_id`: Token ID used for masked positions

## Environment Variables for Training

- `CUDA_VISIBLE_DEVICES`: GPU selection
- `NPROC_PER_NODE`: Number of GPUs for distributed training
- `TARGET_MODEL`: Path or HF model ID
- `DRAFT_CONFIG`: Path to model_config.json
- `TRAIN_DATA_DIR`: Offline data directory
- `OUTPUT_DIR`: Where to save checkpoints
- `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`: Training hyperparameters
- `USE_WANDB`, `WANDB_PROJECT`: W&B logging settings

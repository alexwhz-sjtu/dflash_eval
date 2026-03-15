## DFlash 训练说明

本目录下的数据收集和训练已经拆分为两个独立脚本：

- `run_collect_data.sh`: 使用目标模型生成响应，并可选保存目标模型隐藏层特征。
- `run_training.sh`: 使用已有训练数据和可选离线特征训练 DFlash 草稿模型。

当前训练流程默认支持：

- `nemotron` 数据集从本地 parquet 目录自动加载。
- 训练时草稿模型结构可通过 `train/model_config.json` 单独指定。

## 1. 数据收集

启动脚本：

```bash
cd dflash
bash train/run_collect_data.sh
```

常用环境变量参数：

- `TARGET_MODEL`: 目标模型路径或 Hugging Face 模型 ID。
- `DATASET`: 数据集名称，当前支持 `nemotron`、`gsm8k`、`math500`、`aime24`、`aime25`、`alpaca`、`mt-bench`、`humaneval`、`mbpp`、`lbpp`。
- `DATA_DIR`: 数据输出目录。
- `NUM_SAMPLES`: 采样数量。
- `MAX_NEW_TOKENS`: 每条样本最大生成长度。
- `TEMPERATURE`: 生成温度。
- `NUM_DRAFT_LAYERS`: 草稿层数参考值，用于确定抽取哪些目标层 hidden states。
- `SAVE_HIDDEN_STATES`: 是否保存目标模型隐藏层特征，取值 `true` 或 `false`。
- `NPROC_PER_NODE`: 收集进程数；大于 1 时自动使用 `torchrun` 做多卡数据并行收集。
- `MASTER_PORT`: 多卡收集使用的通信端口。

示例：

```bash
cd dflash
TARGET_MODEL=z-lab/Qwen3-8B-DFlash-b16 \
DATASET=nemotron \
DATA_DIR=./training_data \
NUM_SAMPLES=10000 \
NUM_DRAFT_LAYERS=5 \
SAVE_HIDDEN_STATES=true \
bash train/run_collect_data.sh
```

多卡收集示例：

```bash
cd dflash
CUDA_VISIBLE_DEVICES=3,4,5,6 \
NPROC_PER_NODE=4 \
MASTER_PORT=29511 \
TARGET_MODEL=/share/public/public_models/Qwen3-8B \
DATASET=nemotron \
DATA_DIR=./training_data \
NUM_SAMPLES=9400 \
SAVE_HIDDEN_STATES=true \
bash train/run_collect_data.sh
```

等价的 Python 命令：

```bash
cd dflash
python train/data/collect_data.py \
	--model_path /share/public/public_models/Qwen3-8B \
	--dataset nemotron \
	--output_dir ./training_data \
	--max_new_tokens 4096 \
	--temperature 0.0 \
	--num_samples 10000 \
	--num_draft_layers 5 \
	--save_hidden_states
```

输出内容：

- `training_data/<dataset>_responses.jsonl`: 训练文本数据。
- `training_data/<dataset>_features/`: 每条样本对应的离线目标层 hidden states。

## 2. 训练草稿模型

启动脚本：

```bash
cd dflash
bash train/run_training.sh
```

常用环境变量参数：

- `TARGET_MODEL`: 目标模型路径或 Hugging Face 模型 ID。
- `DRAFT_CONFIG`: 草稿模型配置来源，可为本地 `config.json`、模型目录或 Hugging Face 模型 ID。
- `DATASET`: 数据集名称，仅用于推导默认 `DATA_FILE` 和特征目录。
- `DATA_DIR`: 数据目录。
- `DATA_FILE`: 显式指定训练数据文件路径；若不指定，默认使用 `${DATA_DIR}/${DATASET}_responses.jsonl`。
- `OUTPUT_DIR`: 模型输出目录。
- `NUM_DRAFT_LAYERS`: 可选；若设置，会覆盖 `DRAFT_CONFIG` 中的 `num_hidden_layers`。
- `BLOCK_SIZE`: speculative block 大小。
- `NUM_EPOCHS`: 训练轮数。
- `BATCH_SIZE`: 单卡 batch size。
- `GRAD_ACCUM`: 梯度累积步数。
- `LEARNING_RATE`: 学习率。
- `CACHE_FEATURES`: 是否将离线特征缓存到内存，取值 `true` 或 `false`。
- `NPROC_PER_NODE`: 多卡训练进程数；大于 1 时自动使用 `torchrun`。
- `MASTER_PORT`: 多卡训练端口。
- `USE_WANDB`: 是否启用 wandb，取值 `true` 或 `false`。
- `WANDB_PROJECT`: wandb 项目名。
- `WANDB_RUN_NAME`: wandb 运行名。
- `WANDB_ENTITY`: wandb 用户或组织。
- `WANDB_GROUP`: wandb 分组。
- `WANDB_TAGS`: wandb 标签，逗号分隔。

示例：

多卡示例：

```bash
cd dflash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
MASTER_PORT=29501 \
TARGET_MODEL=/share/public/public_models/Qwen3-8B \
DRAFT_CONFIG=./train/model_config.json \
DATASET=nemotron \
OUTPUT_DIR=./train/output/dflash_nemotron \
MAX_SAMPLES=9400 \
BATCH_SIZE=8 \
bash train/run_training.sh
```

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=3,4,5,6
NPROC_PER_NODE=4
MASTER_PORT=29501
TARGET_MODEL=/share/public/public_models/Qwen3-8B
DRAFT_CONFIG=./train/model_config.json
DATASET=nemotron
OUTPUT_DIR=train/output/dflash_nemotron
MAX_SAMPLES=9400
BATCH_SIZE=8
GRAD_ACCUM=8
ANCHORS_PER_SEQUENCE=512
bash train/run_training.sh

等价的 Python 命令：

```bash
cd dflash
python train/train.py \
	--target_model_path z-lab/Qwen3-8B-DFlash-b16 \
	--draft_config_path ./train/model_config.json \
	--data_file ./training_data/nemotron_responses.jsonl \
	--output_dir ./output/dflash_nemotron \
	--block_size 16 \
	--learning_rate 6e-4 \
	--weight_decay 0.01 \
	--warmup_ratio 0.04 \
	--num_epochs 6 \
	--batch_size 4 \
	--gradient_accumulation_steps 4 \
	--max_grad_norm 1.0 \
	--max_seq_length 4096 \
	--features_dir ./training_data/nemotron_features \
	--save_steps 500 \
	--logging_steps 10 \
	--seed 42
```

如果需要覆盖草稿层数：

```bash
python train/train.py \
	--target_model_path z-lab/Qwen3-8B-DFlash-b16 \
	--draft_config_path ./train/model_config.json \
	--num_draft_layers 5 \
	--data_file ./training_data/nemotron_responses.jsonl \
	--output_dir ./output/dflash_nemotron
```

## 3. 当前默认配置

当前仓库中的默认训练草稿配置文件为：

- `train/model_config.json`

该配置文件会被 `run_training.sh` 默认读取。若未显式传入 `NUM_DRAFT_LAYERS`，训练时将优先使用该配置文件中的：

- `num_hidden_layers`
- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `num_key_value_heads`
- `block_size`
- `dflash_config.target_layer_ids`

训练脚本仍会根据当前目标模型自动刷新：

- `num_target_layers`
- `mask_token_id`
- 与目标模型对齐后的 `target_layer_ids`

## 4. 推荐流程

```bash
cd dflash
bash train/run_collect_data.sh
bash train/run_training.sh
```

如果已经有现成的 `responses.jsonl` 和 `features` 目录，可以直接跳过数据收集，只运行训练脚本。

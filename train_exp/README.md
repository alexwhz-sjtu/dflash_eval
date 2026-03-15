## MTP训练说明

离线收集需占用700TB，因此使用在线收集，即对每个batch样本先用大模型生成一遍response，再训练小模型。

生成序列长度为4096，随机选择512个锚点预测后面一块噪声块。

在特征维度拼接大模型全部层的隐藏状态

## 训练草稿模型

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
CUDA_VISIBLE_DEVICES=0,1,2,3\
NPROC_PER_NODE=4 \
MASTER_PORT=29500 \
TARGET_MODEL=/share/public/public_models/Qwen3-8B \
DRAFT_CONFIG=./train/model_config.json \
DATASET=nemotron \
OUTPUT_DIR=./train_exp/output/dmtp_nemotron \
BATCH_SIZE=2 \
bash train_exp/run_training.sh
```

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

## 当前默认配置

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

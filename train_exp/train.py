"""
训练脚本
实现D-MTP草稿模型的训练流程
"""
import os
import json
import argparse
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm
import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.data.dataset import SimplifiedDataset, simplified_collate_fn, OnlineDataset, online_collate_fn
from train.loss import WeightedBlockLoss
from model.dflash_exp import DFlashDraftModel


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_generation_aligned_hidden_states_all_layers(generation_hidden_states) -> tuple[torch.Tensor, dict]:
    """
    从 generate 返回的 hidden_states 中构造与完整序列对齐的“所有层”特征。

    对于 prompt 部分，保留首步 forward 的逐 token hidden。
    对于 response 部分，使用“生成该 token 之前”的 hidden：
    - 第一个 response token 对应 prompt 最后一个位置的 hidden
    - 后续 response token 对应前一个已生成 token 的 hidden

    返回：
    - target_hidden_flat: [batch, seq_len, num_layers * hidden_size]
    - layer_idx_mapping: 记录每层在最后一维中的切片范围
    """
    # [0] is prefill stage, each step item is tuple: (embeddings, layer0, layer1, ...)
    prefill_step = generation_hidden_states[0]
    num_layers = len(prefill_step) - 1

    # 仅保留各decoder层输出（不含embedding输出）
    prompt_hidden_per_layer = torch.stack(list(prefill_step[1:]), dim=2)
    # shape: [batch, prompt_len, num_layers, hidden_size]

    response_hidden_steps = [prompt_hidden_per_layer[:, -1:, :, :]]
    for step_hidden_states in generation_hidden_states[1:]:
        step_hidden_per_layer = torch.stack(list(step_hidden_states[1:]), dim=2)
        response_hidden_steps.append(step_hidden_per_layer[:, -1:, :, :])

    response_hidden_per_layer = torch.cat(response_hidden_steps, dim=1)
    aligned_hidden_per_layer = torch.cat([prompt_hidden_per_layer, response_hidden_per_layer], dim=1)

    hidden_size = aligned_hidden_per_layer.shape[-1]
    target_hidden_flat = aligned_hidden_per_layer.reshape(
        aligned_hidden_per_layer.shape[0],
        aligned_hidden_per_layer.shape[1],
        num_layers * hidden_size,
    )

    layer_idx_mapping = {
        str(layer_idx): {
            "feature_start": int(layer_idx * hidden_size),
            "feature_end": int((layer_idx + 1) * hidden_size),
        }
        for layer_idx in range(num_layers)
    }

    return target_hidden_flat, layer_idx_mapping


class DFlashTrainer:
    """
    DFlash训练器
    
    训练策略：
    1. 加载目标模型并冻结
    2. 创建草稿模型，共享embedding和lm_head（冻结）
    3. 训练草稿模型的transformer层
    4. 使用加权交叉熵损失
    """
    
    def __init__(
        self,
        target_model_path: str,
        data_file: str,
        output_dir: str,
        num_draft_layers: int = None,
        draft_config_path: str = None,
        draft_hidden_size: int = None,
        draft_intermediate_size: int = None,
        block_size: int = 16,
        anchors_per_sequence: int = 512,
        learning_rate: float = 6e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.04,
        num_epochs: int = 6,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        max_seq_length: int = 4096,
        max_samples: int = None,
        use_torch_compile: bool = False,
        seed: int = 42,
        save_steps: int = 500,
        logging_steps: int = 10,
        device: str = "cuda",
        features_dir: str = None,
        cache_features: bool = False,
        use_wandb: bool = False,
        wandb_project: str = "dflash-training",
        wandb_run_name: str = None,
        wandb_entity: str = None,
        wandb_group: str = None,
        wandb_tags: str = "",
    ):
        args_dict = dict(locals())
        args_dict.pop("self")
        self.args = argparse.Namespace(**args_dict)

        # 分布式训练参数（torchrun）
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1

        if self.distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.is_main_process = (self.rank == 0)
        
        # 设置随机种子
        set_seed(seed)

        if self.distributed and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(device)

        self.output_dir = Path(output_dir)
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.distributed:
            dist.barrier()

        if self.is_main_process:
            print("==> 初始化训练器")
            print(f"   输出目录: {self.output_dir}")
        
        # 加载分词器
        if self.is_main_process:
            print(f"==> 加载分词器: {target_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        
        # 加载目标模型（冻结）
        if self.is_main_process:
            print(f"==> 加载目标模型: {target_model_path}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_path,
            torch_dtype=torch.bfloat16,
        )
        self.target_model.to(self.device)
        self.target_model.eval()
        
        # 冻结目标模型
        for param in self.target_model.parameters():
            param.requires_grad = False
        
        if self.is_main_process:
            print(f"   目标模型层数: {self.target_model.config.num_hidden_layers}")
        
        # 创建草稿模型
        if self.is_main_process:
            print(f"==> 创建草稿模型")
        self.draft_model = self._create_draft_model(
            target_model_path,
            draft_config_path,
            num_draft_layers,
            draft_hidden_size,
            draft_intermediate_size,
            block_size,
        )

        if use_torch_compile and hasattr(torch, "compile"):
            if self.is_main_process:
                print("==> 使用 torch.compile 优化草稿模型")
            self.draft_model = torch.compile(self.draft_model)

        draft_model_dtype = self.target_model.lm_head.weight.dtype
        self.draft_model.to(device=self.device, dtype=draft_model_dtype)

        if self.distributed:
            ddp_kwargs = {}
            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.local_rank]
                ddp_kwargs["output_device"] = self.local_rank
            # 训练路径中存在部分参数按迭代不参与loss（如仅用于prefill的分支），
            # 需要开启unused参数检测以避免DDP reduction状态错误。
            self.draft_model = DDP(self.draft_model, **ddp_kwargs)

        if self.is_main_process:
            print(f"   草稿模型dtype: {draft_model_dtype}")

        actual_num_draft_layers = self._unwrap_draft_model().config.num_hidden_layers
        self.args.num_draft_layers = actual_num_draft_layers
        
        # 提取目标模型所有decoder层，并在特征维度拼接
        self.target_layer_ids = list(range(self.target_model.config.num_hidden_layers))
        if self.is_main_process:
            print(f"   提取的目标层(全部): {self.target_layer_ids}")
        
        # 加载数据集
        if self.is_main_process:
            print(f"==> 加载数据集: {data_file}")
        
        # 用户明确不使用离线特征
        # features_dir = None
        
        if os.path.exists(data_file):
            print(f"==> 加载数据集文件: {data_file}")
            self.dataset = SimplifiedDataset(
                data_file=data_file,
                tokenizer=self.tokenizer,
                block_size=block_size,
                anchors_per_sequence=anchors_per_sequence,
                max_seq_length=max_seq_length,
                max_samples=max_samples,
                features_dir=features_dir,
                cache_features=cache_features,
            )
            self.collate_fn = simplified_collate_fn
        else:
            print(f"==> 加载在线数据集: {data_file}")
            # 对于在线数据集, 传入 dataset_name
            self.dataset = OnlineDataset(
                dataset_name=data_file,
                tokenizer=self.tokenizer,
                max_samples=max_samples,
            )
            self.collate_fn = online_collate_fn

        if features_dir is not None and len(self.dataset) > 0 and isinstance(self.dataset, SimplifiedDataset):
            probe_item = self.dataset[0]
            if "target_hidden" in probe_item and self.is_main_process:
                probe_dim = int(probe_item["target_hidden"].shape[-1])
                expected_dim = int(self._unwrap_draft_model().fc.in_features)
                print(f"   离线特征维度: {probe_dim} | 草稿模型期望维度: {expected_dim}")

        if self.is_main_process:
            print(f"   使用训练样本数量: {len(self.dataset)}")

        # 保存配置（更新features_dir）
        self.args.features_dir = features_dir
        self.args.cache_features = cache_features
        if self.is_main_process:
            with open(self.output_dir / "training_config.json", "w") as f:
                json.dump(vars(self.args), f, indent=2)

        sampler = None
        if self.distributed:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=seed,
            )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=0,  # 使用主进程加载数据
        )
        
        # 创建损失函数
        if self.is_main_process:
            print(f"==> 创建损失函数")
        # 损失函数不包括锚点token (长度减1)
        self.loss_fn = WeightedBlockLoss(block_size=block_size - 1).to(self.device)
        
        # 创建优化器
        if self.is_main_process:
            print(f"==> 创建优化器")
        self.optimizer = AdamW(
            [p for p in self.draft_model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # 创建学习率调度器
        total_steps = len(self.dataloader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        
        if self.is_main_process:
            print(f"   总步数: {total_steps}")
            print(f"   预热步数: {warmup_steps}")
        
        warmup_steps = max(warmup_steps, 1)
        cosine_steps = max(total_steps - warmup_steps, 1)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, cosine_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return max(cosine, 0.1)

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        
        if self.is_main_process:
            print("==> 训练器初始化完成")

        # 可选：初始化 wandb（仅主进程）
        self.wandb_run = None
        if self.args.use_wandb:
            self._init_wandb()

    def _unwrap_draft_model(self) -> nn.Module:
        if isinstance(self.draft_model, DDP):
            return self.draft_model.module
        return self.draft_model

    def _adapt_offline_feature_dim(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """将离线特征维度适配到draft_model.fc的输入维度。"""
        expected_dim = int(self._unwrap_draft_model().fc.in_features)
        got_dim = int(target_hidden.shape[-1])

        if got_dim == expected_dim:
            return target_hidden
        if got_dim > expected_dim:
            return target_hidden[..., :expected_dim]

        raise ValueError(
            f"离线特征维度不足: got={got_dim}, expected={expected_dim}. "
            "请使用与当前训练配置一致的collect_data特征（全层拼接）。"
        )

    def _init_wandb(self):
        if not self.is_main_process:
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "已启用 --use_wandb，但当前环境未安装 wandb。"
                "请先执行: pip install wandb"
            ) from exc

        tags = None
        if self.args.wandb_tags:
            tags = [x.strip() for x in self.args.wandb_tags.split(",") if x.strip()]

        self.wandb_run = wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_run_name,
            entity=self.args.wandb_entity,
            group=self.args.wandb_group,
            dir=str(self.output_dir),
            config=vars(self.args),
            tags=tags,
        )

    def _log_wandb(self, metrics: dict, step: int):
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def _finish_wandb(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None
    
    def _create_draft_model(
        self,
        target_model_path: str,
        draft_config_path: str,
        num_draft_layers: int,
        draft_hidden_size: int,
        draft_intermediate_size: int,
        block_size: int,
    ) -> DFlashDraftModel:
        """
        创建草稿模型
        
        草稿模型特点：
        1. 层数较少（num_draft_layers）
        2. 共享目标模型的embedding和lm_head（冻结）
        3. 仅训练transformer层
        """
        # 加载目标模型配置
        target_config = AutoConfig.from_pretrained(target_model_path)

        config_source = draft_config_path or target_model_path
        config_path = Path(config_source)

        if draft_config_path and config_path.is_file():
            draft_config = Qwen3Config.from_json_file(str(config_path))
        else:
            draft_config = Qwen3Config.from_pretrained(config_source)

        if num_draft_layers is not None:
            draft_config.num_hidden_layers = num_draft_layers
        draft_config.block_size = block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        
        # 可选：修改hidden_size和intermediate_size
        if draft_hidden_size is not None:
            draft_config.hidden_size = draft_hidden_size
        if draft_intermediate_size is not None:
            draft_config.intermediate_size = draft_intermediate_size
        
        # 添加dflash配置
        draft_config.dflash_config = {
            "mask_token_id": self.tokenizer.pad_token_id,
            "target_layer_ids": list(range(target_config.num_hidden_layers)),
        }
        
        # 创建草稿模型
        draft_model = DFlashDraftModel(draft_config)
        
        if self.is_main_process:
            print(f"   草稿配置来源: {config_source}")
            print(f"   草稿模型层数: {draft_config.num_hidden_layers}")
            print(f"   草稿模型hidden_size: {draft_config.hidden_size}")
            print(f"   草稿模型intermediate_size: {draft_config.intermediate_size}")
        
        # 统计参数量
        total_params = sum(p.numel() for p in draft_model.parameters())
        trainable_params = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        
        if self.is_main_process:
            print(f"   总参数量: {total_params / 1e6:.2f}M")
            print(f"   可训练参数量: {trainable_params / 1e6:.2f}M")
        
        return draft_model

    def train_step_online_generation(self, batch: dict) -> dict:
        """
        在线生成并训练
        """
        prompt_input_ids = batch["prompt_input_ids"].to(self.device)
        prompt_attention_mask = batch["prompt_attention_mask"].to(self.device)
        batch_size = prompt_input_ids.shape[0]

        # 1. 在线生成 (Prompt -> Response + Hidden)
        # 使用 args.max_seq_length 作为最大长度限制，或者使用固定 new_tokens
        # collect_data 默认 max_new_tokens=4096。这里为了效率可能需要限制，
        # 但为了覆盖长序列，尽量生成长一点。
        max_new_tokens = self.args.max_seq_length - prompt_input_ids.shape[1]
        if max_new_tokens <= 0:
            return {"loss": 0.0, "num_samples": 0}
        
        with torch.no_grad():
            outputs = self.target_model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False, # 温度0
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # 2. 构造 input_ids 和 target_hidden
        # input_ids: [batch_size, seq_len] (contains prompt + generated)
        input_ids = outputs.sequences
        seq_len = input_ids.shape[1]
        
        # 构造 attention_mask (assuming padding token is 0 or pad_token_id)
        # generate 输出会自动处理 padding，如果 pad_token_id 设置正确。
        # 如果 sequences 中含有 pad_token，需要将其 mask 掉。
        padding_mask = (input_ids != self.tokenizer.pad_token_id)
        # Combine with prompt mask? generate returns full sequence with padding if batching.
        attention_mask = padding_mask
        
        # 提取 target_hidden 并对齐
        # build_generation_aligned_hidden_states_all_layers 返回的是 flat hidden [batch, seq_len, hidden_dim]
        # 且已经对齐好：position i has hidden used to predict token i.
        flat_target_hidden, _ = build_generation_aligned_hidden_states_all_layers(outputs.hidden_states)
        flat_target_hidden = flat_target_hidden.to(self.device)
        
        # 3. 采样锚点
        # 对每个样本，根据其实际生成长度（去掉padding）采样锚点
        anchor_positions_list = []
        for i in range(batch_size):
            valid_len = int(attention_mask[i].sum().item())
            # prompt 长度可能因 padding 而异？ prompt_input_ids 是 left-padded 还是 right-padded?
            # tokenizer default padding side depends. assume left padding for generation usually?
            # But prompt_input_ids comes from online_collate_fn which does right padding (0).
            # "prompt_attention_mask" handles it.
            # outputs.sequences usually preserves input layout (left/right padding).
            # AutoTokenizer usually sets padding side.
            
            # Simple approach: Prompt length is known from batch["prompt_input_ids"] valid length.
            prompt_len = int(batch["prompt_attention_mask"][i].sum().item())
            
            # 采样锚点
            anchor_start = prompt_len
            anchor_end = valid_len - self.args.block_size + 1
            
            candidates = list(range(anchor_start, anchor_end))
            if not candidates:
                anchor_positions_list.append(torch.full((self.args.anchors_per_sequence,), -1, dtype=torch.long))
            elif len(candidates) >= self.args.anchors_per_sequence:
                selected = random.sample(candidates, self.args.anchors_per_sequence)
                selected.sort()
                anchor_positions_list.append(torch.tensor(selected, dtype=torch.long))
            else:
                selected = candidates.copy()
                selected.extend(random.choices(candidates, k=self.args.anchors_per_sequence - len(candidates)))
                selected.sort()
                anchor_positions_list.append(torch.tensor(selected, dtype=torch.long))
                
        anchor_positions = torch.stack(anchor_positions_list).to(self.device)
        
        # 4. 继续原来的训练逻辑
        # 复用后续逻辑，需要构造与 train_step 输入一致的变量
        target_hidden = flat_target_hidden
        
        # 下面逻辑与 train_step 后半部分共用
        # 为了减少重复，可以将后半部分提取出来，或者直接在这里复制
        
        draft_model = self._unwrap_draft_model()
        draft_dtype = draft_model.fc.weight.dtype
        target_hidden = target_hidden.to(dtype=draft_dtype)
        
        total_valid_blocks = 0
        for i in range(batch_size):
            valid_len = int(attention_mask[i].sum().item())
            for anchor_pos in anchor_positions[i].tolist():
                if 0 <= anchor_pos and anchor_pos + self.args.block_size <= valid_len:
                    total_valid_blocks += 1

        if total_valid_blocks == 0:
            return {
                "loss": 0.0,
                "num_samples": 0,
            }

        total_loss_value = 0.0
        num_valid_blocks = 0
        
        for i in range(batch_size):
            valid_len = attention_mask[i].sum().item()
            full_target_hidden = target_hidden[i:i+1, :valid_len, :]

            for anchor_pos in anchor_positions[i].tolist():
                if anchor_pos < 0 or anchor_pos + self.args.block_size > valid_len:
                    continue

                anchor_context_hidden = full_target_hidden[:, anchor_pos:anchor_pos + 1, :]
                
                block_labels = input_ids[i, anchor_pos:anchor_pos + self.args.block_size]
                block_input_ids = input_ids[i, anchor_pos:anchor_pos + self.args.block_size].clone()
                block_input_ids[1:] = draft_model.mask_token_id

                block_embedding = self.target_model.model.embed_tokens(
                    block_input_ids.unsqueeze(0)
                ).to(dtype=draft_dtype)

                block_position_ids = torch.arange(
                    anchor_pos - 1,
                    anchor_pos + self.args.block_size,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                draft_hidden = self.draft_model(
                    target_hidden=anchor_context_hidden,
                    noise_embedding=block_embedding,
                    position_ids=block_position_ids,
                    attention_mask=None,
                    past_key_values=None,
                    use_cache=False,
                )

                block_logits = self.target_model.lm_head(draft_hidden[:, -self.args.block_size:, :])
                loss = self.loss_fn(block_logits[:, 1:, :], block_labels[1:].unsqueeze(0))

                total_loss_value += float(loss.detach().item())
                normalized_loss = loss / (total_valid_blocks * self.args.gradient_accumulation_steps)
                normalized_loss.backward()
                num_valid_blocks += 1
        
        return {
            "loss": total_loss_value / max(num_valid_blocks, 1),
            "num_samples": num_valid_blocks,
        }

    def train_step(self, batch: dict) -> dict:
        """
        训练一步
        
        训练策略：
        1. 使用目标模型提取prompt的最后一个token和响应的完整序列的隐藏层特征
        2. 随机选择锚点token（从prompt的最后一个token开始，直到响应结束前），构造块输入/标签
        3. 提取与锚点token对齐的hiddenstates（语义为“预测该token之前的hidden”）
        4. 锚点token后拼接B-1个mask token
        4. 对锚点块做全注意力预测，只前向一次。
        5. 计算加权损失并更新
        """
        if batch.get("mode") == "online":
            return self.train_step_online_generation(batch)

        input_ids = batch["input_ids"].to(self.device)  # [batch_size, seq_len]

        attention_mask = batch["attention_mask"].to(self.device)
        prompt_lengths = batch["prompt_lengths"].to(self.device)
        anchor_positions = batch["anchor_positions"].to(self.device)
        
        batch_size, seq_len = input_ids.shape
        block_size = self.args.block_size
        
        # 1. 获取目标层隐藏特征 (离线或在线)
        target_hidden = None
        if "target_hidden" in batch:
            target_hidden = batch["target_hidden"].to(self.device).to(dtype=self.target_model.dtype)
        else:
            # 在线计算目标层隐藏特征 (Online Features)
            with torch.no_grad():
                target_outputs = self.target_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

                # 提取目标模型所有decoder层隐藏状态，并在特征维拼接
                # hidden_states: [embeddings, layer1, ..., layerN]
                all_layer_hidden = torch.stack(list(target_outputs.hidden_states[1:]), dim=2)
                # [batch_size, seq_len, num_layers, hidden_size] -> [batch_size, seq_len, num_layers * hidden_size]
                target_hidden = all_layer_hidden.reshape(
                    all_layer_hidden.shape[0],
                    all_layer_hidden.shape[1],
                    -1,
                )

                # 与离线特征语义保持一致：
                # response位置i保存“生成t_i之前”的hidden，即h_{i-1}
                aligned_target_hidden = target_hidden.clone()
                for i in range(batch_size):
                    valid_len = int(attention_mask[i].sum().item())
                    prompt_len = int(prompt_lengths[i].item())
                    if 0 < prompt_len < valid_len:
                        aligned_target_hidden[i, prompt_len:valid_len, :] = target_hidden[i, prompt_len - 1:valid_len - 1, :]
                target_hidden = aligned_target_hidden

            # 释放target_outputs以节省显存
            del target_outputs

        draft_model = self._unwrap_draft_model()
        draft_dtype = draft_model.fc.weight.dtype
        if target_hidden is not None:
            target_hidden = target_hidden.to(dtype=draft_dtype)
        
        # 2. 先统计本batch有效锚点数，用于即时反传时做一致归一化
        total_valid_blocks = 0
        for i in range(batch_size):
            valid_len = int(attention_mask[i].sum().item())
            for anchor_pos in anchor_positions[i].tolist():
                if 0 <= anchor_pos and anchor_pos + block_size <= valid_len:
                    total_valid_blocks += 1

        if total_valid_blocks == 0:
            return {
                "loss": 0.0,
                "num_samples": 0,
            }

        # 3. 对每个样本的随机锚点进行块训练，并按锚点即时反传，避免图累积
        total_loss_value = 0.0
        num_valid_blocks = 0
        
        for i in range(batch_size):
            valid_len = attention_mask[i].sum().item()
            full_target_hidden = target_hidden[i:i+1, :valid_len, :]

            for anchor_pos in anchor_positions[i].tolist():
                if anchor_pos < 0 or anchor_pos + block_size > valid_len:
                    continue

                # 不使用kvcache：取“与锚点token对齐”的target hidden（即预测该token之前的hidden）
                anchor_context_hidden = full_target_hidden[:, anchor_pos:anchor_pos + 1, :]

                # 3. 构造块输入/标签
                block_labels = input_ids[i, anchor_pos:anchor_pos + block_size]
                block_input_ids = input_ids[i, anchor_pos:anchor_pos + block_size].clone()
                block_input_ids[1:] = draft_model.mask_token_id

                block_embedding = self.target_model.model.embed_tokens(
                    block_input_ids.unsqueeze(0)
                ).to(dtype=draft_dtype)

                # position_ids includs the target hidden states, prior to the anchor token
                block_position_ids = torch.arange(
                    anchor_pos - 1,
                    anchor_pos + block_size,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                # 4. 草稿模型前向传播：仅使用 target hidden + 当前块noise embedding，一次前向
                draft_hidden = self.draft_model(
                    target_hidden=anchor_context_hidden,
                    noise_embedding=block_embedding,
                    position_ids=block_position_ids,
                    attention_mask=None,
                    past_key_values=None,
                    use_cache=False,
                )

                block_logits = self.target_model.lm_head(draft_hidden[:, -block_size:, :])
                # 跳过锚点token（第0个），只计算后续生成的token的损失
                loss = self.loss_fn(block_logits[:, 1:, :], block_labels[1:].unsqueeze(0))

                total_loss_value += float(loss.detach().item())
                normalized_loss = loss / (total_valid_blocks * self.args.gradient_accumulation_steps)
                normalized_loss.backward()
                num_valid_blocks += 1
        
        return {
            "loss": total_loss_value / max(num_valid_blocks, 1),
            "num_samples": num_valid_blocks,
        }
    
    def train(self):
        """训练主循环"""
        if self.is_main_process:
            print("==> 开始训练")
        
        self.draft_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        for epoch in range(self.args.num_epochs):
            self.epoch = epoch
            if isinstance(self.dataloader.sampler, DistributedSampler):
                self.dataloader.sampler.set_epoch(epoch)

            if self.is_main_process:
                print(f"\n==> Epoch {epoch + 1}/{self.args.num_epochs}")
            
            epoch_loss = 0.0
            epoch_samples = 0
            
            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=not self.is_main_process,
            )
            
            for step, batch in enumerate(progress_bar):
                should_sync = (step + 1) % self.args.gradient_accumulation_steps == 0
                sync_context = nullcontext()
                if self.distributed and not should_sync:
                    sync_context = self.draft_model.no_sync()

                with sync_context:
                    # 训练一步
                    step_output = self.train_step(batch)
                
                epoch_loss += step_output["loss"]
                epoch_samples += step_output["num_samples"]
                
                # 梯度累积
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self._unwrap_draft_model().parameters(),
                        self.args.max_grad_norm
                    )
                    
                    # 更新参数
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    self.global_step += 1
                    
                    # 日志
                    if self.is_main_process and self.global_step % self.args.logging_steps == 0:
                        avg_loss = epoch_loss / max(epoch_samples, 1)
                        lr = self.optimizer.param_groups[0]['lr']
                        
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        })
                        self._log_wandb(
                            {
                                "train/loss": avg_loss,
                                "train/lr": lr,
                                "train/epoch": epoch + 1,
                                "train/processed_samples": epoch_samples,
                            },
                            step=self.global_step,
                        )
                    
                    # 保存检查点
                    if self.global_step % self.args.save_steps == 0:
                        if self.distributed:
                            dist.barrier()
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
                        if self.distributed:
                            dist.barrier()
            
            # Epoch结束，保存检查点
            avg_epoch_loss = epoch_loss / max(epoch_samples, 1)
            if self.is_main_process:
                print(f"Epoch {epoch + 1} 平均损失: {avg_epoch_loss:.4f}")
                self._log_wandb(
                    {
                        "epoch/loss": avg_epoch_loss,
                        "epoch/index": epoch + 1,
                    },
                    step=self.global_step,
                )
            
            if self.distributed:
                dist.barrier()
            self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")
            if self.distributed:
                dist.barrier()
        
        # 训练结束，保存最终模型
        if self.is_main_process:
            print("\n==> 训练完成!")
        if self.distributed:
            dist.barrier()
        self.save_checkpoint("final_model")
        if self.distributed:
            dist.barrier()
        if self.is_main_process:
            self._finish_wandb()
    
    def save_checkpoint(self, checkpoint_name: str):
        """保存检查点"""
        if not self.is_main_process:
            return

        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存草稿模型
        self._unwrap_draft_model().save_pretrained(checkpoint_dir / "draft_model")
        
        # 保存训练状态
        torch.save({
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, checkpoint_dir / "trainer_state.pt")
        
        print(f"   保存检查点: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="训练DFlash草稿模型")
    
    # 模型参数
    parser.add_argument("--target_model_path", type=str, required=True,
                       help="目标模型路径")
    parser.add_argument("--data_file", type=str, default="nemotron",
                       help="训练数据文件（jsonl格式）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    
    # 草稿模型配置
    parser.add_argument("--num_draft_layers", type=int, default=None,
                       help="草稿模型层数")
    parser.add_argument("--draft_config_path", type=str, default=None,
                       help="草稿模型配置来源，可为模型目录、HF模型ID或config.json文件")
    parser.add_argument("--draft_hidden_size", type=int, default=None,
                       help="草稿模型hidden_size（None表示与目标模型相同）")
    parser.add_argument("--draft_intermediate_size", type=int, default=None,
                       help="草稿模型intermediate_size（None表示与目标模型相同）")
    parser.add_argument("--block_size", type=int, default=16,
                       help="块大小")
    parser.add_argument("--anchors_per_sequence", type=int, default=512,
                       help="每个序列随机采样的锚点数量")
    
    # 训练超参数
    parser.add_argument("--learning_rate", type=float, default=6e-4,
                       help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.04,
                       help="预热比例")
    parser.add_argument("--num_epochs", type=int, default=6,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="梯度裁剪阈值")
    parser.add_argument("--max_seq_length", type=int, default=3072,
                       help="最大序列长度")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="仅使用前N条有效训练样本（默认使用全部）")
    parser.add_argument("--use_torch_compile", action="store_true",
                       help="启用torch.compile优化草稿模型")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="保存检查点的步数")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="日志记录的步数")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备（torchrun多卡时会自动使用LOCAL_RANK）")
    parser.add_argument("--features_dir", type=str, default=None,
                       help="离线特征目录（若存在则跳过目标模型前向）")
    parser.add_argument("--cache_features", action="store_true",
                       help="缓存离线特征到内存（减少IO）")
    parser.add_argument("--use_wandb", action="store_true",
                       help="启用Weights & Biases训练监控")
    parser.add_argument("--wandb_project", type=str, default="dflash-training",
                       help="wandb项目名")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="wandb运行名（默认自动生成）")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="wandb实体名（用户名或团队）")
    parser.add_argument("--wandb_group", type=str, default=None,
                       help="wandb分组名")
    parser.add_argument("--wandb_tags", type=str, default="",
                       help="wandb标签，逗号分隔")
    
    args = parser.parse_args()
    
    trainer = None
    try:
        # 创建训练器
        trainer = DFlashTrainer(
            target_model_path=args.target_model_path,
            data_file=args.data_file,
            output_dir=args.output_dir,
            num_draft_layers=args.num_draft_layers,
            draft_config_path=args.draft_config_path,
            draft_hidden_size=args.draft_hidden_size,
            draft_intermediate_size=args.draft_intermediate_size,
            block_size=args.block_size,
            anchors_per_sequence=args.anchors_per_sequence,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            max_seq_length=args.max_seq_length,
            max_samples=args.max_samples,
            use_torch_compile=args.use_torch_compile,
            seed=args.seed,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            device=args.device,
            features_dir=args.features_dir,
            cache_features=args.cache_features,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
            wandb_tags=args.wandb_tags,
        )

        # 开始训练
        trainer.train()
    finally:
        if trainer is not None and trainer.is_main_process:
            trainer._finish_wandb()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

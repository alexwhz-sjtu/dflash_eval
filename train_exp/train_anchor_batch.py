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
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.data.dataset import SimplifiedDataset, simplified_collate_fn
from train.loss import WeightedBlockLoss
from model.dflash_exp import DFlashDraftModel

def _resolve_offline_dataset_inputs(data_file: str, features_dir: str | None) -> dict:
    """
    解析离线数据输入。

    支持将 data_file 直接指向离线目录：
    - 自动定位 *_responses.jsonl
    - 自动定位 *_features 目录
    - 尝试读取 nemotron_config.json 中的 feature_format / features_per_file
    """
    data_path = Path(data_file)
    result = {
        "resolved_from_dir": False,
        "data_file": data_file,
        "features_dir": features_dir,
        "feature_format": None,
        "features_per_file": 1000,
        "config_path": None,
    }

    if not data_path.exists() or not data_path.is_dir():
        return result

    result["resolved_from_dir"] = True
    config_obj = {}
    config_path = data_path / "nemotron_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_obj = json.load(f)
            result["config_path"] = str(config_path)
        except Exception:
            config_obj = {}

    response_candidates = []
    dataset_name = config_obj.get("dataset_name")
    if isinstance(dataset_name, str) and dataset_name:
        response_candidates.append(data_path / f"{dataset_name}_responses.jsonl")
    response_candidates.append(data_path / "nemotron_responses.jsonl")
    response_candidates.extend(sorted(data_path.glob("*_responses.jsonl")))

    response_file = next((p for p in response_candidates if p.exists()), None)
    if response_file is None:
        raise FileNotFoundError(
            f"离线目录中未找到 *_responses.jsonl: {data_path}"
        )

    resolved_features_dir = Path(features_dir).expanduser() if features_dir else None
    if resolved_features_dir is None:
        cfg_features_dir = config_obj.get("features_dir")
        if isinstance(cfg_features_dir, str) and cfg_features_dir:
            cfg_path = Path(cfg_features_dir)
            cfg_candidates = [
                cfg_path,
                data_path / cfg_path,
                data_path / cfg_path.name,
            ]
            resolved_features_dir = next((p for p in cfg_candidates if p.exists()), None)

    if resolved_features_dir is None:
        feature_dir_candidates = sorted(
            p for p in data_path.iterdir()
            if p.is_dir() and p.name.endswith("_features")
        )
        if feature_dir_candidates:
            resolved_features_dir = feature_dir_candidates[0]

    feature_format = config_obj.get("feature_format")
    features_per_file = int(config_obj.get("features_per_file", 10000))
    if features_per_file <= 0:
        features_per_file = 10000

    if resolved_features_dir is not None:
        has_chunk_zip = any(resolved_features_dir.glob("chunk_*.zip"))
        has_chunk_pt = any(resolved_features_dir.glob("chunk_*.pt"))
        if has_chunk_pt and not has_chunk_zip:
            feature_format = "chunk_pt"
        elif has_chunk_zip:
            feature_format = "chunk_zip"

    result.update(
        {
            "data_file": str(response_file),
            "features_dir": str(resolved_features_dir) if resolved_features_dir is not None else None,
            "feature_format": feature_format,
            "features_per_file": features_per_file,
        }
    )
    return result


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        continue_training: str = None,
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
        mmap_features: bool = True,
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

        # 始终保留 tokenizer + embedding + lm_head，离线特征训练时可裁剪其余backbone。
        self.target_embed_tokens = self.target_model.model.embed_tokens
        self.target_lm_head = self.target_model.lm_head
        self.target_model_dtype = self.target_lm_head.weight.dtype
        self.target_backbone_pruned = False
        
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

        draft_model_dtype = self.target_model_dtype
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
        
        # 加载离线数据集
        if self.is_main_process:
            print(f"==> 加载数据集: {data_file}")

        offline_resolved = _resolve_offline_dataset_inputs(
            data_file=data_file,
            features_dir=features_dir,
        )
        data_file = offline_resolved["data_file"]
        features_dir = offline_resolved["features_dir"]
        offline_feature_format = offline_resolved["feature_format"]
        offline_features_per_file = offline_resolved["features_per_file"]

        self.args.data_file = data_file
        self.args.features_dir = features_dir
        self.args.feature_format = offline_feature_format
        self.args.features_per_file = offline_features_per_file

        if self.is_main_process and offline_resolved["resolved_from_dir"]:
            print(f"   离线目录解析 data_file -> {data_file}")
            print(f"   离线目录解析 features_dir -> {features_dir}")
            if offline_resolved["config_path"] is not None:
                print(f"   使用离线配置: {offline_resolved['config_path']}")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"训练数据路径不存在: {data_file}")

        if features_dir is None:
            raise ValueError(
                "当前训练仅支持离线特征模式，请传入离线数据目录(含*_features)"
                "或显式指定 --features_dir"
            )

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
            mmap_features=mmap_features,
            feature_format=offline_feature_format,
            features_per_file=offline_features_per_file,
        )
        self.collate_fn = simplified_collate_fn

        if len(self.dataset) == 0:
            raise ValueError("离线数据集为空，请检查 responses 与过滤条件")

        probe_item = self.dataset[0]
        if "target_hidden" not in probe_item:
            raise ValueError("离线训练需要 target_hidden 特征，请检查 features_dir")

        if self.is_main_process:
            probe_dim = int(probe_item["target_hidden"].shape[-1])
            expected_dim = int(self._unwrap_draft_model().fc.in_features)
            print(f"   离线特征维度: {probe_dim} | 草稿模型期望维度: {expected_dim}")

        self._prune_target_backbone_if_possible()

        if self.is_main_process:
            print(f"   使用训练样本数量: {len(self.dataset)}")

        # 保存配置（更新features_dir）
        self.args.features_dir = features_dir
        self.args.cache_features = cache_features
        self.args.mmap_features = mmap_features
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
        self.start_epoch = 0

        if continue_training:
            self._resume_from_checkpoint(continue_training)
        
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

    def _resolve_continue_checkpoint(self, continue_training: str) -> Path:
        resume_path = Path(continue_training).expanduser()
        if not resume_path.exists():
            raise FileNotFoundError(f"continue_training 路径不存在: {resume_path}")

        if resume_path.is_file():
            if resume_path.name != "trainer_state.pt":
                raise ValueError(
                    "continue_training 指向文件时，必须是 trainer_state.pt"
                )
            checkpoint_dir = resume_path.parent
            if not (checkpoint_dir / "draft_model").exists():
                raise FileNotFoundError(
                    f"未找到 draft_model 目录: {checkpoint_dir / 'draft_model'}"
                )
            return checkpoint_dir

        if (resume_path / "trainer_state.pt").exists() and (resume_path / "draft_model").exists():
            return resume_path

        checkpoint_dirs = [
            p
            for p in resume_path.iterdir()
            if p.is_dir() and (p / "trainer_state.pt").exists() and (p / "draft_model").exists()
        ]
        if not checkpoint_dirs:
            raise FileNotFoundError(
                f"在 {resume_path} 下未找到可恢复checkpoint（需包含 trainer_state.pt 和 draft_model/）"
            )

        def _score(p: Path) -> tuple[int, int, float]:
            state_path = p / "trainer_state.pt"
            try:
                trainer_state = torch.load(state_path, map_location="cpu")
            except Exception:
                trainer_state = {}
            global_step = int(trainer_state.get("global_step", -1))
            epoch = int(trainer_state.get("epoch", -1))
            return global_step, epoch, p.stat().st_mtime

        checkpoint_dirs.sort(key=_score)
        return checkpoint_dirs[-1]

    def _resume_from_checkpoint(self, continue_training: str):
        checkpoint_dir = self._resolve_continue_checkpoint(continue_training)
        model_dir = checkpoint_dir / "draft_model"
        trainer_state_path = checkpoint_dir / "trainer_state.pt"

        if self.is_main_process:
            print(f"==> 继续训练，加载checkpoint: {checkpoint_dir}")

        restored_model = DFlashDraftModel.from_pretrained(str(model_dir))
        self._unwrap_draft_model().load_state_dict(restored_model.state_dict(), strict=True)
        del restored_model

        trainer_state = torch.load(trainer_state_path, map_location="cpu")
        self.optimizer.load_state_dict(trainer_state["optimizer"])
        self.scheduler.load_state_dict(trainer_state["scheduler"])
        self.global_step = int(trainer_state.get("global_step", 0))
        self.epoch = int(trainer_state.get("epoch", 0))

        if checkpoint_dir.name.startswith("checkpoint-epoch-"):
            self.start_epoch = self.epoch + 1
        else:
            self.start_epoch = self.epoch

        if self.is_main_process:
            print(
                f"   恢复完成: global_step={self.global_step}, "
                f"saved_epoch={self.epoch}, resume_epoch={self.start_epoch}"
            )

    def _prune_target_backbone_if_possible(self):
        """离线特征训练时仅保留 embedding/lm_head，释放 target backbone 显存。"""
        if self.target_backbone_pruned:
            return

        if hasattr(self.target_model, "model") and hasattr(self.target_model.model, "layers"):
            if self.device.type == "cuda":
                for layer in self.target_model.model.layers:
                    layer.to("cpu")
            self.target_model.model.layers = nn.ModuleList()
        self.target_backbone_pruned = True

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.is_main_process:
            print("==> 检测到离线特征，已裁剪 target_model backbone，仅保留 embedding/lm_head/tokenizer")

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

    def _build_joint_sparse_attention_mask(
        self,
        num_blocks: int,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        构造联合块训练的稀疏注意力掩码。

        约束：
        1) 块内双向注意力（query 仅能看同块 noise token）
        2) query 仅能看本块的 target context token
        3) 禁止跨块注意力

        返回形状: [1, 1, q_len, kv_len]
        其中 q_len = num_blocks * block_size, kv_len = num_blocks + q_len
        """
        q_len = num_blocks * block_size
        if q_len == 0:
            return torch.empty((1, 1, 0, 0), dtype=torch.bool, device=device)

        query_block_ids = torch.arange(q_len, device=device) // block_size
        target_block_ids = torch.arange(num_blocks, device=device)
        noise_block_ids = torch.arange(q_len, device=device) // block_size
        key_block_ids = torch.cat([target_block_ids, noise_block_ids], dim=0)

        mask_2d = query_block_ids.unsqueeze(1).eq(key_block_ids.unsqueeze(0))
        return mask_2d.unsqueeze(0).unsqueeze(0)

    def _forward_joint_blocks_for_sequence(
        self,
        seq_target_hidden_list: list[torch.Tensor],
        seq_input_ids_list: list[torch.Tensor],
        seq_labels_list: list[torch.Tensor],
        seq_position_ids_list: list[torch.Tensor],
        draft_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int]:
        """
        对单个序列的多个锚点块进行联合前向（一次前向，稀疏掩码隔离跨块信息）。

        返回：
        - seq_loss: 标量loss（按该序列有效块做平均）
        - num_blocks: 该序列有效块数量
        """
        num_blocks = len(seq_target_hidden_list)
        if num_blocks == 0:
            raise ValueError("_forward_joint_blocks_for_sequence 收到空块列表")

        block_size = self.args.block_size

        # [1, num_blocks, feat_dim]
        seq_target_hidden = torch.cat(seq_target_hidden_list, dim=1)

        # [num_blocks, block_size]
        seq_input_ids = torch.stack(seq_input_ids_list, dim=0)
        seq_labels = torch.stack(seq_labels_list, dim=0)
        seq_position_ids = torch.stack(seq_position_ids_list, dim=0)

        # [1, num_blocks * block_size, hidden_size]
        seq_noise_embedding = self.target_embed_tokens(seq_input_ids).to(dtype=draft_dtype)
        seq_noise_embedding = seq_noise_embedding.reshape(1, num_blocks * block_size, -1)

        # joint position ids: [ctx(num_blocks) + noise(num_blocks * block_size)]
        ctx_position_ids = seq_position_ids[:, 0]
        noise_position_ids = seq_position_ids[:, 1:].reshape(-1)
        joint_position_ids = torch.cat([ctx_position_ids, noise_position_ids], dim=0).unsqueeze(0)

        joint_attention_mask = self._build_joint_sparse_attention_mask(
            num_blocks=num_blocks,
            block_size=block_size,
            device=self.device,
        )

        draft_hidden_joint = self.draft_model(
            target_hidden=seq_target_hidden,
            noise_embedding=seq_noise_embedding,
            position_ids=joint_position_ids,
            attention_mask=joint_attention_mask,
            past_key_values=None,
            use_cache=False,
        )

        # [num_blocks, block_size, hidden_size]
        draft_hidden_blocks = draft_hidden_joint.reshape(num_blocks, block_size, -1)
        block_logits = self.target_lm_head(draft_hidden_blocks)
        seq_loss = self.loss_fn(block_logits[:, 1:, :], seq_labels[:, 1:])

        return seq_loss, num_blocks

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
        input_ids = batch["input_ids"].to(self.device)  # [batch_size, seq_len]
        attention_mask = batch["attention_mask"].to(self.device)
        anchor_positions = batch["anchor_positions"].to(self.device)
        
        batch_size, seq_len = input_ids.shape
        block_size = self.args.block_size
        
        if "target_hidden" not in batch:
            raise ValueError("当前训练仅支持离线特征，请检查数据与features_dir")

        target_hidden = batch["target_hidden"].to(self.device).to(dtype=self.target_model_dtype)
        target_hidden = self._adapt_offline_feature_dim(target_hidden)

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

        # 3. 序列内联合块训练：每个序列一次前向（块拼接+稀疏注意力），块间不互相可见
        total_loss_value = 0.0

        for i in range(batch_size):
            valid_len = int(attention_mask[i].sum().item())
            full_target_hidden = target_hidden[i:i+1, :valid_len, :]

            seq_target_hidden_list = []
            seq_input_ids_list = []
            seq_labels_list = []
            seq_position_ids_list = []

            for anchor_pos in anchor_positions[i].tolist():
                if anchor_pos < 0 or anchor_pos + block_size > valid_len:
                    continue

                # 与锚点token对齐的target hidden（预测该token之前的hidden）
                anchor_context_hidden = full_target_hidden[:, anchor_pos:anchor_pos + 1, :]
                seq_target_hidden_list.append(anchor_context_hidden)

                block_labels = input_ids[i, anchor_pos:anchor_pos + block_size].clone()
                seq_labels_list.append(block_labels)

                block_input_ids = input_ids[i, anchor_pos:anchor_pos + block_size].clone()
                block_input_ids[1:] = draft_model.mask_token_id
                seq_input_ids_list.append(block_input_ids)

                block_position_ids = torch.arange(
                    anchor_pos - 1,
                    anchor_pos + block_size,
                    dtype=torch.long,
                    device=self.device,
                )
                seq_position_ids_list.append(block_position_ids)

            seq_num_blocks = len(seq_target_hidden_list)
            if seq_num_blocks == 0:
                continue

            seq_loss, _ = self._forward_joint_blocks_for_sequence(
                seq_target_hidden_list=seq_target_hidden_list,
                seq_input_ids_list=seq_input_ids_list,
                seq_labels_list=seq_labels_list,
                seq_position_ids_list=seq_position_ids_list,
                draft_dtype=draft_dtype,
            )

            weighted_seq_loss = seq_loss * (seq_num_blocks / float(total_valid_blocks))
            normalized_loss = weighted_seq_loss / self.args.gradient_accumulation_steps
            normalized_loss.backward()

            total_loss_value += float(seq_loss.detach().item()) * seq_num_blocks

        return {
            "loss": total_loss_value / max(total_valid_blocks, 1),
            "num_samples": total_valid_blocks,
        }
    
    def train(self):
        """训练主循环"""
        if self.is_main_process:
            print("==> 开始训练")

        if self.start_epoch >= self.args.num_epochs:
            if self.is_main_process:
                print(
                    f"==> 恢复epoch({self.start_epoch}) 已达到/超过 num_epochs({self.args.num_epochs})，无需继续训练"
                )
            return

        if self.start_epoch > 0 and self.is_main_process:
            print(f"==> 从 Epoch {self.start_epoch + 1}/{self.args.num_epochs} 继续训练")
        
        self.draft_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        for epoch in range(self.start_epoch, self.args.num_epochs):
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
                       help="训练数据文件（jsonl格式）或离线数据目录（自动解析 *_responses.jsonl 与 *_features）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--continue-training", "--continue_training", dest="continue_training", type=str, default=None,
                       help="继续训练路径，可传输出目录或checkpoint目录（自动恢复最近状态）")
    
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
    parser.add_argument("--disable_mmap_features", action="store_true",
                       help="关闭 chunk_pt 的 mmap 加载（默认开启）")
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
            continue_training=args.continue_training,
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
            mmap_features=not args.disable_mmap_features,
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

"""
训练数据集类
实现掩码块构造策略，支持以下功能：
1. 加载目标模型生成的响应数据
2. 构造训练样本：每个token作为锚点，预测后续块内容
3. 支持动态提取目标模型隐藏层特征
"""
import json
import bisect
import random
import io
import zipfile
import os
import tempfile
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Dict, Any


class MaskedBlockDataset(Dataset):
    """
    掩码块训练数据集
    
    训练策略：
    1. 先整个序列以逐token因果方式prefill得到kvcache
    2. 然后以每个token为锚点（从第一个生成token开始顺序遍历），包括其之前的kvcache
    3. 预测后面以此token为首token，后续为噪声的（mask）一个块内的内容
    
    Args:
        data_file: 响应数据文件路径（jsonl格式）
        target_model: 目标模型（用于动态提取隐藏层特征）
        tokenizer: 分词器
        target_layer_ids: 需要提取的目标层ID列表
        block_size: 块大小（默认16）
        mask_token_id: mask token的id
        max_seq_length: 最大序列长度
        min_response_length: 最小响应长度（过滤太短的样本）
    """
    
    def __init__(
        self,
        data_file: str,
        target_model,
        tokenizer,
        target_layer_ids: List[int],
        block_size: int = 16,
        mask_token_id: int = None,
        max_seq_length: int = 4096,
        min_response_length: int = 32,
    ):
        super().__init__()
        self.data_file = Path(data_file)
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.target_layer_ids = target_layer_ids
        self.block_size = block_size
        self.mask_token_id = mask_token_id if mask_token_id is not None else tokenizer.pad_token_id
        self.max_seq_length = max_seq_length
        self.min_response_length = min_response_length
        
        # 加载数据
        self.samples = []
        self.anchor_offsets = []
        self.total_anchors = 0
        self._load_data()
        self._build_anchor_index()
        
        print(f"==> 加载了 {len(self.samples)} 个样本，锚点总数: {self.total_anchors}")
    
    def _load_data(self):
        """加载响应数据并构造训练样本"""
        print(f"==> 从 {self.data_file} 加载数据...")
        
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                
                # 获取input_ids和response_ids
                input_ids = data["input_ids"]
                response_ids = data["response_ids"]
                
                # 过滤太短的响应
                if len(response_ids) < self.min_response_length:
                    continue
                
                # 完整的token序列（prompt + response）
                full_ids = input_ids + response_ids
                
                # 截断到最大长度
                if len(full_ids) > self.max_seq_length:
                    continue
                
                # 为每个可能的锚点位置创建一个训练样本
                # 锚点从prompt结束位置开始，到response结束前block_size位置
                prompt_length = len(input_ids)
                response_length = len(response_ids)
                
                # 确保至少有一个完整的块可以预测
                if response_length < self.block_size:
                    continue
                
                # 保存样本信息
                self.samples.append({
                    "full_ids": full_ids,
                    "prompt_length": prompt_length,
                    "response_length": response_length,
                })
    
    def _build_anchor_index(self):
        """构建锚点索引（按顺序遍历所有可用锚点）"""
        self.anchor_offsets = []
        self.total_anchors = 0
        for sample in self.samples:
            anchor_count = sample["response_length"] - self.block_size + 1
            sample["anchor_start"] = sample["prompt_length"]
            sample["anchor_count"] = anchor_count
            self.total_anchors += anchor_count
            self.anchor_offsets.append(self.total_anchors)

    def __len__(self):
        return self.total_anchors
    
    def __getitem__(self, idx):
        """
        获取一个训练样本
        
        返回:
            dict: 包含以下key
                - input_ids: 完整序列的token ids [seq_len]
                - anchor_pos: 锚点位置（随机选择）
                - block_input_ids: 块的输入token ids [block_size]（第一个token是锚点，其余是mask）
                - block_labels: 块的标签token ids [block_size]（真实应该生成的tokens）
                - attention_mask: 注意力掩码
        """
        sample_idx = bisect.bisect_right(self.anchor_offsets, idx)
        prev_offset = 0 if sample_idx == 0 else self.anchor_offsets[sample_idx - 1]
        local_offset = idx - prev_offset

        sample = self.samples[sample_idx]
        full_ids = sample["full_ids"]
        anchor_pos = sample["anchor_start"] + local_offset
        
        # 构造块的输入和标签
        # 块输入：[anchor_token, mask, mask, ..., mask]
        # 块标签：[token1, token2, ..., token_block_size]（真实的后续tokens）
        block_labels = full_ids[anchor_pos:anchor_pos + self.block_size]
        block_input_ids = [full_ids[anchor_pos]] + [self.mask_token_id] * (self.block_size - 1)
        
        # Prefill部分：从开始到锚点位置（用于kvcache）
        prefill_ids = full_ids[:anchor_pos]
        
        return {
            "prefill_ids": torch.tensor(prefill_ids, dtype=torch.long),
            "anchor_pos": anchor_pos,
            "block_input_ids": torch.tensor(block_input_ids, dtype=torch.long),
            "block_labels": torch.tensor(block_labels, dtype=torch.long),
            "full_ids": torch.tensor(full_ids, dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义collate函数
    
    由于每个样本的prefill长度不同，需要进行padding
    """
    # 找到最长的prefill序列
    max_prefill_len = max(len(item["prefill_ids"]) for item in batch)
    
    # Padding
    batch_size = len(batch)
    block_size = batch[0]["block_input_ids"].shape[0]
    
    # 初始化张量
    prefill_ids = torch.full((batch_size, max_prefill_len), 0, dtype=torch.long)
    prefill_mask = torch.zeros((batch_size, max_prefill_len), dtype=torch.bool)
    block_input_ids = torch.zeros((batch_size, block_size), dtype=torch.long)
    block_labels = torch.zeros((batch_size, block_size), dtype=torch.long)
    anchor_positions = []
    
    for i, item in enumerate(batch):
        prefill_len = len(item["prefill_ids"])
        prefill_ids[i, :prefill_len] = item["prefill_ids"]
        prefill_mask[i, :prefill_len] = True
        block_input_ids[i] = item["block_input_ids"]
        block_labels[i] = item["block_labels"]
        anchor_positions.append(item["anchor_pos"])
    
    return {
        "prefill_ids": prefill_ids,
        "prefill_mask": prefill_mask,
        "block_input_ids": block_input_ids,
        "block_labels": block_labels,
        "anchor_positions": anchor_positions,
    }


class SimplifiedDataset(Dataset):
    """
    简化版数据集，用于快速训练
    直接加载整个序列，并按顺序遍历所有锚点（可选离线特征）
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        block_size: int = 16,
        anchors_per_sequence: int = 512,
        max_seq_length: int = 4096,
        min_response_length: Optional[int] = None,
        max_samples: Optional[int] = None,
        features_dir: Optional[str] = None,
        cache_features: bool = False,
        mmap_features: bool = True,
        feature_format: Optional[str] = None,
        features_per_file: int = 10000,
    ):
        super().__init__()
        self.data_file = Path(data_file)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.anchors_per_sequence = anchors_per_sequence
        self.max_seq_length = max_seq_length
        self.min_response_length = int(min_response_length) if min_response_length is not None else int(block_size)
        self.max_samples = max_samples
        
        self.features_dir = Path(features_dir) if features_dir else None
        self.cache_features = cache_features
        self.mmap_features = bool(mmap_features)
        self._feature_cache_idx = None
        self._feature_cache = None
        self.features_per_file = max(int(features_per_file), 1)
        self._feature_backend = None
        self._zip_cache_chunk_id = None
        self._zip_cache = None
        self._pt_cache_chunk_id = None
        self._pt_chunk_data = None
        self._mmap_fallback_warned = False

        if self.features_dir is not None and self.features_dir.exists():
            if feature_format in {"chunk_zip", "chunk_pt", "single_pt"}:
                self._feature_backend = feature_format
            else:
                chunk_zip_files = sorted(self.features_dir.glob("chunk_*.zip"))
                chunk_pt_files = sorted(self.features_dir.glob("chunk_*.pt"))
                if chunk_zip_files:
                    self._feature_backend = "chunk_zip"
                elif chunk_pt_files:
                    self._feature_backend = "chunk_pt"
                else:
                    self._feature_backend = "single_pt"

        # 加载数据
        self.samples = []
        self._load_data()
        print(f"==> 加载了 {len(self.samples)} 个样本")
    
    def _load_data(self):
        """加载响应数据"""
        print(f"==> 从 {self.data_file} 加载数据...")

        total_count = 0
        dropped_short = 0
        dropped_prompt_too_long = 0
        truncated_count = 0
        
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                total_count += 1
                data = json.loads(line)
                
                # 获取input_ids和response_ids
                input_ids = data["input_ids"]
                response_ids = data["response_ids"]

                max_response_room = self.max_seq_length - len(input_ids)
                if max_response_room <= 0:
                    dropped_prompt_too_long += 1
                    continue

                if len(response_ids) > max_response_room:
                    response_ids = response_ids[:max_response_room]
                    truncated_count += 1
                
                # 过滤太短的响应
                if len(response_ids) < self.min_response_length:
                    dropped_short += 1
                    continue
                
                # 完整的token序列（prompt + response）
                full_ids = input_ids + response_ids
                
                prompt_length = len(input_ids)
                response_length = len(response_ids)
                
                # 确保至少有一个完整的块可以预测
                if response_length < self.block_size:
                    continue
                
                # 保存样本信息
                # 优先使用数据中的idx，如果没有则使用原始行号（line_idx），确保过滤后也能对齐特征
                self.samples.append({
                    "idx": data.get("idx", line_idx),
                    "full_ids": full_ids,
                    "prompt_length": prompt_length,
                    "response_length": response_length,
                })

                if self.max_samples is not None and len(self.samples) >= self.max_samples:
                    break

        print(
            "==> 数据过滤统计: "
            f"total={total_count}, kept={len(self.samples)}, "
            f"truncated={truncated_count}, "
            f"dropped_short={dropped_short}, "
            f"dropped_prompt_too_long={dropped_prompt_too_long}"
        )
    
    def __len__(self):
        return len(self.samples)

    def _load_feature_from_chunk_zip(self, sample_idx: int) -> Dict[str, Any]:
        chunk_id = sample_idx // self.features_per_file
        chunk_file = self.features_dir / f"chunk_{chunk_id:05d}.zip"
        if not chunk_file.exists():
            raise FileNotFoundError(f"未找到特征分块文件: {chunk_file}")

        if self._zip_cache is None or self._zip_cache_chunk_id != chunk_id:
            if self._zip_cache is not None:
                self._zip_cache.close()
            self._zip_cache = zipfile.ZipFile(chunk_file, mode="r")
            self._zip_cache_chunk_id = chunk_id

        member_name = f"{sample_idx}.pt"
        try:
            self._zip_cache.getinfo(member_name)
        except KeyError as exc:
            raise KeyError(f"在 {chunk_file.name} 中未找到样本特征: {member_name}")
        

        raw_bytes = self._zip_cache.read(member_name)
        feature_pack = torch.load(io.BytesIO(raw_bytes), map_location="cpu")
        return feature_pack

    def _load_feature_from_chunk_pt(self, sample_idx: int) -> Dict[str, Any]:
        chunk_id = sample_idx // self.features_per_file
        chunk_file = self.features_dir / f"chunk_{chunk_id:05d}.pt"
        if not chunk_file.exists():
            raise FileNotFoundError(
                f"未找到特征分块文件: {chunk_file}. "
                "请检查 features_per_file 是否与收集阶段一致。"
            )

        if self._pt_chunk_data is None or self._pt_cache_chunk_id != chunk_id:
            # 使用 mmap 让 tensor storage 由操作系统按需分页，避免一次性常驻大块内存。
            load_kwargs = {"map_location": "cpu"}
            if self.mmap_features:
                load_kwargs["mmap"] = True
            try:
                chunk_data = torch.load(chunk_file, **load_kwargs)
            except TypeError:
                if self.mmap_features and not self._mmap_fallback_warned:
                    print("==> 当前PyTorch不支持 torch.load(..., mmap=True)，回退到普通加载")
                    self._mmap_fallback_warned = True
                chunk_data = torch.load(chunk_file, map_location="cpu")
            if not isinstance(chunk_data, dict):
                raise TypeError(f"chunk_pt 格式错误: {chunk_file} 不是dict")
            self._pt_chunk_data = chunk_data
            self._pt_cache_chunk_id = chunk_id

        key = str(sample_idx)
        if key in self._pt_chunk_data:
            return self._pt_chunk_data[key]
        if sample_idx in self._pt_chunk_data:
            return self._pt_chunk_data[sample_idx]

        raise KeyError(f"在 {chunk_file.name} 中未找到样本特征: {sample_idx}")

    @staticmethod
    def _align_feature_to_full_sequence(
        target_hidden: torch.Tensor,
        seq_len: int,
        prompt_length: int,
        align_start: Optional[int] = None,
    ) -> torch.Tensor:
        """
        兼容两种离线特征格式：
        1) 旧格式：target_hidden 长度 == seq_len
        2) 新格式：仅保存 prompt最后一个token + response全部token
        """
        if target_hidden.shape[0] == seq_len:
            return target_hidden

        if align_start is None:
            align_start = max(prompt_length - 1, 0)

        aligned_hidden = torch.zeros(
            (seq_len, target_hidden.shape[-1]),
            dtype=target_hidden.dtype,
        )

        start = max(int(align_start), 0)
        end = min(start + int(target_hidden.shape[0]), seq_len)
        if end > start:
            aligned_hidden[start:end] = target_hidden[: end - start]

        return aligned_hidden

    def __del__(self):
        if self._zip_cache is not None:
            self._zip_cache.close()
            self._zip_cache = None
        self._pt_chunk_data = None

    def _sample_anchor_positions(self, sample: Dict) -> torch.Tensor:
        """为单个序列随机采样锚点位置（最多 anchors_per_sequence 个）。"""
        anchor_start = sample["prompt_length"]
        anchor_end = len(sample["full_ids"]) - self.block_size + 1
        candidates = list(range(anchor_start, anchor_end))

        if not candidates:
            return torch.full((self.anchors_per_sequence,), -1, dtype=torch.long)

        if len(candidates) >= self.anchors_per_sequence:
            selected = random.sample(candidates, self.anchors_per_sequence)
        else:
            selected = candidates.copy()
            selected.extend(random.choices(candidates, k=self.anchors_per_sequence - len(candidates)))

        selected.sort()
        return torch.tensor(selected, dtype=torch.long)
    
    def __getitem__(self, idx):
        """获取一个训练样本"""
        sample = self.samples[idx]
        anchor_positions = self._sample_anchor_positions(sample)

        item = {
            "full_ids": torch.tensor(sample["full_ids"], dtype=torch.long),
            "prompt_length": sample["prompt_length"],
            "anchor_positions": anchor_positions,
            "sample_idx": idx,
        }

        # 如果有离线特征，按需加载
        if self.features_dir is not None and self.features_dir.exists():
            if self.cache_features and self._feature_cache_idx == idx:
                target_hidden = self._feature_cache
            else:
                if self._feature_backend == "chunk_zip":
                    feature_pack = self._load_feature_from_chunk_zip(sample["idx"])
                elif self._feature_backend == "chunk_pt":
                    feature_pack = self._load_feature_from_chunk_pt(sample["idx"])
                else:
                    feature_path = self.features_dir / f"{sample['idx']}.pt"
                    feature_pack = torch.load(feature_path, map_location="cpu")

                if isinstance(feature_pack, dict) and "target_hidden" in feature_pack:
                    target_hidden = feature_pack["target_hidden"]
                    align_start = feature_pack.get("align_start")
                elif isinstance(feature_pack, torch.Tensor):
                    target_hidden = feature_pack
                    align_start = None
                else:
                    raise KeyError(
                        f"样本 {sample['idx']} 的特征中缺少 target_hidden，"
                        f"特征类型: {type(feature_pack)}"
                    )

                target_hidden = self._align_feature_to_full_sequence(
                    target_hidden=target_hidden,
                    seq_len=len(sample["full_ids"]),
                    prompt_length=sample["prompt_length"],
                    align_start=align_start,
                )

                if self.cache_features:
                    self._feature_cache_idx = idx
                    self._feature_cache = target_hidden

            item["target_hidden"] = target_hidden

        return item


def simplified_collate_fn(batch: List[Dict]) -> Dict:
    """简化版collate函数"""
    # 找到最长的序列
    max_len = max(len(item["full_ids"]) for item in batch)
    
    batch_size = len(batch)
    
    # Padding
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    prompt_lengths = []
    anchors_per_sequence = batch[0]["anchor_positions"].shape[0]
    anchor_positions = torch.full((batch_size, anchors_per_sequence), -1, dtype=torch.long)

    has_features = "target_hidden" in batch[0]
    if has_features:
        feature_dim = batch[0]["target_hidden"].shape[-1]
        target_hidden = torch.zeros((batch_size, max_len, feature_dim), dtype=batch[0]["target_hidden"].dtype)
    
    for i, item in enumerate(batch):
        seq_len = len(item["full_ids"])
        input_ids[i, :seq_len] = item["full_ids"]
        attention_mask[i, :seq_len] = True
        prompt_lengths.append(item["prompt_length"])
        anchor_positions[i] = item["anchor_positions"]
        if has_features:
            target_hidden[i, :seq_len] = item["target_hidden"][:seq_len]
    
    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        "anchor_positions": anchor_positions,
    }

    if has_features:
        batch_dict["target_hidden"] = target_hidden

    return batch_dict


class OnlineDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        max_samples=None,
        balanced_total_samples: int = 400000,
        balanced_task_sampling: bool = True,
        seed: int = 42,
        selection_file: Optional[str] = None,
    ):
        try:
            from model.utils import load_and_process_dataset
        except ImportError:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from model.utils import load_and_process_dataset

        self.dataset = load_and_process_dataset(dataset_name)
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.selection_file = Path(selection_file) if selection_file else None
        target_samples = max_samples if max_samples is not None else balanced_total_samples

        if target_samples is not None and target_samples < len(self.dataset):
            if balanced_task_sampling:
                selected_indices = self._get_or_create_indices(
                    dataset=self.dataset,
                    total_samples=target_samples,
                    seed=seed,
                    mode="balanced_by_task",
                    index_builder=self._build_balanced_indices,
                )
            else:
                selected_indices = self._get_or_create_indices(
                    dataset=self.dataset,
                    total_samples=target_samples,
                    seed=seed,
                    mode="uniform_random",
                    index_builder=self._build_uniform_random_indices,
                )

            self.dataset = self.dataset.select(selected_indices)

        print(f"==> OnlineDataset loaded {len(self.dataset)} samples from {dataset_name}")

    def _selection_key(self, mode: str, total_samples: int) -> str:
        return f"{self.dataset_name}::{mode}::{int(total_samples)}"

    def _load_selection_store(self) -> Dict[str, Any]:
        if self.selection_file is None or not self.selection_file.exists():
            return {}

        try:
            with open(self.selection_file, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            return {}

        if isinstance(obj, dict):
            return obj
        return {}

    def _save_selection_store(self, store: Dict[str, Any]):
        if self.selection_file is None:
            return

        self.selection_file.parent.mkdir(parents=True, exist_ok=True)
        # 多进程/多卡同时写入时，固定tmp文件名会产生竞态，
        # 这里改为每个进程独立临时文件后再原子替换目标文件。
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=f"{self.selection_file.name}.",
            suffix=f".{os.getpid()}.tmp",
            dir=str(self.selection_file.parent),
        )
        os.close(tmp_fd)
        tmp_file = Path(tmp_path)

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)

        os.replace(tmp_file, self.selection_file)

    def _validate_indices(self, indices: List[int], dataset_size: int, expected_count: int) -> bool:
        if not isinstance(indices, list):
            return False
        if len(indices) != expected_count:
            return False
        if expected_count == 0:
            return True
        if not all(isinstance(x, int) for x in indices):
            return False
        if min(indices) < 0 or max(indices) >= dataset_size:
            return False
        return True

    def _get_or_create_indices(
        self,
        dataset,
        total_samples: int,
        seed: int,
        mode: str,
        index_builder,
    ) -> List[int]:
        dataset_size = len(dataset)
        total_samples = min(int(total_samples), dataset_size)
        key = self._selection_key(mode=mode, total_samples=total_samples)

        store = self._load_selection_store()
        entry = store.get(key)
        if isinstance(entry, dict):
            cached_indices = entry.get("indices")
            if self._validate_indices(cached_indices, dataset_size=dataset_size, expected_count=total_samples):
                print(
                    "==> OnlineDataset 复用历史采样: "
                    f"key={key}, 数量={len(cached_indices)}"
                )
                return sorted(cached_indices)

        indices, stats = index_builder(dataset=dataset, total_samples=total_samples, seed=seed)
        indices = sorted(indices)

        if self.selection_file is not None:
            store[key] = {
                "dataset_name": self.dataset_name,
                "mode": mode,
                "total_samples": total_samples,
                "dataset_size_when_selected": dataset_size,
                "seed_when_selected": seed,
                "stats": stats,
                "indices": indices,
            }
            self._save_selection_store(store)
            print(f"==> OnlineDataset 已保存采样记录: {self.selection_file}")

        return indices

    @staticmethod
    def _build_uniform_random_indices(dataset, total_samples: int, seed: int) -> tuple[List[int], Dict[str, Any]]:
        dataset_size = len(dataset)
        rng = random.Random(seed)
        selected_indices = rng.sample(range(dataset_size), total_samples)
        stats = {
            "sampling_strategy": "uniform_random",
            "dataset_size": dataset_size,
            "selected_count": len(selected_indices),
        }
        return selected_indices, stats

    @staticmethod
    def _normalize_task_value(raw_value) -> str:
        if raw_value is None:
            return "unknown"
        if isinstance(raw_value, str):
            value = raw_value.strip()
            return value if value else "unknown"
        if isinstance(raw_value, (list, tuple)):
            for item in raw_value:
                norm = OnlineDataset._normalize_task_value(item)
                if norm != "unknown":
                    return norm
            return "unknown"
        if isinstance(raw_value, dict):
            for key in ("task_type", "task", "category", "name", "type"):
                if key in raw_value:
                    return OnlineDataset._normalize_task_value(raw_value[key])
            return "unknown"
        return str(raw_value)

    @staticmethod
    def _find_task_key(column_names: List[str]) -> Optional[str]:
        candidates = [
            "task_type",
            "task",
            "category",
            "source",
            "domain",
            "dataset",
            "subset",
            "subtask",
        ]
        for key in candidates:
            if key in column_names:
                return key
        return None

    def _build_balanced_indices(self, dataset, total_samples: int, seed: int) -> tuple[List[int], Dict[str, Any]]:
        dataset_size = len(dataset)
        if dataset_size == 0:
            return [], {
                "sampling_strategy": "balanced_by_task",
                "task_key": None,
                "num_tasks": 0,
                "selected_count": 0,
            }

        total_samples = min(int(total_samples), dataset_size)
        rng = random.Random(seed)
        task_key = self._find_task_key(dataset.column_names)

        if task_key is None:
            selected_indices = rng.sample(range(dataset_size), total_samples)
            print(
                "==> OnlineDataset 未找到任务类型字段，"
                f"退化为随机采样 {total_samples} 条"
            )
            return selected_indices, {
                "sampling_strategy": "uniform_random_fallback",
                "task_key": None,
                "num_tasks": 0,
                "selected_count": len(selected_indices),
            }

        grouped_indices = defaultdict(list)
        for idx, example in enumerate(dataset):
            task_type = self._normalize_task_value(example.get(task_key))
            grouped_indices[task_type].append(idx)

        task_types = sorted(grouped_indices.keys())
        num_tasks = len(task_types)
        if num_tasks == 0:
            selected_indices = rng.sample(range(dataset_size), total_samples)
            return selected_indices, {
                "sampling_strategy": "uniform_random_fallback",
                "task_key": task_key,
                "num_tasks": 0,
                "selected_count": len(selected_indices),
            }

        per_task_quota = total_samples // num_tasks
        selected_indices = []
        remaining_pool = []

        for task_type in task_types:
            indices = grouped_indices[task_type]
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            take = min(per_task_quota, len(shuffled))
            selected_indices.extend(shuffled[:take])
            if take < len(shuffled):
                remaining_pool.extend(shuffled[take:])

        if len(selected_indices) < total_samples and remaining_pool:
            remaining_needed = total_samples - len(selected_indices)
            if remaining_needed >= len(remaining_pool):
                selected_indices.extend(remaining_pool)
            else:
                selected_indices.extend(rng.sample(remaining_pool, remaining_needed))

        print(
            "==> OnlineDataset 任务均衡采样: "
            f"task_key={task_key}, 任务类型数={num_tasks}, "
            f"采样数={len(selected_indices)}/{dataset_size}"
        )
        task_counts = {task_type: len(grouped_indices[task_type]) for task_type in task_types}
        stats = {
            "sampling_strategy": "balanced_by_task",
            "task_key": task_key,
            "num_tasks": num_tasks,
            "selected_count": len(selected_indices),
            "task_counts": task_counts,
        }
        return selected_indices, stats

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        prompt = example['turns'][0] if isinstance(example['turns'], list) else example['turns']
        
        messages = [{"role": "user", "content": prompt}]
        # 即使tokenizer没有chat_template，apply_chat_template通常也会报错或有默认行为。
        # 假设model/utils.py里的model对应的tokenizer配置正确。
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
             
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
        
        return {
            "input_ids": input_ids,
            "idx": idx
        }

def online_collate_fn(batch: List[Dict]) -> Dict:
    # prompt batch padding
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    
    for i, item in enumerate(batch):
        l = len(item["input_ids"])
        input_ids[i, :l] = item["input_ids"]
        attention_mask[i, :l] = True
        
    return {
        "prompt_input_ids": input_ids,
        "prompt_attention_mask": attention_mask,
        "mode": "online",
    }


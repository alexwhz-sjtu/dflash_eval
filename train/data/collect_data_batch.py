"""
数据收集脚本
使用目标模型在指定数据集上生成响应，并保存以下内容：
1. 生成的响应文本（作为训练标签）
2. 目标模型“所有decoder层”的隐藏层特征
3. token与隐藏层对应关系：t_n->LLM->h_n->lm_head->t_n+1。保存时把“用于预测当前token的hidden”与当前token对齐。
4. 在特征文件中显式保存层idx与特征切片范围的映射信息。
"""
import os
import sys
import json
import io
import zipfile
import argparse
import time
import random
import tempfile
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from typing import Any, Dict, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model.utils import load_and_process_dataset


SAMPLE_SELECTION_DIR = Path("/share/wanghanzhen/SpeculativeDecoding/NIPS26/dflash/train_exp/sample_selection")


class FeatureChunkPtWriter:
    def __init__(self, features_dir: Path, rank: int, features_per_file: int):
        self.features_dir = features_dir
        self.rank = rank
        self.features_per_file = features_per_file
        self.current_chunk_id = None
        self.current_chunk_data = {}

    def _pt_path(self, chunk_id: int) -> Path:
        return self.features_dir / f"chunk_{chunk_id:05d}.rank{self.rank}.pt"

    def _switch_chunk(self, chunk_id: int):
        if self.current_chunk_id == chunk_id:
            return
        if self.current_chunk_id is not None:
            torch.save(self.current_chunk_data, self._pt_path(self.current_chunk_id))
        self.current_chunk_id = chunk_id
        self.current_chunk_data = {}

    def write(self, sample_idx: int, feature_pack: dict):
        chunk_id = sample_idx // self.features_per_file
        self._switch_chunk(chunk_id)
        self.current_chunk_data[str(sample_idx)] = feature_pack

    def close(self):
        if self.current_chunk_id is not None:
            torch.save(self.current_chunk_data, self._pt_path(self.current_chunk_id))
            self.current_chunk_id = None
            self.current_chunk_data = {}


def init_distributed(enable_process_group: bool = False, backend: str = "gloo", timeout_seconds: int = 7200):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        pg_initialized = False
        if enable_process_group and world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                timeout=timedelta(seconds=timeout_seconds),
            )
            pg_initialized = True
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank, pg_initialized
    return False, 0, 1, 0, False


def finalize_distributed(pg_initialized: bool):
    if pg_initialized and dist.is_initialized():
        dist.destroy_process_group()


def write_rank_done_marker(
    done_dir: Path,
    dataset_name: str,
    rank: int,
    num_processed: int,
    status: str = "ok",
    error: str | None = None,
):
    done_dir.mkdir(parents=True, exist_ok=True)
    marker_file = done_dir / f"{dataset_name}.rank{rank}.done.json"
    tmp_file = marker_file.with_suffix(".tmp")
    payload = {
        "rank": rank,
        "status": status,
        "num_processed": int(num_processed),
        "error": error,
        "timestamp": int(time.time()),
    }
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_file, marker_file)


def wait_for_all_rank_markers(
    done_dir: Path,
    dataset_name: str,
    world_size: int,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> list[dict]:
    expected_files = [
        done_dir / f"{dataset_name}.rank{rank}.done.json"
        for rank in range(world_size)
    ]
    deadline = time.time() + timeout_seconds

    while True:
        if all(marker.exists() for marker in expected_files):
            statuses = []
            for marker in expected_files:
                with open(marker, "r", encoding="utf-8") as f:
                    statuses.append(json.load(f))
            return statuses

        if time.time() >= deadline:
            missing = [str(marker.name) for marker in expected_files if not marker.exists()]
            raise TimeoutError(
                f"等待 rank 完成标记超时（{timeout_seconds}s），缺失标记: {missing}"
            )

        time.sleep(max(0.5, poll_interval_seconds))


def _selection_key(dataset_name: str, mode: str, total_samples: int) -> str:
    return f"{dataset_name}::{mode}::{int(total_samples)}"


def _load_selection_store(selection_file: Path) -> Dict[str, Any]:
    if not selection_file.exists():
        return {}

    try:
        with open(selection_file, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}

    if isinstance(obj, dict):
        return obj
    return {}


def _save_selection_store(selection_file: Path, store: Dict[str, Any]):
    selection_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f"{selection_file.name}.",
        suffix=f".{os.getpid()}.tmp",
        dir=str(selection_file.parent),
    )
    os.close(tmp_fd)
    tmp_file = Path(tmp_path)

    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)

    os.replace(tmp_file, selection_file)


def _validate_indices(indices: List[int], dataset_size: int, expected_count: int) -> bool:
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


def _normalize_task_value(raw_value) -> str:
    if raw_value is None:
        return "unknown"
    if isinstance(raw_value, str):
        value = raw_value.strip()
        return value if value else "unknown"
    if isinstance(raw_value, (list, tuple)):
        for item in raw_value:
            norm = _normalize_task_value(item)
            if norm != "unknown":
                return norm
        return "unknown"
    if isinstance(raw_value, dict):
        for key in ("task_type", "task", "category", "name", "type"):
            if key in raw_value:
                return _normalize_task_value(raw_value[key])
        return "unknown"
    return str(raw_value)


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


def _build_balanced_indices(dataset, total_samples: int, seed: int) -> tuple[List[int], Dict[str, Any]]:
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
    task_key = _find_task_key(dataset.column_names)

    if task_key is None:
        selected_indices = rng.sample(range(dataset_size), total_samples)
        print(
            "==> 未找到任务类别字段，退化为随机采样: "
            f"{len(selected_indices)}/{dataset_size}"
        )
        return selected_indices, {
            "sampling_strategy": "uniform_random_fallback",
            "task_key": None,
            "num_tasks": 0,
            "selected_count": len(selected_indices),
        }

    grouped_indices: Dict[str, List[int]] = {}
    for idx, example in enumerate(dataset):
        task_type = _normalize_task_value(example.get(task_key))
        grouped_indices.setdefault(task_type, []).append(idx)

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
    selected_indices: List[int] = []
    remaining_pool: List[int] = []

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

    task_counts = {task_type: len(grouped_indices[task_type]) for task_type in task_types}
    stats = {
        "sampling_strategy": "balanced_by_task",
        "task_key": task_key,
        "num_tasks": num_tasks,
        "selected_count": len(selected_indices),
        "task_counts": task_counts,
    }
    print(
        "==> 按任务均衡采样: "
        f"task_key={task_key}, 任务类型数={num_tasks}, "
        f"采样数={len(selected_indices)}/{dataset_size}"
    )
    return sorted(selected_indices), stats


def get_or_create_balanced_sample_indices(
    dataset,
    dataset_name: str,
    total_samples: int,
    seed: int,
    selection_file: Path,
) -> tuple[List[int], Dict[str, Any], bool]:
    dataset_size = len(dataset)
    total_samples = min(int(total_samples), dataset_size)
    key = _selection_key(dataset_name=dataset_name, mode="balanced_by_task", total_samples=total_samples)

    store = _load_selection_store(selection_file)
    entry = store.get(key)
    if isinstance(entry, dict):
        cached_indices = entry.get("indices")
        if _validate_indices(cached_indices, dataset_size=dataset_size, expected_count=total_samples):
            print(
                "==> 复用历史采样索引: "
                f"key={key}, 数量={len(cached_indices)}"
            )
            return sorted(cached_indices), entry.get("stats", {}), True

    indices, stats = _build_balanced_indices(dataset=dataset, total_samples=total_samples, seed=seed)
    store[key] = {
        "dataset_name": dataset_name,
        "mode": "balanced_by_task",
        "total_samples": total_samples,
        "dataset_size_when_selected": dataset_size,
        "seed_when_selected": seed,
        "stats": stats,
        "indices": indices,
    }
    _save_selection_store(selection_file, store)
    print(f"==> 已保存采样索引文件: {selection_file}")
    return indices, stats, False


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


def collect_data_from_target_model(
    model_path: str,
    dataset_name: str,
    output_dir: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    batch_size: int = 1,
    num_samples: int = None,
    sampling_seed: int = 42,
    selection_file: str = None,
    num_draft_layers: int = 8,
    save_hidden_states: bool = False,
    features_per_file: int = 10000,
    sync_mode: str = "file",
    dist_backend: str = "gloo",
    dist_timeout_seconds: int = 7200,
    merge_wait_timeout_seconds: int = 10800,
    merge_poll_interval_seconds: float = 5.0,
):
    """
    使用目标模型生成响应并收集训练数据
    
    Args:
        model_path: 目标模型路径
        dataset_name: 数据集名称
        output_dir: 输出目录
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        batch_size: 批大小
        num_samples: 采样数量（None表示全部）
        sampling_seed: 任务均衡采样随机种子
        selection_file: 采样索引记录文件（相同采样条数可复用）
        num_draft_layers: 草稿模型层数（用于确定需要提取的目标层）
        save_hidden_states: 是否保存目标模型隐藏状态（离线训练）
        features_per_file: 每个特征文件保存的样本数
    """
    use_process_group = sync_mode == "barrier"
    is_distributed, rank, world_size, local_rank, pg_initialized = init_distributed(
        enable_process_group=use_process_group,
        backend=dist_backend,
        timeout_seconds=dist_timeout_seconds,
    )
    is_main_process = rank == 0

    if is_distributed:
        print(f"==> 启用分布式收集: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print(f"==> [rank {rank}] 加载目标模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)
    model.eval()
    
    # 收集目标模型所有decoder层
    num_target_layers = model.config.num_hidden_layers
    target_layer_ids = list(range(num_target_layers))
    print(f"==> 目标模型层数: {num_target_layers}")
    print(f"==> 草稿模型层数(仅记录配置用途): {num_draft_layers}")
    print(f"==> 保存全部目标层: {target_layer_ids}")
    
    # 准备输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process:
        print(f"==> 输出目录设置为: {output_dir}")

    # 加载数据集
    print(f"==> 加载数据集: {dataset_name}")
    try:
        dataset = load_and_process_dataset(dataset_name)
    except Exception:
        finalize_distributed(pg_initialized)
        raise

    original_total_size = len(dataset)
    print(f"==> 原始数据集大小: {original_total_size}")

    selected_global_indices = None
    sampling_stats = None
    reused_sampling = False
    effective_total_samples = min(int(num_samples), original_total_size) if num_samples is not None else original_total_size
    sample_tag = f"n{effective_total_samples}"
    if selection_file:
        effective_selection_file = Path(selection_file)
    else:
        SAMPLE_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
        effective_selection_file = SAMPLE_SELECTION_DIR / f"{dataset_name}_sample_selection_{sample_tag}.json"

    if num_samples is not None and num_samples < original_total_size:
        selected_global_indices, sampling_stats, reused_sampling = get_or_create_balanced_sample_indices(
            dataset=dataset,
            dataset_name=dataset_name,
            total_samples=num_samples,
            seed=sampling_seed,
            selection_file=effective_selection_file,
        )
        dataset = dataset.select(selected_global_indices)
        print(
            "==> 采样后数据集大小: "
            f"{len(dataset)} (seed={sampling_seed}, reused={reused_sampling})"
        )
    else:
        print("==> 未启用子采样，使用全部样本")

    total_size = len(dataset)
    print(f"==> 本次收集样本数: {total_size}")

    global_indices = list(range(total_size))
    shard_indices = global_indices[rank::world_size]
    print(f"==> [rank {rank}] 分配样本数: {len(shard_indices)}")

    # 准备输出文件
    responses_file = output_dir / f"{dataset_name}_responses.jsonl"
    shard_responses_file = output_dir / f"{dataset_name}_responses.rank{rank}.jsonl"
    done_dir = output_dir / "_rank_done"
    features_dir = output_dir / f"{dataset_name}_{num_samples}_features"
    if save_hidden_states:
        features_dir.mkdir(parents=True, exist_ok=True)
    feature_writer = FeatureChunkPtWriter(features_dir, rank, features_per_file) if save_hidden_states else None
    
    # 清空或创建分片响应文件
    with open(shard_responses_file, "w", encoding="utf-8") as f:
        pass
        
    num_processed_responses = 0

    if batch_size < 1:
        raise ValueError(f"batch_size 必须 >= 1，当前为: {batch_size}")
    
    print(f"==> [rank {rank}] 开始生成响应...")
    
    try:
        # 按 batch_size 分批处理，每张卡一次生成多条样本
        num_local_samples = len(shard_indices)
        processed_local = 0
        progress_iter = range(0, num_local_samples, batch_size)

        for batch_start in tqdm(progress_iter, disable=not is_main_process):
            batch_shard_indices = shard_indices[batch_start:batch_start + batch_size]
            batch_examples = [dataset[idx] for idx in batch_shard_indices]
            batch_source_indices = [
                int(selected_global_indices[idx]) if selected_global_indices is not None else int(idx)
                for idx in batch_shard_indices
            ]

            batch_prompts = [
                ex["turns"][0] if isinstance(ex["turns"], list) else ex["turns"]
                for ex in batch_examples
            ]
            batch_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for prompt in batch_prompts
            ]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
            ).to(device)

            # left padding 时，每条样本真实prompt最后一个token位于同一索引
            prompt_max_len = int(inputs["input_ids"].shape[1])

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            full_target_hidden = None
            layer_idx_mapping = None
            if save_hidden_states:
                # 一次前向同时拿到整批样本的各层hidden
                full_target_hidden, layer_idx_mapping = build_generation_aligned_hidden_states_all_layers(
                    outputs.hidden_states
                )

            batch_sequences = outputs.sequences
            for i, shard_idx in enumerate(batch_shard_indices):
                generated_ids = batch_sequences[i]
                prompt_len_i = int(inputs["attention_mask"][i].sum().item())
                prompt_start_i = max(prompt_max_len - prompt_len_i, 0)
                unpadded_input_ids = inputs["input_ids"][i, prompt_start_i:prompt_max_len]

                # 去掉末尾pad，保留中间eos等真实生成token
                if tokenizer.pad_token_id is not None:
                    non_pad_pos = (generated_ids != tokenizer.pad_token_id).nonzero(as_tuple=False).squeeze(-1)
                    if non_pad_pos.numel() > 0:
                        valid_end = int(non_pad_pos[-1].item()) + 1
                    else:
                        valid_end = 0
                else:
                    valid_end = int(generated_ids.shape[0])

                response_start = prompt_max_len
                response_ids = generated_ids[response_start:valid_end]

                if save_hidden_states and valid_end > 0:
                    # 仅保留：prompt最后一个token + response全部token
                    align_start = max(prompt_max_len - 1, 0)
                    hidden_end = min(valid_end, int(full_target_hidden.shape[1]))
                    aligned_target_hidden = full_target_hidden[i:i + 1, align_start:hidden_end, :]
                    aligned_token_ids = generated_ids[align_start:valid_end]

                    target_hidden = aligned_target_hidden.squeeze(0).cpu()
                    feature_writer.write(
                        shard_idx,
                        {
                            "target_hidden": target_hidden,
                            "aligned_token_ids": aligned_token_ids.cpu(),
                            "align_start": int(align_start),
                            "layer_idx_mapping": layer_idx_mapping,
                            "num_layers": num_target_layers,
                            "hidden_size": int(model.config.hidden_size),
                            "hidden_layout": "concat_all_decoder_layers",
                            "token_hidden_alignment": "for token t_i, target_hidden[i] is the hidden used to predict t_i",
                            "token_span": "prompt_last_token_and_all_response_tokens",
                        },
                    )

                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                response_data = {
                    "idx": shard_idx,
                    "source_idx": batch_source_indices[i],
                    "prompt": batch_prompts[i],
                    "response": response,
                    "input_ids": unpadded_input_ids.cpu().tolist(),
                    "response_ids": response_ids.cpu().tolist(),
                }
                with open(shard_responses_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(response_data, ensure_ascii=False) + "\n")
                num_processed_responses += 1

            processed_local += len(batch_shard_indices)
            if processed_local % 100 == 0:
                print(f"[rank {rank}] 已处理样本数: {processed_local}/{num_local_samples}")
    except Exception as e:
        write_rank_done_marker(
            done_dir=done_dir,
            dataset_name=dataset_name,
            rank=rank,
            num_processed=num_processed_responses,
            status="error",
            error=f"{type(e).__name__}: {e}",
        )
        finalize_distributed(pg_initialized)
        raise

    if feature_writer is not None:
        feature_writer.close()

    if world_size > 1:
        write_rank_done_marker(
            done_dir=done_dir,
            dataset_name=dataset_name,
            rank=rank,
            num_processed=num_processed_responses,
            status="ok",
            error=None,
        )

    if is_distributed and use_process_group:
        dist.barrier()

    # 主进程合并响应分片
    if is_main_process:
        if world_size > 1 and sync_mode == "file":
            print(f"==> [rank {rank}] 等待所有rank完成标记...")
            statuses = wait_for_all_rank_markers(
                done_dir=done_dir,
                dataset_name=dataset_name,
                world_size=world_size,
                timeout_seconds=merge_wait_timeout_seconds,
                poll_interval_seconds=merge_poll_interval_seconds,
            )
            failed = [item for item in statuses if item.get("status") != "ok"]
            if failed:
                raise RuntimeError(f"检测到失败rank标记: {failed}")

        merged_responses = []
        for shard_rank in range(world_size):
            shard_file = output_dir / f"{dataset_name}_responses.rank{shard_rank}.jsonl"
            if not shard_file.exists():
                continue
            with open(shard_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    merged_responses.append(json.loads(line))

        merged_responses.sort(key=lambda x: x["idx"])

        print(f"==> 保存合并响应到: {responses_file}")
        with open(responses_file, "w", encoding="utf-8") as f:
            for response_data in merged_responses:
                f.write(json.dumps(response_data, ensure_ascii=False) + "\n")

        print(f"==> 数据收集完成!")
        print(f"   - 响应文件: {responses_file}")
        print(f"   - 共收集 {len(merged_responses)} 个样本")
        if save_hidden_states:
            total_chunks = (total_size + features_per_file - 1) // features_per_file
            for chunk_id in range(total_chunks):
                merged_chunk = features_dir / f"chunk_{chunk_id:05d}.pt"
                merged_data = {}
                for shard_rank in range(world_size):
                    shard_chunk = features_dir / f"chunk_{chunk_id:05d}.rank{shard_rank}.pt"
                    if not shard_chunk.exists():
                        continue
                    shard_data = torch.load(shard_chunk, map_location="cpu", weights_only=False)
                    merged_data.update(shard_data)
                    shard_chunk.unlink(missing_ok=True)
                if merged_data:
                    torch.save(merged_data, merged_chunk)
            print(f"   - 特征目录: {features_dir}")
    
    # 保存训练配置信息
    if is_main_process:
        config_file = output_dir / f"{dataset_name}_config.json"
        config = {
            "model_path": model_path,
            "dataset_name": dataset_name,
            "num_samples": total_size,
            "original_dataset_size": original_total_size,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "sampling_seed": sampling_seed,
            "selection_file": str(effective_selection_file),
            "sampling_reused": reused_sampling,
            "sampling_stats": sampling_stats,
            "num_target_layers": num_target_layers,
            "num_draft_layers": num_draft_layers,
            "target_layer_ids": target_layer_ids,
            "saved_hidden_layers": "all_decoder_layers",
            "layer_idx_mapping": {
                str(layer_idx): {
                    "feature_start": int(layer_idx * model.config.hidden_size),
                    "feature_end": int((layer_idx + 1) * model.config.hidden_size),
                }
                for layer_idx in range(num_target_layers)
            },
            "hidden_layout": "concat_all_decoder_layers",
            "token_hidden_alignment": "for token t_i, target_hidden[i] is the hidden used to predict t_i",
            "save_hidden_states": save_hidden_states,
            "features_dir": str(features_dir) if save_hidden_states else None,
            "feature_format": "chunk_pt" if save_hidden_states else None,
            "features_per_file": features_per_file if save_hidden_states else None,
            "world_size": world_size,
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"   - 配置文件: {config_file}")

    if is_distributed and use_process_group:
        dist.barrier()

    finalize_distributed(pg_initialized)
    
    # 因为前面取消了 all_responses 全局内存保留，这里改为主进程按需返回聚合数据或 None 
    if is_main_process and 'merged_responses' in locals():
        return merged_responses
    return None


def main():
    parser = argparse.ArgumentParser(description="收集训练数据")
    parser.add_argument("--model_path", type=str, required=True, help="目标模型路径")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["gsm8k", "math500", "aime24", "aime25", "alpaca", 
                   "mt-bench", "humaneval", "mbpp", "lbpp", "nemotron"],
                       help="数据集名称")
    parser.add_argument("--output_dir", type=str, default="./training_data", 
                       help="输出目录")
    parser.add_argument("--max_new_tokens", type=int, default=4096, 
                       help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.0, 
                       help="采样温度")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="批大小")
    parser.add_argument("--num_samples", type=int, default=None, 
                       help="采样数量（None表示全部）")
    parser.add_argument("--sampling_seed", type=int, default=42,
                       help="任务均衡采样随机种子")
    parser.add_argument("--selection_file", type=str, default=None,
                       help="采样索引记录文件（默认: train_exp/sample_selection/<dataset>_sample_selection_n<样本数>.json）")
    parser.add_argument("--num_draft_layers", type=int, default=8, 
                       help="草稿模型层数")
    parser.add_argument("--save_hidden_states", action="store_true",
                       help="保存目标模型隐藏状态用于离线训练")
    parser.add_argument("--features_per_file", type=int, default=1000,
                       help="每个特征文件保存的样本数量")
    parser.add_argument("--sync_mode", type=str, default="file", choices=["file", "barrier"],
                       help="多卡同步方式：file(默认，文件标记同步) / barrier(进程组barrier)")
    parser.add_argument("--dist_backend", type=str, default="gloo", choices=["gloo", "nccl"],
                       help="当 sync_mode=barrier 时使用的分布式后端")
    parser.add_argument("--dist_timeout_seconds", type=int, default=7200,
                       help="当 sync_mode=barrier 时，进程组通信超时秒数")
    parser.add_argument("--merge_wait_timeout_seconds", type=int, default=10800,
                       help="主进程等待所有rank完成的最大秒数（sync_mode=file）")
    parser.add_argument("--merge_poll_interval_seconds", type=float, default=5.0,
                       help="主进程轮询rank完成标记的间隔秒数（sync_mode=file）")
    
    args = parser.parse_args()
    
    collect_data_from_target_model(
        model_path=args.model_path,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        sampling_seed=args.sampling_seed,
        selection_file=args.selection_file,
        num_draft_layers=args.num_draft_layers,
        save_hidden_states=args.save_hidden_states,
        features_per_file=args.features_per_file,
        sync_mode=args.sync_mode,
        dist_backend=args.dist_backend,
        dist_timeout_seconds=args.dist_timeout_seconds,
        merge_wait_timeout_seconds=args.merge_wait_timeout_seconds,
        merge_poll_interval_seconds=args.merge_poll_interval_seconds,
    )


if __name__ == "__main__":
    main()

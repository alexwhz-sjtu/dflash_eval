# DFlash 投机解码框架说明

## 一、核心设计思想

### 1.1 任务定义
小模型（Draft Model）接收大模型（Target Model）最后一个token的**所有层hidden states拼接**，预测接下来**B个token**（B=block_size，默认16）。

### 1.2 关键特点
- **无自回归依赖**：小模型预测块内token时，**不依赖**ground truth token的embedding
- **双向注意力**：块内token之间是双向注意力（非因果）
- **mask token填充**：除第一个token外，其余B-1个位置都用mask_token_id填充作为输入

---

## 二、推理流程（Inference Pipeline）

### 2.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Prefill 阶段                                    │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 输入: prompt tokens     │ ──────┐
│ [t1, t2, ..., tn]       │       │
└─────────────────────────┘       │
    │                             │
    ▼                             │
┌─────────────────────────┐       │
│ Target Model Forward    │       │
│ - output_hidden_states  │       │
│ - use_cache=True        │       │
└─────────────────────────┘       │
    │                             │
    ▼                             │
┌─────────────────────────┐       │
│ 提取 target_hidden      │       │
│ (拼接所有层的最后位置)   │       │
└─────────────────────────┘       │
    │                             │
    ▼                             │
┌─────────────────────────┐       │
│ 采样第一个token         │◄──────┘
│ (大模型生成，非投机)     │
└─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              Decode 阶段（循环）                             │
└─────────────────────────────────────────────────────────────────────────────┘

第1步: 小模型生成候选块
┌────────────────────────────────────────────────────────┐
│ 输入:                                                  │
│   - target_hidden: [1, 1, H] (大模型最后一个token特征) │
│   - block_output_ids: [1, B] = [real_token, mask, ...]│
│   - noise_embedding: 通过大模型embedding层获取         │
│   - position_ids: [start-1 : start+B-1]               │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │ Draft Model Forward │
              │ - target_hidden作为K/V │
              │ - noise_embedding作为Q │
              │ - 块内双向注意力     │
              └────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │ draft_output: [1, B, H] │
              └────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │ 过大模型lm_head    │
              │ draft_logits: [1, B-1, V] │
              │ (只取后B-1个位置)   │
              └────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │ sample() 采样B-1个token │
              │ 填入block_output_ids │
              └────────────────────┘

第2步: 大模型验证
┌────────────────────────────────────────────────────────┐
│ 输入: block_output_ids [1, B] (小模型生成的完整块)     │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │ Target Model Forward │
              │ (使用KV Cache加速)  │
              └────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │ posterior_logits   │
              │ 采样B个token作为验证 │
              └────────────────────┘

第3步: 接受/拒绝逻辑
┌────────────────────────────────────────────────────────┐
│ 对比: block_output_ids[1:] vs posterior[:-1]           │
│                                                        │
│ acceptance_length = 连续匹配数量                       │
│   = (draft_tokens == posterior_tokens).cumprod().sum() │
│                                                        │
│ 更新序列:                                              │
│   - 接受: block_output_ids[:acceptance_length+1]      │
│   - 拒绝位置: 使用posterior[acceptance_length]        │
└────────────────────────────────────────────────────────┘

第4步: 更新状态
┌────────────────────────────────────────────────────────┐
│ - start += acceptance_length + 1                       │
│ - 裁剪target KV Cache到start位置                       │
│ - 裁剪draft KV Cache到start位置                        │
│ - 提取新的target_hidden (从验证结果中)                  │
└────────────────────────────────────────────────────────┘
```

### 2.2 核心代码实现

```python
# ========== Prefill 阶段 ==========
output = target_model(
    input_ids,                              # [1, N] prompt
    position_ids=position_ids[:, :N],
    past_key_values=past_key_values_target, # DynamicCache
    use_cache=True,
    output_hidden_states=True,              # 必须输出hidden states
)

# 提取最后一个token的所有层hidden states并拼接
# extract_context_feature在model/utils.py中实现
def extract_context_feature(hidden_states, layer_ids):
    selected_states = []
    for layer_id in layer_ids:  # 如 [0, 9, 17, 25, 33]
        selected_states.append(hidden_states[layer_id + 1])  # +1因为第0层是embedding
    return torch.cat(selected_states, dim=-1)  # [batch, seq_len, sum(hidden_sizes)]

target_hidden = extract_context_feature(
    output.hidden_states,
    model.target_layer_ids  # 配置中指定要提取哪些层
)[:, -1:, :]  # 只取最后一个位置: [1, 1, H]

# 采样第一个token（非投机）
first_token = sample(output.logits, temperature)
output_ids[:, N:N+1] = first_token

# ========== Decode 阶段（循环）==========
while start < max_length:
    # --- Step 1: 构造块输入 ---
    # block_output_ids: [1, B] = [real_token, mask_id, mask_id, ...]
    block_output_ids = output_ids[:, start:start+block_size].clone()

    # 通过大模型embedding层获取noise_embedding
    noise_embedding = target_model.model.embed_tokens(block_output_ids)

    # --- Step 2: 小模型生成 ---
    draft_hidden = draft_model(
        target_hidden=target_hidden,    # [1, 1, H] 来自大模型最后一个token
        noise_embedding=noise_embedding, # [1, B, hidden_size]
        position_ids=position_ids[:, start-1:start+block_size-1],
        past_key_values=past_key_values_draft,
        use_cache=True,
        is_causal=False,                # 关键！块内双向注意力
    )
    # draft_hidden: [1, B, hidden_size]

    # 过大模型lm_head得到logits
    draft_logits = target_model.lm_head(draft_hidden[:, -(block_size-1):, :])
    # draft_logits: [1, B-1, vocab_size]

    # 采样B-1个token
    draft_tokens = sample(draft_logits, temperature)
    block_output_ids[:, 1:] = draft_tokens

    # 裁剪draft KV Cache
    past_key_values_draft.crop(start)

    # --- Step 3: 大模型验证 ---
    output = target_model(
        block_output_ids,               # [1, B] 小模型生成的块
        position_ids=block_position_ids,
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
    )

    posterior = sample(output.logits, temperature)  # [1, B] 大模型验证结果

    # --- Step 4: 接受/拒绝 ---
    # 计算接受长度：连续匹配的数量
    matches = (block_output_ids[:, 1:] == posterior[:, :-1])  # [1, B-1]
    acceptance_length = matches.cumprod(dim=1).sum(dim=1).item()  # 连续True的数量

    # 更新输出序列
    output_ids[:, start:start+acceptance_length+1] = block_output_ids[:, :acceptance_length+1]
    output_ids[:, start+acceptance_length+1] = posterior[:, acceptance_length]

    # --- Step 5: 更新状态 ---
    start += acceptance_length + 1
    past_key_values_target.crop(start)

    # 提取新的target_hidden用于下一轮
    # 从验证结果中提取已接受token的hidden states
    target_hidden = extract_context_feature(
        output.hidden_states,
        model.target_layer_ids
    )[:, :acceptance_length+1, :]
```

### 2.3 小模型输入细节

```python
# DFlashDraftModel.forward 输入参数
def forward(
    self,
    position_ids: torch.LongTensor,           # 位置编码 [1, B+1] 或 [1, B]
    attention_mask: Optional[torch.Tensor],    # 稀疏注意力掩码（训练时用）
    noise_embedding: Optional[torch.Tensor],   # 块token的embedding [1, B, hidden_size]
    target_hidden: Optional[torch.Tensor],     # 核心输入！大模型hidden states [1, ctx_len, H]
    past_key_values: Optional[Cache],          # KV Cache
    use_cache: bool = False,
):
    # 关键处理：通过fc层投影target_hidden
    # fc: Linear(num_target_layers * hidden_size, hidden_size)
    target_hidden = self.hidden_norm(self.fc(target_hidden))

    # 在每一层Transformer中，attention的K/V来自target_hidden + noise_embedding拼接
    # Q只来自noise_embedding
```

### 2.4 Attention机制

```python
# Qwen3DFlashAttention.forward 核心逻辑
def forward(
    self,
    hidden_states: torch.Tensor,      # noise_embedding [1, B, hidden_size]
    target_hidden: torch.Tensor,      # 大模型特征 [1, ctx_len, H]
    position_embeddings: tuple,       # RoPE
    attention_mask: Optional[torch.Tensor],
    ...
):
    # Q投影：只来自noise_embedding（要生成的token）
    q = self.q_proj(hidden_states)

    # K/V投影：来自target_hidden + noise_embedding拼接
    kv_input = torch.cat([target_hidden, hidden_states], dim=1)
    k = self.k_proj(kv_input)  # [1, ctx_len+B, ...]
    v = self.v_proj(kv_input)

    # 应用RoPE
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # Attention计算
    attn_output = attention_fn(q, k, v, attention_mask, ...)
```

---

## 三、训练流程（Training Pipeline）

### 3.1 数据准备

#### 离线数据收集
```python
# collect_data_batch.py 收集的数据结构
{
    "idx": 0,                                    # 样本索引
    "input_ids": [...],                          # prompt token ids
    "response_ids": [...],                       # 大模型生成的response
    "target_hidden": torch.Tensor,               # 关键！所有层的hidden states
}
```

目录结构：
```
train_data/
├── nemotron_responses.jsonl      # tokenized responses
└── nemotron_features/
    ├── chunk_00000.pt            # 每10000条一个chunk
    ├── chunk_00001.pt
    └── ...
```

#### target_hidden格式
```python
# 形状: [seq_len, num_target_layers * hidden_size]
# 例如：对于36层模型，hidden_size=4096
# target_hidden.shape = [seq_len, 36 * 4096] = [seq_len, 147456]

# 收集时保存所有层的最后一维hidden state并拼接
target_hidden = torch.cat([
    hidden_states[1],   # layer 0
    hidden_states[2],   # layer 1
    ...
    hidden_states[37],  # layer 36
], dim=-1)
```

### 3.2 训练数据流程

```python
# SimplifiedDataset 加载流程

1. 加载response jsonl文件
2. 对每个样本随机采样anchor_positions（最多512个）
3. 从features目录加载对应的target_hidden
4. collate_fn将batch内的样本padding到相同长度

返回的batch结构：
{
    "input_ids": [batch_size, max_seq_len],
    "attention_mask": [batch_size, max_seq_len],
    "anchor_positions": [batch_size, anchors_per_sequence],  # 每个序列的锚点位置
    "target_hidden": [batch_size, max_seq_len, feature_dim], # 离线特征
}
```

### 3.3 训练Step流程

```python
def train_step(self, batch):
    # 输入
    input_ids = batch["input_ids"]           # [B, L]
    anchor_positions = batch["anchor_positions"]  # [B, N_anchors]
    target_hidden = batch["target_hidden"]   # [B, L, H] 离线特征

    # 对每个样本，遍历其anchor_positions
    for i in range(batch_size):
        valid_len = attention_mask[i].sum().item()

        seq_target_hidden_list = []
        seq_input_ids_list = []
        seq_labels_list = []
        seq_position_ids_list = []

        for anchor_pos in anchor_positions[i]:
            if anchor_pos + block_size > valid_len:
                continue

            # 1. 提取target_hidden（锚点位置）
            anchor_context_hidden = target_hidden[i:i+1, anchor_pos:anchor_pos+1, :]
            seq_target_hidden_list.append(anchor_context_hidden)

            # 2. 构造块标签（ground truth）
            block_labels = input_ids[i, anchor_pos:anchor_pos+block_size]
            seq_labels_list.append(block_labels)

            # 3. 构造块输入（第一个token是真实的，其余是mask）
            block_input_ids = input_ids[i, anchor_pos:anchor_pos+block_size].clone()
            block_input_ids[1:] = mask_token_id  # 关键！后续位置用mask填充
            seq_input_ids_list.append(block_input_ids)

            # 4. 构造position_ids（锚点前一个位置开始）
            block_position_ids = torch.arange(
                anchor_pos-1, anchor_pos+block_size, device=device
            )
            seq_position_ids_list.append(block_position_ids)

        # 联合块训练：一个序列的所有锚点块一起前向
        seq_loss, _ = self._forward_joint_blocks_for_sequence(
            seq_target_hidden_list,
            seq_input_ids_list,
            seq_labels_list,
            seq_position_ids_list,
        )

        # 加权并反向传播
        weighted_seq_loss = seq_loss * (num_blocks / total_valid_blocks)
        weighted_seq_loss.backward()
```

### 3.4 联合块训练（Joint Block Training）

```python
def _forward_joint_blocks_for_sequence(self, ...):
    # 将多个锚点块拼接在一起，一次前向

    # 1. 拼接target_hidden: [1, num_blocks, H]
    seq_target_hidden = torch.cat(seq_target_hidden_list, dim=1)

    # 2. 堆叠input_ids: [num_blocks, block_size]
    seq_input_ids = torch.stack(seq_input_ids_list, dim=0)
    seq_labels = torch.stack(seq_labels_list, dim=0)

    # 3. 获取noise embedding
    seq_noise_embedding = target_embed_tokens(seq_input_ids)
    seq_noise_embedding = seq_noise_embedding.reshape(1, num_blocks * block_size, -1)

    # 4. 构造联合position_ids
    ctx_position_ids = seq_position_ids[:, 0]  # [num_blocks]
    noise_position_ids = seq_position_ids[:, 1:].reshape(-1)  # [num_blocks * (block_size-1)]
    joint_position_ids = torch.cat([ctx_position_ids, noise_position_ids], dim=0).unsqueeze(0)

    # 5. 构造稀疏注意力掩码（块间隔离）
    joint_attention_mask = self._build_joint_sparse_attention_mask(
        num_blocks=num_blocks,
        block_size=block_size,
    )
    # mask shape: [1, 1, q_len, kv_len]
    # q_len = num_blocks * block_size
    # kv_len = num_blocks + q_len (ctx + noise)

    # 6. 前向
    draft_hidden_joint = draft_model(
        target_hidden=seq_target_hidden,        # [1, num_blocks, H]
        noise_embedding=seq_noise_embedding,     # [1, num_blocks*B, hidden_size]
        position_ids=joint_position_ids,
        attention_mask=joint_attention_mask,     # 稀疏掩码确保块间不互相看到
        past_key_values=None,
        use_cache=False,                         # 训练时不使用KV Cache
    )

    # 7. 分割回块并计算损失
    draft_hidden_blocks = draft_hidden_joint.reshape(num_blocks, block_size, -1)
    block_logits = target_lm_head(draft_hidden_blocks)

    # 使用WeightedBlockLoss（块内位置加权）
    seq_loss = loss_fn(block_logits[:, 1:, :], seq_labels[:, 1:])
```

### 3.5 稀疏注意力掩码

```python
def _build_joint_sparse_attention_mask(num_blocks, block_size, device):
    """
    构造块隔离的稀疏注意力掩码：
    - 每个query只能看到同块的noise token
    - query只能看到本块的target context token
    - 禁止跨块注意力
    """
    q_len = num_blocks * block_size

    # query属于哪个块
    query_block_ids = torch.arange(q_len, device=device) // block_size  # [0,0,0...,1,1,1...]

    # target context属于哪个块
    target_block_ids = torch.arange(num_blocks, device=device)  # [0,1,2...]

    # noise属于哪个块
    noise_block_ids = torch.arange(q_len, device=device) // block_size

    # 合并key的位置
    key_block_ids = torch.cat([target_block_ids, noise_block_ids], dim=0)

    # mask: query只能看到同块的key
    mask_2d = query_block_ids.unsqueeze(1).eq(key_block_ids.unsqueeze(0))

    return mask_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]
```

### 3.6 损失函数（WeightedBlockLoss）

```python
class WeightedBlockLoss(nn.Module):
    """
    块内位置加权的交叉熵损失
    - 早期位置错误影响更大，使用指数衰减权重
    - w_k = exp(-(k-1)/gamma)，gamma=7（block_size=16时）
    """
    def __init__(self, block_size=16, gamma=7.0):
        # 预计算权重
        k = torch.arange(1, block_size + 1, dtype=torch.float32)
        weights = torch.exp(-(k - 1) / gamma)
        weights = weights / weights.mean()  # 归一化使平均为1

    def forward(self, logits, labels):
        # logits: [batch, block_size-1, vocab_size]
        # labels: [batch, block_size-1]

        # 计算每个位置的CE loss
        loss_per_token = F.cross_entropy(logits_flat, labels_flat, reduction='none')
        loss_per_token = loss_per_token.reshape(batch_size, block_size-1)

        # 应用位置权重
        weighted_loss = loss_per_token * self.position_weights.unsqueeze(0)

        return weighted_loss.mean()
```

---

## 四、框架集成要点

### 4.1 模型配置（model_config.json）

```json
{
  "num_hidden_layers": 5,        // 小模型层数
  "hidden_size": 4096,           // 小模型hidden size
  "intermediate_size": 12288,    // FFN中间层
  "block_size": 16,              // 块大小B
  "num_target_layers": 36,       // 大模型层数（用于target_hidden维度计算）
  "dflash_config": {
    "mask_token_id": 151669,     // mask token
    "target_layer_ids": [1,9,17,25,33]  // 提取哪些层的hidden（可选）
  }
}
```

### 4.2 关键接口总结

```python
# ========== 推理接口 ==========

# 1. 加载模型
draft_model = DFlashDraftModel.from_pretrained(
    "path/to/draft_model",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)

target_model = AutoModelForCausalLM.from_pretrained(
    "path/to/target_model",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)

# 2. 生成
output_ids = draft_model.spec_generate(
    target=target_model,
    input_ids=input_ids,           # [1, N] prompt
    max_new_tokens=2048,
    stop_token_ids=[eos_token_id],
    temperature=0.0,
)

# ========== 训练接口 ==========

# 1. 准备数据（离线收集）
# 见 train/data/collect_data_batch.py

# 2. 训练
python train_exp/train_anchor_batch.py \
    --target_model_path /path/to/target \
    --draft_config_path ./train/model_config.json \
    --data_file /path/to/offline_data \
    --output_dir ./output \
    --block_size 16 \
    --num_epochs 6 \
    --batch_size 4

# 3. 关键类
- DFlashTrainer: 训练器主类
- SimplifiedDataset: 数据集
- WeightedBlockLoss: 损失函数
```

### 4.3 数据流总结

```
推理时：
  Input: prompt tokens
  Target Model (prefill) → hidden_states → extract_context_feature → target_hidden
  Draft Model (target_hidden, mask_tokens) → draft_hidden
  Target Model lm_head (draft_hidden) → draft_logits → draft_tokens
  Target Model (verify) → posterior → accept/reject

训练时：
  Offline Data: (input_ids, response_ids, target_hidden)
  Sample anchors → construct blocks (mask token填充)
  Joint forward with sparse attention mask
  WeightedBlockLoss backward
```

---

## 五、与标准投机解码的区别

| 特性 | 标准投机解码 (EAGLE等) | DFlash |
|------|------------------------|--------|
| 输入依赖 | 依赖已生成token的embedding | 只依赖mask token + target_hidden |
| 注意力 | 因果注意力 | 块内双向注意力 |
| 训练数据 | 需要在线生成 | 离线预收集 |
| 收敛速度 | 较慢（自回归依赖） | 更快（块级训练） |
| 适用场景 | 通用 | 特定target model |

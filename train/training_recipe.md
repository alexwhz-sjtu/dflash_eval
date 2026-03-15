# Role
你是一位精通 PyTorch 和 LLM 训练架构的资深算法工程师。我需要进行一个以扩散基础草稿模型加速自回归模型投机解码的训练过程。

# Context
我已经有了模型架构定义和推理代码dflash.spec_generate函数，现在需要完善的**训练脚本 (Training Script)** 和 **数据加载器 (DataLoader)**。
请基于以下论文细节编写代码，确保逻辑严密且高效。

# 1. 模型架构要求 (Model Architecture)
- **草稿模型 (Draft Model)**：轻量级 Block Diffusion Transformer。
  - 层数：默认 5 层 (Qwen3-Coder 用 8 层)。
  - **共享参数**：与目标模型 (Target Model) 共享 Token Embedding 和 LM Head，且这两部分在训练时**冻结 (Frozen)**。仅训练 Transformer 层。
- **KV Injection (核心机制)**：
  - 从目标模型提取隐藏状态 (Hidden States)：均匀采样 5 层 (从第 2 层到倒数第 3 层)。
  - 特征融合：将提取的隐藏状态拼接并通过轻量级投影层融合为目标上下文特征 (Target Context Feature)。
  - **注入方式**：将这些特征直接注入到草稿模型每一层的 **Key 和 Value 投影** 中 (存入 KV Cache)，而不仅仅是作为输入嵌入。
  - 代码需体现 `KVCache` 的修改以支持额外注入的 context feature。
  这部分我的推理代码中已经完成

# 2. 数据构造与掩码策略 (Data & Masking)
- **数据来源**：
 - NVIDIA Nemotron Post-Training Dataset V2 (约800K)
 - CodeAlpaca

使用目标模型生成的响应 (Response) 作为训练标签，而非原始数据集标签。
- **掩码块构造 (Random Sampling of Masked Blocks)**：
  - 不要均匀分块。从响应中随机采样锚点 token (Anchor Tokens)。
  - 每个锚点作为块的起始位置，块内剩余位置被掩码 (Masked)。
  - 块大小 (Block Size)：默认 16 (LLaMA 实验为 10)。
  - 每个序列随机采样 512 个 anchor 位置。
- **注意力掩码 (Attention Mask)**：
  - 使用稀疏注意力 (Sparse Attention)。
  - 块内：双向注意力 (Bidirectional)。
  - 块间：无注意力 (No Attention)，防止信息泄漏。
  - 请使用 `Flex Attention` 或自定义 `attention_mask` 实现。

# 3. 损失函数 (Loss Function)
- 使用加权交叉熵损失 (Weighted Cross-Entropy Loss)。
- 块内早期位置的错误影响更大，需应用指数衰减权重。
- 对于块内位置 $k$ (从 1 开始)，权重公式为：
  $$
  w_k = \exp\left(-\frac{k - 1}{\gamma}\right)
  $$
  其中 $\gamma$ 控制衰减率 (块大小 16 时 $\gamma=7$；块大小 10 时 $\gamma=5$)。
- 请在代码中明确实现该加权逻辑。

# 4. 训练超参数 (Hyperparameters)
- **优化器**：AdamW
- **学习率**：$6 \times 10^{-4}$
- **训练轮数**：6 epochs
- **梯度裁剪**：阈值 1.0
- **调度策略**：余弦调度 (Cosine Schedule)，warmup 比例 0.04
- **最大序列长度**：3072 tokens (Qwen3-Coder 为 4096)
- **训练模式**：支持离线模式 (预先计算并缓存目标隐藏特征，训练时加载) 以降低开销。这部分在/share/wanghanzhen/SpeculativeDecoding/NIPS26/dflash/train/data中已完成。检查是否有错误即可

# 5. 代码输出要求
1. **模块化**：将 Dataset、Model、Loss、Trainer 分离为不同类或函数。
2. **效率**：使用 `torch.compile` 或 FlashAttention 优化注意力计算。。
3. **公式**：代码文档字符串中的数学公式请使用 LaTeX 格式。
4. **兼容性**：假设我已有一个冻结的 `target_model` 实例可用于提取特征。
5. 不要再写新的md文档了
请先生成核心代码结构，重点展示 `KVInjectionLayer`、`MaskedBlockDataset` 和 `WeightedLoss` 的实现。

"""
训练一步

训练策略：
1. 使用目标模型提取完整序列的隐藏层特征
2. 先用草稿模型因果注意力prefill得到kvcache
3. 使用锚点顺序遍历所有token
4. 块内双向注意力预测块内容
5. 计算加权损失并更新
"""
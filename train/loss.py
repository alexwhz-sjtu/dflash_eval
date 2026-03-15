"""
加权损失函数
实现块内位置加权的交叉熵损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBlockLoss(nn.Module):
    r"""
    块内位置加权的交叉熵损失
    
    损失函数说明：
    - 块内早期位置的错误影响更大，需应用指数衰减权重
    - 对于块内位置 k (从 1 开始)，权重公式为：
      
      $$w_k = \exp\left(-\frac{k - 1}{\gamma}\right)$$
      
      其中 $\gamma$ 控制衰减率：
      - 块大小 16 时 $\gamma=7$
      - 块大小 10 时 $\gamma=5$
      
    Args:
        block_size: 块大小
        gamma: 衰减率参数（None时自动根据block_size设置）
        ignore_index: 忽略的索引（默认-100）
    """
    
    def __init__(
        self, 
        block_size: int = 16, 
        gamma: float = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.block_size = block_size
        self.ignore_index = ignore_index
        
        # 自动设置gamma
        if gamma is None:
            if block_size == 16:
                gamma = 7.0
            elif block_size == 10:
                gamma = 5.0
            else:
                # 线性插值估算
                gamma = 5.0 + (block_size - 10) * (7.0 - 5.0) / (16 - 10)
        
        self.gamma = gamma
        
        # 预计算权重
        # k 从 1 到 block_size
        # w_k = exp(-(k-1)/gamma)
        k = torch.arange(1, block_size + 1, dtype=torch.float32)
        weights = torch.exp(-(k - 1) / gamma)
        
        # 归一化权重（使得平均权重为1）
        weights = weights / weights.mean()
        
        # 注册为buffer（不参与梯度更新但会随模型移动）
        self.register_buffer('position_weights', weights)
        
        print(f"==> 初始化加权损失函数")
        print(f"   - 块大小: {block_size}")
        print(f"   - gamma: {gamma:.2f}")
        print(f"   - 位置权重: {weights.tolist()}")
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算加权交叉熵损失
        
        Args:
            logits: 模型输出 [batch_size, block_size, vocab_size]
            labels: 标签 [batch_size, block_size]
        
        Returns:
            loss: 标量损失值
        """
        batch_size, block_size, vocab_size = logits.shape
        
        # 确保block_size匹配
        assert block_size == self.block_size, \
            f"输入的block_size ({block_size}) 与初始化的block_size ({self.block_size}) 不匹配"
        
        # 计算每个位置的交叉熵损失（不进行reduction）
        # logits: [batch_size, block_size, vocab_size]
        # labels: [batch_size, block_size]
        logits_flat = logits.view(-1, vocab_size)  # [batch_size * block_size, vocab_size]
        labels_flat = labels.view(-1)  # [batch_size * block_size]
        
        # 计算每个位置的loss（使用reduction='none'）
        loss_per_token = F.cross_entropy(
            logits_flat, 
            labels_flat, 
            reduction='none',
            ignore_index=self.ignore_index
        )  # [batch_size * block_size]
        
        # 重塑为 [batch_size, block_size]
        loss_per_token = loss_per_token.view(batch_size, block_size)
        
        # 应用位置权重
        # position_weights: [block_size]
        # 扩展为 [1, block_size] 以便广播
        weights = self.position_weights.unsqueeze(0)  # [1, block_size]
        
        # 加权损失
        weighted_loss = loss_per_token * weights  # [batch_size, block_size]
        
        # 计算平均损失
        # 注意：需要考虑ignore_index的情况
        if self.ignore_index >= 0:
            # 创建mask，标记非ignore的位置
            mask = (labels != self.ignore_index).float()  # [batch_size, block_size]
            
            # 计算有效位置的加权损失
            weighted_loss = (weighted_loss * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            # 直接取平均
            weighted_loss = weighted_loss.mean()
        
        return weighted_loss
    
    def get_position_weights(self):
        """返回位置权重（用于分析）"""
        return self.position_weights.cpu().tolist()


class SimpleWeightedLoss(nn.Module):
    """
    简化版加权损失函数
    可以直接应用于标准的语言模型输出
    """
    
    def __init__(
        self, 
        block_size: int = 16, 
        gamma: float = None,
    ):
        super().__init__()
        self.block_size = block_size
        
        # 自动设置gamma
        if gamma is None:
            if block_size == 16:
                gamma = 7.0
            elif block_size == 10:
                gamma = 5.0
            else:
                gamma = 5.0 + (block_size - 10) * (7.0 - 5.0) / (16 - 10)
        
        self.gamma = gamma
        
        # 预计算权重
        k = torch.arange(1, block_size + 1, dtype=torch.float32)
        weights = torch.exp(-(k - 1) / gamma)
        weights = weights / weights.mean()
        
        self.register_buffer('position_weights', weights)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        计算加权交叉熵损失（更灵活的版本）
        
        Args:
            logits: 模型输出 [batch_size, seq_len, vocab_size]
            labels: 标签 [batch_size, seq_len]
            mask: 可选的掩码 [batch_size, seq_len]，标记哪些位置需要计算损失
        
        Returns:
            loss: 标量损失值
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # 计算交叉熵损失（不进行reduction）
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        loss_per_token = F.cross_entropy(
            logits_flat, 
            labels_flat, 
            reduction='none'
        )
        
        loss_per_token = loss_per_token.view(batch_size, seq_len)
        
        # 应用掩码（如果提供）
        if mask is not None:
            loss_per_token = loss_per_token * mask.float()
            
            # 对每个样本的每个块应用位置权重
            # 这里假设seq_len是block_size的整数倍
            num_blocks = seq_len // self.block_size
            
            if num_blocks > 0:
                # 重塑为 [batch_size, num_blocks, block_size]
                loss_reshaped = loss_per_token[:, :num_blocks * self.block_size].view(
                    batch_size, num_blocks, self.block_size
                )
                mask_reshaped = mask[:, :num_blocks * self.block_size].view(
                    batch_size, num_blocks, self.block_size
                )
                
                # 应用位置权重
                weights = self.position_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, block_size]
                weighted_loss = loss_reshaped * weights
                
                # 计算平均损失
                total_loss = (weighted_loss * mask_reshaped.float()).sum()
                total_count = mask_reshaped.float().sum().clamp(min=1.0)
                
                return total_loss / total_count
            else:
                # 如果seq_len < block_size，直接返回平均损失
                return (loss_per_token * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        else:
            # 没有掩码，假设是标准的块大小
            if seq_len == self.block_size:
                weights = self.position_weights.unsqueeze(0)
                weighted_loss = loss_per_token * weights
                return weighted_loss.mean()
            else:
                # 直接返回平均损失
                return loss_per_token.mean()


def test_weighted_loss():
    """测试加权损失函数"""
    print("==> 测试加权损失函数")
    
    # 创建损失函数
    loss_fn = WeightedBlockLoss(block_size=16, gamma=7.0)
    
    # 打印权重
    print(f"位置权重: {loss_fn.get_position_weights()}")
    
    # 创建模拟数据
    batch_size = 2
    block_size = 16
    vocab_size = 1000
    
    logits = torch.randn(batch_size, block_size, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, block_size))
    
    # 计算损失
    loss = loss_fn(logits, labels)
    print(f"损失值: {loss.item():.4f}")
    
    # 测试梯度
    logits.requires_grad_(True)
    loss = loss_fn(logits, labels)
    loss.backward()
    print(f"梯度范数: {logits.grad.norm().item():.4f}")
    
    print("==> 测试完成!")


if __name__ == "__main__":
    test_weighted_loss()

import torch
import torch.nn.functional as F
import math

def attention(Q, K, V):
    d_k = Q.size(-1)

    scores = Q @ K.transpose(-2, -1)   # 相似度
    scores = scores / math.sqrt(d_k)   # 缩放

    weights = F.softmax(scores, dim=-1)  # 概率

    output = weights @ V  # 加权求和
    return output, weights

# 示例
if __name__ == "__main__":
    Q = torch.rand(2, 4, 8)  # (batch_size, seq_len, d_k)
    K = torch.rand(2, 4, 8)
    V = torch.rand(2, 4, 8)

    output, weights = attention(Q, K, V)
    print("Output shape:", output.shape)  # (batch_size, seq_len, d_k)
    print("Weights shape:", weights.shape)  # (batch_size, seq_len, seq_len)
    print("Output:", output)
    print("Weights:", weights)
    
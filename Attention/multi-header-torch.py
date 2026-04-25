# write multiple headers to a file using torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k).float())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        print(f"Attention Weights shape: {attn_weights.shape}, Attention Output shape: {attn_output.shape}")
        
        return self.linear_out(attn_output)

# Example usage
if __name__ == "__main__":
    d_model = 4 # Dimension of the model
    num_heads = 2 # Each head will have a dimension of 2 (d_k = d_model // num_heads)
    batch_size = 1 # One word vector
    seq_length = 1 # One word vector
    
    # Create a MultiHeadAttention instance
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Example input (one word vector)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # Shape: (1, 4)
    
    # Forward pass through multi-head attention
    output = mha(x, x, x)  # Using the same input for query, key, and value
    print("Output of Multi-Head Attention:\n", output)

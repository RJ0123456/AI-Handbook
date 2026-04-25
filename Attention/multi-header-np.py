import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# 1. Setup Parameters
d_model = 4
num_heads = 2
d_k = d_model // num_heads  # 2

# Input: One word vector (e.g., "Apple")
x = np.array([[1.0, 2.0, 3.0, 4.0]]) # Shape: (1, 4)

# 2. Weights for 2 Heads (Randomly initialized for example)
# In reality, these are learned during training
W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model)
W_v = np.random.randn(d_model, d_model)
W_o = np.random.randn(d_model, d_model)

print("--- Multi-Head Attention ---")
# 3. Linear Transformations (Q, K, V)
Q = np.dot(x, W_q)
K = np.dot(x, W_k)
V = np.dot(x, W_v)
print(f"Q Shape: {Q.shape}, K Shape: {K.shape}, V Shape: {V.shape}")
print(f"Q:\n {Q}")
print(f"K:\n {K}")
print(f"V:\n {V}")

# 4. Multi-Head Split
# Reshape to (batch, num_heads, d_k)
Q_heads = Q.reshape(1, num_heads, d_k)
K_heads = K.reshape(1, num_heads, d_k)
V_heads = V.reshape(1, num_heads, d_k)
print(f"Q Heads Shape: {Q_heads.shape}")  # (1, 2, 2)
print(f"K Heads Shape: {K_heads.shape}")  # (1, 2, 2)
print(f"V Heads Shape: {V_heads.shape}")  # (1, 2, 2)
print(f"Q Heads:\n {Q_heads}")
print(f"K Heads:\n {K_heads}")
print(f"V Heads:\n {V_heads}")

print(f"Original Vector: {x}")
print(f"Head 1 Query: {Q_heads[0, 0, :]}")
print(f"Head 2 Query: {Q_heads[0, 1, :]}")

# 5. Scaled Dot-Product Attention (Simplified for one word)
# Score = Q * K^T / sqrt(d_k)
scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 2)) / np.sqrt(d_k)
weights = softmax(scores)
context = np.matmul(weights, V_heads) # Shape: (1, 2, 2)
print(f"Attention Scores:\n {scores}")
print(f"Attention Weights:\n {weights}")
print(f"Context after Attention:\n {context}")

q_k_score = np.dot(Q, K.T) / np.sqrt(d_k)
q_k_weights = softmax(q_k_score)
q_k_context = np.dot(q_k_weights, V)
print(f"Q*K^T Score:\n {q_k_score}")
print(f"Q*K^T Weights:\n {q_k_weights}")
print(f"Context from Q*K^T:\n {q_k_context}")

# 6. Concatenate & Final Linear Transform (W_o)
concat_out = context.reshape(1, d_model)
final_output = np.dot(concat_out, W_o)

print(f"Concatenated Output: {concat_out}")
print(f"Final Output after W_o: {final_output}")


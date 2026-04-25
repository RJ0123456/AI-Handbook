import numpy as np
#import matplotlib.pyplot as plt

np.random.seed(42)


def softmax(x, axis=-1):
    """Compute softmax along the specified axis."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def self_attention(X, W_Q, W_K, W_V):
    """
    Compute self-attention output.

    Args:
        X: Input embeddings of shape (seq_len, embed_dim)
        W_Q, W_K, W_V: Projection matrices

    Returns:
        Output representations of shape (seq_len, d_v)
    """
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    d_k = K.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    output = weights @ V

    return output, weights

# Create embeddings for a 4-token sequence
seq_len = 4
embed_dim = 8
d_k = d_v = 6

# Random embeddings representing different tokens
X = np.random.randn(seq_len, embed_dim)

# Initialize projection matrices
W_Q = np.random.randn(embed_dim, d_k) * 0.1
W_K = np.random.randn(embed_dim, d_k) * 0.1
W_V = np.random.randn(embed_dim, d_v) * 0.1

# Compute attention on original sequence
output_original, weights_original = self_attention(X, W_Q, W_K, W_V)

# Create a permutation: swap positions 1 and 2
permutation = [0, 2, 1, 3]
X_permuted = X[permutation]

# Compute attention on permuted sequence
output_permuted, weights_permuted = self_attention(X_permuted, W_Q, W_K, W_V)

# Apply the same permutation to the original output for comparison
output_original_reordered = output_original[permutation]
# Print results
print("Original Output:\n", np.round(output_original, 2))
print("\nPermuted Output:\n", np.round(output_permuted, 2))

# Print Max difference between original and permuted outputs
max_diff = np.max(np.abs(output_original_reordered - output_permuted))
print("Max difference between original and permuted outputs:", max_diff)

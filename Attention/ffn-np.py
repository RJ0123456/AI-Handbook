import numpy as np

np.random.seed(42)


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def ffn(x, W1, b1, W2, b2, activation=relu):
    """
    Position-wise feed-forward network.

    Args:
        x: Input tensor, shape (n, d_model) or (d_model,)
        W1: First layer weights, shape (d_model, d_ff)
        b1: First layer bias, shape (d_ff,)
        W2: Second layer weights, shape (d_ff, d_model)
        b2: Second layer bias, shape (d_model,)
        activation: Nonlinear activation function

    Returns:
        Output tensor, same shape as x
    """
    hidden = activation(x @ W1 + b1)
    output = hidden @ W2 + b2
    return output


# Example dimensions (from original transformer)
d_model = 512  # Model dimension
d_ff = 2048  # Hidden dimension (4x expansion)

# Initialize weights with Xavier/Glorot initialization
W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / (d_model + d_ff))
print("W1 shape:", W1.shape)  # Should be (512, 2048)
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / (d_ff + d_model))
print("W2 shape:", W2.shape)  # Should be (2048, 512)
b2 = np.zeros(d_model)

# Process a single position
x_single = np.random.randn(d_model)
y_single = ffn(x_single, W1, b1, W2, b2)
print("Input shape:", x_single.shape)  # Should be (512,)
print("Output shape:", y_single.shape)  # Should be (512,)
print("Input norm:", np.linalg.norm(x_single))
print("Output norm:", np.linalg.norm(y_single))


#-- Demonstrate position independence of FFN ---
# Demonstrate position independence
seq_len = 5

X = np.random.randn(seq_len, d_model)

# Process all positions at once (batch processing)
Y_batch = ffn(X, W1, b1, W2, b2)

# Process each position individually
Y_individual = np.zeros_like(X)
for i in range(seq_len):
    Y_individual[i] = ffn(X[i], W1, b1, W2, b2)

# Check they're identical
difference = np.abs(Y_batch - Y_individual).max()
print("Max difference between batch and individual processing:", difference)

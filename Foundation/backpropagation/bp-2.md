# Backpropagation (BP)

Backpropagation (BP) is essentially a large-scale application of the chain rule in multi-layer neural networks. Its core objective is to calculate the partial derivatives of the loss function $L$ with respect to each parameter (weights $w$ and biases $b$) in the network, thereby updating the parameters using gradient descent.

To give you a clear mathematical overview, let's assume a simple multilayer perceptron model.

------------------------------

## 1. Notation Definitions
Assume the network has $L$ layers, and the current layer is layer $l$:

* $w^l_{jk}$: The weights from the $k$-th neuron in layer $l-1$ to the $j$-th neuron in layer $l$.

* $b^l_j$: The bias of the $j$-th neuron in layer $l$.

* $z^l_j$: The linear output (unactivated) of the $j$-th neuron in layer $l$: $z^l = w^l a^{l-1} + b^l$. * $a^l_j$: The activation output of the $j$-th neuron in layer $l$: $a^l = \sigma(z^l)$.

* $C$: Loss function (such as cross-entropy or MSE).

------------------------------

## 2. Four Core Equations of Backpropagation

The essence of backpropagation lies in defining an intermediate variable $\delta^l_j$, which we call the error of the $j$-th neuron in layer $l$:

$$\delta^l_j = \frac{\partial C}{\partial z^l_j}$$

## Equation 1: Error of the output layer ($L$ layer) $\delta^L$

This is the starting point of backpropagation.

$$\delta^L = \frac{\partial C}{\partial a^L} \odot \sigma'(z^L)$$

* Explanation: The error depends on the derivative of the loss function with respect to the output, and the sensitivity (derivative) of the activation function.

* Note: If you use "cross-entropy + Softmax", this term simplifies to $(\hat{y} - y)$ derived earlier.

## Equation 2: Propagation of Hidden Layer Error $\delta^l$ How to derive the error of the current layer $l$ from the error of the next layer $l+1$?

$$\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$$

* Explanation: The error of the current layer is "propagated back" from the error of the next layer through the weight matrix, and then multiplied by the derivative of the current layer's activation function.

## Equation 3: Gradient of Loss with Respect to Bias

$$\frac{\partial C}{\partial b^l_j} = \delta^l_j$$

* Explanation: The gradient of the bias is directly equal to the error of the neuron.

## Equation 4: Gradient of Loss with Respect to Weights

$$\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j$$

* Explanation: The gradient of the weights is equal to the "input of the previous layer" multiplied by the "error produced by this layer". This is why the larger the value (feature) of the input layer, the more drastic the parameter updates.

------------------------------

## 3. Complete Algorithm Flow

1. Forward Pass:

Input $x$, calculate layer by layer until output $\hat{y}$. Store all intermediate variables $z^l$ and $a^l$ (because they are needed for backpropagation).

2. Calculate the output layer error $\delta^L$:

Calculate the error of the last layer based on the predicted and actual values.

3. Backpropagation Error:

Using Equation 2, starting from the $L-1$-th layer, calculate $\delta^L, \delta^{L-1}, \dots, \delta^2$ layer by layer in reverse propagation.

4. Calculate and update the gradient:

Using Equations 3 and 4, calculate all $\frac{\partial C}{\partial w}$ and $\frac{\partial C}{\partial b}$, and then update according to the following rule:

$$w \leftarrow w - \eta \frac{\partial C}{\partial w}$$ ($\eta$ is the learning rate)

------------------------------

## 4. Intuitive Summary: Why is it called "backpropagation"?

* Forward propagation is the "flow of information": data to weights to prediction.

* Backpropagation is the "tracing of responsibility":

1. The last step predicted incorrectly (error).

2. How much of this error is the responsibility of the last layer of weights? (Calculate delta^L).

3. By assigning weights, trace back to the second-to-last layer to see how much responsibility that layer bears.

4. Tracing back all the way to the first layer.

This mathematical technique of "layer-by-layer responsibility allocation" avoids the enormous computational burden of independently differentiating each parameter, and is the foundation upon which deep learning is possible.

## Python (NumPy) Implementation

To implement these four equations from scratch, we’ll build a simple 2-layer neural network (one hidden layer and one output layer) to solve a binary classification problem.
We will use Cross-Entropy as the loss and Sigmoid as the activation function, which, as we discussed, leads to the elegant "$\hat{y} - y$" gradient.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# 1. Setup Data (X: Input, y: True Label)
X = np.array([[0.5, 0.2]])  # 1x2 Input
y = np.array([[1.0]])       # 1x1 Target

# 2. Initialize Parameters (Weights & Biases)# Layer 1: 2 inputs -> 3 neurons
w1 = np.random.randn(2, 3) 
b1 = np.zeros((1, 3))
# Layer 2: 3 neurons -> 1 output
w2 = np.random.randn(3, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.1

# --- FORWARD PASS ---
# Layer 1
z1 = np.dot(X, w1) + b1
a1 = sigmoid(z1)
# Layer 2 (Output Layer)
z2 = np.dot(a1, w2) + b2
a2 = sigmoid(z2)  # This is our y_hat

print(f"Initial Prediction (y_hat): {a2[0][0]:.4f}")

# --- BACKPROPAGATION (The 4 Equations) ---

# Equation 1: Output Error (delta_L) 
# For Cross-Entropy + Sigmoid, this simplifies to (a2 - y)
delta2 = a2 - y 

# Equation 2: Hidden Layer Error (delta_l)
# Propagate delta back: (delta_next * W_next^T) * sigma_prime(z_current)
delta1 = np.dot(delta2, w2.T) * sigmoid_derivative(z1)

# Equation 4 & 3: Gradients for Weights and Biases
# dL/dw = a_prev.T * delta
# dL/db = delta
dw2 = np.dot(a1.T, delta2)
db2 = delta2

dw1 = np.dot(X.T, delta1)
db1 = delta1

# --- PARAMETER UPDATE ---
w2 -= learning_rate * dw2
b2 -= learning_rate * db2
w1 -= learning_rate * dw1
b1 -= learning_rate * db1

# --- VERIFY ---
new_z1 = np.dot(X, w1) + b1
new_a1 = sigmoid(new_z1)
new_z2 = np.dot(new_a1, w2) + b2
new_a2 = sigmoid(new_z2)
print(f"New Prediction after 1 step: {new_a2[0][0]:.4f} (Closer to {y[0][0]})")
```

## Breakdown of the Code vs. Theory

   1. Equation 1 ($\delta^L$): We used delta2 = a2 - y. Mathematically, this is the result of the "cancelation" we derived earlier. Without that cancelation, you would have had to calculate (loss_derivative) * sigmoid_derivative(z2).
   2. Equation 2 ($\delta^l$): delta1 = np.dot(delta2, w2.T) * sigmoid_derivative(z1). This shows how the error from the output layer is distributed back to the hidden neurons based on their connection weights.
   3. Equation 4 ($\frac{\partial C}{\partial w}$): np.dot(a1.T, delta2). Notice how we use the transpose of the activation. This aligns the dimensions so that every weight gets exactly the gradient it deserves based on the input it received and the error it caused.
   4. Learning Rate ($\eta$): We subtract the gradient. If the gradient is positive (prediction > truth), we decrease the weight to lower the next prediction.

A quick tip: In modern frameworks like PyTorch, you don't write these four equations. You just call .backward(), and the engine builds a "Computation Graph" to solve these chain-rule steps for you automatically.

## Other Resource
[Reference 1](https://visionbook.mit.edu/backpropagation.html) 
[Reference 2](https://en.wikipedia.org/wiki/Backpropagation)
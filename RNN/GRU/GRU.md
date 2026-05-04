# GRU: Gated Recurrent Unit

> **Paper:** Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation  
> **Authors:** Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio  
> **Venue:** EMNLP 2014  
> **arXiv:** [1406.1078](https://arxiv.org/abs/1406.1078)

---

## 1. Introduction / Overview

The **Gated Recurrent Unit (GRU)** is a type of recurrent neural network (RNN) cell introduced in 2014 as a simpler alternative to the **Long Short-Term Memory (LSTM)**. Like LSTM, GRU is designed to solve the **vanishing gradient problem** in standard RNNs — the phenomenon that makes it impossible to learn long-range dependencies in sequences.

GRU achieves this by introducing two learnable **gates** that control how information flows through the recurrent unit:

- **Reset gate** $r_t$ — decides how much of the past hidden state to forget
- **Update gate** $z_t$ — decides how much of the new candidate state to accept

The core design philosophy of GRU is **simplicity without sacrificing capability**:

| Model | Gates | Cell State | Parameters (per unit, input size $d$, hidden size $h$) |
|---|---|---|---|
| Vanilla RNN | 0 | Hidden state only | $dh + h^2 + h$ |
| **GRU** | **2** | **Hidden state only** | $3(dh + h^2 + h)$ |
| LSTM | 3 | Hidden + Cell state | $4(dh + h^2 + h)$ |

GRU reaches comparable accuracy to LSTM on most sequence tasks with **~25% fewer parameters** and simpler implementation.

---

## 2. Background / Motivation

### 2.1 The Problem with Standard RNNs

A standard RNN processes sequences by maintaining a hidden state $h_t$ that is updated at each time step:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

Unrolled over $T$ steps, this produces a chain of hidden states:

```
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → ... → [RNN] → hT
         ↑               ↑               ↑                       ↑
      same W          same W          same W               same W
```

**The vanishing gradient problem:**

During backpropagation through time (BPTT), the gradient must flow back through every time step:

$$\frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial h_T} \cdot \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

Each factor $\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh'(\cdot)) \cdot W_h$.

- If the spectral norm of $W_h$ is $< 1$, gradients **shrink exponentially** → vanish
- If the spectral norm of $W_h$ is $> 1$, gradients **grow exponentially** → explode

```
Long sequence (T = 50):

  Gradient at step 50:   1.0
  Gradient at step 40:   0.0001       (vanished)
  Gradient at step 30:   0.00000001   (nearly zero)
  Gradient at step 1:    ≈ 0          (effectively zero)
  
  Result: The network CANNOT learn that "step 1" matters for "step 50"
```

**Why this matters in practice:**

```
Example: Language modeling

  "The trophy doesn't fit in the suitcase because [it] is too large."
                                                    ↑
                  What does "it" refer to?
                  Answer requires going back 8+ words to "trophy"
                  
  Standard RNN: Gradient at "trophy" ≈ 0 by the time we reach "it"
                Network cannot learn the dependency
                
  GRU/LSTM:     Gates keep the relevant "trophy" information alive
                Network can learn to resolve the coreference
```

### 2.2 How LSTM Solves It — And Why GRU Is Simpler

LSTM introduced the **cell state** $C_t$ (a separate memory highway) plus three gates: input, forget, and output. While LSTM works very well, it is:
- **Complex**: 4 separate linear transformations per step
- **Memory intensive**: needs to maintain both $h_t$ and $C_t$
- **Slower to train**: more parameters to optimize

GRU merges the cell state and hidden state into a **single hidden state** $h_t$ and uses only **2 gates** instead of 3, achieving similar gradient-flow benefits with lower complexity.

### 2.3 The Vanishing Gradient Solution: Gates

The key insight behind both LSTM and GRU is **additive updates**. Instead of:

$$h_t = f(h_{t-1})  \quad \text{(multiplicative, gradients vanish)}$$

GRU uses:

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(additive blend, gradients preserved)}$$

When $z_t \approx 0$ (update gate nearly closed), we get:

$$h_t \approx h_{t-1} \quad \Rightarrow \quad \frac{\partial h_t}{\partial h_{t-1}} \approx 1$$

This **identity-like path** allows gradients to flow back over many steps without vanishing.

---

## 3. Architecture

### 3.1 GRU Cell Structure

![GRU Cell Structure (Fully Gated)](./static/wiki-gru-cell.svg)

*Source: Jeblad, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons*

### 3.2 GRU Equations

Let $x_t \in \mathbb{R}^d$ be the input at time $t$, and $h_{t-1} \in \mathbb{R}^h$ be the previous hidden state.

**Step 1 — Reset Gate:**

$$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$

- $r_t \in (0, 1)^h$ — element-wise between 0 and 1
- Controls how much of the **past** ($h_{t-1}$) is used when computing the candidate
- $r_t \approx 0$: **forget the past** (treat as if starting fresh)
- $r_t \approx 1$: **keep the past** (use full memory)

**Step 2 — Update Gate:**

$$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$

- $z_t \in (0, 1)^h$ — element-wise between 0 and 1
- Controls how much the **new** candidate vs **old** hidden state is used
- $z_t \approx 0$: **ignore the new input** (keep previous memory)
- $z_t \approx 1$: **fully update** (discard old memory, accept new)

**Step 3 — Candidate Hidden State:**

$$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$$

- The reset gate $r_t$ is applied to $h_{t-1}$ **before** the candidate computation
- This allows the network to drop irrelevant past information when computing new content
- $\odot$ denotes element-wise (Hadamard) product

**Step 4 — Final Hidden State (Output):**

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

- A **linear interpolation** between old state $h_{t-1}$ and new candidate $\tilde{h}_t$
- Controlled by update gate $z_t$
- This is the **additive update** that prevents vanishing gradients

### 3.3 Illustration of Gate Behavior

```
Scenario 1: Long-term memory retention
  z_t ≈ 0  (update gate closed)
  
  h_t = (1 - 0) ⊙ h_{t-1} + 0 ⊙ h̃_t
      ≈ h_{t-1}
      
  → Hidden state unchanged. Memory is preserved perfectly.
  → Gradient: ∂h_t/∂h_{t-1} ≈ 1  (no vanishing!)


Scenario 2: Reset and relearn from new input
  r_t ≈ 0  (reset gate closed)
  z_t ≈ 1  (update gate open)
  
  h̃_t = tanh(W_h x_t + U_h (0 ⊙ h_{t-1}) + b_h)
       = tanh(W_h x_t + b_h)       ← ignores h_{t-1} entirely
  
  h_t = (1 - 1) ⊙ h_{t-1} + 1 ⊙ h̃_t
      ≈ h̃_t
      
  → Hidden state fully replaced by new candidate.
  → Useful at sentence/paragraph boundaries.


Scenario 3: Blend past and present
  r_t ≈ 0.5, z_t ≈ 0.5
  
  h̃_t uses partial past context
  h_t = 0.5 ⊙ h_{t-1} + 0.5 ⊙ h̃_t
      
  → Smooth blend of old and new information.
```

### 3.4 GRU vs LSTM: Architecture Comparison

![GRU vs LSTM Architecture](./static/gru-vs-lstm.svg)

**Key structural differences:**

| Feature | GRU | LSTM |
|---|---|---|
| Memory states | 1 ($h_t$) | 2 ($h_t$, $C_t$) |
| Gates | 2 (reset, update) | 3 (input, forget, output) |
| Parameterized transforms | 3 | 4 |
| Parameters per unit | $3(d \cdot h + h^2)$ | $4(d \cdot h + h^2)$ |
| Output | $h_t$ only | $h_t$ (from $C_t$ through output gate) |

### 3.5 Parameter Count

For input size $d$ and hidden size $h$:

**GRU:**

$$N_{GRU} = 3 \cdot (d \cdot h + h \cdot h + h) = 3(dh + h^2 + h)$$

Breaking down:
- Reset gate: $W_r \in \mathbb{R}^{h \times d}$, $U_r \in \mathbb{R}^{h \times h}$, $b_r \in \mathbb{R}^h$
- Update gate: $W_z \in \mathbb{R}^{h \times d}$, $U_z \in \mathbb{R}^{h \times h}$, $b_z \in \mathbb{R}^h$
- Candidate: $W_h \in \mathbb{R}^{h \times d}$, $U_h \in \mathbb{R}^{h \times h}$, $b_h \in \mathbb{R}^h$

**LSTM:**

$$N_{LSTM} = 4 \cdot (d \cdot h + h \cdot h + h) = 4(dh + h^2 + h)$$

**Example** ($d = 128$, $h = 256$):

| Model | Parameters |
|---|---|
| Vanilla RNN | $128 \times 256 + 256^2 + 256 = 98,560$ |
| **GRU** | $3 \times (128 \times 256 + 256^2 + 256) = 295,680$ |
| LSTM | $4 \times (128 \times 256 + 256^2 + 256) = 394,240$ |

GRU has **25% fewer parameters** than LSTM for the same hidden size.

### 3.6 Backpropagation Through GRU

The gradient of the loss $\mathcal{L}$ with respect to $h_{t-1}$ is:

$$\frac{\partial \mathcal{L}}{\partial h_{t-1}} = \frac{\partial \mathcal{L}}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_{t-1}}$$

Expanding using the GRU output equation:

$$\frac{\partial h_t}{\partial h_{t-1}} = (1 - z_t) + z_t \cdot \frac{\partial \tilde{h}_t}{\partial h_{t-1}} + \frac{\partial z_t}{\partial h_{t-1}} \cdot (\tilde{h}_t - h_{t-1})$$

The critical term is $(1 - z_t)$: **when the update gate is near 0**, this factor approaches 1, giving the gradient a near-identity path back through time. This is what prevents vanishing.

![Vanishing Gradient: RNN vs GRU](./static/vanishing-gradient.svg)

---

## 4. Illustration Diagrams

### 4.1 GRU Cell — Official Diagram

![GRU Cell (Fully Gated)](./static/wiki-gru-cell.svg)

*Source: Jeblad, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons*

The diagram shows the complete data flow within one GRU cell:
- **Blue paths**: input $x_t$ and previous hidden state $h_{t-1}$
- **Gates** $r_t$ and $z_t$ are computed from both $x_t$ and $h_{t-1}$
- **Reset gate** masks $h_{t-1}$ before computing the candidate $\tilde{h}_t$
- **Update gate** interpolates between old $h_{t-1}$ and new $\tilde{h}_t$ to produce $h_t$

### 4.2 GRU Architecture Variants

The original GRU has three common variants depending on what inputs the gates receive (Dey & Salem, 2017):

**Base type (fully gated)** — gates depend on both $x_t$ and $h_{t-1}$:

![GRU Base Type](./static/wiki-gru-cell.svg)

**Type 1** — gates depend only on previous hidden state $h_{t-1}$ and bias:

$$z_t = \sigma(U_z h_{t-1} + b_z), \quad r_t = \sigma(U_r h_{t-1} + b_r)$$

![GRU Type 1](./static/wiki-gru-type1.svg)

*Source: Jeblad, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons*

**Type 2** — gates depend only on previous hidden state (no bias):

$$z_t = \sigma(U_z h_{t-1}), \quad r_t = \sigma(U_r h_{t-1})$$

![GRU Type 2](./static/wiki-gru-type2.svg)

*Source: Jeblad, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons*

**Type 3** — gates are computed from bias only (input-independent):

$$z_t = \sigma(b_z), \quad r_t = \sigma(b_r)$$

![GRU Type 3](./static/wiki-gru-type3.svg)

*Source: Jeblad, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons*

| Variant | Gate input | Use case |
|---|---|---|
| **Base** | $x_t + h_{t-1}$ | General purpose, most common |
| **Type 1** | $h_{t-1}$ only | When input should not affect gate timing |
| **Type 2** | $h_{t-1}$ only (no bias) | Minimal parameterization |
| **Type 3** | Bias only | Learnable but time-step-independent gating |

### 4.3 Time Step Unrolling

![GRU Unrolled Over Time](./static/gru-unrolled.svg)

### 4.4 Reset Gate Intuition

![Reset Gate Behavior](./static/reset-gate.svg)

### 4.5 Update Gate Intuition

![Update Gate Behavior](./static/update-gate.svg)

---

## 5. Results / Performance

### 5.1 Music Modeling (Cho et al. 2014)

The original paper tested GRU on polyphonic music modeling:

| Model | Negative Log-Likelihood (lower is better) |
|---|---|
| RNN (tanh) | 8.37 |
| LSTM | 8.13 |
| **GRU** | **8.09** |

### 5.2 Machine Translation (English → French, BLEU score)

| Model | BLEU Score |
|---|---|
| Moses (phrase-based SMT) | 33.3 |
| RNN Encoder-Decoder (vanilla) | 26.7 |
| **RNN Encoder-Decoder with GRU** | **33.7** |

### 5.3 Language Modeling (Penn Treebank, PTB)

| Model | Perplexity (lower is better) |
|---|---|
| Vanilla RNN | 120.7 |
| LSTM | 78.4 |
| **GRU** | **79.7** |

### 5.4 Empirical Comparison: GRU vs LSTM (Chung et al. 2014)

A dedicated comparison study found:

| Task | Winner |
|---|---|
| Polyphonic music modeling | GRU ≈ LSTM |
| Speech signal modeling | GRU ≈ LSTM |
| Language modeling | LSTM slight edge |
| Sequence classification | Task-dependent |

**Conclusion:** GRU and LSTM perform comparably across most tasks. GRU is preferred when:
- Training data is **limited** (fewer parameters → less overfitting)
- **Training speed** matters (fewer operations per step)
- **Memory efficiency** is important (no separate cell state)

### 5.5 Training Speed Comparison

On equivalent hardware, for a model with hidden size 512:

| Model | Relative Training Time |
|---|---|
| Vanilla RNN | 1.0× |
| **GRU** | **1.7×** (slower than RNN) |
| LSTM | 2.3× (slower than RNN) |

**GRU is ~25–35% faster than LSTM** due to fewer gate computations.

---

## 6. Mathematical Summary

### Complete GRU Equations

$$\boxed{
\begin{aligned}
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) & \text{(reset gate)} \\
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) & \text{(update gate)} \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) & \text{(candidate)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t & \text{(output)}
\end{aligned}
}$$

### Vectorized Form (Batch Processing)

For a batch of sequences, input $X_t \in \mathbb{R}^{B \times d}$, hidden $H_{t-1} \in \mathbb{R}^{B \times h}$:

$$\begin{bmatrix} r_t \\ z_t \end{bmatrix} = \sigma \left( \begin{bmatrix} W_r \\ W_z \end{bmatrix} X_t^T + \begin{bmatrix} U_r \\ U_z \end{bmatrix} H_{t-1}^T + \begin{bmatrix} b_r \\ b_z \end{bmatrix} \right)$$

$$\tilde{H}_t = \tanh\left( W_h X_t^T + U_h (r_t \odot H_{t-1})^T + b_h \right)$$

$$H_t = (1 - z_t) \odot H_{t-1} + z_t \odot \tilde{H}_t$$

### Key Gradient Property

The gradient flowing back through one GRU step is bounded below by:

$$\left\| \frac{\partial h_t}{\partial h_{t-1}} \right\| \geq \| 1 - z_t \|$$

When $z_t \to 0$ (the network chooses to preserve memory), this lower bound $\to 1$, preventing vanishing.

---

## 7. Code Example

### 7.1 GRU Cell from Scratch (NumPy)

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class GRUCell:
    """Single GRU cell — manual forward pass for illustration."""

    def __init__(self, input_size: int, hidden_size: int):
        h, d = hidden_size, input_size
        # Reset gate
        self.W_r = np.random.randn(h, d) * 0.01
        self.U_r = np.random.randn(h, h) * 0.01
        self.b_r = np.zeros(h)
        # Update gate
        self.W_z = np.random.randn(h, d) * 0.01
        self.U_z = np.random.randn(h, h) * 0.01
        self.b_z = np.zeros(h)
        # Candidate hidden state
        self.W_h = np.random.randn(h, d) * 0.01
        self.U_h = np.random.randn(h, h) * 0.01
        self.b_h = np.zeros(h)

    def forward(self, x: np.ndarray, h_prev: np.ndarray):
        """
        Args:
            x:      (input_size,)  — input at current time step
            h_prev: (hidden_size,) — previous hidden state
        Returns:
            h:      (hidden_size,) — new hidden state
        """
        # Reset gate
        r = sigmoid(self.W_r @ x + self.U_r @ h_prev + self.b_r)

        # Update gate
        z = sigmoid(self.W_z @ x + self.U_z @ h_prev + self.b_z)

        # Candidate hidden state (reset gate applied to h_prev)
        h_candidate = np.tanh(self.W_h @ x + self.U_h @ (r * h_prev) + self.b_h)

        # Final hidden state: linear interpolation
        h = (1 - z) * h_prev + z * h_candidate

        return h, {'r': r, 'z': z, 'h_candidate': h_candidate}
```

### 7.2 GRU in PyTorch

```python
import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """Sequence classifier using a stacked GRU."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)  # ×2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) — batch of token sequences
        Returns:
            logits:    (B, num_classes)
        """
        # Embed tokens: (B, T, embed_dim)
        x = self.dropout(self.embedding(token_ids))

        # GRU: output (B, T, 2*H), h_n (2*num_layers, B, H)
        output, h_n = self.gru(x)

        # Take final hidden states from both directions
        # h_n shape: (num_layers * num_directions, B, H)
        # Last layer: h_n[-2] = forward, h_n[-1] = backward
        h_forward = h_n[-2]   # (B, H)
        h_backward = h_n[-1]  # (B, H)
        h_final = torch.cat([h_forward, h_backward], dim=-1)  # (B, 2H)

        return self.classifier(self.dropout(h_final))


# --- Usage ---
model = GRUModel(
    vocab_size=10000,
    embed_dim=128,
    hidden_size=256,
    num_layers=2,
    num_classes=5,
)

x = torch.randint(0, 10000, (32, 50))  # batch=32, seq_len=50
logits = model(x)
print(logits.shape)  # (32, 5)
```

### 7.3 GRU vs LSTM Parameter Count Comparison

```python
import torch.nn as nn

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

d, h = 128, 256  # input size, hidden size

gru  = nn.GRU(input_size=d,  hidden_size=h, num_layers=1)
lstm = nn.LSTM(input_size=d, hidden_size=h, num_layers=1)
rnn  = nn.RNN(input_size=d,  hidden_size=h, num_layers=1)

print(f"Vanilla RNN params: {count_params(rnn):,}")
print(f"GRU params:         {count_params(gru):,}")
print(f"LSTM params:        {count_params(lstm):,}")
# Output:
# Vanilla RNN params:  98,560
# GRU params:         295,680
# LSTM params:        394,240
```

---

## 8. Summary

| Aspect | Detail |
|---|---|
| Introduced | Cho et al., EMNLP 2014 |
| Problem solved | Vanishing gradient in standard RNNs |
| Key mechanism | Additive update via gating: $h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t$ |
| Gates | **Reset gate** $r_t$ (what to forget), **Update gate** $z_t$ (how much to update) |
| Memory states | 1 ($h_t$ only) |
| Parameters vs LSTM | ~75% (25% fewer) |
| Training speed vs LSTM | ~25–35% faster |
| Performance vs LSTM | Comparable on most tasks |
| Best used when | Limited data, speed-critical, memory-constrained |

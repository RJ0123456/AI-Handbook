# Self-Attention

> **Core idea:** In self-attention, every token builds a new representation by looking at other tokens in the same sequence and weighting them by relevance.
> **Why it matters:** It lets the model capture both nearby and long-range relationships without recurrence, which is a core reason Transformers scale well.
> **Mental model:** Each token asks, *"Which other tokens in this sequence should influence me right now?"*

---

## 1. What Self-Attention Means

Self-attention is an attention mechanism where the query, key, and value all come from the same input sequence.

If the input sequence is:

`The cat sat on the mat`

then when computing the representation for `sat`, the model can look at `The`, `cat`, `on`, `the`, and `mat`, and decide how much each token should contribute.

This is called *self*-attention because the sequence attends to itself.

---

## 2. Why We Need Self-Attention

Language and many other sequential signals contain dependencies that are not purely local.

Example:

`The book on the table near the window was old.`

To understand the word `was`, the model should connect it to `book`, even though several words appear in between.

Traditional RNN-based models can, in principle, learn such dependencies, but information must pass through many sequential steps. Self-attention creates a direct path between any pair of tokens in one layer.

This gives three major advantages:

1. It models long-range interactions more easily.
2. It allows parallel computation across all tokens.
3. It adapts dynamically depending on the current token and context.

---

## 3. Query, Key, and Value Inside One Sequence

Suppose the input embeddings are stacked into a matrix $X \in \mathbb{R}^{n \times d}$, where:

- $n$ is sequence length,
- $d$ is embedding dimension.

Self-attention first produces three learned projections:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

where:

- $Q$ contains queries,
- $K$ contains keys,
- $V$ contains values.

Even though all three come from the same input $X$, they serve different roles:

- **Query:** what this token is looking for.
- **Key:** what this token makes available to others.
- **Value:** the information this token contributes if selected.

---

## 4. Core Computation

For a single token $i$, we compare its query $q_i$ with every key $k_j$ in the sequence.

The raw compatibility score is often a dot product:

$$
s_{ij} = q_i \cdot k_j
$$

For all tokens together, this becomes:

$$
S = QK^\top
$$

Then we scale and normalize:

$$
A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)
$$

Finally, we mix values using these attention weights:

$$
\text{SelfAttention}(X) = AV
$$

or equivalently:

$$
\text{SelfAttention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Each row of $A$ tells us how one token distributes its attention over all tokens in the sequence.

---

## 5. Intuition Behind the Attention Matrix

If the input has $n$ tokens, then the attention matrix has shape $n \times n$.

- Row $i$: where token $i$ looks.
- Column $j$: how much other tokens look at token $j$.
- Entry $A_{ij}$: how strongly token $i$ attends to token $j$.

This makes self-attention a content-based interaction map over the whole sequence.

For example, in:

`The animal did not cross the street because it was tired.`

the token `it` may assign large weight to `animal`, because that token best matches the information needed to resolve the reference.

---

## 6. Why the Scaling Factor Appears

The factor $\sqrt{d_k}$ prevents dot products from growing too large when key and query dimensions increase.

Without scaling, the logits entering softmax can become very large in magnitude. That makes softmax distributions overly sharp and gradients less stable.

So the scaled form:

$$
\frac{QK^\top}{\sqrt{d_k}}
$$

is mainly a numerical stability trick that improves optimization.

---

## 7. Tiny Numerical Example

Assume one token compares itself against three tokens and gets raw scores:

$$
[2.2,\ 0.8,\ 0.0]
$$

After softmax, the weights are approximately:

$$
[0.723,\ 0.177,\ 0.100]
$$

This means:

- the first token contributes most,
- the second contributes a little,
- the third contributes least.

If the value vectors are $v_1, v_2, v_3$, then the new representation is:

$$
0.723v_1 + 0.177v_2 + 0.100v_3
$$

So the output token is a learned weighted mixture of the whole sequence.

---

## 8. Self-Attention vs Cross-Attention

The key difference is where $Q$, $K$, and $V$ come from.

### Self-attention

- $Q$, $K$, and $V$ all come from the same sequence.
- Used to model internal relationships inside one sequence.

### Cross-attention

- Queries come from one sequence.
- Keys and values come from another sequence.
- Used when one representation needs to retrieve information from a different source.

In Transformers, encoder layers commonly use self-attention, and encoder-decoder models use cross-attention so the decoder can attend to encoder outputs.

---

## 9. Bidirectional vs Causal Self-Attention

Self-attention can be used in two common ways.

### Bidirectional self-attention

Each token can attend to tokens on both the left and the right.

This is useful when the full sequence is visible, such as in encoders like BERT.

### Causal self-attention

Each token can only attend to itself and earlier tokens.

This is required for autoregressive language modeling, where token $t$ must not see future tokens $t+1, t+2, \dots$ during prediction.

This restriction is implemented using a mask over the attention scores.

---

## 10. Attention Masking

Before softmax, we can modify the score matrix to block illegal positions.

For causal masking:

- positions in the future are assigned $-\infty$,
- softmax then turns those entries into zero probability.

Conceptually:

$$
A = \text{softmax}\!\left(\frac{QK^\top + M}{\sqrt{d_k}}\right)
$$

where $M$ is the mask matrix.

Masks are also used for:

- padding tokens,
- task-specific constraints,
- structured attention patterns.

---

## 11. Multi-Head Self-Attention

A single attention map may focus on only one kind of relationship. Multi-head self-attention solves this by running several self-attention operations in parallel with different learned projections.

For head $h$:

$$
\text{head}_h = \text{Attention}(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)})
$$

Then the heads are concatenated and projected again:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)W_O
$$

Different heads can capture different patterns, such as:

- local syntax,
- subject-verb agreement,
- coreference,
- positional or structural relations.

---

## 12. What Self-Attention Buys Us

1. **Global receptive field in one layer**  
Any token can directly incorporate information from any other token.

2. **Dynamic context selection**  
The relevant context changes depending on the token and sentence.

3. **Parallelization**  
All pairwise token interactions can be computed as matrix operations on modern hardware.

4. **Flexible representation learning**  
A token representation is no longer fixed after embedding; it becomes context-aware.

---

## 13. Limitations

Self-attention is powerful, but not free.

### Quadratic cost

For sequence length $n$, the attention matrix is $n \times n$, so compute and memory cost are typically $O(n^2)$.

### No built-in order awareness

Self-attention by itself is permutation-invariant. Without positional encoding, the model would not know whether a token came first or last.

### Attention is not perfect explanation

Attention maps can be informative, but high attention weight does not automatically mean true causal importance.

These issues motivated efficient attention variants and better positional encoding methods.

---

## 14. Self-Attention in the Transformer Block

Inside a Transformer block, self-attention is usually followed by:

1. residual connection,
2. layer normalization,
3. feed-forward network,
4. another residual and normalization step.

So self-attention is not the whole model. It is the mechanism that mixes information across tokens before additional nonlinear transformation.

---

## 15. One-Sentence Summary

Self-attention lets each token look across the entire sequence, score which tokens matter most, and update itself as a weighted combination of the most relevant contextual information.

---

## 16. References

- Vaswani, A., et al. (2017). *Attention Is All You Need*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
- Standard Transformer lecture notes and visual explainers for intuition and implementation details.

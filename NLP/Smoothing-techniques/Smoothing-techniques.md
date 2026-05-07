# Smoothing Techniques for N-gram Language Models

> **Core idea:** Redistribute probability mass from observed N-grams to unseen ones, so that no event is assigned zero probability.  
> **Why it matters:** MLE assigns $P=0$ to any unseen N-gram, making the entire sequence probability $0$ and perplexity infinite.  
> **Key trade-off:** Smoothing borrows probability from rich events and gives it to sparse or unseen ones — the less data you have, the more you need to smooth.

---

## 1. Background and Motivation

N-gram language models estimate the probability of a token sequence using Maximum Likelihood Estimation (MLE):

$$
\hat{P}_\text{MLE}(w \mid h) = \frac{C(h,\, w)}{C(h)}
$$

This works well for frequent N-grams, but natural language is **highly sparse**. Even with large corpora:

- Most trigrams or higher-order N-grams appear **zero times** in training data.
- At test time, any unseen N-gram yields $P = 0$.
- Multiplying zero into a sequence probability collapses the entire sentence to $P = 0$.

This is the **data sparsity problem**, and smoothing is the family of techniques designed to fix it.

---

## 2. The Zero Probability Problem

### 2.1 Why MLE Fails

Consider a bigram model trained on a small corpus. Many valid bigrams in English will never appear in training. For example:

$$
P_\text{MLE}(\text{zebra} \mid \text{the}) = \frac{C(\text{the, zebra})}{C(\text{the})} = \frac{0}{C(\text{the})} = 0
$$

At test time, the sentence *"I saw the zebra"* gets:

$$
P(\text{I saw the zebra}) = \cdots \times P(\text{zebra} \mid \text{the}) = \cdots \times 0 = 0
$$

The entire probability collapses to zero regardless of all the other well-estimated factors.

### 2.2 Effect on Perplexity

Perplexity is defined as:

$$
\text{PP}(W) = P(w_1, w_2, \dots, w_N)^{-\frac{1}{N}}
$$

A single zero probability makes $P(W) = 0$, which means $\text{PP}(W) = \infty$. This renders the model completely untrainable and unevaluable on realistic test sets.

### 2.3 The Core Goal of Smoothing

Every smoothing technique tries to produce a valid probability distribution:

$$
\sum_{w \in V} P_\text{smooth}(w \mid h) = 1, \quad P_\text{smooth}(w \mid h) > 0 \;\; \forall\, w \in V
$$

The mass assigned to unseen events must come from somewhere — smoothing takes it from seen events proportionally.

---

## 3. Laplace (Add-One) Smoothing

The simplest idea: pretend every N-gram was seen **at least once** by adding 1 to every count.

### 3.1 Formula

$$
\boxed{P_\text{Laplace}(w \mid h) = \frac{C(h,\, w) + 1}{C(h) + V}}
$$

Where:
- $C(h, w)$: raw count of the N-gram
- $C(h)$: raw count of the history
- $V$: vocabulary size (number of unique tokens)

The denominator increases by $V$ because we add 1 for each of the $V$ possible next tokens, preserving the sum-to-one property.

### 3.2 Worked Example

Toy corpus (bigrams):

| History $h$ | Next word $w$ | $C(h,w)$ |
|-------------|--------------|----------|
| `I`         | `love`       | 2        |
| `I`         | `enjoy`      | 1        |
| `I`         | `hate`       | 0        |

Vocabulary size $V = 5$ (assume 5 tokens total).  
History count $C(\text{I}) = 3$.

**MLE estimates:**

$$
P_\text{MLE}(\text{love} \mid \text{I}) = \frac{2}{3} \approx 0.667, \quad
P_\text{MLE}(\text{hate} \mid \text{I}) = \frac{0}{3} = 0
$$

**Laplace estimates:**

$$
P_\text{Laplace}(\text{love} \mid \text{I}) = \frac{2+1}{3+5} = \frac{3}{8} = 0.375
$$

$$
P_\text{Laplace}(\text{enjoy} \mid \text{I}) = \frac{1+1}{3+5} = \frac{2}{8} = 0.250
$$

$$
P_\text{Laplace}(\text{hate} \mid \text{I}) = \frac{0+1}{3+5} = \frac{1}{8} = 0.125
$$

All probabilities are now positive and sum to $\leq 1$ (they sum to 1 over all $V$ tokens).

### 3.3 Adjusted Counts

A useful way to reason about Laplace smoothing is through **adjusted counts** — what effective count does a smoothed probability correspond to?

$$
c^*(h, w) = \frac{(C(h,w) + 1) \cdot C(h)}{C(h) + V}
$$

For example, the bigram `(I, love)` with $C=2$, $C(h)=3$, $V=5$:

$$
c^*(\text{I, love}) = \frac{3 \cdot 3}{8} = 1.125
$$

The effective count dropped from 2 to 1.125 — smoothing transferred probability mass away from frequent events.

### 3.4 The Problem with Laplace Smoothing

Adding 1 is a very large perturbation when $V$ is large (typical vocabularies: $10^4$–$10^5$ tokens):

- Unseen N-grams receive exactly as much probability mass as a count of 1.
- High-frequency N-grams have their effective counts **drastically deflated**.
- The total mass given to zero-count events can exceed the mass for observed events.

In practice, Laplace smoothing produces **poor probability estimates** for language models. It is mainly taught for conceptual clarity.

---

## 4. Add-$k$ Smoothing

A direct generalization of Laplace: instead of adding exactly 1, add a fractional $k$ (also called **Lidstone smoothing**).

### 4.1 Formula

$$
\boxed{P_\text{add-k}(w \mid h) = \frac{C(h,\, w) + k}{C(h) + kV}}
$$

Where $0 < k \leq 1$ is a hyperparameter tuned on a held-out set.

- $k = 1$: reduces to Laplace smoothing.
- $k \to 0$: approaches MLE (less smoothing).

### 4.2 Choosing $k$

$k$ is selected by minimizing perplexity on a **development set** (not the test set):

$$
k^* = \arg\min_{k} \; \text{PP}_{\text{dev}}(k)
$$

Typical values that work in practice: $k \approx 0.01$–$0.1$ for bigrams.

### 4.3 Limitation

Even with optimal $k$, add-$k$ smoothing does not model the **distribution of unseen events well**. It assigns the same probability to all unseen N-grams, regardless of how plausible they are. Better methods (interpolation, backoff) use lower-order N-grams to inform these estimates.

---

## 5. Good-Turing Estimation

Good-Turing re-estimates counts using the **frequency of frequencies** — how often each count value itself appears in the corpus.

### 5.1 Intuition

Suppose in your corpus, 10 bigrams appear exactly once. These "hapax legomena" are similar to unseen bigrams — both are rare. Good-Turing uses the count of singletons to estimate how much probability the unseen events should collectively receive.

### 5.2 Notation

Define $N_r$ as the number of distinct N-grams that appear exactly $r$ times:

$$
N_r = \left|\{(h, w) : C(h, w) = r\}\right|
$$

In particular:
- $N_0$: number of unseen N-grams (typically enormous)
- $N_1$: number of singletons (N-grams seen exactly once)
- $N_r$: number of N-grams seen exactly $r$ times

### 5.3 Adjusted Count Formula

Good-Turing replaces the raw count $r$ with an adjusted count $r^*$:

$$
\boxed{r^* = (r+1) \cdot \frac{N_{r+1}}{N_r}}
$$

This shrinks higher counts and uses singleton information to estimate the unseen mass.

For $r = 0$ (unseen events):

$$
0^* = 1 \cdot \frac{N_1}{N_0}
$$

This means the total probability mass assigned to all unseen events is $\frac{N_1}{N}$, where $N = \sum_r r \cdot N_r$ is the total number of N-gram tokens.

### 5.4 Worked Example

Suppose a corpus has:

| $r$ | $N_r$ |
|-----|-------|
| 0   | 100,000 |
| 1   | 3,000  |
| 2   | 1,500  |
| 3   | 800    |

Adjusted counts:

$$
0^* = 1 \cdot \frac{N_1}{N_0} = \frac{3000}{100000} = 0.03
$$

$$
1^* = 2 \cdot \frac{N_2}{N_1} = 2 \cdot \frac{1500}{3000} = 1.0
$$

$$
2^* = 3 \cdot \frac{N_3}{N_2} = 3 \cdot \frac{800}{1500} = 1.6
$$

Each count is deflated. Singletons are effectively treated as having count 1.0 (unchanged here by coincidence), while the unseen events each get a tiny sliver $0.03$.

### 5.5 Reliability at High Counts

For large $r$, $N_{r+1}$ becomes noisy or zero. In practice, **Simple Good-Turing** uses a linear regression on $\log N_r$ vs $\log r$ to smooth the frequency-of-frequency distribution and extrapolate reliably.

---

## 6. Interpolation: Combining N-gram Orders

Rather than choosing one N-gram order, **interpolation** mixes probabilities from multiple orders using weights $\lambda$.

### 6.1 Linear Interpolation

For a trigram model combining trigram, bigram, and unigram estimates:

$$
\boxed{P_\text{interp}(w \mid w_{t-2}, w_{t-1}) = \lambda_3 \hat{P}(w \mid w_{t-2}, w_{t-1}) + \lambda_2 \hat{P}(w \mid w_{t-1}) + \lambda_1 \hat{P}(w)}
$$

Subject to:

$$
\lambda_1 + \lambda_2 + \lambda_3 = 1, \quad \lambda_i \geq 0
$$

Because the unigram model $\hat{P}(w) > 0$ for all words in vocabulary, the interpolated model is also always positive.

### 6.2 Why It Works

- Trigram: best estimate when context is **frequent** — high data support.
- Bigram: backs off when trigram context is sparse.
- Unigram: always positive, guarantees no zero.

Interpolation always has the unigram as a safety net.

### 6.3 Learning the Weights

$\lambda$ values are optimized on a **held-out set** using the **EM algorithm** (Expectation-Maximization), which iteratively finds $\lambda_i$ that maximize held-out log-likelihood.

EM update (E-step then M-step):

$$
\hat{\lambda}_i \propto \sum_{w, h} C_\text{heldout}(h, w) \cdot \frac{\lambda_i \hat{P}_i(w \mid h)}{P_\text{interp}(w \mid h)}
$$

Then normalize so the $\lambda_i$ sum to 1.

### 6.4 Context-Dependent $\lambda$ (Deleted Interpolation)

An extension allows $\lambda$ to depend on the count of the current context:

$$
\lambda_3 = \lambda_3\!\left(C(w_{t-2}, w_{t-1})\right)
$$

When a context is seen frequently, trust the high-order estimate more ($\lambda_3 \uparrow$); when it is rare, rely more on lower orders ($\lambda_2, \lambda_1 \uparrow$). This is known as **deleted interpolation**.

---

## 7. Backoff: Using Lower Orders as Fallback

**Backoff** differs from interpolation: instead of always mixing all orders, it uses the highest reliable order and falls back to lower orders **only** when the high-order count is zero (or below a threshold).

### 7.1 Stupid Backoff (Brants et al., 2007)

A simple, unnormalized version often used at large scale:

$$
S(w \mid h) =
\begin{cases}
\dfrac{C(h, w)}{C(h)} & \text{if } C(h, w) > 0 \\[8pt]
0.4 \cdot S(w \mid h') & \text{otherwise}
\end{cases}
$$

Where $h'$ is the history with the oldest word removed. The constant $0.4$ is a fixed discount (not learned). Note: this does **not** produce a proper probability distribution, but works well empirically at web scale.

### 7.2 Katz Backoff (Proper Normalization)

Katz backoff normalizes using a **back-off weight** $\alpha(h)$ to ensure a valid distribution:

$$
P_\text{Katz}(w \mid h) =
\begin{cases}
P^*(w \mid h) & \text{if } C(h, w) > 0 \\[6pt]
\alpha(h) \cdot P_\text{Katz}(w \mid h') & \text{otherwise}
\end{cases}
$$

Where:
- $P^*(w \mid h)$: discounted probability for seen events (using Good-Turing discounts)
- $\alpha(h)$: a normalization factor that distributes the remaining mass over unseen events

The back-off weight is computed as:

$$
\alpha(h) = \frac{1 - \sum_{w:\, C(h,w)>0} P^*(w \mid h)}{\sum_{w:\, C(h,w)=0} P_\text{Katz}(w \mid h')}
$$

This ensures the probabilities sum to exactly 1.

---

## 8. Kneser-Ney Smoothing

Kneser-Ney is widely regarded as the best practical smoothing method for N-gram language models.

### 8.1 The Continuation Probability Insight

Standard backoff uses unigram counts $C(w)$ when backing off. But this is suboptimal. Consider the word **"Francisco"**: it appears often, but almost always after "San". Its raw unigram count is high, but it should not be used as a fallback for arbitrary contexts.

Kneser-Ney uses a **continuation probability** instead:

$$
P_\text{continuation}(w) \propto \left|\{h : C(h, w) > 0\}\right|
$$

This counts the number of **distinct contexts** $h$ in which $w$ has appeared — a better measure of how versatile the word is as a fill-in for unseen bigrams.

### 8.2 Interpolated Kneser-Ney

$$
\boxed{P_\text{KN}(w \mid h) = \frac{\max\!\left(C(h,w) - d,\; 0\right)}{C(h)} + \lambda(h) \cdot P_\text{continuation}(w)}
$$

Where:
- $d \in (0,1)$: a fixed discount (typically $d \approx 0.75$), subtracted from every positive count
- $\lambda(h)$: the interpolation weight, which distributes the discounted mass to the continuation distribution

The interpolation weight is:

$$
\lambda(h) = \frac{d \cdot \left|\{w : C(h, w) > 0\}\right|}{C(h)}
$$

This equals the total discounted mass, ensuring the distribution sums to 1.

### 8.3 Recursive Lower-Order Estimates

The continuation probability itself is estimated the same way at the lower order, recursively:

$$
P_\text{KN}(w \mid h') = \frac{\max\!\left(C(h', w) - d,\; 0\right)}{C(h')} + \lambda(h') \cdot P_\text{continuation}(w)
$$

Until the base case: the unigram continuation distribution normalized over $V$.

### 8.4 Modified Kneser-Ney

In practice, the best variant uses **different discount values** for counts 1, 2, and 3+:

$$
d(r) = \begin{cases} d_1 & r = 1 \\ d_2 & r = 2 \\ d_{3+} & r \geq 3 \end{cases}
$$

These discounts are estimated from the training data using counts of counts:

$$
Y = \frac{N_1}{N_1 + 2 N_2}
$$

$$
d_1 = 1 - 2Y\frac{N_2}{N_1}, \quad d_2 = 2 - 3Y\frac{N_3}{N_2}, \quad d_{3+} = 3 - 4Y\frac{N_4}{N_3}
$$

Modified Kneser-Ney consistently achieves the **lowest perplexity** of all classical N-gram smoothing methods.

---

## 9. Comparison Summary

| Method | Zero probs? | Key idea | Practical quality |
|---|---|---|---|
| MLE | Yes | Raw counts | Baseline only |
| Laplace (Add-1) | No | Add 1 to all counts | Poor for large $V$ |
| Add-$k$ | No | Add tuned $k$ | Marginal improvement |
| Good-Turing | No | Frequency of frequencies | Decent; unstable at high counts |
| Stupid Backoff | No | Fixed 0.4 discount, fall back | Effective at web scale |
| Katz Backoff | No | Good-Turing + normalized backoff | Good; complex normalization |
| Interpolated KN | No | Discounting + continuation probability | Very good |
| Modified KN | No | Per-count discounts + continuation | **Best classical method** |

---

## 10. Practical Guidelines

1. **Small data (< 1 M tokens):** Use interpolated or modified Kneser-Ney. Smoothing has the highest impact here.
2. **Large data (> 1 B tokens):** Stupid Backoff is fast and surprisingly competitive; Modified KN still slightly better in quality.
3. **Choosing $n$:** Higher $n$ = richer context but more sparsity. Trigrams with KN smoothing are the standard baseline.
4. **Vocabulary handling:** Apply a `<UNK>` token for out-of-vocabulary words to avoid zero probabilities at the word-identity level, separate from the smoothing over histories.
5. **Always tune on a dev set:** Never select smoothing hyperparameters ($k$, $d$, $\lambda$) on the test set.


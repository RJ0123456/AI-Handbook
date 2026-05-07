# Cross-entropy relationship with KL-divergence

## Cross-entropy

In information theory, the **cross-entropy** between two probability distributions ${\displaystyle p}$ and ${\displaystyle q}$, over the same underlying set of events, measures the average number of bits needed to identify an event drawn from the set when the coding scheme used for the set is optimized for an estimated probability distribution ${\displaystyle q}$, rather than the true distribution $
{\displaystyle p}$.

## Definition and Relationship

The cross-entropy of the distribution ${\displaystyle q}$ relative to a distribution ${\displaystyle p}$ over a given set is defined as follows:

${\displaystyle H(p,q)=-\operatorname {E} _{p}[\log q],}$

where

${\displaystyle \operatorname {E} _{p}[\cdot ]}$ is the expected value operator with respect to the distribution $
{\displaystyle p}$.

The definition may be formulated using the Kullback–Leibler divergence ${\displaystyle D_{\mathrm {KL} }(p\parallel q)}$, divergence of ${\displaystyle p}$ from ${\displaystyle q}$ also known as the relative entropy of ${\displaystyle p}$ with respect to ${\displaystyle q}$. ${\displaystyle H(p,q)=H(p)+D_{\mathrm {KL} }(p\parallel q),}$

where

${\displaystyle H(p)}$ is the entropy of ${\displaystyle p}$.

From a physics and information theory perspective, cross-entropy essentially measures "information loss" or "communication efficiency."

## Cross-entropy essentially measures "information loss" or "communication efficiency

### 1. Background: Shannon Entropy (Optimal Code Length)

In physics/information theory, **entropy $H(p)$** represents the average uncertainty of a system.

- **Physical Meaning**: If you fully understand a distribution $p$ and design an optimal encoding scheme for it (short codes for high-frequency events, long codes for low-frequency events), **entropy** is the **theoretically shortest average number of bits required to transmit each symbol**.

- This is analogous to: you are the weather controller, and you know there's a 100% chance of rain tomorrow; the information content is 0, and no encoding is needed for transmission.

### 2. Cross-Entropy: A Costly "Misunderstanding"

**Cross-entropy $H(p, q)$ describes**: when you **assume** the distribution is $q$ (predicted) but the actual distribution is $p$ (real), you design an encoding scheme based on this incorrect understanding, resulting in an excessively low average number of bits required to transmit information.

- **Physical Scenario**: Reality ($p$):
    - In a forest, the probability of a tiger appearing is 80%, and the probability of a rabbit is 20%.

    - Your Perception ($q$): You believe the probability of a tiger appearing is only 10%, and the probability of a rabbit is 90%.

- **Encoding Scheme**:
    - Because you believe there are more rabbits, you assign the shortest code (e.g., 1 bit) to rabbits and a long code (e.g., 10 bits) to tigers.

- **Physical Consequence**:
    - In reality, tigers appear every day! You have to send a 10-bit long code every time.

    - Cross-entropy is the average cost of doing this. Because your perception (prediction) deviates from reality, you incur a huge communication cost.

### 3. KL Divergence: Wasted Bandwidth

According to the mathematical formula: Cross-entropy = Entropy + KL Divergence.

$$H(p, q) = H(p) + D_{KL}(p || q)$$
In a physical sense:

1. **$H(p)$ (entropy)**: This is the minimum communication cost (background noise) that is defined by nature and cannot be reduced.
2. **$D_{KL}(p || q)$ (KL divergence)**: This is the extra cost (wasted bandwidth) incurred because your model isn't smart enough and misunderstands the world.

## 4. Why should machine learning minimize it?

When training a model, our goal is to make the model's view of the world ($q$) coincide with the real world ($p$).

- Physically, when we minimize cross-entropy, we are eliminating redundancy.

- When cross-entropy is reduced to its minimum (equal to entropy), it means your model perfectly captures the distribution of the data, no longer incurring any additional information costs due to "misunderstandings".

## Summary

- **Entropy**: The theoretical limit (the insurmountable bottom line).

- **Cross-entropy**: The actual cost under flawed perception.

- **Physical intuition**: The larger the cross-entropy, the more "inefficient" your description of the system, and the higher your "surprise" at the truth.

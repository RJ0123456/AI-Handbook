Mathematically, LSTM doesn't impose a strict range constraint on the input $x_t$ (unlike some algorithms that immediately throw an error if the input exceeds 1). However, in practical engineering applications, to accommodate the mathematical properties of sigmoid and tanh, normalization or standardization of the input data is usually required.

The specific reasons are as follows:

## 1. Avoiding Gradient Saturation

This is the most crucial requirement.

* Characteristics of $Sigmoid/Tanh$: The derivatives of these two functions rapidly approach 0 when the input is far from 0 (the function curve becomes very flat).

* Consequence: If your $x_t$ value is very large (e.g., 1000), after multiplication by the weight matrix $W$, the result will fall into the saturation region of the activation function. At this point, the gradient is almost 0, the model parameters cannot be updated, leading to "training stagnation".

* Requirements: It is generally recommended to map the input $x_t$ to $[-1, 1]$ or $[0, 1]$, or to standardize it to conform to a distribution with a mean of 0 and a variance of 1.

## 2. Maintaining Feature Sensitivity

* tanh is nearly linear in the interval $(-1, 1)$, with the largest gradient, making the model most sensitive to subtle changes in the input.

* If the input range is not uniform (e.g., feature A is 0.1, feature B is 100), the weights $W$ must learn to have a large order of magnitude difference to compensate for this imbalance, which increases training difficulty and instability.

## 3. Matching the Order of Magnitude with the Hidden State $h_{t-1}$

In the formula $\sigma(W \cdot [h_{t-1}, x_t] + b)$, $x_t$ is concatenated with $h_{t-1}$. * Since $h_{t-1}$ is output via tanh, its range is always between $(-1, 1)$.

* If the magnitude of $x_t$ is too large, it will completely "overwhelm" the information from $h_{t-1}$ in the previous time step, causing LSTM to lose its ability to utilize historical information.

------------------------------

## A Visual Analogy

You can think of the activation function as a "field of vision":

* The "high-resolution field of vision" of sigmoid and tanh is concentrated in a narrow region near 0.

* If your input $x_t$ is a behemoth (e.g., 5000), it will be at the very edge of the field of vision, and the activation function will only see a blurry outline (output close to 1, derivative close to 0).

* Only by shrinking the data to $[-1, 1]$ can the activation function see its details.

## Recommended Practices:

Before feeding data into an LSTM, typically one of the following two processing methods should be used:

1. Min-Max Scaling: $x' = \frac{x - min}{max - min}$ (Scaling to $[0, 1]$)

2. Z-Score Normalization: $x' = \frac{x - \mu}{\sigma}$ (Standardizing to mean 0, variance 1)

# Cross-Entropy with Softmax

While the derivation of Softmax is slightly more complex than that of Sigmoid (due to the involvement of vectors), its "match-3" result is equally elegant.

In multi-class classification, assuming there are $K$ classes, for the output $z_i$ of the $i$-th class, the output of Softmax is:

$$\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

## 1. Key Premise: Partial Derivatives of Softmax

We need to calculate $\frac{\partial \hat{y}_k}{\partial z_i}$. There are two cases here:

1. When $k=i$: The derivative is $\hat{y}_i(1 - \hat{y}_i)$

2. When $k \neq i$: The derivative is $-\hat{y}_i \hat{y}_k$

## 2. Cross-entropy Loss Function
The cross-entropy for multi-class classification is defined as:

$$L = - \sum_{k=1}^K y_k \ln \hat{y}_k$$
<span style="color:red">Note: Under One-hot encoding, only the true class $j$ has $y_j=1$, and the rest have $y_k=0$. Therefore, $L = -\ln \hat{y}_j$.</span>

## 3. Starting the "Cancellation" Derivation
We need to calculate the derivative of the loss $L$ with respect to the $i$-th linear output $z_i$, $\frac{\partial L}{\partial z_i}$.

Using the chain rule, we need to sum over all $\hat{y}_k$, because changing one $z_i$ affects all $\hat{y}_k$ (where $z_i$ is in the denominator):

$$\frac{\partial L}{\partial z_i} = \sum_{k=1}^K \frac{\partial L}{\partial \hat{y}_k} \frac{\partial \hat{y}_k}{\partial z_i}$$

1. Substituting the loss derivative: $\frac{\partial L}{\partial \hat{y}_k} = -\frac{y_k}{\hat{y}_k}$

2. Expanding and summing:

$$\frac{\partial L}{\partial z_i} = -\frac{y_i}{\hat{y}_i} \cdot \underbrace{\hat{y}_i(1 - \hat{y}_i)}_{k=i \text{ case}} + \sum_{k \neq i} \left( -\frac{y_k}{\hat{y}_k} \cdot \underbrace{(-\hat{y}_k \hat{y}_i)}_{k \neq i \text{ case}} \right)$$ 
3. Simplify: 
$$\frac{\partial L}{\partial z_i} = -y_i(1 - \hat{y}_i) + \sum_{k \neq i} y_k \hat{y}_i$$ $$\frac{\partial L}{\partial z_i} = -y_i + y_i \hat{y}_i + \hat{y}_i \sum_{k \neq i} y_k$$ 
4. Final merging:

Since the sum of all true labels $\sum_{k=1}^K y_k = 1$, then $y_i + \sum_{k \neq i} y_k = 1$.

$$\frac{\partial L}{\partial z_i} = \hat{y}_i (y_i + \sum_{k \neq i} y_k) - y_i = \hat{y}_i(1) - y_i$$

## 4. Conclusion

$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$

This is wonderful: Whether it's the binary classification Sigmoid or the multi-class classification Softmax, with cross-entropy, the gradient form is completely unified as "predicted value - true value".

This simplicity is not only mathematically pleasing, but also greatly simplifies the computation of backpropagation in engineering implementation.
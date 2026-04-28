# Cross-Entropy with Sigmoid

This derivation process is quite elegant and is known as "gradient cancellation." We'll use binary cross-entropy as an example to guide you through this "match-3" scenario step by step.

## 1. Preparation

Assume our model's last layer output is $z$, which, after Sigmoid activation, yields the predicted value $\hat{y}$:

* Activation function: $\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$

* Important property (to be used later): $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$

## 2. Cross-entropy Loss Function

The cross-entropy formula for binary classification is:

$$L = -[y \ln \hat{y} + (1 - y) \ln(1 - \hat{y})]$$

## 3. Starting Differentiation (Chain Rule)

We need to calculate the gradient $\frac{\partial L}{\partial z}$ of the loss $L$ with respect to the output $z$. 
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$$

### Step 1: Calculate $\frac{\partial L}{\partial \hat{y}}$ (derivative of loss with respect to predicted value)

Take the derivatives of $\ln \hat{y}$ and $\ln(1-\hat{y})$ in the $L$ formula:

$$\frac{\partial L}{\partial \hat{y}} = - \left( \frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}} \right)$$ 

Combine denominators:

$$\frac{\partial L}{\partial \hat{y}} = - \frac{y(1-\hat{y}) - \hat{y}(1-y)}{\hat{y}(1-\hat{y})} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

### Step 2: Calculate $\frac{\partial \hat{y}}{\partial z}$ (the derivative of the activation function)

Using the Sigmoid derivative property mentioned in Step 1:

$$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y})$$

## 4. The Moment of Miracle

Now multiply the two parts:

$$\frac{\partial L}{\partial z} = \underbrace{\frac{\hat{y} - y}{\hat{y}(1-\hat{y})}}_{\frac{\partial L}{\partial \hat{y}}} \cdot \underbrace{\hat{y}(1-\hat{y})}_{\frac{\partial \hat{y}}{\partial z}}$$

You'll find that the denominator $\hat{y}(1-\hat{y})$ and the derivative term $\hat{y}(1-\hat{y})$ perfectly cancel each other out!

Finally, only:

$$\frac{\partial L}{\partial z} = \hat{y} - y$$

------------------------------

## 5. Conclusion

This result is extremely concise: Gradient = Predicted value - True value.

* No saturation region: In the derivation of MSE, $\hat{y}(1-\hat{y})$ is retained, causing gradient vanishing when $z$ is large. However, in cross-entropy, this term is mathematically canceled out.

* Larger error, faster update: As long as the difference between $\hat{y}$ and $y$ is large, the gradient is large, and the model updates rapidly. This ensures that the model can quickly escape even with poor initialization (falling into the saturation region).

In multi-class classification tasks, Softmax combined with cross-entropy has the same "cancellation" property.
Choosing $tanh$ instead of $sigmoid$ in the candidate value stage of LSTM ($\tilde{C}_t$) is primarily based on the following two core logics:

## 1. Expanding the numerical range (feature directionality)

* The output of $tanh$ is $(-1, 1)$: This means that the "new content" it produces can be positive (increasing the strength of a feature) or negative (weakening or canceling out a feature).

* Compared to $sigmoid (0, 1)$: If you use sigmoid, you can only "add things" to the cell state. Mathematically, this causes the value of the cell state $C_t$ to only increase and never decrease as the sequence grows, eventually leading to a numerical explosion, and it cannot express the meaning of "reverse".

## 2. The "Vitality" of Gradients

* Avoiding Vanishing Gradients: While both tanh and sigmoid suffer from vanishing gradients when the input is extremely large or small, tanh's derivative is 1 near $0$, while sigmoid's maximum derivative is only $0.25$.

* Zero Symmetry: tanh is zero-centered. During backpropagation, this property allows for faster convergence of weight updates, preventing "Z-shaped" oscillations during updates.

## Visual Analogy: Imagine the cell state $C_t$ as the level in a water tank:

* The forget gate acts like a drain valve (determining how much water to drain).

* The input gate (the sigmoid part) acts like a control switch (determining whether to add water).

The candidate value ($\tilde{C}_t$ uses tanh) determines what is added:

* A positive value is like adding water (enhancing information);

* A negative value is like draining water (weakening the influence of old information).

If you also use sigmoid here, you can only add water to the tank, and you can't achieve a precise "balance of addition and subtraction".

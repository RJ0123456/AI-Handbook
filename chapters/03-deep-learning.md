# Chapter 3: Deep Learning

## What Is Deep Learning?

Deep Learning is a subfield of machine learning that uses **artificial neural networks** with many layers (hence "deep") to learn representations of data. These networks are inspired by the structure and function of the human brain, though the analogy is loose.

Deep learning has driven the most significant AI breakthroughs of the past decade, enabling machines to recognize images, understand speech, translate languages, and generate content at or beyond human level.

## Neural Networks: The Building Block

### The Artificial Neuron (Perceptron)

A single neuron takes weighted inputs, sums them, and applies an **activation function** to produce an output:

```
output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + bias)
```

### Activation Functions

| Function | Formula | Typical Use |
|----------|---------|-------------|
| Sigmoid | 1 / (1 + e⁻ˣ) | Binary output layers |
| Tanh | (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ) | Hidden layers (older networks) |
| ReLU | max(0, x) | Hidden layers (default choice) |
| Leaky ReLU | max(αx, x) | When dying ReLU is a problem |
| Softmax | eˣⁱ / Σeˣʲ | Multi-class output layers |

### Layers

- **Input Layer:** Receives raw features.
- **Hidden Layers:** Transform the data through learned weights. More layers = deeper network.
- **Output Layer:** Produces the final prediction.

## Training Neural Networks

### Forward Pass

Data flows through the network, producing a prediction.

### Loss Function

Measures how wrong the prediction is.
- **Cross-entropy loss** for classification
- **Mean Squared Error (MSE)** for regression

### Backpropagation

The algorithm that calculates the gradient of the loss with respect to every weight using the chain rule of calculus.

### Gradient Descent

Weights are updated to reduce the loss:

```
w = w − learning_rate × gradient
```

**Variants:**
| Optimizer | Description |
|-----------|-------------|
| SGD | Stochastic Gradient Descent — one sample at a time |
| Mini-batch SGD | Small batches (most common in practice) |
| Momentum | Accelerates gradients in the right direction |
| Adam | Adaptive learning rates; default for most tasks |
| AdamW | Adam with decoupled weight decay |

## Key Architectures

### Convolutional Neural Networks (CNNs)

Designed for **grid-like data** (images, video). Use convolutional filters to detect local patterns (edges, textures, shapes).

**Key layers:**
- **Convolutional layer:** Applies learnable filters across the input.
- **Pooling layer:** Downsamples the feature map (max pooling, average pooling).
- **Fully connected layer:** Flattens features for final classification.

**Landmark models:** LeNet, AlexNet, VGG, ResNet, EfficientNet, Vision Transformer (ViT)

**Applications:** Image classification, object detection, medical imaging, facial recognition

### Recurrent Neural Networks (RNNs)

Designed for **sequential data** (time series, text). Maintain a hidden state that captures context from previous time steps.

**Variants:**
- **LSTM (Long Short-Term Memory):** Addresses the vanishing gradient problem with gates.
- **GRU (Gated Recurrent Unit):** Simplified LSTM with fewer parameters.

**Applications:** Time-series forecasting, speech recognition (largely replaced by Transformers for NLP)

### Transformers

The dominant architecture for NLP — and increasingly for vision, audio, and multi-modal tasks.

**Key innovation:** The **self-attention mechanism**, which allows every token to attend to every other token in the input.

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V
```

**Landmark models:**
| Model | Year | Key Contribution |
|-------|------|-----------------|
| Transformer | 2017 | Introduced self-attention |
| BERT | 2018 | Bidirectional pre-training |
| GPT-2 | 2019 | Autoregressive language generation |
| GPT-3 | 2020 | 175B parameters, few-shot learning |
| T5 | 2020 | Text-to-text framework |
| GPT-4 | 2023 | Multimodal, near-human reasoning |
| LLaMA / Mistral | 2023+ | Open-weight large language models |

**Applications:** Text generation, translation, summarization, question answering, code generation

### Generative Models

- **GANs (Generative Adversarial Networks):** Generator vs. discriminator compete to create realistic data.
- **VAEs (Variational Autoencoders):** Learn a latent space for generating new samples.
- **Diffusion Models:** Gradually denoise random noise into structured data (DALL-E, Stable Diffusion).

## Practical Tips for Training Deep Networks

| Challenge | Solution |
|-----------|---------|
| Vanishing gradients | Use ReLU, batch normalization, residual connections |
| Overfitting | Dropout, data augmentation, early stopping |
| Slow convergence | Learning rate scheduling, Adam optimizer |
| Class imbalance | Weighted loss, oversampling (SMOTE), focal loss |
| Limited data | Transfer learning, data augmentation |

## Transfer Learning

Instead of training from scratch, start from a model pre-trained on a large dataset (e.g., ImageNet, large text corpora) and **fine-tune** it on your specific task.

**Why it works:** Lower layers learn general features (edges, textures) that are useful across tasks.

**Common patterns:**
- **Feature extraction:** Freeze pre-trained layers; train only the final layer(s).
- **Fine-tuning:** Unfreeze some or all layers and train with a small learning rate.

## Summary

Deep learning's power comes from learning hierarchical representations directly from raw data. CNNs, RNNs, and Transformers each excel in different domains. Transfer learning has democratized access to state-of-the-art performance even with limited data.

---

**Previous:** [Chapter 2 – Machine Learning Fundamentals](02-machine-learning.md)  
**Next:** [Chapter 4 – Natural Language Processing](04-nlp.md)

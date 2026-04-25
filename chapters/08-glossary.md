# Chapter 8: Glossary

A reference glossary of key terms used throughout this handbook.

---

## A

**Activation Function**  
A mathematical function applied to a neuron's output to introduce non-linearity. Common examples: ReLU, sigmoid, tanh, softmax.

**Adversarial Attack**  
A technique that adds small, carefully crafted perturbations to inputs to fool a machine learning model into making incorrect predictions.

**Agent (AI)**  
An autonomous system that perceives its environment, makes decisions, and takes actions to achieve a goal. Used in reinforcement learning and LLM-based AI assistants.

**Attention Mechanism**  
A component in neural networks (especially Transformers) that allows the model to focus on relevant parts of the input when producing each element of the output.

**Autoregressive Model**  
A model that generates sequences one element at a time, with each element conditioned on the previous ones. Used in GPT-style language models.

---

## B

**Backpropagation**  
The algorithm used to train neural networks by computing the gradient of the loss function with respect to each weight using the chain rule, then updating weights via gradient descent.

**Batch Normalization**  
A technique that normalizes layer inputs to have zero mean and unit variance, improving training stability and speed.

**BERT (Bidirectional Encoder Representations from Transformers)**  
A pre-trained Transformer model from Google that reads text bidirectionally, excelling at understanding tasks like question answering and classification.

**Bias (model)**  
The systematic error a model makes due to wrong assumptions. High bias leads to underfitting.

**Bias (in training data)**  
Systematic unfairness or skew in a dataset that causes a model to make discriminatory or incorrect predictions for certain groups.

---

## C

**Classification**  
A supervised learning task where the model assigns an input to one of a predefined set of categories.

**CNN (Convolutional Neural Network)**  
A neural network architecture that uses convolutional filters, well-suited for grid-structured data like images.

**Cross-Entropy Loss**  
A common loss function for classification tasks that measures the difference between the predicted probability distribution and the true distribution.

**Cross-Validation**  
A technique that evaluates a model by training and testing it on multiple subsets of the data to get a more reliable performance estimate.

---

## D

**Data Augmentation**  
Artificially expanding a training dataset by applying transformations (e.g., flips, rotations, crops) to existing samples.

**Diffusion Model**  
A generative model that learns to reverse a gradual noising process to generate high-quality data. Used in Stable Diffusion and DALL-E.

**Dimensionality Reduction**  
The process of reducing the number of features while preserving as much information as possible. Examples: PCA, t-SNE, UMAP.

**Dropout**  
A regularization technique that randomly sets a fraction of neuron activations to zero during training to prevent overfitting.

---

## E

**Embedding**  
A dense, low-dimensional vector representation of discrete inputs (words, images, users) that captures semantic relationships.

**Encoder-Decoder**  
An architecture where an encoder compresses the input into a latent representation and a decoder generates the output from it. Used in translation, summarization, and image segmentation.

**Ensemble**  
A technique that combines multiple models to produce better predictions than any single model. Examples: random forests, gradient boosting, model stacking.

**Epoch**  
One complete pass through the entire training dataset during model training.

---

## F

**F1 Score**  
The harmonic mean of precision and recall. A balanced metric useful when class imbalance is present.

**Feature**  
An individual measurable property or characteristic used as input to a machine learning model.

**Feature Engineering**  
The process of creating, selecting, or transforming raw data into features that improve model performance.

**Fine-Tuning**  
Adapting a pre-trained model to a new task by continuing training on task-specific data, usually with a smaller learning rate.

**Foundation Model**  
A large model trained on broad data at scale that can be adapted (fine-tuned or prompted) for a wide range of downstream tasks. Examples: GPT-4, LLaMA, CLIP.

---

## G

**GAN (Generative Adversarial Network)**  
A generative model consisting of a generator and a discriminator trained adversarially. The generator creates realistic samples; the discriminator tries to distinguish real from fake.

**Gradient**  
The vector of partial derivatives of a loss function with respect to model parameters. Used to update weights during training.

**Gradient Descent**  
An optimization algorithm that iteratively updates model parameters in the direction that minimizes the loss function.

**GPT (Generative Pre-trained Transformer)**  
A family of autoregressive language models from OpenAI trained on large text corpora.

---

## H

**Hallucination**  
When a language model generates plausible-sounding but factually incorrect or fabricated information.

**Hyperparameter**  
A configuration setting for a model or training process (e.g., learning rate, number of layers) that is set before training, not learned from data.

---

## I

**Inference**  
Using a trained model to make predictions on new, unseen data.

**Instance Segmentation**  
A computer vision task that identifies and delineates individual objects at the pixel level.

---

## K

**k-NN (k-Nearest Neighbors)**  
A non-parametric algorithm that classifies a new data point based on the majority class among its k closest training examples.

**Knowledge Distillation**  
A compression technique where a smaller "student" model is trained to mimic the outputs of a larger "teacher" model.

---

## L

**Label**  
The target output for a training example in supervised learning.

**Latent Space**  
A compressed, abstract representation of data learned by a model (e.g., the internal representation in a VAE or diffusion model).

**Learning Rate**  
A hyperparameter that controls how large the parameter updates are during gradient descent.

**LLM (Large Language Model)**  
A language model with billions of parameters trained on vast text corpora, capable of few-shot learning across diverse tasks.

**Loss Function**  
A function that measures how far the model's predictions are from the true labels. Training aims to minimize it.

**LoRA (Low-Rank Adaptation)**  
A parameter-efficient fine-tuning method that injects trainable low-rank matrices into a pre-trained model, reducing memory and compute requirements.

---

## M

**mAP (mean Average Precision)**  
A metric for object detection that averages precision across different recall thresholds and object classes.

**Model Card**  
A document that describes a model's intended use, performance across demographic groups, limitations, and ethical considerations.

**Multi-modal Model**  
A model that processes and integrates multiple types of data (text, images, audio). Examples: GPT-4o, Gemini.

---

## N

**NLP (Natural Language Processing)**  
The subfield of AI concerned with enabling computers to understand, process, and generate human language.

**Neural Network**  
A computational model loosely inspired by the brain, consisting of layers of interconnected nodes (neurons) that learn representations from data.

**Normalization**  
Scaling input features to a standard range (e.g., 0–1) or distribution to improve training stability.

---

## O

**Object Detection**  
A computer vision task that identifies and localizes objects in an image with bounding boxes and class labels.

**Overfitting**  
When a model learns the training data too well, including its noise, causing poor performance on new data.

**ONNX (Open Neural Network Exchange)**  
An open format for representing machine learning models, enabling interoperability between frameworks.

---

## P

**Perceptron**  
The simplest neural network unit, computing a weighted sum of inputs and applying an activation function.

**Precision**  
The fraction of positive predictions that are actually correct: TP / (TP + FP).

**Pre-training**  
Training a model on a large, general dataset before fine-tuning it on a specific task.

**Prompt**  
The input text provided to a language model to elicit a desired response.

**Prompt Engineering**  
The practice of designing effective prompts to improve language model outputs without modifying model weights.

---

## Q

**Quantization**  
Reducing the numerical precision of model weights (e.g., from 32-bit float to 8-bit integer) to decrease memory usage and increase inference speed.

---

## R

**RAG (Retrieval-Augmented Generation)**  
A technique that augments a generative model with retrieved documents at inference time to ground responses in external knowledge.

**Recall**  
The fraction of actual positives that are correctly identified: TP / (TP + FN).

**Regularization**  
Techniques that penalize model complexity to prevent overfitting. Examples: L1 (Lasso), L2 (Ridge), dropout.

**Reinforcement Learning (RL)**  
A learning paradigm where an agent learns to make decisions by receiving rewards or penalties from an environment.

**Residual Connection**  
A shortcut connection that adds the input of a layer directly to its output, enabling training of very deep networks (ResNet).

**RLHF (Reinforcement Learning from Human Feedback)**  
A technique for aligning language models with human preferences by training a reward model on human ratings and optimizing the LLM with RL.

**RNN (Recurrent Neural Network)**  
A neural network with recurrent connections that maintains a hidden state, making it suitable for sequential data.

---

## S

**Semantic Segmentation**  
A computer vision task that assigns a class label to every pixel in an image.

**Self-Attention**  
A mechanism in Transformers that allows each position in a sequence to attend to all other positions, capturing long-range dependencies.

**Self-Supervised Learning**  
A form of unsupervised learning where labels are generated automatically from the data itself (e.g., predicting masked words).

**Softmax**  
A function that converts a vector of raw scores into a probability distribution over classes.

**Supervised Learning**  
A machine learning paradigm where models are trained on labeled examples with known input-output pairs.

---

## T

**Temperature**  
A hyperparameter that controls the randomness of a language model's output. Higher temperature = more creative/random; lower = more deterministic.

**Token**  
The basic unit of text a language model processes. Can be a word, subword, or character depending on the tokenizer.

**Tokenization**  
The process of splitting text into tokens.

**Transfer Learning**  
Applying knowledge gained from one task to improve performance on a different but related task.

**Transformer**  
A neural network architecture based on self-attention, introduced in "Attention Is All You Need" (2017). The dominant architecture for NLP and increasingly for vision.

**Training Set**  
The subset of data used to fit model parameters.

---

## U

**Underfitting**  
When a model is too simple to capture the underlying patterns in the data, leading to poor performance on both training and test data.

**Unsupervised Learning**  
A machine learning paradigm where models find patterns in unlabeled data.

---

## V

**Validation Set**  
A subset of data used to tune hyperparameters and monitor model performance during training.

**Variance (model)**  
The sensitivity of a model to fluctuations in the training data. High variance leads to overfitting.

**VAE (Variational Autoencoder)**  
A generative model that encodes inputs into a learned latent distribution and decodes samples from that distribution into outputs.

**Vector Database**  
A database optimized for storing and querying high-dimensional vectors (embeddings), used in semantic search and RAG systems.

---

## W

**Weight**  
A learnable parameter in a neural network that is updated during training.

**Weight Decay**  
An L2 regularization technique applied as a penalty on large weights during optimization.

---

## Z

**Zero-Shot Learning**  
The ability of a model to perform a task it was not explicitly trained on, given only a description of the task.

---

**Previous:** [Chapter 7 – AI Ethics and Responsible AI](07-ethics.md)  
**Back to:** [Table of Contents](../README.md)

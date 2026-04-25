# Chapter 2: Machine Learning Fundamentals

## What Is Machine Learning?

Machine Learning (ML) is a subfield of AI that gives computers the ability to learn from data without being explicitly programmed. Instead of writing rules by hand, you provide examples, and the algorithm figures out the rules itself.

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."
> — Tom Mitchell, 1997

## The Machine Learning Workflow

```
1. Define the problem
       ↓
2. Collect & prepare data
       ↓
3. Choose a model
       ↓
4. Train the model
       ↓
5. Evaluate the model
       ↓
6. Deploy & monitor
```

## Types of Machine Learning

### Supervised Learning

The model learns from **labeled** examples — each input comes with a correct output.

**Common tasks:**
- **Classification:** Predict a category (spam/not spam, cat/dog)
- **Regression:** Predict a continuous value (house price, temperature)

**Common algorithms:**
| Algorithm | Use Case |
|-----------|----------|
| Linear Regression | Predicting continuous values |
| Logistic Regression | Binary classification |
| Decision Trees | Interpretable classification/regression |
| Random Forest | Ensemble of decision trees |
| Support Vector Machines (SVM) | High-dimensional classification |
| k-Nearest Neighbors (k-NN) | Instance-based learning |
| Gradient Boosting (XGBoost, LightGBM) | Tabular data competitions |

### Unsupervised Learning

The model learns patterns from **unlabeled** data — there are no predefined answers.

**Common tasks:**
- **Clustering:** Group similar data points (k-means, DBSCAN)
- **Dimensionality Reduction:** Compress data while preserving structure (PCA, t-SNE, UMAP)
- **Anomaly Detection:** Identify outliers
- **Generative Modeling:** Learn the data distribution to create new samples

### Reinforcement Learning

An agent learns to make decisions by interacting with an environment and receiving **rewards** or **penalties**.

**Key concepts:**
- **Agent:** The learner/decision-maker
- **Environment:** What the agent interacts with
- **State:** The current situation
- **Action:** What the agent can do
- **Reward:** Feedback signal (positive or negative)
- **Policy:** The strategy the agent follows

**Applications:** Game playing (AlphaGo), robotics, recommendation systems, autonomous driving.

### Semi-Supervised and Self-Supervised Learning

- **Semi-supervised:** Uses a small amount of labeled data and a large amount of unlabeled data.
- **Self-supervised:** Creates labels from the data itself (e.g., predicting a masked word in a sentence — the basis for BERT).

## Key Concepts

### Training, Validation, and Test Sets

| Split | Purpose |
|-------|---------|
| Training set | Fit the model parameters |
| Validation set | Tune hyperparameters and model selection |
| Test set | Final, unbiased evaluation of model performance |

A common split is **70% train / 15% validation / 15% test**, though this varies with dataset size.

### Overfitting and Underfitting

- **Overfitting:** The model memorizes the training data and performs poorly on new data. (High variance)
- **Underfitting:** The model is too simple to capture the underlying pattern. (High bias)
- **Good fit:** The model generalizes well to unseen data.

**Techniques to reduce overfitting:**
- Gather more training data
- Use regularization (L1/Lasso, L2/Ridge)
- Apply dropout (neural networks)
- Simplify the model
- Use cross-validation

### Evaluation Metrics

**Classification:**
| Metric | Description |
|--------|-------------|
| Accuracy | Fraction of correct predictions |
| Precision | True positives / (True positives + False positives) |
| Recall | True positives / (True positives + False negatives) |
| F1 Score | Harmonic mean of precision and recall |
| AUC-ROC | Area under the ROC curve |

**Regression:**
| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of determination |

### Hyperparameters vs. Parameters

- **Parameters:** Learned from training data (e.g., weights in a neural network).
- **Hyperparameters:** Set before training (e.g., learning rate, number of trees, regularization strength).

### Feature Engineering

The process of transforming raw data into features that better represent the underlying problem to the model.

Common techniques:
- Normalization / Standardization
- One-hot encoding of categorical variables
- Handling missing values (imputation)
- Creating interaction features
- Log transforms for skewed distributions

## The Bias-Variance Tradeoff

| | High Bias | Low Bias |
|-|-----------|----------|
| **High Variance** | — | Overfitting |
| **Low Variance** | Underfitting | Good fit |

The goal is to find a model complex enough to capture the signal, but not so complex that it captures the noise.

## Summary

Machine learning is the backbone of modern AI. Understanding supervised, unsupervised, and reinforcement learning — and the core concepts of evaluation, overfitting, and feature engineering — gives you the foundation to tackle nearly any ML problem.

---

**Previous:** [Chapter 1 – Introduction to AI](01-introduction.md)  
**Next:** [Chapter 3 – Deep Learning](03-deep-learning.md)

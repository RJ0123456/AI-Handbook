# Chapter 6: AI Tools and Frameworks

## Overview

The modern AI ecosystem is rich with open-source and commercial tools that lower the barrier to building, training, and deploying AI systems. This chapter surveys the most important tools across the full AI lifecycle.

## Deep Learning Frameworks

### PyTorch

The most widely adopted framework for research and increasingly for production.

**Key features:**
- Dynamic computation graph (define-by-run)
- Strong Python integration
- TorchScript and ONNX for deployment
- Large ecosystem (torchvision, torchaudio, torchtext, HuggingFace)

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)
```

### TensorFlow / Keras

Google's framework — popular in production deployments.

**Key features:**
- Static and dynamic graph modes (TF 2.x defaults to eager)
- Keras high-level API for rapid prototyping
- TensorFlow Serving, TF Lite for deployment
- Strong TPU support

### JAX

Google's high-performance framework for research.

**Key features:**
- Functional transformations: `jit`, `grad`, `vmap`, `pmap`
- NumPy-compatible API
- Best performance on TPUs
- Used by DeepMind, Google Brain

## Machine Learning Libraries

| Library | Focus |
|---------|-------|
| **scikit-learn** | Classical ML: decision trees, SVMs, pipelines, evaluation |
| **XGBoost** | Gradient boosting; benchmark for tabular data |
| **LightGBM** | Faster gradient boosting; large datasets |
| **CatBoost** | Gradient boosting with native categorical feature support |
| **statsmodels** | Statistical modeling, hypothesis testing |

## Data Processing

| Library | Purpose |
|---------|---------|
| **NumPy** | Numerical arrays; foundation of scientific Python |
| **pandas** | DataFrames; tabular data manipulation |
| **Polars** | Fast DataFrame library (Rust-based) |
| **Dask** | Parallel computation on datasets larger than memory |
| **Apache Spark (PySpark)** | Distributed data processing at scale |

## LLM and NLP Tooling

### Hugging Face Ecosystem

The central hub for pre-trained models and NLP tooling.

| Package | Purpose |
|---------|---------|
| `transformers` | Pre-trained models for NLP, vision, audio |
| `datasets` | 10,000+ standardized datasets |
| `tokenizers` | Fast tokenization |
| `peft` | Parameter-Efficient Fine-Tuning (LoRA, prefix tuning) |
| `trl` | Training LLMs with RLHF |
| `accelerate` | Distributed training made simple |
| `diffusers` | Diffusion models for images and audio |

### LLM APIs

| Provider | Models | Key Use Cases |
|----------|--------|--------------|
| OpenAI | GPT-4o, o1, o3 | Chat, code generation, vision |
| Anthropic | Claude 3.5 / 3.7 | Long context, reasoning |
| Google | Gemini 1.5 / 2.0 | Multimodal, long context |
| Meta (open) | LLaMA 3 | Self-hosted, fine-tuning |
| Mistral (open) | Mistral, Mixtral | Efficient, multilingual |

### LLM Orchestration

| Library | Purpose |
|---------|---------|
| **LangChain** | Chains, agents, RAG pipelines |
| **LlamaIndex** | Data ingestion and retrieval for LLMs |
| **DSPy** | Programming LMs with optimized prompts |
| **Semantic Kernel** | Microsoft's SDK for AI app development |

## MLOps and Experiment Tracking

| Tool | Purpose |
|------|---------|
| **MLflow** | Experiment tracking, model registry, serving |
| **Weights & Biases (W&B)** | Experiment tracking, model versioning, sweeps |
| **DVC** | Data and model versioning (Git for data) |
| **Kubeflow** | ML workflows on Kubernetes |
| **Airflow** | Workflow scheduling and orchestration |
| **Prefect / Dagster** | Modern data pipelines |

## Model Deployment

### Serving Frameworks

| Tool | Description |
|------|-------------|
| **FastAPI** | Build REST APIs for model inference |
| **TorchServe** | PyTorch model serving |
| **TF Serving** | TensorFlow model serving |
| **Triton Inference Server** | NVIDIA's high-performance multi-framework server |
| **vLLM** | High-throughput LLM inference with PagedAttention |
| **Ollama** | Run open LLMs locally |

### Cloud ML Platforms

| Platform | Provider |
|----------|---------|
| SageMaker | AWS |
| Vertex AI | Google Cloud |
| Azure ML | Microsoft Azure |
| Databricks | Multi-cloud data + AI platform |

## Vector Databases (for RAG)

| Database | Notes |
|---------|-------|
| **Pinecone** | Managed, fast vector search |
| **Weaviate** | Open-source, multi-modal |
| **Qdrant** | Open-source, Rust-based |
| **Chroma** | Simple open-source for prototyping |
| **pgvector** | Vector extension for PostgreSQL |
| **FAISS** | Facebook's library for similarity search |

## Hardware

| Hardware | Use Case |
|---------|---------|
| **NVIDIA GPUs (A100, H100, RTX)** | Training and inference; most widely used |
| **Google TPUs** | Optimized for TensorFlow/JAX; Google Cloud |
| **Apple Silicon (M-series)** | Efficient local inference via Metal |
| **AMD GPUs (ROCm)** | Growing alternative to NVIDIA |
| **AWS Trainium / Inferentia** | Custom ASICs for training and inference |

## Choosing the Right Tool

```
Starting out with ML?          → scikit-learn + pandas
Deep learning research?        → PyTorch
Production vision/audio?       → PyTorch or TensorFlow
Working with LLMs?             → HuggingFace Transformers + LangChain
Tabular data competitions?     → XGBoost / LightGBM
Experiment tracking?           → MLflow (open-source) or W&B (cloud)
Deploying LLMs at scale?       → vLLM + FastAPI
```

## Summary

The AI tooling landscape evolves rapidly. PyTorch and the Hugging Face ecosystem dominate research and NLP. For production systems, MLOps tools like MLflow and W&B enable reproducibility and collaboration. Cloud platforms provide scalable infrastructure, while open-source options give full control over the stack.

---

**Previous:** [Chapter 5 – Computer Vision](05-computer-vision.md)  
**Next:** [Chapter 7 – AI Ethics and Responsible AI](07-ethics.md)

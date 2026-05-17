# Transformer Encoder Architecture

## Table of Contents
1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Mathematical Formulas](#mathematical-formulas)
4. [Components Explanation](#components-explanation)
5. [Use Cases](#use-cases)
6. [Open Source Models](#open-source-models)
7. [Implementation Example](#implementation-example)

## Overview

The Transformer Encoder is a neural network architecture component that processes input sequences and converts them into rich contextual representations. Introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017), the encoder uses self-attention mechanisms and feed-forward networks to capture long-range dependencies efficiently.

### Key Characteristics:
- **Parallel Processing**: All positions in a sequence are processed simultaneously
- **Attention-Based**: Uses multi-head self-attention to capture relationships between all pairs of positions
- **Position-Independent**: Relies on positional encodings to maintain sequential information
- **Bidirectional Context**: Has access to both left and right context when processing any position

## Architecture Diagram

The encoder consists of a stack of N identical layers (typically 6-12 in practice), where each layer contains:

**See diagrams in static folder:**
- [Encoder Block Structure](static/encoder-block-structure.md)
- [Attention Mechanism](static/attention-mechanism.md)
- [Encoder Layer Detail](static/encoder-layer-detail.md)
- [Positional Encoding](static/positional-encoding.md)
- [Encoder Models Comparison](static/encoder-comparison.md)

## Mathematical Formulas

### 1. Scaled Dot-Product Attention

The fundamental building block of the encoder is attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ (Query): $[B, L, d_k]$ - batch size, sequence length, dimension
- $K$ (Key): $[B, L, d_k]$
- $V$ (Value): $[B, L, d_v]$
- $d_k$: dimension of keys (scaled by $\sqrt{d_k}$ to prevent vanishing gradients)

**Derivation**:
- Compute attention weights: $\text{scores} = QK^T / \sqrt{d_k}$ where higher scores indicate stronger relevance
- Apply softmax to get normalized attention weights
- Compute weighted sum of values

### 2. Multi-Head Attention

Instead of single attention, use multiple attention heads in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Parameters:
- $h$ = number of attention heads (typically 8 or 12)
- $W_i^Q, W_i^K, W_i^V$: projection matrices for each head
- $W^O$: output projection matrix

**Dimensions**:
- Input: $[B, L, d_{model}]$
- Each head dimension: $d_v = d_k = d_{model} / h$
- Output: $[B, L, d_{model}]$

### 3. Feed-Forward Network (FFN)

Applied to each position separately and identically:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Or with modern activations (e.g., GELU, SwiGLU):

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

Typical configuration:
- First layer expands: $d_{model} \rightarrow 4 \times d_{model}$
- Second layer projects back: $4 \times d_{model} \rightarrow d_{model}$

### 4. Layer Normalization

Applied before/after each sub-layer:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- $\mu$ = mean of $x$ across feature dimension
- $\sigma^2$ = variance across feature dimension
- $\gamma, \beta$ = learnable scale and shift parameters
- $\epsilon$ = small constant for numerical stability

### 5. Residual Connection (Skip Connection)

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Or with Pre-LN (modern variant):

$$\text{Output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

### 6. Positional Encoding

Adds position information to embeddings (Sinusoidal):

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- $pos$ = position in sequence (0, 1, 2, ..., L-1)
- $i$ = dimension index (0, 1, 2, ..., d_{model}/2 - 1)

### 7. Complete Encoder Layer

Combining all components (Post-LN):

$$\text{EncoderLayer}(x) = \text{FFN}(\text{Attention}(x) + x) + \text{Attention}(x) + x$$

Or more formally:

$$x' = x + \text{MultiHeadAttention}(\text{LayerNorm}(x))$$
$$\text{EncoderOutput} = x' + \text{FFN}(\text{LayerNorm}(x'))$$

### 8. Full Encoder Stack

$$\text{Encoder}(X) = \text{EncoderLayer}_N(\text{EncoderLayer}_{N-1}(...\text{EncoderLayer}_1(X)))$$

## Components Explanation

### Multi-Head Self-Attention

**Purpose**: Allows the model to attend to different representation subspaces
- Each head learns different patterns and relationships
- Enables simultaneous focus on different parts of the sequence

**Advantages**:
- Captures diverse linguistic phenomena
- Improves gradient flow during backpropagation
- Reduces computational cost per head

**Complexity**: $O(L^2 \cdot d_{model})$ where $L$ is sequence length

### Feed-Forward Network

**Purpose**: Introduces non-linearity and increases model capacity
- Position-wise fully connected layers
- Applied identically to each position

**Configuration**:
- Two linear transformations with non-linearity
- Typically expands to 4x the model dimension
- Example: $512 \rightarrow 2048 \rightarrow 512$

### Layer Normalization

**Purpose**: Stabilizes training and improves convergence
- Normalizes across feature dimension
- Learnable scale and shift restore representational capacity

**Variants**:
- **Post-LN** (Original): LN after residual connection
- **Pre-LN** (Modern): LN before sub-layer (more stable)

### Residual Connections

**Purpose**: Enable deep networks and improve gradient flow
- Allows gradients to flow directly through layers
- Helps prevent vanishing gradient problem
- Enables training of very deep models (50+ layers)

## Use Cases

### 1. **Language Understanding Tasks**
- **Text Classification**: Sentiment analysis, topic classification
- **Question Answering**: SQuAD, TriviaQA
- **Natural Language Inference**: MNLI, SNLI
- **Semantic Similarity**: Measuring sentence/document similarity

### 2. **Information Extraction**
- **Named Entity Recognition (NER)**: Identifying persons, organizations, locations
- **Relation Extraction**: Extracting relationships between entities
- **Semantic Role Labeling**: Understanding predicate-argument structures

### 3. **Representation Learning**
- **Embeddings**: Creating contextual embeddings for downstream tasks
- **Transfer Learning**: Pre-training on large corpora, fine-tuning on specific tasks
- **Semantic Search**: Finding similar documents/queries

### 4. **Sequence Tagging**
- **Part-of-Speech Tagging**: Identifying word types
- **Chunking**: Identifying noun/verb phrases
- **Sequence Labeling**: Various token-level classification tasks

### 5. **Feature Extraction**
- Generating representations for clustering
- Creating features for downstream ML models
- Dimensionality reduction for large texts

### 6. **Multi-modal Applications**
- **Vision-Language**: Processing images + text (ViT encoder)
- **Cross-modal Retrieval**: Matching images to descriptions
- **Scene Understanding**: Combining visual and textual features

### 7. **Domain-Specific Applications**
- **Biomedical NLP**: Processing scientific papers, clinical notes
- **Code Understanding**: Analyzing source code
- **Legal Document Analysis**: Processing contracts, precedents

## Open Source Models

### 1. **BERT (Bidirectional Encoder Representations from Transformers)**
- **Organization**: Google
- **Repository**: [google-research/bert](https://github.com/google-research/bert)
- **Language**: PyTorch, TensorFlow
- **Details**:
  - Pure encoder architecture
  - 12-24 layers, 768-1024 hidden dimensions
  - Trained on MLM (Masked Language Modeling)
  - Base model: 110M parameters
  - Large model: 340M parameters
- **Use Cases**: Classification, tagging, similarity tasks
- **Citation**: Devlin et al., 2018

### 2. **RoBERTa (Robustly Optimized BERT)**
- **Organization**: Facebook AI
- **Repository**: [pytorch/fairseq](https://github.com/pytorch/fairseq)
- **Language**: PyTorch
- **Details**:
  - Improved BERT with better training
  - Same architecture with optimized pre-training
  - 110M-355M parameters
- **Improvements**: Better performance on GLUE, SuperGLUE benchmarks
- **Citation**: Liu et al., 2019

### 3. **DistilBERT (Distilled BERT)**
- **Organization**: Hugging Face
- **Repository**: [huggingface/transformers](https://github.com/huggingface/transformers)
- **Language**: PyTorch, TensorFlow
- **Details**:
  - 40% smaller, 60% faster than BERT
  - 6 layers instead of 12
  - 66M parameters
  - Uses knowledge distillation
- **Use Cases**: Fast inference, edge devices, real-time applications

### 4. **ALBERT (A Lite BERT)**
- **Organization**: Google
- **Repository**: [google-research/albert](https://github.com/google-research/albert)
- **Language**: PyTorch, TensorFlow
- **Details**:
  - Parameter-efficient via factorization
  - Shared layers across encoder stack
  - 12M-223M parameters
  - Faster training convergence
- **Use Cases**: Resource-constrained environments

### 5. **ELECTRA**
- **Organization**: Google Research
- **Repository**: [google-research/electra](https://github.com/google-research/electra)
- **Language**: PyTorch, TensorFlow
- **Details**:
  - Discriminator-based pre-training
  - More sample efficient than MLM
  - 110M-340M parameters
- **Advantages**: Better performance with less data

### 6. **DeBERTa (Decoding-enhanced BERT)**
- **Organization**: Microsoft
- **Repository**: [microsoft/DeBERTa](https://github.com/microsoft/DeBERTa)
- **Language**: PyTorch
- **Details**:
  - Disentangled attention mechanism
  - Improved attention mechanism design
  - 110M-900M parameters
  - State-of-the-art on SuperGLUE
- **Innovation**: Separate content and position attention

### 7. **XLNet**
- **Organization**: Google/CMU
- **Repository**: [zihangdai/xlnet](https://github.com/zihangdai/xlnet)
- **Language**: PyTorch, TensorFlow
- **Details**:
  - Transformer-XL based
  - Autoregressive pre-training (PLM)
  - 340M parameters
  - Longer context (1600 tokens)
- **Advantages**: Captures longer dependencies

### 8. **Sentence-BERT (SBERT)**
- **Organization**: Hugging Face
- **Repository**: [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- **Language**: PyTorch
- **Details**:
  - BERT with Siamese network design
  - Optimized for sentence embeddings
  - Fast semantic search
  - 110M-355M parameters
- **Use Cases**: Semantic similarity, clustering, search

### 9. **Domain-Specific Models**

#### BioBERT
- **Domain**: Biomedical NLP
- **Repository**: [dmis-lab/biobert](https://github.com/dmis-lab/biobert)
- **Use Cases**: Biomedical text mining, drug discovery

#### ClinicalBERT
- **Domain**: Clinical NLP
- **Use Cases**: Medical text classification, clinical notes analysis

#### CodeBERT
- **Domain**: Code understanding
- **Repository**: [microsoft/CodeBERT](https://github.com/microsoft/CodeBERT)
- **Use Cases**: Code search, documentation generation

#### LEGAL-BERT
- **Domain**: Legal documents
- **Use Cases**: Contract analysis, legal research

### 10. **Multilingual Models**

#### mBERT (Multilingual BERT)
- **Languages**: 104 languages
- **Parameters**: 110M
- **Use Cases**: Cross-lingual transfer

#### XLM-RoBERTa
- **Organization**: Facebook
- **Languages**: 100+ languages
- **Parameters**: 550M
- **Use Cases**: Multilingual classification, translation

## Implementation Example

### PyTorch Implementation (Simplified)

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, L, D = x.shape
        
        # Project and reshape for multi-head attention
        Q = self.q_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L, D)
        output = self.out_proj(attn_output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.attn(self.norm1(x))
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def create_positional_encoding(self, max_seq_len, d_model):
        pos = torch.arange(max_seq_len).unsqueeze(1)
        i = torch.arange(d_model)[::2].unsqueeze(0)
        
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos / (10000 ** (2 * i / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (2 * i / d_model)))
        
        return pe.unsqueeze(0)  # [1, max_seq_len, d_model]
    
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        seq_len = x.shape[1]
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return x

# Usage
model = Encoder(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    vocab_size=10000,
    max_seq_len=512
)

input_ids = torch.randint(0, 10000, (2, 128))  # batch_size=2, seq_len=128
output = model(input_ids)
print(output.shape)  # [2, 128, 512]
```

### Using Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained BERT encoder
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize input
text = "The encoder architecture is fundamental to transformers."
inputs = tokenizer(text, return_tensors="pt")

# Get encoder output
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state: [1, num_tokens, 768]
embeddings = outputs.last_hidden_state
print(embeddings.shape)
```

## Key Insights

1. **Scalability**: Encoder can process sequences in parallel, making training efficient
2. **Bidirectional**: Full access to context (unlike autoregressive models) enables better representations
3. **Composability**: Stacking identical layers allows for deep models with residual connections
4. **Versatility**: Pre-trained encoders transfer well to diverse downstream tasks
5. **Attention Mechanism**: Provides interpretability through attention weights

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*
- Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *arXiv:1810.04805*
- Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." *arXiv:1907.11692*
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*

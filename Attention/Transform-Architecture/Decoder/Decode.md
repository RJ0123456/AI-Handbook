# Transformer Decoder Architecture

## Table of Contents
1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Mathematical Formulas](#mathematical-formulas)
4. [Components Explanation](#components-explanation)
5. [Use Cases](#use-cases)
6. [Open Source Models](#open-source-models)
7. [Implementation Example](#implementation-example)
8. [Encoder-Decoder Integration](#encoder-decoder-integration)

## Overview

The Transformer Decoder is a neural network architecture component that generates output sequences autoregressively using attention mechanisms. Unlike the encoder which processes the entire input sequence in parallel, the decoder generates outputs one token at a time, attending to previously generated tokens and the encoder's output representations.

Introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017), the decoder combines three types of attention:
1. **Masked Self-Attention**: Attends only to previous tokens
2. **Cross-Attention**: Attends to encoder outputs
3. **Multi-Head Design**: Parallel attention mechanisms

### Key Characteristics:
- **Autoregressive Generation**: Generates tokens sequentially, one at a time
- **Causal Masking**: Can only attend to previous positions, not future ones
- **Cross-Attention**: Incorporates information from encoder via cross-attention
- **Bidirectional Encoder Access**: Full access to bidirectional encoder context
- **Token-by-Token**: Produces one output token per forward pass during inference

## Architecture Diagram

The decoder consists of a stack of N identical layers (typically 6-12, matching encoder), where each layer contains:

**See diagrams in static folder:**
- [Decoder Block Structure](static/decoder-block-structure.md)
- [Masked Self-Attention](static/masked-attention.md)
- [Cross-Attention Mechanism](static/cross-attention.md)
- [Decoder Layer Detail](static/decoder-layer-detail.md)
- [Encoder-Decoder Interaction](static/encoder-decoder-interaction.md)
- [Autoregressive Generation](static/autoregressive-generation.md)

## Mathematical Formulas

### 1. Masked Scaled Dot-Product Attention

Decoder uses masked attention to prevent attending to future tokens:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V$$

Where:
- $M$ is the causal mask: $M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$
- This ensures attention weights are 0 for future positions
- $Q, K, V$: Query, Key, Value matrices
- $d_k$: Key dimension

**Effect of Masking**:
- Position $i$ can only attend to positions $0, 1, ..., i$
- Future information is prevented during training and inference
- Enables parallel training on sequences with ground truth

### 2. Multi-Head Masked Attention

Parallel masked attention heads in decoder:

$$\text{MultiHeadMasked}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head applies causal masking:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Dimensions**:
- Input: $[B, L, d_{model}]$ where $L$ is target sequence length
- Each head: operates on $d_k = d_{model} / h$ dimensions
- Output: $[B, L, d_{model}]$

### 3. Cross-Attention Mechanism

Decoder attends to encoder outputs:

$$\text{CrossAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ comes from decoder layer: $[B, L_{target}, d_k]$
- $K, V$ come from encoder: $[B, L_{source}, d_k]$
- **No masking** applied - full access to encoder context
- Enables information flow from encoder to decoder

**Key Difference from Self-Attention**:
- Self-attention: Q, K, V all from same source (decoder)
- Cross-attention: Q from decoder, K, V from encoder
- Cross-attention shape: decoder_length × encoder_length

### 4. Multi-Head Cross-Attention

$$\text{MultiHeadCross}(Q_{dec}, K_{enc}, V_{enc}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(Q_{dec}W_i^Q, K_{enc}W_i^K, V_{enc}W_i^V)$$

**Information Flow**:
- Enables decoder to access all encoder positions simultaneously
- Multiple heads capture different encoder aspects
- Output maintains decoder sequence length

### 5. Feed-Forward Network (Same as Encoder)

Applied to each position separately:

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

Configuration:
- Expansion factor: $d_{model} \rightarrow 4 \times d_{model}$
- Projection: $4 \times d_{model} \rightarrow d_{model}$

### 6. Layer Normalization and Residuals

Pre-normalization variant (modern):

$$x' = x + \text{MaskedAttention}(\text{LayerNorm}(x))$$
$$x'' = x' + \text{CrossAttention}(\text{LayerNorm}(x'))$$
$$\text{Output} = x'' + \text{FFN}(\text{LayerNorm}(x''))$$

**Residual Paths**: Three independent residual connections per layer

### 7. Positional Encoding (Decoder)

Same sinusoidal encoding as encoder:

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Important: Encoding based on **output position**, not input position

### 8. Complete Decoder Layer

Combining masked attention, cross-attention, and FFN:

$$\text{DecoderLayer}(x, encoder\_output) = $$
$$\text{FFN}(x + \text{CrossAttn}(x, encoder\_output) + \text{MaskedSelfAttn}(x))$$

More formally:

$$x_1 = x + \text{MaskedMultiHeadAttention}(\text{LayerNorm}(x))$$
$$x_2 = x_1 + \text{MultiHeadCrossAttention}(\text{LayerNorm}(x_1), encoder\_output)$$
$$x_3 = x_2 + \text{FFN}(\text{LayerNorm}(x_2))$$
$$\text{Output} = x_3$$

### 9. Full Decoder Stack

$$\text{Decoder}(Y, encoder\_output) = $$
$$\text{DecoderLayer}_N(...\text{DecoderLayer}_1(Y, encoder\_output))$$

Where:
- $Y$ = target sequence (or previously generated tokens)
- $encoder\_output$ = fixed output from encoder

### 10. Output Projection and Distribution

Final output logits for vocabulary:

$$\text{logits} = \text{Decoder}(Y, encoder\_output) \times W_{vocab}^T + b_{vocab}$$

$$P(y_t | y_{<t}, x) = \text{softmax}(\text{logits}_t)$$

Where:
- $W_{vocab}$: $[d_{model} \times |V|]$ projection to vocabulary
- Often tied with embedding weights
- $|V|$ = vocabulary size

## Components Explanation

### Masked Self-Attention

**Purpose**: Prevents decoder from seeing future tokens during training

**Mechanism**:
- At position $i$, attention can only attend to positions $0...i$
- Future positions receive $-\infty$ attention scores (become 0 after softmax)
- Implements causality constraint

**Training vs Inference**:
- **Training**: Process entire target sequence with mask (parallel)
- **Inference**: Generate one token at a time, add to sequence
- Mask prevents information leakage during training

**Complexity**: $O(L^2 \cdot d_{model})$ same as encoder attention

### Cross-Attention

**Purpose**: Incorporates encoder information into decoder generation

**Key Features**:
- Decoder queries can attend to all encoder positions
- No causal masking on encoder attention
- Enables sequence-to-sequence translation
- Different queries per decoder position

**Attention Matrix**:
- Shape: $[target\_len, source\_len]$
- Allows decoder to retrieve relevant source information
- Multiple heads capture different alignments

**Use in Generation**:
- During training: full cross-attention matrix
- During inference: same attention for all generated tokens
- Enables copying mechanism and alignment

### Embedding and Positional Encoding

**Target Embedding**:
- Embeds target tokens into $d_{model}$ space
- Typically shared weights with output projection
- Applied to ground truth during training
- Applied to generated tokens during inference

**Positional Encoding**:
- Based on **output sequence position**, not input
- Helps decoder understand generation order
- Sinusoidal encoding allows variable length sequences

### Output Projection

**Purpose**: Maps decoder hidden states to vocabulary probabilities

**Architecture**:
- Linear layer: $d_{model} \rightarrow |V|$
- Softmax to get probability distribution
- Often shares weights with embedding matrix

**Generation Process**:
- Sample from distribution: $y_t \sim P(y_t | ...)$
- Greedy decoding: $y_t = \arg\max_v P(y_t = v | ...)$
- Beam search: keep $k$ highest probability sequences

## Use Cases

### 1. **Machine Translation (Seq2Seq)**
- **Encoder**: Processes source language
- **Decoder**: Generates target language
- **Cross-Attention**: Aligns source and target
- **Examples**: English→French, English→German
- **Models**: Transformer, mBART, mT5

### 2. **Text Summarization**
- **Abstractive Summarization**: Decoder generates summary
- **Encoder**: Processes source document
- **Decoder**: Creates condensed output
- **Examples**: PEGASUS, BART, T5
- **Applications**: News summarization, document condensing

### 3. **Question Answering (Generative)**
- **Encoder**: Processes context passage
- **Decoder**: Generates answer
- **Cross-Attention**: Retrieves answer information from context
- **Examples**: SQuAD with seq2seq, ELI5
- **Models**: BART, T5, Pegasus

### 4. **Text Generation**
- **Decoder-Only**: Some models use only decoder (GPT-like)
- **Conditional Generation**: Decoder generates conditioned on prompt
- **Examples**: Story generation, code generation
- **Beam Search**: Multiple candidate sequences

### 5. **Image Captioning**
- **Encoder**: Vision encoder (CNN or ViT)
- **Decoder**: Generates caption text
- **Cross-Attention**: Attends to image regions
- **Applications**: Scene description, accessibility
- **Models**: ViT + Transformer decoder

### 6. **Dialogue Systems**
- **Context Encoding**: Encoder processes conversation history
- **Response Generation**: Decoder generates next response
- **Long-term Dependency**: Cross-attention handles coherence
- **Examples**: Seq2Seq chatbots, conversational AI

### 7. **Speech Recognition & Synthesis**
- **Speech-to-Text**: Encoder processes audio, decoder generates text
- **Text-to-Speech**: Encoder processes text, decoder generates audio
- **Cross-Attention**: Aligns modalities
- **Examples**: Transformer-based ASR, TTS systems

### 8. **Semantic Parsing**
- **Source**: Natural language text
- **Target**: Logical form or SQL queries
- **Cross-Attention**: Maps NL to formal structures
- **Applications**: SQL generation, semantic understanding

### 9. **Data-to-Text Generation**
- **Input**: Structured data (tables, graphs)
- **Output**: Natural language description
- **Encoder**: Processes structured data
- **Decoder**: Generates fluent text

### 10. **Document Understanding**
- **Encoder**: Processes document sections
- **Decoder**: Generates answers, summaries, or transformations
- **Cross-Attention**: Links document parts to output

## Open Source Models

### 1. **BART (Bidirectional and Auto-Regressive Transformers)**
- **Organization**: Facebook AI
- **Repository**: [pytorch/fairseq](https://github.com/pytorch/fairseq)
- **Language**: PyTorch
- **Details**:
  - Combines encoder (BERT-like) + decoder (autoregressive)
  - Encoder-decoder architecture
  - 139M-405M parameters
  - Pre-trained on denoising objective
- **Use Cases**: Summarization, translation, QA
- **Citation**: Lewis et al., 2019

### 2. **T5 (Text-to-Text Transfer Transformer)**
- **Organization**: Google
- **Repository**: [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)
- **Language**: TensorFlow, PyTorch
- **Details**:
  - Unified text-to-text framework
  - Encoder-decoder for all tasks
  - Base: 220M, Large: 770M, 3B, 11B parameters
  - Transfer learning on 750GB corpus
- **Use Cases**: Translation, summarization, QA, classification
- **Citation**: Raffel et al., 2019

### 3. **Pegasus (Pre-training with Extracted Gap-sentences)**
- **Organization**: Google Research
- **Repository**: [google-research/pegasus](https://github.com/google-research/pegasus)
- **Language**: TensorFlow
- **Details**:
  - Optimized for abstractive summarization
  - Pre-trained on document-summary pairs
  - Base: 223M, Large: 568M parameters
  - SOTA on summarization benchmarks
- **Use Cases**: Abstractive summarization, document generation
- **Citation**: Zhang et al., 2019

### 4. **mBART (Multilingual BART)**
- **Organization**: Facebook AI
- **Repository**: [pytorch/fairseq](https://github.com/pytorch/fairseq)
- **Language**: PyTorch
- **Details**:
  - Multilingual encoder-decoder
  - 50+ languages supported
  - Denoising pre-training
  - 610M parameters
- **Use Cases**: Multilingual translation, cross-lingual summarization
- **Citation**: Liu et al., 2020

### 5. **mT5 (Multilingual T5)**
- **Organization**: Google
- **Repository**: [google-research/multilingual-t5](https://github.com/google-research/multilingual-t5)
- **Language**: TensorFlow, PyTorch
- **Details**:
  - T5 for 101 languages
  - Base: 392M, Large: 1.2B, 3B, 13B parameters
  - Unified multilingual framework
- **Use Cases**: Translation, summarization across languages
- **Citation**: Xue et al., 2020

### 6. **MarianMT (Machine Translation)**
- **Organization**: Hugging Face
- **Repository**: [huggingface/transformers](https://github.com/huggingface/transformers)
- **Language**: PyTorch, TensorFlow
- **Details**:
  - 1500+ language pairs
  - ~50M parameters per model
  - Built on Marian framework
  - Easy deployment
- **Use Cases**: Production machine translation
- **Supported**: Hugging Face Hub with pre-trained models

### 7. **GPT-2, GPT-3, GPT-4 (Decoder-Only)**
- **Organization**: OpenAI
- **Repository**: [openai/gpt-2](https://github.com/openai/gpt-2)
- **Language**: PyTorch, TensorFlow
- **Details**:
  - Pure decoder architecture (no encoder)
  - GPT-2: 1.5B parameters
  - GPT-3: 175B parameters (proprietary)
  - GPT-4: Multimodal (proprietary)
- **Use Cases**: Text generation, in-context learning
- **Citation**: Radford et al., 2018-2023

### 8. **LLaMA (Large Language Model Meta AI)**
- **Organization**: Meta
- **Repository**: [facebookresearch/llama](https://github.com/facebookresearch/llama)
- **Language**: PyTorch
- **Details**:
  - Decoder-only architecture
  - 7B, 13B, 33B, 65B parameters
  - Optimized for efficiency
  - Open-source weights
- **Use Cases**: General text generation, instruction following
- **Citation**: Touvron et al., 2023

### 9. **Mistral, Mixtral (Sparse Decoder Models)**
- **Organization**: Mistral AI
- **Repository**: [mistralai/mistral-src](https://github.com/mistralai/mistral-src)
- **Language**: PyTorch
- **Details**:
  - Mistral-7B: Efficient decoder
  - Mixtral-8x7B: Mixture of Experts (MoE)
  - High performance with efficiency
  - Sliding window attention
- **Use Cases**: Fast inference, instruction following

### 10. **Llama 2, CodeLlama**
- **Organization**: Meta
- **Repository**: [facebookresearch/llama](https://github.com/facebookresearch/llama)
- **Language**: PyTorch
- **Details**:
  - Llama 2: 7B-70B parameters
  - CodeLlama: Specialized for code
  - Fine-tuned for chat and instruction
- **Use Cases**: Chat, code generation, general tasks

### 11. **Claude (Decoder-Only, Proprietary)**
- **Organization**: Anthropic
- **Details**:
  - Decoder-only transformer
  - Constitutional AI training
  - 100K token context window
  - Multimodal variants
- **Use Cases**: Conversational AI, analysis, coding

### 12. **BLOOM (BigScience Large Open-science Open-access Multilingual)**
- **Organization**: BigScience
- **Repository**: [bigscience/bloom](https://github.com/bigscience/bloom)
- **Language**: PyTorch
- **Details**:
  - 176B parameters
  - 46 languages + 13 programming languages
  - Open weights and license
- **Use Cases**: Multilingual generation, cross-lingual tasks

### 13. **Flan-T5**
- **Organization**: Google
- **Repository**: [google-research/FLAN](https://github.com/google-research/FLAN)
- **Language**: TensorFlow
- **Details**:
  - T5 fine-tuned with instruction following
  - Base: 220M, Large: 770M, 3B, 11B parameters
  - Improved instruction understanding
- **Use Cases**: Zero-shot tasks, instruction following

### 14. **UniLM (Unified Language Model)**
- **Organization**: Microsoft
- **Repository**: [microsoft/unilm](https://github.com/microsoft/unilm)
- **Language**: PyTorch
- **Details**:
  - Unified encoder, decoder, and encoder-decoder
  - 110M-355M parameters
  - Shared transformer backbone
- **Use Cases**: NLU + NLG, multi-task learning

### 15. **Domain-Specific Decoder Models**

#### CodeGen
- **Purpose**: Code generation from natural language
- **Repository**: [Salesforce/CodeGen](https://github.com/Salesforce/CodeGen)
- **Parameters**: Up to 16.3B
- **Use Cases**: Program synthesis, code completion

#### Medical-T5
- **Purpose**: Medical text generation and summarization
- **Domain**: Healthcare and biomedical
- **Use Cases**: Clinical note generation, medical summarization

#### ViT-GPT2
- **Purpose**: Image captioning (Vision + Decoder)
- **Architecture**: Vision Transformer encoder + GPT-2 decoder
- **Use Cases**: Image description, accessibility

## Implementation Example

### PyTorch Implementation (Simplified)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMultiHeadAttention(nn.Module):
    """Masked self-attention for decoder"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def create_causal_mask(self, seq_len, device):
        """Create lower triangular matrix for causal masking"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(self, x):
        B, L, D = x.shape
        
        Q = self.q_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply causal mask
        mask = self.create_causal_mask(L, x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0)  # Handle NaN from -inf
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L, D)
        output = self.out_proj(attn_output)
        
        return output

class CrossMultiHeadAttention(nn.Module):
    """Cross-attention: attend to encoder outputs"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)  # From decoder
        self.k_proj = nn.Linear(d_model, d_model)  # From encoder
        self.v_proj = nn.Linear(d_model, d_model)  # From encoder
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, query, key_value):
        """
        query: [B, L_target, d_model] - decoder hidden states
        key_value: [B, L_source, d_model] - encoder hidden states
        """
        B, L_q, D = query.shape
        L_k = key_value.shape[1]
        
        Q = self.q_proj(query).view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key_value).view(B, L_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(key_value).view(B, L_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention (no masking)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L_q, D)
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

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attn = MaskedMultiHeadAttention(d_model, num_heads)
        self.cross_attn = CrossMultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output):
        # Masked self-attention
        masked_attn_output = self.masked_attn(self.norm1(x))
        x = x + self.dropout(masked_attn_output)
        
        # Cross-attention with encoder
        cross_attn_output = self.cross_attn(self.norm2(x), encoder_output)
        x = x + self.dropout(cross_attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(self.norm3(x))
        x = x + self.dropout(ffn_output)
        
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def create_positional_encoding(self, max_seq_len, d_model):
        pos = torch.arange(max_seq_len).unsqueeze(1)
        i = torch.arange(d_model)[::2].unsqueeze(0)
        
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos / (10000 ** (2 * i / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (2 * i / d_model)))
        
        return pe.unsqueeze(0)
    
    def forward(self, target_ids, encoder_output):
        """
        target_ids: [B, L_target]
        encoder_output: [B, L_source, d_model]
        """
        seq_len = target_ids.shape[1]
        d_model = encoder_output.shape[-1]
        
        # Embedding and positional encoding
        x = self.embedding(target_ids)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output)
        
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits

# Usage Example
encoder_output = torch.randn(2, 20, 512)  # [batch, source_len, d_model]
target_ids = torch.randint(0, 10000, (2, 15))  # [batch, target_len]

decoder = Decoder(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    vocab_size=10000,
    max_seq_len=512
)

logits = decoder(target_ids, encoder_output)
print(logits.shape)  # [2, 15, 10000] - batch, seq_len, vocab
```

### Using Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load BART (encoder-decoder model)
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Source and target
source_text = "Machine learning is a subset of artificial intelligence."
target_text = "ML is AI subset."

# Tokenize
source_ids = tokenizer(source_text, return_tensors="pt")
target_ids = tokenizer(target_text, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(
        input_ids=source_ids["input_ids"],
        decoder_input_ids=target_ids["input_ids"]
    )

logits = outputs.logits
print(logits.shape)  # [1, target_len, vocab_size]

# Generation
generated_ids = model.generate(source_ids["input_ids"], max_length=50)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
```

### Inference with Beam Search

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """
Machine learning is a subset of artificial intelligence. 
It focuses on the development of algorithms and statistical models 
that enable computers to learn and make predictions without being 
explicitly programmed for every scenario.
"""

# Tokenize
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

# Beam search generation
generated_ids = model.generate(
    inputs["input_ids"],
    max_length=150,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=2
)

summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Summary: {summary}")
```

## Encoder-Decoder Integration

### Architecture Connection

```
Input Sequence
    ↓
[Encoder: 6 layers]
    ↓
Encoder Output [B, L_source, d_model]
    ↓
    ├──────────────────────┐
    ↓                      ↓
[Decoder Layer i]  [Cross-Attention]
    ↓                      ↑
    ├──────────────────────┘
    ↓
Target: [B, L_target, vocab_size]
```

### Information Flow

1. **Encoding Phase**:
   - Encoder processes entire input sequence in parallel
   - Generates bidirectional context representations
   - Output shape: `[batch, source_length, d_model]`

2. **Decoding Phase**:
   - Decoder generates output sequentially
   - Uses masked self-attention on previous tokens
   - Uses cross-attention to retrieve encoder information
   - Output shape at each step: `[batch, 1, vocab_size]` (during inference)

3. **Cross-Attention Bridge**:
   - Decoder queries attention from encoder outputs
   - Enables information transfer from encoder to decoder
   - Multiple heads capture different alignments
   - Full access (no masking) to encoder context

### Training vs Inference

**Training**:
- Encoder: Process full input `[B, L_source, d_model]`
- Decoder: Process full target with teacher forcing `[B, L_target, d_model]`
- Causal mask prevents future peeking
- Parallel processing for efficiency
- Compute full cross-attention matrix

**Inference**:
- Encoder: Process full input once, cache output
- Decoder: Generate one token at a time
- Feed previous tokens to decoder
- Use greedy or beam search
- Cached cross-attention values

## Key Insights

1. **Autoregressive Generation**: Decoder generates sequentially, enabling controllable generation
2. **Causal Masking**: Prevents information leakage while allowing parallel training
3. **Cross-Attention**: Enables sequence-to-sequence modeling and conditioning
4. **Bidirectional Encoder + Autoregressive Decoder**: Best of both worlds
5. **Flexible Architecture**: Can use encoder-decoder for all NLP tasks (T5)
6. **Efficient Inference**: Encoder computed once, decoder iterative
7. **Attention Weights**: Provide interpretability through alignment visualization

## Comparison: Encoder vs Decoder

| Aspect | Encoder | Decoder |
|--------|---------|---------|
| **Processing** | Parallel all positions | Sequential token-by-token |
| **Self-Attention** | Unmasked (bidirectional) | Masked (causal) |
| **Context** | Full sequence context | Only previous tokens |
| **Purpose** | Understanding | Generation |
| **Training Speed** | Fast (parallel) | Slower (sequential) |
| **Inference** | Compute once | Iterative (slower) |
| **Attention Types** | Self-attention only | Self + Cross-attention |

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*
- Lewis, M., et al. (2019). "BART: Denoising Sequence-to-Sequence Pre-training." *arXiv:1910.13461*
- Raffel, C., et al. (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *arXiv:1910.10683*
- Zhang, J., et al. (2019). "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization." *arXiv:1912.08777*
- Xue, L., et al. (2020). "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer." *arXiv:2010.11934*


# Decoder Architecture - Quick Reference Card

## 🔑 Key Formulas

| Name | Formula |
|------|---------|
| **Masked Attention** | $\text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V$ where $M_{ij} = -\infty$ if $j > i$ |
| **Cross-Attention** | $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ (Q from decoder, K,V from encoder) |
| **Multi-Head** | $\text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$ |
| **FFN** | $\text{GELU}(xW_1 + b_1)W_2 + b_2$ |
| **LayerNorm** | $\gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ |
| **Residual** | $x + \text{Sublayer}(x)$ |
| **Output** | $\text{softmax}(\text{Decoder}(Y, Enc) \times W_{vocab}^T)$ |

## 📐 Common Configurations

| Model | Type | Layers | Hidden | Heads | Params |
|-------|------|--------|--------|-------|--------|
| BART-Base | Enc-Dec | 6-6 | 768 | 12 | 139M |
| BART-Large | Enc-Dec | 12-12 | 1024 | 16 | 405M |
| T5-Base | Enc-Dec | 12-12 | 768 | 12 | 220M |
| T5-Large | Enc-Dec | 24-24 | 1024 | 16 | 770M |
| GPT-2 | Decoder | 12 | 768 | 12 | 1.5B |
| GPT-3 | Decoder | 96 | 12288 | 96 | 175B |
| LLaMA-7B | Decoder | 32 | 4096 | 32 | 7B |
| Mistral-7B | Decoder | 32 | 4096 | 32 | 7B |

## 🎯 Complexity Analysis

| Operation | Complexity | Note |
|-----------|-----------|------|
| Masked Self-Attn | $O(L^2 \cdot d)$ | Quadratic in target length |
| Cross-Attn | $O(L_{target} \times L_{source} \cdot d)$ | Connects two sequences |
| FFN | $O(L \cdot d \cdot d_{ff})$ | Linear in sequence length |
| Full Layer | $O(L^2 \cdot d)$ | Dominated by attention |
| Full Decoder | $O(N \cdot L^2 \cdot d)$ | N layers stacked |
| Inference (1 token) | $O(L_{cache} \cdot d)$ | Uses cached encoder output |

## 🔄 Processing Pipeline

```
Source Text
   ↓
[Encoder Processing]
   ↓
Encoder Output [B, L_source, d]  ← CACHED
   ↓
Target Start Token [B, 1]
   ↓
Embedding [B, 1, d]
   ↓
+ Positional Encoding [B, 1, d]
   ↓
Decoder Layer 1: [Masked Self-Attn] → [Cross-Attn with Encoder] → [FFN]
   ↓
...
   ↓
Decoder Layer N
   ↓
Output Projection [B, 1, vocab_size]
   ↓
Softmax → Probability Distribution
   ↓
Sample/Argmax → Next Token
   ↓
Feed Back + Previous Tokens
   ↓
REPEAT until END token
```

## 💾 Typical Sizes

### Encoder-Decoder Models
- **Sequence Length (L)**: 128-512 (standard), up to 2048 (extended)
- **Hidden Dimension (d)**: 768-1024 (base), 1024-2048 (large)
- **Number of Heads (h)**: 8-12
- **FFN Expansion**: 4x hidden (typically 3072)
- **Layers (N)**: 6-24 (typically equal encoder and decoder)
- **Total Params**: 139M-13B

### Decoder-Only Models
- **Hidden Dimension (d)**: 768-12288
- **Number of Heads (h)**: 12-96
- **Intermediate Size**: 3-4x hidden
- **Layers (N)**: 12-96
- **Total Params**: 1.5B-175B+

## 🎓 Use Case Quick Selector

| Task | Best Model | Key Feature |
|------|-----------|------------|
| **Translation** | MarianMT | 1500+ language pairs |
| **Summarization** | Pegasus | Optimized for abstractive |
| **General Seq2Seq** | BART, T5 | Balanced performance |
| **Code Generation** | CodeLlama | Code-specific training |
| **Chat/Dialog** | LLaMA 2, Claude | Instruction-tuned |
| **Long Context** | GPT-4, Claude | 8K-100K+ tokens |
| **Multilingual** | mBART, mT5 | 50-100+ languages |
| **Image Captioning** | ViT-GPT2 | Vision + Language |
| **SQL Generation** | CodeLlama, T5 | Semantic parsing |
| **Fast Inference** | Mistral-7B | Small & efficient |

## 🚀 Generation Methods

| Method | Description | Speed | Quality |
|--------|-------------|-------|---------|
| **Greedy** | Pick max probability at each step | ⚡ Fastest | ⭐ Good |
| **Beam Search** | Keep K best paths, expand each | 🔸 Medium | ⭐⭐⭐ Best |
| **Sampling** | Sample from full distribution | 🔸 Medium | ⭐⭐ Good |
| **Top-K** | Sample from top K probabilities | 🔸 Medium | ⭐⭐ Good |
| **Top-P** | Sample from top-p cumulative probability | 🔸 Medium | ⭐⭐ Good |
| **Temperature** | Scale logits before softmax | ⚡ Fast | Adjustable |

## 🎯 Decoding Hyperparameters

```python
# Generation configuration
max_length = 150              # Maximum output length
num_beams = 4                 # Beam search width
early_stopping = True         # Stop when all beams reach end
temperature = 1.0             # Logit scaling (>1: diversity, <1: confident)
top_p = 0.9                   # Nucleus sampling threshold
top_k = 50                    # Top-K sampling
repetition_penalty = 1.2      # Penalize repeated n-grams
length_penalty = 1.0          # Length normalization
no_repeat_ngram_size = 2      # Block repeating n-grams

# Output
num_return_sequences = 1      # Number of alternatives
output_scores = False         # Return log probabilities
return_dict_in_generate = True # Get attention weights
```

## 🔗 Common Hugging Face Patterns

```python
# Load and generate with BART
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

# Encoding
inputs = tokenizer("Hello world", return_tensors="pt")

# Decoding with beam search
outputs = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_beams=4,
    early_stopping=True
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## ⚡ Inference Optimization

| Technique | Benefit | Trade-off |
|-----------|---------|-----------|
| **KV Caching** | Avoid recomputing keys/values | Memory usage |
| **Quantization** | Smaller model size | Slight quality loss |
| **Distillation** | Smaller, faster model | Accuracy decrease |
| **Pruning** | Remove redundant parameters | Architecture change |
| **Batch Processing** | Process multiple inputs | Latency increase |
| **Flash Attention** | Faster attention computation | Hardware specific |

## 🚨 Common Pitfalls

❌ Forgetting to mask future positions in decoder self-attention
❌ Using encoder output without caching (compute repeatedly)
❌ Not handling [CLS], [SEP], [PAD] tokens correctly
❌ Wrong decoding strategy for the task (greedy vs beam search)
❌ High temperature causing incoherent outputs
❌ Low temperature causing repetitive outputs
❌ Using wrong tokenizer for model
❌ Forgetting to set model.eval() mode for inference
❌ Not using attention_mask for padded sequences
❌ Confusion between input_ids and decoder_input_ids

## 🎓 Training Tips

- **Learning Rate**: 2e-5 to 5e-5 (Adam optimizer)
- **Batch Size**: 8-32 (depends on GPU memory)
- **Epochs**: 3-10 (more for small datasets)
- **Warmup Steps**: 10% of total training steps
- **Weight Decay**: 0.01 (regularization)
- **Dropout**: 0.1 (already in model)
- **Label Smoothing**: 0.1 (for generation tasks)
- **Gradient Accumulation**: Simulate larger batches
- **Mixed Precision**: Use fp16 for memory efficiency
- **Gradient Clipping**: Clip at 1.0 to prevent explosion

## 📊 Performance Benchmarks

### Translation (BLEU Score)
| Model | EN→FR | EN→DE | EN→ZH |
|-------|-------|-------|-------|
| BART | 38.1 | 34.5 | 28.3 |
| mBART | 39.2 | 35.8 | 29.5 |
| MarianMT | 40.1 | 36.9 | 30.2 |

### Summarization (ROUGE-1/ROUGE-L)
| Model | CNN/DM | XSUM |
|-------|--------|------|
| BART | 44.1/41.0 | 45.1/37.1 |
| Pegasus | 44.2/41.5 | 47.6/39.5 |
| T5-Large | 43.5/40.7 | 44.8/37.3 |

### Language Understanding (Perplexity)
| Model | WikiText2 | WikiText103 |
|-------|----------|------------|
| GPT-2 | 18.34 | 24.37 |
| GPT-3 | 12.89 | 15.23 |
| LLaMA-7B | 11.0 | 13.5 |

---

**For detailed information, refer to [Decode.md](Decode.md)**

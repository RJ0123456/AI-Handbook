# Encoder Architecture - Quick Reference Card

## 🔑 Key Formulas

| Name | Formula |
|------|---------|
| **Attention** | $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ |
| **Multi-Head** | $\text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$ |
| **FFN** | $\text{GELU}(xW_1 + b_1)W_2 + b_2$ |
| **LayerNorm** | $\gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ |
| **Residual** | $\text{Output} = x + \text{Sublayer}(x)$ |
| **PosEnc Sine** | $\sin\left(\frac{\text{pos}}{10000^{2i/d_{model}}}\right)$ |
| **PosEnc Cos** | $\cos\left(\frac{\text{pos}}{10000^{2i/d_{model}}}\right)$ |

## 📐 Common Configurations

| Model | Layers | Hidden | Heads | FFN Size | Params |
|-------|--------|--------|-------|----------|--------|
| BERT-Base | 12 | 768 | 12 | 3072 | 110M |
| BERT-Large | 24 | 1024 | 16 | 4096 | 340M |
| DistilBERT | 6 | 768 | 12 | 3072 | 66M |
| ALBERT | 12 | 768 | 12 | 3072 | 110M+ |
| RoBERTa-Base | 12 | 768 | 12 | 3072 | 125M |

## 🎯 Complexity Analysis

| Operation | Complexity | Note |
|-----------|-----------|------|
| Single Attention | $O(L^2 \cdot d)$ | Quadratic in sequence length |
| Multi-Head Attn | $O(h \cdot L^2 \cdot d/h) = O(L^2 \cdot d)$ | Same as single with parallelization |
| FFN | $O(L \cdot d \cdot d_{ff})$ | Linear in sequence length |
| Full Layer | $O(L^2 \cdot d + L \cdot d \cdot d_{ff})$ | Dominated by attention if $d_{ff} \sim 4d$ |

## 🔄 Processing Pipeline

```
Raw Text
   ↓
Tokenization
   ↓
Token IDs [B, L]
   ↓
Embedding [B, L, d]
   ↓
+ Positional Encoding [B, L, d]
   ↓
Encoder Layer 1 [B, L, d]
   ↓
...
   ↓
Encoder Layer N [B, L, d]
   ↓
LayerNorm [B, L, d]
   ↓
Contextual Embeddings ✓
```

## 💾 Typical Sizes

- **Sequence Length (L)**: 128-512 (BERT), up to 2048 (XLNet)
- **Hidden Dimension (d)**: 768-1024
- **Number of Heads (h)**: 8-12
- **FFN Expansion**: 4x hidden (typically 3072)
- **Layers (N)**: 6-24

## 🎓 Use Case Quick Selector

| Task | Best Model | Characteristics |
|------|-----------|-----------------|
| **General Purpose** | RoBERTa, BERT | Balanced performance |
| **Fast Inference** | DistilBERT | 60% faster, smaller |
| **Lightweight** | ALBERT | Parameter sharing |
| **Biomedical** | BioBERT | Domain-specific vocab |
| **Code** | CodeBERT | Programming language tokens |
| **Legal** | LegalBERT | Legal terminology |
| **Multilingual** | XLM-R | 100+ languages |
| **Semantic Search** | Sentence-BERT | Optimized embeddings |

## 🚀 Implementation Checklist

- [ ] Choose architecture (base, large, efficient)
- [ ] Select pre-trained checkpoint
- [ ] Load tokenizer and model
- [ ] Prepare input data (tokenize, truncate)
- [ ] Forward pass through encoder
- [ ] Extract embeddings (`[CLS]` token or mean pooling)
- [ ] Adapt for downstream task (add classifier head)
- [ ] Fine-tune on task-specific data
- [ ] Evaluate on validation set
- [ ] Deploy with batch inference

## 📊 Performance Benchmarks (GLUE Score)

| Model | Score | Size | Inference Time |
|-------|-------|------|-----------------|
| BERT-Base | 78.3 | 110M | 1x |
| RoBERTa-Base | 82.1 | 125M | 1x |
| ALBERT-Base | 82.3 | 12M | 0.9x |
| DistilBERT | 77.0 | 66M | 0.6x |
| DeBERTa-Base | 86.8 | 140M | 1.3x |

## 🔗 Common Hugging Face Commands

```python
# Load encoder model
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode text
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state

# Extract [CLS] token embedding
cls_embedding = embeddings[:, 0, :]  # [batch_size, 768]

# Mean pooling across tokens
mean_embedding = outputs.last_hidden_state.mean(dim=1)  # [batch_size, 768]
```

## 📚 Hyperparameter Tuning Tips

- **Learning Rate**: 2e-5 to 5e-5 (Adam optimizer)
- **Batch Size**: 16-32 (GPU dependent)
- **Epochs**: 3-5 for fine-tuning
- **Warmup Steps**: 10% of total steps
- **Weight Decay**: 0.01
- **Dropout**: 0.1 (already in model)
- **Max Sequence Length**: 128-512 (truncate/pad)

## ⚠️ Common Pitfalls

❌ Using wrong tokenizer (use matching model's tokenizer)
❌ Not handling [CLS], [SEP], [PAD] tokens correctly
❌ Training with too high learning rate (exploding gradients)
❌ Not padding sequences to same length
❌ Using raw embeddings instead of fine-tuning for new tasks
❌ Forgetting to set model to eval mode for inference
❌ Confusion between hidden states and embeddings dimensions

---

**For detailed information, refer to [Encoder.md](Encoder.md)**

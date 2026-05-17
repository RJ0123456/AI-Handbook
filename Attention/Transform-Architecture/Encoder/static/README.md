# Encoder Architecture - Visual Resources

## Mermaid Diagrams

This directory contains visual representations of the Transformer Encoder Architecture:

### 1. **Encoder Block Structure** (`encoder-block-structure.md`)
Shows the overall flow of input sequence through the encoder stack:
- Input embedding and positional encoding
- Stacking of multiple encoder layers
- Final output generation

### 2. **Multi-Head Attention Mechanism** (`attention-mechanism.md`)
Illustrates how multi-head self-attention works:
- Splitting input into Query, Key, Value
- Parallel attention heads (typically 8 or 12)
- Head concatenation and output projection

### 3. **Encoder Layer Detail** (`encoder-layer-detail.md`)
Deep dive into a single encoder layer:
- Layer normalization before/after sub-layers
- Multi-head self-attention branch
- Feed-forward network branch
- Residual connections

### 4. **Positional Encoding** (`positional-encoding.md`)
Shows how position information is added to embeddings:
- Position index computation
- Sinusoidal encoding for even/odd dimensions
- Combination with token embeddings

### 5. **Encoder Models Comparison** (`encoder-comparison.md`)
Comparison of various encoder-based models:
- BERT-based models (BERT, RoBERTa, DistilBERT, ALBERT)
- Domain-specific models (BioBERT, CodeBERT, LegalBERT)
- Multilingual models (mBERT, XLM-R)

## How to Use

1. Open any `.md` file in VS Code
2. The Mermaid diagram will be rendered automatically
3. Click on the preview pane to see the visual representation
4. Use the diagrams as reference while studying the main [Encoder.md](../Encoder.md) document

## Related Files

- **Parent Document**: [../Encoder.md](../Encoder.md) - Complete Encoder Architecture guide
- **Project Index**: [../../../README.md](../../../README.md) - AI Handbook main index

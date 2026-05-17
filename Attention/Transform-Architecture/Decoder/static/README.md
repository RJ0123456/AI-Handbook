# Decoder Architecture - Visual Resources

## Mermaid Diagrams

This directory contains visual representations of the Transformer Decoder Architecture:

### 1. **Decoder Block Structure** (`decoder-block-structure.md`)
Shows the overall flow of target sequence through the decoder stack:
- Target embedding and positional encoding
- Stacking of multiple decoder layers
- Final output logits generation

### 2. **Masked Self-Attention (Causal)** (`masked-attention.md`)
Illustrates causal masking mechanism:
- Attention allowed to previous positions only
- Future positions masked to -∞
- Lower triangular attention matrix

### 3. **Cross-Attention Mechanism** (`cross-attention.md`)
Shows how decoder attends to encoder:
- Query from decoder
- Key and Value from encoder
- No masking - full encoder access

### 4. **Decoder Layer Detail** (`decoder-layer-detail.md`)
Deep dive into a single decoder layer:
- Masked self-attention branch
- Cross-attention with encoder branch
- Feed-forward network branch
- Three independent residual connections

### 5. **Encoder-Decoder Interaction** (`encoder-decoder-interaction.md`)
Illustrates information flow between components:
- Encoder processes source language
- Decoder processes target language
- Cross-attention bridges encoder and decoder

### 6. **Autoregressive Generation** (`autoregressive-generation.md`)
Shows token-by-token generation process:
- Start token fed to decoder
- Generate one token at a time
- Use previously generated tokens for next step
- Stop at end-of-sequence token

### 7. **Decoder Types Comparison** (`decoder-types-comparison.md`)
Comparison of architecture styles:
- Encoder-Decoder models (BART, T5, mBART)
- Decoder-Only models (GPT-2/3, LLaMA, Mistral)
- Encoder-Only models (BERT, RoBERTa)

## How to Use

1. Open any `.md` file in VS Code
2. The Mermaid diagram will be rendered automatically
3. Click on the preview pane to see the visual representation
4. Use the diagrams as reference while studying the main [Decode.md](../Decode.md) document

## Related Files

- **Parent Document**: [../Decode.md](../Decode.md) - Complete Decoder Architecture guide
- **Project Index**: [../../../README.md](../../../README.md) - AI Handbook main index

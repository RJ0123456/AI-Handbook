#!/usr/bin/env python3
"""
Generate Mermaid diagrams for Transformer Encoder Architecture
These diagrams are rendered in VS Code directly
"""

import os

MERMAID_DIAGRAMS = {
    "encoder-block-structure.md": """# Encoder Block Structure Diagram

```mermaid
graph TD
    A["Input Sequence<br/>[B, L, d_model]"] --> B["Embedding + Positional Encoding"]
    B --> C["Encoder Layer Stack"]
    C --> D["Layer 1"]
    D --> E["Layer 2"]
    E --> F["..."]
    F --> G["Layer N"]
    G --> H["Output<br/>[B, L, d_model]"]
    
    style A fill:#e1f5ff
    style H fill:#c8e6c9
```
""",
    
    "attention-mechanism.md": """# Multi-Head Attention Mechanism

```mermaid
graph LR
    A["Input<br/>[B, L, d]"] --> B["Q = x @ W_q"]
    A --> C["K = x @ W_k"]
    A --> D["V = x @ W_v"]
    
    B --> E["Split into heads<br/>h=8"]
    C --> F["Split into heads<br/>h=8"]
    D --> G["Split into heads<br/>h=8"]
    
    E --> H["head_i<br/>Attention"]
    F --> H
    G --> H
    
    H --> I["Concat heads"]
    I --> J["Output Projection<br/>W_o"]
    J --> K["Output<br/>[B, L, d]"]
    
    style A fill:#e1f5ff
    style K fill:#c8e6c9
    style H fill:#fff9c4
```
""",
    
    "encoder-layer-detail.md": """# Encoder Layer Detail

```mermaid
graph TD
    A["Input x<br/>[B, L, d]"] --> B["LayerNorm"]
    B --> C["Multi-Head<br/>Self-Attention"]
    C --> D["Dropout"]
    D --> E["Residual Add"]
    A --> E
    
    E --> F["x' = x + attn"]
    F --> G["LayerNorm"]
    G --> H["Feed-Forward<br/>Network"]
    H --> I["Dropout"]
    I --> J["Residual Add"]
    F --> J
    
    J --> K["Output<br/>[B, L, d]"]
    
    style A fill:#e1f5ff
    style K fill:#c8e6c9
    style C fill:#fff9c4
    style H fill:#fff9c4
```
""",
    
    "positional-encoding.md": """# Positional Encoding

```mermaid
graph LR
    A["Position Index<br/>pos = 0,1,2,...,L-1"] --> B["Even Dimensions<br/>PE_even = sin"]
    A --> C["Odd Dimensions<br/>PE_odd = cos"]
    
    B --> D["Combine PE dimensions"]
    C --> D
    
    D --> E["PE shape:<br/>[L, d_model]"]
    F["Token Embeddings<br/>[L, d_model]"] --> G["x + PE"]
    E --> G
    
    G --> H["Positional<br/>Embeddings<br/>[L, d_model]"]
    
    style A fill:#e1f5ff
    style H fill:#c8e6c9
    style E fill:#fff9c4
```
""",
    
    "encoder-comparison.md": """# Encoder Types Comparison

```mermaid
graph TD
    A["Encoder Models<br/>Pure Encoder Architecture"]
    
    A --> B["BERT-based"]
    A --> C["Domain-Specific"]
    A --> D["Multilingual"]
    
    B --> B1["BERT"]
    B --> B2["RoBERTa"]
    B --> B3["DistilBERT"]
    B --> B4["ALBERT"]
    
    C --> C1["BioBERT"]
    C --> C2["CodeBERT"]
    C --> C3["LegalBERT"]
    
    D --> D1["mBERT"]
    D --> D2["XLM-R"]
    
    style A fill:#e1f5ff
    style B fill:#fff9c4
    style C fill:#fff9c4
    style D fill:#fff9c4
```
"""
}

def create_diagram_files(output_dir="Attention/Transform-Architecture/Encoder/static"):
    """Create markdown files with Mermaid diagrams"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Mermaid Diagrams for Encoder Architecture")
    print("=" * 60)
    
    for filename, content in MERMAID_DIAGRAMS.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created: {filepath}")
    
    print("=" * 60)
    print(f"Generated {len(MERMAID_DIAGRAMS)} diagram files")
    print("=" * 60)

if __name__ == "__main__":
    create_diagram_files()

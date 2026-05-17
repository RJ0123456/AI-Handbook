#!/usr/bin/env python3
"""
Generate Mermaid diagrams for Transformer Decoder Architecture
"""

import os

MERMAID_DIAGRAMS = {
    "decoder-block-structure.md": """# Decoder Block Structure Diagram

```mermaid
graph TD
    A["Target Sequence<br/>[B, L, d_model]"] --> B["Embedding + Positional Encoding"]
    B --> C["Decoder Layer Stack"]
    C --> D["Layer 1"]
    D --> E["Masked Self-Attention"]
    E --> F["Cross-Attention with Encoder"]
    F --> G["Feed-Forward Network"]
    G --> H["Layer 2"]
    H --> I["..."]
    I --> J["Layer N"]
    J --> K["Output Logits<br/>[B, L, vocab_size]"]
    
    style A fill:#fff9c4
    style K fill:#c8e6c9
    style E fill:#ffccbc
    style F fill:#b3e5fc
```
""",
    
    "masked-attention.md": """# Masked Self-Attention (Causal)

```mermaid
graph LR
    A["Current Position i"] --> B["Can Attend To"]
    B --> C["Position 0"]
    B --> D["Position 1"]
    B --> E["..."]
    B --> F["Position i"]
    
    A --> G["Cannot Attend To"]
    G --> H["Position i+1"]
    G --> I["Position i+2"]
    G --> J["..."]
    G --> K["Position N"]
    
    M["Mask Matrix:<br/>Lower Triangular"] --> L["Attention weights<br/>for future = 0"]
    
    style B fill:#c8e6c9
    style G fill:#ffccbc
    style M fill:#e1f5ff
```
""",
    
    "cross-attention.md": """# Cross-Attention Mechanism

```mermaid
graph TD
    A["Decoder Hidden State<br/>[B, L_target, d]"] --> B["Query Projection"]
    C["Encoder Output<br/>[B, L_source, d]"] --> D["Key Projection"]
    C --> E["Value Projection"]
    
    B --> F["Attention Computation"]
    D --> F
    E --> F
    
    F --> G["No Masking<br/>Full Access"]
    G --> H["Output<br/>[B, L_target, d]"]
    
    style A fill:#fff9c4
    style C fill:#e1f5ff
    style G fill:#c8e6c9
    style H fill:#b3e5fc
```
""",
    
    "decoder-layer-detail.md": """# Decoder Layer Detail

```mermaid
graph TD
    A["Input x<br/>[B, L_target, d]"] --> B["LayerNorm"]
    B --> C["Masked Self-Attention<br/>Only previous tokens"]
    C --> D["Dropout"]
    D --> E["Residual Add"]
    A --> E
    
    E --> F["x' = x + masked_attn"]
    F --> G["LayerNorm"]
    G --> H["Cross-Attention<br/>with Encoder"]
    H --> I["Dropout"]
    I --> J["Residual Add"]
    F --> J
    
    J --> K["x'' = x' + cross_attn"]
    K --> L["LayerNorm"]
    L --> M["Feed-Forward<br/>Network"]
    M --> N["Dropout"]
    N --> O["Residual Add"]
    K --> O
    
    O --> P["Output<br/>[B, L_target, d]"]
    
    style A fill:#fff9c4
    style P fill:#c8e6c9
    style C fill:#ffccbc
    style H fill:#b3e5fc
    style M fill:#f0f4c3
```
""",
    
    "encoder-decoder-interaction.md": """# Encoder-Decoder Interaction

```mermaid
graph LR
    subgraph Encoder["Encoder"]
        A["Input: 'Hello world'<br/>[B, 2, d]"]
        B["Processing"]
        C["Output<br/>[B, 2, d]"]
        A --> B --> C
    end
    
    subgraph Decoder["Decoder"]
        D["Target: '你好'<br/>[B, 2, d]"]
        E["Masked<br/>Self-Attn"]
        F["Cross-Attn<br/>with Encoder"]
        G["FFN"]
        H["Output<br/>[B, 2, vocab]"]
        D --> E --> F --> G --> H
    end
    
    C -->|"Encoder Output"| F
    
    style C fill:#e1f5ff
    style F fill:#b3e5fc
    style H fill:#c8e6c9
```
""",
    
    "autoregressive-generation.md": """# Autoregressive Generation Process

```mermaid
graph TD
    A["Encoder Output<br/>Cached"] 
    
    B["Step 1: Start Token<br/>&lt;s&gt;"]
    B --> C["Decoder<br/>+ Cross-Attention"]
    C --> D["Output: Logits"]
    D --> E["Sample/Argmax<br/>Token: 机"]
    
    F["Step 2: Start + Token 1<br/>&lt;s&gt; 机"]
    E --> F
    F --> G["Decoder<br/>+ Cross-Attention"]
    G --> H["Output: Logits"]
    H --> I["Sample/Argmax<br/>Token: 器"]
    
    J["Step 3: Previous Tokens<br/>&lt;s&gt; 机 器"]
    I --> J
    J --> K["Decoder<br/>+ Cross-Attention"]
    K --> L["Output: Logits"]
    L --> M["Sample/Argmax<br/>Token: &lt;END&gt;"]
    
    A -.-> C
    A -.-> G
    A -.-> K
    
    M --> N["Final Output<br/>机器"]
    
    style A fill:#e1f5ff
    style E fill:#c8e6c9
    style I fill:#c8e6c9
    style M fill:#c8e6c9
    style N fill:#90ee90
```
""",

    "decoder-types-comparison.md": """# Decoder Architecture Types

```mermaid
graph TD
    A["Transformer Architectures"]
    
    A --> B["Encoder-Decoder"]
    A --> C["Decoder-Only"]
    A --> D["Encoder-Only"]
    
    B --> B1["BART"]
    B --> B2["T5"]
    B --> B3["mBART"]
    B --> B4["Pegasus"]
    
    C --> C1["GPT-2/3"]
    C --> C2["LLaMA"]
    C --> C3["Mistral"]
    C --> C4["Claude"]
    
    D --> D1["BERT"]
    D --> D2["RoBERTa"]
    
    style B fill:#b3e5fc
    style C fill:#fff9c4
    style D fill:#e1f5ff
```
"""
}

def create_diagram_files(output_dir="Attention/Transform-Architecture/Decoder/static"):
    """Create markdown files with Mermaid diagrams"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Mermaid Diagrams for Decoder Architecture")
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

# Vision Transformer (ViT)

## Table of Contents
1. [Overview](#overview)
2. [From an Image to Tokens](#from-an-image-to-tokens)
3. [Architecture](#architecture)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Training and Fine-Tuning](#training-and-fine-tuning)
6. [Variants and Related Models](#variants-and-related-models)
7. [Strengths and Limitations](#strengths-and-limitations)
8. [Practical Example](#practical-example)
9. [Key Takeaways](#key-takeaways)
10. [References](#references)

## Overview

A Vision Transformer (ViT) applies the Transformer encoder to an image. Instead
of processing pixels with convolutional filters, it divides the image into
fixed-size patches, embeds each patch as a vector, and treats the resulting
vectors as a sequence of tokens.

The central idea is simple:

> An image can be represented as a sequence of patch tokens, and self-attention
> can learn relationships between any pair of patches.

The original ViT showed that a standard Transformer can perform strongly on
image classification when trained with enough data and compute. Unlike a
convolutional network, the basic ViT does not build in locality or
translation-equivariant filters; it learns visual relationships from its
training data.

### Where ViT is used

- Image classification and feature extraction
- Image retrieval and visual similarity
- Object detection and image segmentation, usually with task-specific heads
- Vision-language systems, where a vision encoder is paired with a text model

ViT refers to the vision encoder architecture, not by itself to a complete
vision-language model.

## From an Image to Tokens

Let an image have height $H$, width $W$, and $C$ channels. Split it into
non-overlapping square patches of side length $P$. If $H$ and $W$ are divisible
by $P$, the number of patches is:

$$N = \frac{H}{P} \cdot \frac{W}{P}$$

Each patch has $P^2C$ scalar values. Flatten it into a vector and map it to the
Transformer hidden dimension $D$ with a learned linear projection:

$$\mathbf{e}_i = \operatorname{flatten}(\text{patch}_i) \mathbf{E} + \mathbf{b}$$

where $\mathbf{E} \in \mathbb{R}^{(P^2C) \times D}$.

### Shape example

For an RGB image of $224 \times 224$, using $16 \times 16$ patches:

- Patches per side: $224 / 16 = 14$
- Number of patches: $N = 14 \times 14 = 196$
- Values in each flattened patch: $16 \times 16 \times 3 = 768$
- Projected patch sequence: $[B, 196, D]$

Here $B$ is the batch size. Implementations commonly prepend one learned
**class token** (CLS), making the sequence length $N+1=197$. The final hidden
state of this token is used as the image representation for classification.

### Patch projection as a convolution

The same patch embedding can be implemented as a convolution with kernel size
$P$ and stride $P$, with $D$ output channels. The convolution produces a
$D \times (H/P) \times (W/P)$ feature map, which is flattened across its spatial
dimensions to form the patch-token sequence. This is an implementation
equivalence; the basic ViT still uses global self-attention after patchifying.

## Architecture

The original classification architecture follows these steps:

1. **Patchify** the image and project every patch into a $D$-dimensional token.
2. **Prepend a CLS token** to the patch tokens.
3. **Add positional embeddings** so the model can distinguish patch locations.
4. **Apply a stack of Transformer encoder blocks.**
5. **Normalize and classify** the final CLS-token representation.

Conceptually:

```text
image [B, C, H, W]
  -> patch embedding [B, N, D]
  -> prepend CLS token and add position embeddings [B, N + 1, D]
  -> Transformer encoder blocks [B, N + 1, D]
  -> final CLS representation [B, D]
  -> classification head [B, number_of_classes]
```

### Positional embeddings

Self-attention alone does not encode the order or 2-D location of tokens. ViT
adds a learned embedding for each sequence position:

$$\mathbf{z}_0 = [\mathbf{x}_{\mathrm{cls}};\mathbf{e}_1;\dots;\mathbf{e}_N]
                 + \mathbf{E}_{\mathrm{pos}}$$

The original model uses learned 1-D positional embeddings for the flattened
patch sequence. The patch sequence itself is arranged in a consistent
row-by-row order. Other vision Transformers use 2-D relative position
information or other position schemes.

When fine-tuning at a different image resolution, the number of patch
positions changes. A common approach is to interpolate the pretrained
positional embeddings on their 2-D patch grid; the CLS position is handled
separately.

### Transformer encoder block

The encoder uses multi-head self-attention (MSA) and a position-wise
feed-forward network (MLP), each with a residual connection. In the
pre-normalization layout used by ViT:

$$\mathbf{z}'_\ell =
  \mathbf{z}_{\ell-1} +
  \operatorname{MSA}(\operatorname{LN}(\mathbf{z}_{\ell-1}))$$

$$\mathbf{z}_\ell =
  \mathbf{z}'_\ell +
  \operatorname{MLP}(\operatorname{LN}(\mathbf{z}'_\ell))$$

The MLP is applied independently to every token, using the same weights:

$$\operatorname{MLP}(\mathbf{x}) =
  \operatorname{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)
  \mathbf{W}_2 + \mathbf{b}_2$$

Its hidden dimension is often larger than $D$ (for example, approximately
$4D$ in a conventional Transformer block).

### Multi-head self-attention

For each head, queries, keys, and values are learned projections of the token
sequence:

$$\operatorname{Attention}(Q,K,V) =
  \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The attention matrix has shape $[B, h, N', N']$, where $N'=N+1$ includes the
CLS token and $h$ is the number of heads. Thus, every patch can exchange
information directly with every other patch in one attention layer. Multiple
heads learn different relationships in parallel.

### Classification head

After the final encoder block and normalization, let $\mathbf{h}_{\mathrm{cls}}$
be the CLS-token output. A linear head maps it to class logits:

$$\mathbf{y} = \mathbf{h}_{\mathrm{cls}}\mathbf{W}_{\mathrm{head}}
               + \mathbf{b}_{\mathrm{head}}$$

For $K$ classes, the logits have shape $[B,K]$. Softmax is applied when class
probabilities are needed; cross-entropy loss is commonly used during
supervised classification training.

## Mathematical Formulation

For an image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$, let
$\mathbf{x}_p^i \in \mathbb{R}^{P^2C}$ denote its $i$-th flattened patch.
Including a learned class embedding, the input to the encoder is:

$$\mathbf{z}_0 =
  [\mathbf{x}_{\mathrm{cls}};\,
   \mathbf{x}_p^1\mathbf{E};\,
   \mathbf{x}_p^2\mathbf{E};\,
   \dots;\,
   \mathbf{x}_p^N\mathbf{E}]
  + \mathbf{E}_{\mathrm{pos}}$$

where $\mathbf{E} \in \mathbb{R}^{(P^2C)\times D}$ and
$\mathbf{E}_{\mathrm{pos}} \in \mathbb{R}^{(N+1)\times D}$ are learned
parameters. The encoder applies $L$ blocks:

$$\mathbf{z}_L =
  \operatorname{TransformerEncoder}(\mathbf{z}_0)$$

The classifier predicts from the CLS token:

$$\mathbf{y} = \operatorname{Head}(\operatorname{LN}(\mathbf{z}_L^0))$$

where $\mathbf{z}_L^0$ is the output at sequence position zero. Some ViT
implementations use the mean of patch-token outputs or another pooling method
instead of a CLS token.

### Compute and memory

For sequence length $N'$, hidden dimension $D$, and $L$ layers, the attention
score calculation has a term proportional to $L(N')^2D$. The complete block
also includes projections and MLP computation, which scale approximately
linearly with sequence length for fixed dimensions. Attention's quadratic
dependence on the number of patches can become expensive at high image
resolutions.

Because $N \propto HW/P^2$, reducing patch size increases sequence length
rapidly. For example, halving $P$ approximately quadruples $N$, and therefore
can increase the attention-matrix size by roughly sixteen times.

## Training and Fine-Tuning

### Supervised training

With labeled images, a classification ViT is trained end-to-end using the
classification loss, commonly cross-entropy. The model learns patch features,
attention patterns, and the classification head jointly.

The original ViT work found that large-scale pretraining was important: with
less training data, a plain ViT can be more prone to overfitting than a
convolutional model. Data augmentation, regularization, suitable optimization,
and pretrained weights are common parts of practical training.

### Transfer learning

A common workflow is:

1. Load a ViT pretrained on a large image dataset.
2. Replace or adapt its classification head for the target labels.
3. Fine-tune the model on the target dataset, optionally freezing some layers
   when data or compute is limited.
4. Evaluate on data that was not used for training or model selection.

For a different input resolution, use the model or library's supported
positional-embedding resizing procedure rather than silently truncating or
reshaping learned positions.

### Self-supervised pretraining

ViTs are also pretrained without human class labels. Masked Autoencoders (MAE),
for example, train an encoder-decoder system to reconstruct masked image
patches from visible patches. Self-supervised pretraining can make use of large
unlabeled image collections, after which the encoder can be fine-tuned for
downstream tasks.

## Variants and Related Models

| Model or family | Main idea | Typical use |
|---|---|---|
| **ViT** | Global self-attention over a sequence of image patches | Classification and general visual features |
| **DeiT** | Training and data-efficient techniques for ViT; includes a distillation-token variant | ViT when training data or compute is more limited |
| **MAE** | Masked image-patch reconstruction for self-supervised pretraining | Learning transferable visual representations |
| **Swin Transformer** | Local window attention with shifted windows and hierarchical features | Detection, segmentation, and multiscale vision tasks |
| **Hybrid ViT** | Uses convolutional features as the Transformer input instead of raw image patches | Combining convolutional inductive bias with attention |

These models are related but not interchangeable: for example, Swin's local,
hierarchical attention is different from the original ViT's global attention
over a flat patch sequence.

## Strengths and Limitations

### Strengths

- **Global context:** Any patch can attend to any other patch in one layer.
- **Scalable pretraining:** The architecture can benefit substantially from
  large datasets and compute.
- **Transferable features:** Pretrained representations can be adapted to many
  vision tasks.
- **Simple sequence-based design:** Image processing can reuse Transformer
  components and training infrastructure.

### Limitations

- **Data hungry:** A plain ViT may need more data or stronger regularization
  than a comparable CNN to generalize well.
- **High-resolution cost:** Global attention scales quadratically with the
  number of patch tokens.
- **Weak built-in locality:** Unlike CNNs, the basic ViT does not inherently
  prioritize nearby pixels or translation equivariance.
- **Patch-size trade-off:** Smaller patches preserve finer detail but sharply
  increase sequence length and compute.
- **Resolution handling:** Learned positional embeddings must be adapted
  carefully when the patch grid changes.

## Practical Example

The following uses a pretrained Hugging Face image-classification checkpoint.
The image processor applies the model's expected resizing, normalization, and
formatting; use the processor associated with the checkpoint rather than
assuming all ViTs share the same input preprocessing.

```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

checkpoint = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForImageClassification.from_pretrained(checkpoint)

image = Image.open("image.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

model.eval()
with torch.inference_mode():
    logits = model(**inputs).logits

predicted_id = logits.argmax(dim=-1).item()
print(model.config.id2label[predicted_id])
```

The example assumes the checkpoint and its dependencies have been downloaded
and installed. To fine-tune on a custom dataset, load a checkpoint with a
classification head configured for the target number of labels and train it
with labeled examples.

## Key Takeaways

- ViT converts an image into patch tokens, then processes those tokens with a
  Transformer encoder.
- Patch projection creates token features; positional embeddings supply
  information about where patches came from.
- The CLS token is a learned summary position often used for image
  classification, but other pooling strategies are also used.
- Global self-attention captures long-range relationships but its memory and
  compute grow quadratically with patch count.
- Pretrained ViTs are commonly fine-tuned; the best architecture depends on
  data scale, resolution, task, and compute budget.

## References

- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), 2020.
- Touvron et al., [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877), 2021.
- He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), 2021.
- Liu et al., [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030), 2021.
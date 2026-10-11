# mask = torch.triu(torch.ones(x.size(0), x.size(0), device=x.device), diagonal=1).bool() 这个mask是用什么值填充的，通常mask用什么值填充，为什么？

这里这个 `mask` 很关键。你代码里的：

```python
mask = torch.triu(
    torch.ones(x.size(0), x.size(0), device=x.device),
    diagonal=1
).bool()
```

实际上**不是用 0 和 1 作为最终的 mask 值**，而是先生成 `0/1` 矩阵，最后通过 `.bool()` 变成 **`False/True`**。你上传的代码中就是这样使用的。transformer_decode

---

## 1. 先一步一步看这行代码

假设：

```python
x.size(0) = 5
```

那么：

```python
torch.ones(5, 5)
```

得到：

```text
1 1 1 1 1
1 1 1 1 1
1 1 1 1 1
1 1 1 1 1
1 1 1 1 1
```

然后：

```python
torch.triu(..., diagonal=1)
```

取**主对角线以上**的部分：

```text
0 1 1 1 1
0 0 1 1 1
0 0 0 1 1
0 0 0 0 1
0 0 0 0 0
```

最后：

```python
.bool()
```

于是变成：

```text
False True  True  True  True
False False True  True  True
False False False True  True
False False False False True
False False False False False
```

所以这里的 mask 实际上是：

> **True = 这个位置被屏蔽（不能注意）**  
> **False = 这个位置允许注意**

---

## 2. 为什么上三角是 True？

假设序列：

```text
位置：   0    1    2    3    4
token:  I   love  AI   very  much
```

在 Decoder 中，位置 2 的 `AI` **不能看到未来的 token**：

```text
I love AI very much
0   1   2    3    4
        ↑
```

位置 2 可以看：

```text
0  1  2
```

但不能看：

```text
3  4
```

所以 attention mask：

```text
          Key
        0  1  2  3  4
Query 0  F  T  T  T  T
      1  F  F  T  T  T
      2  F  F  F  T  T
      3  F  F  F  F  T
      4  F  F  F  F  F
```

这正好就是一个**上三角矩阵**。

---

## 3. 但是一个非常重要的问题：True 到底怎么“屏蔽”？

你可能会产生一个疑问：

> `True` 又没有把 attention score 变成一个很小的数，它怎么就“屏蔽”了？

答案是：

**`mask` 本身只是告诉 `MultiheadAttention` 哪些位置不能参与 attention。**

你这里：

```python
self.self_attn(
    x, x, x,
    attn_mask=mask
)
```

PyTorch 的 `MultiheadAttention` 会根据这个 `bool mask` 处理 attention score。

概念上可以理解成：

```python
scores = Q @ K.T / sqrt(d_k)

scores[mask] = -∞

attn = softmax(scores)
```

也就是说：

```text
True
 ↓
屏蔽
 ↓
score → -∞
 ↓
softmax(-∞) → 0
 ↓
这个位置的 attention weight = 0
```

---

## 4. 为什么通常填 `-∞`？

这就和你上一问的：

```python
attn = F.softmax(scores, dim=-1)
```

直接联系起来了。

假设：

```text
scores = [2.0, 1.0, 3.0]
```

softmax：

```text
softmax(scores)
≈ [0.245, 0.090, 0.665]
```

三个位置都有概率。

现在假设第三个位置不能看：

```text
scores = [2.0, 1.0, -∞]
```

那么：

```text
e^2     ≈ 7.39
e^1     ≈ 2.72
e^-∞   = 0
```

所以：

```text
softmax
=
[7.39, 2.72, 0]
----------------
  10.11
```

得到：

```text
[0.731, 0.269, 0]
```

注意：

> 被 mask 的位置经过 softmax 后，权重严格变成 0。

因此：

```text
attention = 0.731 × V₀
          + 0.269 × V₁
          + 0 × V₂
```

所以 `V₂` 完全不会影响当前 token。

---

## 5. 为什么不是填 0？

这是一个非常重要的区别。

很多刚开始学习 Transformer 的人会想：

> “既然不想让它参与，那我把 score 填成 0 不就行了吗？”

**不行。**

例如：

```text
scores = [2, 1, 0]
```

softmax 后：

```text
[0.665, 0.245, 0.090]
```

你看：

```text
score = 0
```

并没有被屏蔽。

它仍然获得了：

```text
9%
```

左右的 attention。

所以：

```text
mask → 0
```

**不是正确的 attention masking 方法。**

---

## 6. 为什么也不能简单填一个很大的负数？

理论上可以。

例如：

```text
-100
```

那么：

```text
e^-100 ≈ 0
```

实际计算中基本就是 0。

因此你经常会看到：

```python
scores = scores.masked_fill(mask, -1e9)
```

或者：

```python
scores = scores.masked_fill(mask, float('-inf'))
```

两者思想一样：

```text
允许的位置：保留原来的 score

禁止的位置：变成极小值
```

例如：

```text
原始 scores：

  2.1   0.7   1.5   3.2
  1.2   2.3   0.8   1.9
  0.5   1.1   2.7   0.9
  1.4   0.6   1.8   2.2
```

Causal mask：

```text
F T T T
F F T T
F F F T
F F F F
```

mask 后：

```text
  2.1  -∞   -∞   -∞
  1.2   2.3 -∞   -∞
  0.5   1.1  2.7 -∞
  1.4   0.6  1.8  2.2
```

然后：

```python
softmax(scores)
```

就会得到：

```text
1.0   0     0     0

0.25  0.75  0     0

...   ...   ...   0

...   ...   ...   ...
```

---

## 7. 所以常见的 Mask 有两种“表现形式”

这里特别容易混淆。

### 第一种：Boolean mask

也就是你代码中的：

```python
mask = ...
.bool()
```

例如：

```text
False True  True
False False True
False False False
```

通常约定：

```text
True  = mask掉
False = 不mask
```

然后由框架内部完成：

```text
True → -∞
```

---

### 第二种：Additive mask

直接把 mask 加到 attention score 上。

例如：

```text
0    -∞    -∞
0     0    -∞
0     0     0
```

然后：

```python
scores = scores + mask
```

结果：

```text
允许位置：
score + 0 = score

禁止位置：
score + (-∞) = -∞
```

这也是 Transformer 中非常常见的写法。

---

## 8. 为什么 `-∞` 是最漂亮的数学设计？

因为它和 softmax 完美配合：

$$
\operatorname{softmax}(x_i)
=
\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

对于被屏蔽的位置：

$$
x_i=-\infty
$$

那么：

$$
e^{-\infty}=0
$$

所以：

$$
\operatorname{softmax}(-\infty)=0
$$

于是：

```text
mask
 ↓
score = -∞
 ↓
exp(-∞) = 0
 ↓
attention weight = 0
 ↓
这个 token 不参与信息聚合
```

这就是 **Attention Mask 最核心的数学机制**。

---

## 9. 回到你这段代码

你代码：

```python
mask = torch.triu(
    torch.ones(x.size(0), x.size(0), device=x.device),
    diagonal=1
).bool()

attn_output, _ = self.self_attn(
    x, x, x,
    attn_mask=mask
)
```

transformer_decode

可以把它**脑补成下面这个过程**：

```text
              QKᵀ
               ↓
       ┌─────────────────┐
       │ attention score │
       └─────────────────┘
               ↓
          应用 mask
               ↓
       ┌─────────────────┐
       │ 未来位置 → -∞   │
       └─────────────────┘
               ↓
            softmax
               ↓
       未来位置权重 = 0
               ↓
             × V
               ↓
       Attention Output
```

因此，你现在可以记住一句非常重要的话：

> **Attention mask 通常不是“把数据填成什么值”，而是“把不允许关注的位置对应的 attention score 变成一个极小值（通常概念上是 −∞），让 softmax 后它的权重变成 0”。**

而你这段 PyTorch 代码采用的是更方便的 **Boolean mask：`True = 屏蔽，False = 保留`**。

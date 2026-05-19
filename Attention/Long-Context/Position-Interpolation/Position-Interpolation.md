# Position Interpolation for RoPE Long Context

> How to extend RoPE-based Transformers beyond training context length while maintaining stability and performance.

---

## 一、问题：RoPE 为什么在长序列上失效

### 1.1 RoPE 的核心限制

RoPE 在位置编码时，使用：

$$
\theta_i = 10000^{-2i/d}
$$

每个频率维度在第 $p$ 个位置时的旋转角为：

$$
\theta_i \cdot p
$$

**关键问题**：

如果训练时最大位置是 $L_{train}$，而推理时需要处理 $L_{test} > L_{train}$，则：

- 新的角度 $\theta_i \cdot L_{test}$ 远超训练分布
- 模型从未见过这些角度组合
- 注意力分数完全错乱

### 1.2 具体症状

在长序列上：

* Attention 权重分布崩溃
* Token 之间无法正确定位
* 模型性能大幅下降
* 有时甚至无法完成生成

**例如**：
- 训练：4096 tokens
- 推理：32768 tokens
- 结果：模型表现大幅下滑甚至失败

---

## 二、位置插值（Position Interpolation）的核心思想

### 2.1 基本策略

而不是让位置 $p$ 直接映射到 $\theta_i \cdot p$，我们采用：

$$
\theta_i \cdot \frac{p}{s}
$$

其中 $s$ 是**缩放因子**。

**直观理解**：

> 把新位置"压缩"回训练分布内的范围。

### 2.2 数学形式

原始 RoPE：

$$
q_p = q \, e^{ip\theta_i}
$$

位置插值后：

$$
q'_p = q \, e^{i(p/s)\theta_i}
$$

等价于降低频率：

$$
\theta'_i = \frac{\theta_i}{s}
$$

### 2.3 缩放因子的选择

最直观的方法：

$$
s = \frac{L_{test}}{L_{train}}
$$

**例如**：
- 训练长度：4096
- 测试长度：32768
- 缩放因子：$s = 32768 / 4096 = 8$

这样所有位置都被映射到 $[0, 4096]$ 内的有效范围。

---

## 三、Position Interpolation 的几何直觉

### 3.1 复平面视角

在复平面中，RoPE 让每个 token 在不同维度上按频率旋转。

**原始情况**：
- 位置 $p$ 时，角度为 $\theta_i \cdot p$
- 位置 $L_{train}$ 时，角度已经很大
- 位置 $L_{test} > L_{train}$ 时，角度超出训练范围→ **陌生区域**

**位置插值后**：
- 位置 $p$ 时，角度为 $\theta_i \cdot (p/s)$
- 最大角度被限制在 $\theta_i \cdot L_{train}/s = \theta_i \cdot L_{train} / s$
- 如果 $s = L_{test}/L_{train}$，最大角度 $= \theta_i \cdot L_{train}$ → **训练分布内**

### 3.2 Fourier 频率视角

位置插值等价于：

$$
e^{i\theta_i p} \to e^{i(\theta_i/s)p}
$$

即降低所有频率分量，使得高频分量不再快速增长。

**效果**：
- 高频："局部关注"仍然保留（相位差仍在）
- 低频："全局定位"的周期被拉长
- 结果：模型可以在更长序列上工作，但本质仍是 RoPE

---

## 四、Position Interpolation 的实现

### 4.1 最简单的形式

在计算位置编码时，只需修改位置值：

```python
# 原始
pos = position

# 位置插值
s = max_seq_len / training_seq_len
pos = position / s
```

### 4.2 融入 RoPE 计算

在注意力中应用旋转时：

```python
def apply_rotary_emb(x, positions, freq_base=10000, seq_len_scale=1.0):
    dim = x.shape[-1]
    inv_freq = 1.0 / (freq_base ** (torch.arange(0, dim, 2).float() / dim))
    
    # 关键修改：位置缩放
    t = positions / seq_len_scale
    
    freqs = torch.einsum("..., f -> ... f", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    
    # 应用旋转
    return (x * emb.cos()) + (rotate_half(x) * emb.sin())
```

### 4.3 缩放因子计算

```python
# 方法1：固定最大长度
max_training_length = 4096
max_eval_length = 32768
scale = max_eval_length / max_training_length

# 方法2：动态
scale = current_seq_len / training_seq_len
```

---

## 五、Position Interpolation 为什么有效

### 5.1 保持相对位置信息

即使 $p \to p/s$，相对位置差仍然保留：

$$
e^{i\theta_i p/s} \cdot e^{-i\theta_i q/s} = e^{i\theta_i(p-q)/s}
$$

相位差 $(p-q)/s$ 仍然反映相对距离。

**关键**：Attention 本质上只需要相对位置，不需要绝对位置。

### 5.2 避免 Phase Aliasing

原始问题：

$$
e^{i\theta_i p} = e^{i(\theta_i p + 2k\pi)}
$$

不同的 $p$ 可能产生同一相位（周期性混叠）。

位置插值后：

$$
e^{i\theta_i p/s}
$$

通过降低频率，相位增长变慢，**延缓** phase aliasing 出现。

### 5.3 平滑的分布外推

虽然 $p/s$ 的值超出训练范围，但：
- 角度增长是 **连续** 和 **平滑** 的
- 不是陡然跳跃
- 模型的参数能够 **泛化** 到新角度区间

---

## 六、Position Interpolation 的局限与改进

### 6.1 性能下降

虽然 PI 延长了上下文长度，但通常会牺牲性能：

* 注意力精度下降
* 长距离依赖能力减弱
* 困惑度（perplexity）增加 5%~15%

**原因**：频率被均匀缩放，高频分量损失了。

### 6.2 改进方向

#### 6.2.1 YaRN（Yet another RoPE extensioN）

不均匀缩放不同频率：

$$
\theta'_i = 
\begin{cases}
\theta_i & \text{if } i < i_{crit} \\
\theta_i / s & \text{if } i \geq i_{crit}
\end{cases}
$$

**思想**：保留高频（局部信息），只缩放低频（全局定位）。

**效果**：性能损失大幅降低。

#### 6.2.2 NTK（Neural Tangent Kernel）

基于 NTK 理论的动态缩放：

$$
\theta'_i = 
\begin{cases}
\theta_i & \text{if } i < i_{crit} \\
\theta_i \cdot s^{(d-1-2i)/(d-1-2i_{crit})} & \text{otherwise}
\end{cases}
$$

**特点**：平滑的频率调整，避免剧烈变化。

#### 6.2.3 其他方向

* **ALiBi**：完全放弃绝对位置，用相对位置偏置
* **Rotary + Interpolation**：先插值再应用额外位置偏置
* **Dynamic Position Scaling**：根据当前序列长度动态调整

---

## 七、Position Interpolation vs 其他长上下文方法

| 方法 | 原理 | 长度扩展 | 性能 | 计算代价 |
| ---- | ---- | ------ | ---- | ------ |
| PI（Position Interpolation） | 缩放位置坐标 | 4K→32K | 中等损失 | 无 |
| YaRN | 非均匀频率缩放 | 4K→100K+ | 低损失 | 无 |
| NTK | 频率微调 | 4K→32K+ | 低损失 | 无 |
| ALiBi | 相对位置偏置 | 无限制 | 较好 | 低 |
| Flash-Attention + KV Cache 压缩 | 降低计算复杂度 | 4K→128K+ | 变化 | 中等 |
| Sparse Attention | 稀疏模式 | 无限制 | 中等 | 中等 |

---

## 八、实战应用：如何在模型上使用 PI

### 8.1 在推理时应用

```python
# Llama、Mistral 等开源模型通常在 model.py 中

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x, seq_len=None, position_ids=None):
        seq_len = seq_len or x.shape[1]
        
        # PI: 添加缩放因子
        rope_scale = seq_len / self.max_position_embeddings
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # 应用缩放
        position_ids = position_ids / rope_scale
        
        # 计算频率
        freqs = torch.einsum("ij,k->ik", position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        return emb
```

### 8.2 在 HuggingFace 中调整

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# 调整绳索缩放
rope_scaling = {
    "type": "linear",
    "factor": 8.0  # 4K -> 32K
}

# 在模型配置中应用
model.config.rope_scaling = rope_scaling
```

### 8.3 性能评估

```python
# 测试困惑度
def evaluate_perplexity(model, dataset, max_length):
    total_loss = 0
    total_tokens = 0
    
    for batch in dataset:
        with torch.no_grad():
            outputs = model(batch, labels=batch)
            total_loss += outputs.loss.item() * batch.shape[1]
            total_tokens += batch.shape[1]
    
    ppl = torch.exp(torch.tensor(total_loss / total_tokens))
    return ppl
```

---

## 九、Position Interpolation 的实验结果

### 9.1 LLaMA 扩展例子

原始 LLaMA-7B：4096 token 上下文

使用 PI 扩展到 32768 token：

| 长度 | 困惑度增长 | 长距离任务准确率 |
| ---- | -------- | ----------- |
| 4K（训练） | 1.0x | 95% |
| 8K（2x）  | 1.05x | 92% |
| 16K（4x） | 1.12x | 85% |
| 32K（8x） | 1.25x | 72% |
| 64K（16x） | 1.45x | 45% |

使用 YaRN 改进：

| 长度 | 困惑度增长 | 长距离任务准确率 |
| ---- | -------- | ----------- |
| 4K（训练） | 1.0x | 95% |
| 32K（8x） | 1.08x | 88% |
| 64K（16x） | 1.15x | 82% |

### 9.2 关键观察

* PI 可以安全地扩展 **2-4 倍**
* 超过 **8 倍** 扩展时，性能下降明显
* 结合微调或 YaRN，可以达到 **16+ 倍**
* 对于某些任务（如搜索、检索），甚至 **32 倍** 也可接受

---

## 十、Position Interpolation 的数学直观

### 10.1 频率降低的含义

位置插值 $p \to p/s$ 等价于：

$$
\theta_i^{new} = \frac{\theta_i}{s}
$$

即：

$$
10000^{-2i/d} \to 10000^{-2i/(ds)}
$$

**含义**：
- 维度 $i$ 对应的"周期"被拉长 $s$ 倍
- 高维度（低频）周期本来就很长，更加拉长
- 低维度（高频）周期被大幅拉长

### 10.2 为什么性能会下降

**权衡**：

* **优点**：模型可以处理更长的序列
* **缺点**：
  - 高频信息（局部细节）丢失
  - 注意力分辨率降低
  - 某些任务需要精细的局部信息

**例如**：
- 在长文档中找"第一个"匹配项：高频信息很重要
- 在长上下文中做总结：低频信息足够

---

## 十一、何时使用 Position Interpolation

### 11.1 适用场景

✓ **文本摘要**：需要理解全局上下文
✓ **长文档分类**：需要整体语义
✓ **对话系统**：多轮对话历史
✓ **代码理解**：长源文件
✗ **细粒度检索**：需要精确匹配
✗ **精确问答**：需要定位精确信息

### 11.2 选择策略

| 需求 | 推荐方法 |
| ---- | ------ |
| 快速原型 | Position Interpolation |
| 高性能 | YaRN 或 NTK |
| 无限长度 | ALiBi 或 Sparse Attention |
| 计算受限 | Flash-Attention |

---

## 十二、最终总结

**Position Interpolation 是**：

**一种简单、低成本的方法，通过缩放位置坐标，将 RoPE 基座的 Transformer 扩展到更长的序列。**

**核心公式**：

$$
\theta_i \cdot p \to \theta_i \cdot \frac{p}{s}, \quad s = \frac{L_{test}}{L_{train}}
$$

**优点**：
* 实现简单（一行代码）
* 零额外计算代价
* 立即可用

**缺点**：
* 性能有损失
* 有最大扩展倍数限制
* 对某些任务不够稳定

**建议**：
* 快速实验用 PI
* 生产环境用 YaRN 或 NTK
* 极长序列或特殊场景考虑其他方案

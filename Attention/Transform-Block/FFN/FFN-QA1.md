# FFN在transfrom中作用

在 Transformer 中，FFN（Feed Forward Network，前馈神经网络）是每一层里非常核心的组成部分。
很多人刚开始学习 Transformer 时，会把注意力（Attention）当成主角，但实际上：

> Attention 负责“信息交换”，FFN 负责“信息加工”。

可以把 Transformer 理解成：

```text
Attention：不同 token 之间互相沟通
FFN：每个 token 自己进行深度思考
```

---

## 一、Transformer 一层的结构

经典 Transformer Block：

```text
输入
 ↓
Multi-Head Attention
 ↓
Add & Norm
 ↓
FFN
 ↓
Add & Norm
 ↓
输出
```

其中 FFN 通常写成：

$$\mathrm{FFN}(x)=W_2,\sigma(W_1x+b_1)+b_2$$

这里：

* $x$：输入向量
* $W_1,W_2$：两层线性变换
* $\sigma$：激活函数（ReLU/GELU/SwiGLU 等）

---

## 二、FFN 到底在干什么？

FFN 的本质：

> 对每个 token 做“非线性特征变换”。

注意：

* Attention 是 token 与 token 之间交互
* FFN 是单个 token 内部计算

即：

```text
Attention:
“我该关注谁？”

FFN:
“拿到信息后，我该怎么理解？”
```

---

## 三、Attention 为什么还不够？

很多初学者会问：

> “既然 Attention 已经能混合信息了，为什么还要 FFN？”

因为：

### Attention 本身更像“路由器”

Attention 的核心：
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

它做的是：

* 计算相关性
* 加权聚合信息

本质偏线性组合。

如果只有 Attention：

```text
token之间能通信
但无法形成复杂非线性表达
```

模型表达能力会严重不足。

---

## 四、FFN 提供“非线性表达能力”

这是 FFN 最重要的作用。
神经网络真正强大的地方：

> 非线性（Non-linearity）

FFN 中的激活函数：

* ReLU
* GELU
* SwiGLU

让模型可以学习：

* 复杂模式
* 高阶特征
* 语义抽象
* 条件逻辑

否则整个 Transformer 会接近“大型线性系统”。

---

## 五、为什么 FFN 通常会“升维再降维”？

Transformer 中经典结构：

```text
d_model = 768

768
 ↓
3072
 ↓
768
```

即：

$$d_{ff}\approx 4\times d_{model}$$

例如：

* GPT-2:

  * hidden = 768
  * FFN = 3072

---

### 为什么这样设计？

因为：

#### 1. 扩展特征空间

类似：

```text
低维空间 → 高维空间 → 压缩回去
```

高维空间更容易：

* 分离特征
* 组合概念
* 学习复杂函数

这类似 kernel trick 的思想。

---

#### 2. 提高模型容量

Transformer 参数里：

> FFN 往往占大部分参数量。

例如很多 GPT 模型：

```text
FFN参数 > Attention参数
```

因为：

```text
768 × 3072 × 2
```

非常大。
所以：

> Attention 决定信息流动，
> FFN 决定模型“智力容量”。

---

## 六、FFN 为什么是“逐 token 独立计算”？

FFN 的一个重要特征：

```text
每个 token 单独计算
```

即：

```text
token A → FFN
token B → FFN
token C → FFN
```

互不影响。
数学上：
如果输入：

```text
(batch, seq_len, hidden)
```

FFN 只对最后一个维度做变换。

---

### 为什么这样设计？

因为：
Attention 已经完成：

```text
token之间的信息融合
```

FFN 接下来只需：

```text
对融合后的结果做深加工
```

---

## 七、FFN 可以理解成“知识存储器”

这是现代研究里的一个重要观点。
很多论文发现：

> Transformer 的 FFN 内部会存储大量知识。

例如：

* 事实知识
* 语义模式
* 语言规则

有研究发现：

```text
某些 neuron 专门对应：
- 国家
- 人名
- 语法
- 时间
```

因此有人把 FFN 称为：

```text
Key-Value Memory
```

Attention 像检索：

```text
“去哪里找信息”
```

FFN 像存储：

```text
“真正记住了什么”
```

---

## 八、为什么现在很多模型改进 FFN？

因为 FFN 非常重要。
现代 LLM 对 FFN 改进很多：

---

### 1. GELU

BERT/GPT 常见：
$$\mathrm{GELU}(x)=x\Phi(x)$$
比 ReLU 更平滑。

---

### 2. SwiGLU（Llama 常用）

Llama 使用：

```text
SwiGLU FFN
```

比传统 FFN 更强。
形式类似：

$$
\text{SwiGLU}(x) = (W_1x)\odot \text{Swish}(W_2x)
$$

优点：

* 更强表达能力
* 更稳定训练
* 参数利用率更高

---

### 3. MoE（Mixture of Experts）

现在超大模型：

* Mixtral
* DeepSeek
* GPT-4 类架构

大量采用：

```text
MoE FFN
```

即：

```text
不同 token 走不同 FFN 专家
```

因为：

> FFN 是 Transformer 中最“耗参数”的部分。

MoE 可以：

* 扩大容量
* 不增加太多计算

---

## 九、一个非常形象的理解

可以把 Transformer 想象成一个会议室：

---

### Attention

大家互相交流：

```text
“谁说的话重要？”
“我该听谁？”
```

这是信息流动。

---

### FFN

每个人回去自己思考：

```text
“这些信息意味着什么？”
“我要如何更新自己的理解？”
```

这是内部认知加工。

---

## 十、一句话总结

Transformer 中：

* Attention：

  * 负责 token 间的信息交互
  * 决定“看哪里”

* FFN：

  * 负责非线性特征加工
  * 决定“怎么理解”

二者组合：

```text
Attention → 聚合信息
FFN → 深度理解
```

缺少任何一个：
Transformer 都无法达到现在 LLM 的能力。

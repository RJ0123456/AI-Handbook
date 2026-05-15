# GLU of FFN

在 Transformer 的 FFN（Feed Forward Network）里，GLU（Gated Linear Unit）本质上是一种“带门控的非线性前馈层”。

它可以理解成：

> 普通 FFN：
> “把所有信息都统一处理”

而 GLU：

> “先判断哪些信息值得通过，再进行增强”

这和 Transformer 中 Attention 的思想有点类似，都是一种“选择性信息流动”。

---

## 1. 先回顾普通 FFN

标准 Transformer FFN：

$$
\mathrm{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2
$$

通常：

* $\sigma$ 是 ReLU / GELU / SiLU
* 中间维度会扩大 4 倍

例如：

$$
d_{model}=4096
$$

则：

$$
4096 \rightarrow 16384 \rightarrow 4096
$$

结构：

$$
x \rightarrow \text{Linear} \rightarrow \text{Activation} \rightarrow \text{Linear}
$$

---

## 2. GLU 的核心思想

GLU 增加了一个“门（Gate）”。

它不是直接激活：

$$
\sigma(Wx)
$$

而是：

$$
\mathrm{Value}\times\mathrm{Gate}
$$

即：

$$
a \odot g
$$

其中：

* $a$：真正的信息
* $g$：控制通过多少信息

这非常像：

* LSTM 的门
* Attention 的权重
* 神经元动态开关

---

## 3. GLU 数学形式

经典 GLU：

$$
\mathrm{GLU}(x)=(W_a x)\odot\sigma(W_g x)
$$

其中：

* $W_a x$：value 分支
* $W_g x$：gate 分支
* $\sigma$：sigmoid

结构：

$$
x \rightarrow
\begin{cases}
W_a x \\
W_g x
\end{cases}
$$

然后：

$$
(W_a x)\odot\sigma(W_g x)
$$

---

## 4. 几何直觉

普通 FFN：

> “所有方向统一放大/压缩”

而 GLU：

> “先判断当前方向是否重要”

即：

$$
\mathrm{输出}=\mathrm{信息}\times\mathrm{重要性}
$$

---

## 5. Transformer 中常见的 GLU 变体

现代 LLM 很少直接用原始 GLU，更常见：

* GeGLU（Google PaLM）
* SwiGLU（Llama）
* ReGLU

---

## 6. SwiGLU（Llama 使用）

Llama 系列大量使用 SwiGLU。

公式：

$$
\mathrm{SwiGLU}(x)=(W_1x)\odot\mathrm{SiLU}(W_2x)
$$

其中：

$$
\mathrm{SiLU}(x)=x\sigma(x)
$$

Transformer FFN 变成：

$$
\mathrm{FFN}(x)=W_o\left[(W_vx)\odot\mathrm{SiLU}(W_gx)\right]
$$

结构：

$$
x \rightarrow
\begin{cases}
W_vx \\
W_gx
\end{cases}
\rightarrow \text{SiLU Gate}
\rightarrow \text{逐元素乘法}
\rightarrow W_o
$$

---

## 7. 为什么 GLU 比 GELU FFN 更强？

核心原因：它让网络拥有“条件计算能力”。

普通 GELU：

$$
\mathrm{output}=\mathrm{activation}(Wx)
$$

每个神经元：

* 总会参与计算
* 只是强弱不同

而 GLU：

$$
\mathrm{output}=\mathrm{value}\times\mathrm{gate}
$$

gate 可以：

* 接近 0（关闭）
* 接近 1（打开）
* 动态调节

于是网络能动态选择信息路径，这会：

* 提高表达能力
* 提高参数效率
* 提高训练稳定性

---

## 8. 为什么 Llama 用 SwiGLU

因为它比 ReLU、GELU 通常：

* 更平滑
* 梯度更稳定
* 表达能力更强

论文与实践发现：在相同参数量下，SwiGLU 往往 perplexity 更低，token 预测更好。

因此：

* PaLM 用 GeGLU
* Llama 用 SwiGLU
* 很多现代 LLM 都采用 gated FFN

---

## 9. 一个语言上的直觉例子

输入句子：

> “苹果发布了新芯片”

普通 FFN 会对“苹果、发布、新、芯片”统一进行非线性变换。

而 GLU 可能会：

* 强打开：“苹果”“芯片”
* 弱打开：“了”“新”

即模型动态决定：

> “哪些特征更值得传播”

---

## 10. 从信息流角度理解

普通 FFN：

$$
x \rightarrow f(x)
$$

GLU：

$$
x \rightarrow f(x)\times g(x)
$$

即系统不仅学习：

* 信息内容

还学习：

* 信息流控制

这非常重要，因为深度网络真正困难的部分之一是：

> 什么信息应该继续传播

---

## 11. 参数量为什么会变化？

普通 FFN 通常：

$$
d \rightarrow 4d \rightarrow d
$$

SwiGLU 因为有两条分支，常见参数匹配设置会采用：

$$
d \rightarrow \frac{8}{3}d
$$

这样总体 FLOPs 与参数量可接近标准 FFN（取决于具体实现与常数项）。

这是 Llama 中的常见设计思路。

---

## 12. 本质总结

GLU 类 FFN 的核心本质：

| 普通 FFN | GLU FFN |
| --- | --- |
| 固定非线性 | 动态门控非线性 |
| 所有信息统一处理 | 选择性传播 |
| 激活函数控制强弱 | gate 控制通路 |
| 单一路径 | value + gate 双路径 |

可以把它理解成：

> FFN 从“静态特征变换”升级成“动态信息路由器”。

而这也是现代大模型越来越强的重要原因之一。

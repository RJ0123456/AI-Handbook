# 从复数视角理解 RoPE

RoPE（Rotary Position Embedding）最本质的理解方式，不是“旋转矩阵”，而是：

**相位旋转（phase rotation）**

它本质上是在复平面中，对 token 向量施加位置相关的复指数旋转。

---

## 1. 二维旋转 = 复数乘法

RoPE 中每两个维度：

$$
(x_1,x_2)
$$

可以看成一个复数：

$$
z=x_1+ix_2
$$

二维旋转矩阵：

$$
R(\theta)
$$

等价于：

$$
z' = ze^{i\theta}
$$

依据欧拉公式：

$$
e^{i\theta}=\cos\theta+i\sin\theta
$$

因此：

RoPE 的本质不是“加位置”，而是：

**给 embedding 加一个位置相关的相位。**

---

## 2. RoPE 的核心形式

位置为 $p$ 时：

$$
z_p = z\,e^{ip\theta}
$$

其中：

* $z$：token 表示
* $p$：位置
* $\theta$：该维度对应频率

即：

随着位置变化，token 在复平面持续旋转。

---

## 3. Attention 为什么天然变成“相对位置”？

设：

$$
q_p=qe^{ip\theta}
$$

$$
k_t=ke^{it\theta}
$$

attention：

$$
q_p\overline{k_t}
$$

展开：

$$
qe^{ip\theta}\cdot\overline{ke^{it\theta}} = q\bar{k}\,e^{i(p-t)\theta}
$$

于是：

**绝对位置自动消失，**

**Attention 只依赖：**

$$
p-t
$$

即：

**相对位置 = 相位差。**

这是 RoPE 最核心的数学性质。

---

## 4. 高频与低频

RoPE 不同维度使用不同频率：

$$
\theta_i = 10000^{-2i/d}
$$

因此：

| 频率     | 特性          |
| ------ | ----------- |
| 高频（低维） | 旋转快，擅长局部距离  |
| 低频（高维） | 旋转慢，保持长距离稳定 |

本质上：

**高频擅长“局部解析”**

**低频擅长“全局保持”**

这与 Fourier 系统完全一致。

---

## 5. 为什么会远距离衰减？

Attention 实际是：

$$
\sum_i c_i e^{i(p-t)\theta_i}
$$

即：

**多频率复指数叠加。**

短距离：

* 各频率相位接近
* 相长干涉（constructive interference）
* attention 强

长距离：

* 不同频率逐渐失步
* 相位随机化
* 相消干涉（destructive interference）

最终：

**attention 自然衰减。**

本质：

**phase decoherence（相位退相干）**

---

## 6. 为什么会出现长上下文混叠（aliasing）？

因为复指数具有周期性：

$$
e^{i\theta}=e^{i(\theta+2\pi)}
$$

因此：

不同距离可能产生相同相位：

$$
(p-t)\theta = (p'-t')\theta + 2k\pi
$$

模型将无法区分。

这就是：

**长距离 phase aliasing。**

---

## 7. Position Interpolation 的复数本质

PI：

$$
p \to p/s
$$

于是：

$$
e^{ip\theta}\to e^{i(p/s)\theta}
$$

等价于：

**降低频率**

**减缓相位增长**

从而：

* 延缓 phase explosion
* 减缓 decoherence
* 改善长上下文稳定性

---

## 8. Fourier 视角下的 RoPE

RoPE 的：

$$
e^{i\omega x}
$$

本质就是：

**Fourier basis（傅里叶基）**

因此：

Transformer 某种意义上是在：

**频域（frequency domain）中处理序列关系。**

RoPE：

本质上是：

**用 Fourier 相位编码位置。**

---

## 9. 最终本质

RoPE 最核心的数学思想：

**位置不是“标签”，**

**而是“相位”。**

token 之间的位置关系：

**通过相位差自然表达。**

因此：

$$
z_p = z\,e^{ip\theta}
$$

以及：

$$
e^{ip\theta}e^{-it\theta}=e^{i(p-t)\theta}
$$

就是 RoPE 的最深层本质。

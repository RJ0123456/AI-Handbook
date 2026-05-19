# RoPE 的“远距离衰减”数学本质

如果要扒开 RoPE 的底层数学推导，你会发现它的“远程衰减”并不是作者刻意设计出来的，而是**利用“等比步长的频率序列”在进行多波叠加时，数学上自然涌现出的“黎曼-勒贝格引理（Riemann-Lebesgue lemma）”和“狄利克雷核（Dirichlet Kernel）”效应**。

我们可以用严谨的数学语言，把这个“震荡相消”的推导过程拆解为三个核心步骤：

---

## 1. 核心数学表达式的建立

在 RoPE 中，Query 向量 $\mathbf{q}$ 和 Key 向量 $\mathbf{k}$ 分别位于位置 $m$ 和 $n$。为了简化推导且不失一般性，我们假设 $\mathbf{q}$ 和 $\mathbf{k}$ 的每个二维子空间上的模长均为 1，且初始相位相同。

那么，第 $m$ 位的 Query 和第 $n$ 位的 Key 的内积 $f(m, n)$ 可以写为 $d/2$ 个二维复平面内积的和：

$$f(m, n) = \text{Re} \sum_{i=1}^{d/2} e^{j(m-n)\theta_i} = \sum_{i=1}^{d/2} \cos(\Delta m \cdot \theta_i)$$

其中：

* $\Delta m = m - n$ 是相对距离。
* $\theta_i = \theta^{-2(i-1)/d}$ 是频率，基底通常取 $\theta = 10000$。

此时，**所谓的“远程衰减”，在数学上的等价命题就是：证明当 $\Delta m \to \infty$ 时，上面这个余弦级数和 $f(\Delta m)$ 的上界（Upper Bound）是趋于 0 的。**

---

## 2. 连续化近似与积分转换

当维度 $d$ 非常大时（例如大模型的 Hidden Size 通常是 4096，意味着有 2048 个项求和），我们可以把这个离散的求和**近似看作一个连续的积分**。

我们令 $t = \frac{i-1}{d/2}$，其范围从 $0$ 到 $1$。那么频率可以写为关于 $t$ 的函数：


$$\theta(t) = \theta^{-t} = 10000^{-t}$$

那么，原先的求和公式就可以转化为在 $t \in [0, 1]$ 上的定积分：

$$f(\Delta m) \approx \frac{d}{2} \int_{0}^{1} \cos\left( \Delta m \cdot 10000^{-t} \right) dt$$

为了求解这个积分，我们引入换元法。令 $x = 10000^{-t}$，则 $t = -\frac{\ln x}{\ln 10000}$，微分 $dt = -\frac{1}{\ln 10000} \cdot \frac{1}{x} dx$。积分上下限随之变为从 $1$ 到 $10000^{-1}$。

带入后，公式变为：

$$f(\Delta m) \approx \frac{d}{2 \ln 10000} \int_{10000^{-1}}^{1} \frac{\cos(\Delta m \cdot x)}{x} dx$$

---

## 3. 衰减上界的数学证明

重点就在于最后得到的这个积分：


$$\int_{10000^{-1}}^{1} \frac{\cos(\Delta m \cdot x)}{x} dx$$

这是一个在信号处理和数学分析中非常经典的积分（与**余弦积分 Ci** 密切相关）。我们可以通过分部积分法（Integration by parts）来观察它随 $\Delta m$ 变化的衰减速度：

设 $u = \frac{1}{x} \implies du = -\frac{1}{x^2}dx$
设 $dv = \cos(\Delta m \cdot x)dx \implies v = \frac{\sin(\Delta m \cdot x)}{\Delta m}$

利用分部积分公式 $\int u dv = uv - \int v du$：

$$\int_{10000^{-1}}^{1} \frac{\cos(\Delta m \cdot x)}{x} dx = \left[ \frac{\sin(\Delta m \cdot x)}{\Delta m \cdot x} \right]_{10000^{-1}}^{1} - \int_{10000^{-1}}^{1} \frac{\sin(\Delta m \cdot x)}{\Delta m \cdot x^2} dx$$

现在我们对这个结果提取 $\Delta m$：

$$= \frac{1}{\Delta m} \left( \sin(\Delta m) - 10000 \cdot \sin\left(\frac{\Delta m}{10000}\right) \right) - \frac{1}{\Delta m} \int_{10000^{-1}}^{1} \frac{\sin(\Delta m \cdot x)}{x^2} dx$$

由于 $\sin(\cdot)$ 函数的值域永远被限制在 $[-1, 1]$ 之间，我们可以利用**三角不等式**放大上界。原式的大小可以被缩放为：

$$\left| f(\Delta m) \right| \le \frac{C}{\Delta m}$$

其中 $C$ 是一个与维度 $d$ 和基底 $10000$ 相关的常数。

---

## 结论：数学本质是什么？

通过上述推导，RoPE 远距离衰减的数学本质可以总结为一句话：

> **通过将频率设置为底数为 10000 的等比数列，RoPE 将内积计算转化为了一个关于相对距离 $\Delta m$ 的高频震荡积分。根据分部积分法，该积分的包络线（Upper Bound）具有 $\mathcal{O}\left(\frac{1}{\Delta m}\right)$ 的渐进衰减速度。**

### 补遗：为什么大模型的外推会失效？

从上面的数学公式也能看出，当大模型遇到超长文本（$\Delta m > 10000$）时，高维度（低频段）的 $\sin(\frac{\Delta m}{10000})$ 也会开始剧烈震荡（跨越了超过 $2\pi$ 的周期）。这会导致原本用来稳定长距离的低频信号也陷入了“破坏性干涉”，破坏了原有的衰减规律，这就是为什么长文本大模型必须通过 **RoPE Scaling**（即改变基底 10000，或对频率乘以缩放因子）来强行把这个积分拉回稳定区的数学原因。

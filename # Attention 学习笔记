# Attention 学习笔记：从直觉到公式，再到代码

> 适合已经理解基础神经网络、CNN、RNN 或 LSTM 的学习者。  
> 本笔记重点回答：Attention 为什么出现、如何计算、Q/K/V 是什么、Self-Attention 与 Transformer 有什么关系。

---

# 1. Attention 是什么

Attention 可以理解为一种：

```text
根据当前任务，从一组信息中动态选择并聚合相关内容的机制。
```

RNN 的思路是：

```text
把过去的信息逐步压缩进隐藏状态。
```

Attention 的思路是：

```text
当前需要什么信息，就直接从所有候选信息中检索。
```

例如句子：

```text
小明昨天没有去学校，因为他生病了。
```

当模型处理“他”时，需要判断“他”指的是谁。

Attention 可以让“他”直接对“小明”分配较高权重，而不必像普通 RNN 那样让信息经过多个时间步逐步传播。

---

# 2. 为什么需要 Attention

## 2.1 RNN 的信息压缩问题

早期序列到序列模型通常采用：

```text
输入序列
→ Encoder RNN/LSTM
→ 固定长度向量
→ Decoder RNN/LSTM
→ 输出序列
```

假设输入序列为：

\[
x_1,x_2,\dots,x_T
\]

Encoder 最终把整个序列压缩为一个固定长度向量：

\[
h_T
\]

问题在于：

```text
无论输入序列有多长，都需要压缩进一个固定长度向量。
```

当输入很长时，这个向量可能无法完整保存所有细节。

---

## 2.2 Attention 的改进

Attention 不要求 Decoder 只依赖最后一个 Encoder 状态。

Encoder 会保留所有时间步的隐藏状态：

\[
h_1,h_2,\dots,h_T
\]

Decoder 在生成每个输出时，都可以重新查看这些状态，并决定：

```text
当前位置最应该关注输入中的哪些位置？
```

---

# 3. Attention 的三个基本步骤

Attention 的核心流程只有三步：

```text
计算相关性
→ softmax 得到权重
→ 对信息加权求和
```

假设当前查询向量为：

\[
q
\]

候选信息对应的 Key 和 Value 为：

\[
k_1,k_2,\dots,k_n
\]

\[
v_1,v_2,\dots,v_n
\]

先计算相关性分数：

\[
e_i=\operatorname{score}(q,k_i)
\]

再使用 softmax 转成权重：

\[
\alpha_i=
\frac{\exp(e_i)}
{\sum_j\exp(e_j)}
\]

最后加权聚合 Value：

\[
c=\sum_i\alpha_i v_i
\]

其中：

| 符号 | 含义 |
|---|---|
| \(q\) | 当前查询 |
| \(k_i\) | 第 \(i\) 条信息的匹配表示 |
| \(v_i\) | 第 \(i\) 条信息的实际内容 |
| \(e_i\) | 未归一化相关性分数 |
| \(\alpha_i\) | 注意力权重 |
| \(c\) | 聚合后的上下文向量 |

---

# 4. Query、Key、Value

现代 Attention 最重要的三个概念是：

- Query
- Key
- Value

## 4.1 直觉解释

可以类比搜索系统。

假设你搜索：

```text
深度学习课程
```

那么：

```text
Query：你输入的搜索内容
Key：每个网页的标题、标签和关键词
Value：网页实际包含的内容
```

搜索系统先用 Query 和 Key 进行匹配，再返回对应的 Value。

---

## 4.2 三者的职责

### Query

```text
我想找什么？
```

### Key

```text
我应该如何被匹配？
```

### Value

```text
匹配成功后，我实际提供什么内容？
```

因此 Attention 不是直接聚合 Key，而是：

```text
用 Query 和 Key 计算权重，再用权重聚合 Value。
```

---

# 5. Attention 的矩阵形式

输入序列通常表示为矩阵：

\[
X\in\mathbb{R}^{n\times d_{\text{model}}}
\]

其中：

- \(n\)：序列长度；
- \(d_{\text{model}}\)：每个 token 的向量维度。

通过三个可学习矩阵得到：

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

其中：

\[
W_Q\in\mathbb{R}^{d_{\text{model}}\times d_k}
\]

\[
W_K\in\mathbb{R}^{d_{\text{model}}\times d_k}
\]

\[
W_V\in\mathbb{R}^{d_{\text{model}}\times d_v}
\]

所以：

\[
Q\in\mathbb{R}^{n\times d_k}
\]

\[
K\in\mathbb{R}^{n\times d_k}
\]

\[
V\in\mathbb{R}^{n\times d_v}
\]

---

# 6. Scaled Dot-Product Attention

Transformer 中使用的核心公式为：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
\]

下面分步理解。

---

## 6.1 第一步：计算匹配分数

\[
QK^\top
\]

如果：

\[
Q\in\mathbb{R}^{n_q\times d_k}
\]

\[
K\in\mathbb{R}^{n_k\times d_k}
\]

则：

\[
QK^\top\in\mathbb{R}^{n_q\times n_k}
\]

第 \(i,j\) 个元素表示：

```text
第 i 个 Query 与第 j 个 Key 的匹配程度。
```

---

## 6.2 第二步：缩放

\[
\frac{QK^\top}{\sqrt{d_k}}
\]

为什么要除以 \(\sqrt{d_k}\)？

当向量维度较大时，点积结果的绝对值容易变大。

例如 softmax 输入：

```text
[1, 2, 3]
```

输出仍然比较平缓。

但如果输入变成：

```text
[10, 20, 30]
```

softmax 会极度集中在最大值位置，容易进入饱和区域，导致梯度变小。

因此使用：

\[
\sqrt{d_k}
\]

对点积进行缩放，使训练更加稳定。

---

## 6.3 第三步：softmax

\[
A=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
\]

其中：

\[
A\in\mathbb{R}^{n_q\times n_k}
\]

softmax 通常沿最后一个维度计算，因此每一行满足：

\[
\sum_j A_{ij}=1
\]

每一行表示：

```text
某一个 Query 对所有 Key 的注意力分布。
```

---

## 6.4 第四步：聚合 Value

\[
AV
\]

如果：

\[
A\in\mathbb{R}^{n_q\times n_k}
\]

\[
V\in\mathbb{R}^{n_k\times d_v}
\]

则：

\[
AV\in\mathbb{R}^{n_q\times d_v}
\]

每个 Query 最终得到一个 Value 的加权组合。

---

# 7. 一个完整手算示例

假设：

\[
Q=
\begin{bmatrix}
1&0
\end{bmatrix}
\]

\[
K=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\]

\[
V=
\begin{bmatrix}
10&0\\
0&20
\end{bmatrix}
\]

暂时忽略缩放因子。

## 7.1 计算匹配分数

\[
QK^\top
=
\begin{bmatrix}
1&0
\end{bmatrix}
\]

## 7.2 计算 softmax

\[
\operatorname{softmax}([1,0])
\approx
[0.731,0.269]
\]

## 7.3 聚合 Value

\[
0.731[10,0]
+
0.269[0,20]
\]

得到：

\[
[7.31,5.38]
\]

解释：

- Query 与第一个 Key 更相似；
- 因此第一个 Value 权重更高；
- 第二个 Value 仍然保留了一部分影响。

Attention 通常是“软选择”，不是只选择一个位置。

---

# 8. Self-Attention

## 8.1 定义

如果 Q、K、V 都来自同一个输入序列：

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

这就是 Self-Attention。

“Self”表示：

```text
序列中的每个位置都在关注同一序列中的其他位置。
```

## 8.2 例子

句子：

```text
小明没有去学校，因为他生病了。
```

在更新“他”的表示时，Self-Attention 可以直接读取“小明”的信息。

RNN 中：

```text
“小明”的信息需要经过多个时间步才能到达“他”。
```

Self-Attention 中：

```text
“他”可以直接与“小明”建立连接。
```

---

# 9. 为什么同一个输入需要 Q、K、V 三种表示

虽然 Q、K、V 都来自同一个 token，但角色不同。

以句子为例：

```text
银行提高了贷款利率。
```

“贷款”作为 Query 时，可能表示：

```text
我现在需要查找与贷款相关的信息。
```

“银行”作为 Key 时，可能表示：

```text
我可能是动作执行者，可以被相关 Query 匹配。
```

“银行”作为 Value 时，则提供：

```text
银行这一实体的语义内容。
```

因此三个线性映射不是多余的，它们让模型学习三种不同的功能表示。

---

# 10. Self-Attention 的注意力矩阵

假设句子有 4 个 token：

```text
小明 喜欢 深度 学习
```

那么 Attention 矩阵为：

\[
A\in\mathbb{R}^{4\times4}
\]

每一行表示一个 token 在更新自身表示时，对所有 token 的关注权重。

例如“深度”对应的一行可能是：

| 被关注位置 | 权重 |
|---|---:|
| 小明 | 0.05 |
| 喜欢 | 0.10 |
| 深度 | 0.20 |
| 学习 | 0.65 |

这表示“深度”在更新自身表示时，主要读取“学习”的信息。

因此经过 Self-Attention 后，一个 token 的表示不再只包含自己，而是融合了上下文。

---

# 11. Self-Attention 与 RNN/LSTM 的区别

| 特征 | RNN/LSTM | Self-Attention |
|---|---|---|
| 信息传播 | 按时间逐步传递 | 任意位置直接交互 |
| 并行能力 | 较弱 | 较强 |
| 长距离依赖 | 传播路径长 | 传播路径短 |
| 顺序信息 | 结构天然包含 | 需要位置编码 |
| 标准复杂度 | 近似 \(O(n)\) | 近似 \(O(n^2)\) |
| 典型模型 | LSTM、GRU | Transformer、BERT、GPT |

核心区别：

```text
RNN：通过状态传递信息。
Attention：通过动态关系矩阵读取信息。
```

---

# 12. 为什么 Self-Attention 需要位置编码

Self-Attention 主要根据内容相关性进行匹配，本身不天然知道 token 的先后顺序。

例如：

```text
猫追狗
狗追猫
```

两句话包含相同的词，但顺序不同，含义也不同。

因此 Transformer 的输入通常是：

\[
z_i=x_i+p_i
\]

其中：

- \(x_i\)：token embedding；
- \(p_i\)：位置编码。

常见位置编码：

- 固定正弦余弦位置编码；
- 可学习绝对位置编码；
- 相对位置编码；
- RoPE 旋转位置编码。

---

# 13. Mask

Attention 中常使用 Mask 屏蔽某些位置。

## 13.1 Padding Mask

批量训练时，不同句子长度不同，需要补齐：

```text
我 喜欢 机器 学习
他 喜欢 NLP [PAD]
```

`[PAD]` 不是真实内容，不应该被关注。

做法是在 softmax 前将对应位置的分数设置为：

\[
-\infty
\]

因为：

\[
\exp(-\infty)=0
\]

所以 softmax 后对应权重为 0。

## 13.2 Causal Mask

自回归生成模型不能看到未来 token。

例如：

```text
今 天 天 气
```

模型预测第二个 token 时，只能看到前面的 token，不能看到后面内容。

因果 Mask 通常是下三角结构：

\[
M=
\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}
\]

计算变为：

\[
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V
\]

GPT 类模型使用的就是 Masked Self-Attention。

---

# 14. Multi-Head Attention

## 14.1 为什么需要多头

单个 Attention 头只能在一个表示子空间中学习关系。

但语言中同时存在多种关系：

- 主谓关系；
- 指代关系；
- 修饰关系；
- 位置关系；
- 语义相似关系。

Multi-Head Attention 使用多组独立参数。

第 \(i\) 个头：

\[
\operatorname{head}_i
=
\operatorname{Attention}
(QW_i^Q,KW_i^K,VW_i^V)
\]

最后拼接：

\[
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}
(\operatorname{head}_1,\dots,\operatorname{head}_h)W^O
\]

## 14.2 直觉

不同头可能分别关注：

```text
头 1：局部相邻关系
头 2：主语与谓语
头 3：代词与指代对象
头 4：长距离语义关系
```

但不能保证每个头都具有清晰、固定、可解释的功能。

---

# 15. Cross-Attention

Cross-Attention 指 Q 与 K、V 来自不同序列。

典型 Encoder-Decoder 模型中：

```text
Q：来自 Decoder
K：来自 Encoder
V：来自 Encoder
```

作用是：

```text
Decoder 生成当前位置时，从 Encoder 输出中检索相关信息。
```

Self-Attention 与 Cross-Attention 的区别：

| 类型 | Q 来源 | K/V 来源 |
|---|---|---|
| Self-Attention | 当前序列 | 当前序列 |
| Cross-Attention | 一个序列 | 另一个序列 |

---

# 16. PyTorch 实现基础 Attention

```python
import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    """
    query: [batch, q_len, d_k]
    key:   [batch, k_len, d_k]
    value: [batch, k_len, d_v]

    mask:
        可以广播到 [batch, q_len, k_len]
        True 表示允许关注
        False 表示屏蔽
    """

    d_k = query.size(-1)

    # 1. 计算 Query 与 Key 的匹配分数
    scores = torch.matmul(
        query,
        key.transpose(-2, -1),
    )

    # 2. 缩放
    scores = scores / math.sqrt(d_k)

    # 3. 应用 Mask
    if mask is not None:
        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    # 4. 转换为注意力权重
    weights = F.softmax(scores, dim=-1)

    # 5. 聚合 Value
    output = torch.matmul(weights, value)

    return output, weights
```

---

# 17. 张量形状分析

假设：

```text
Q: [batch, q_len, d_k]
K: [batch, k_len, d_k]
V: [batch, k_len, d_v]
```

则：

```text
K.transpose(-2, -1)
→ [batch, d_k, k_len]
```

矩阵乘法：

```text
Q @ K^T
→ [batch, q_len, k_len]
```

经过 softmax 后：

```text
weights
→ [batch, q_len, k_len]
```

最后：

```text
weights @ V
→ [batch, q_len, d_v]
```

理解 Attention 时，必须熟悉这些形状变化。

---

# 18. 简单 Self-Attention 模块

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
    ):
        super().__init__()

        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_k)
        self.out_proj = nn.Linear(d_k, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        x: [batch, seq_len, d_model]
        """

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores / math.sqrt(q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(
                ~mask,
                float("-inf"),
            )

        weights = F.softmax(scores, dim=-1)

        context = torch.matmul(weights, v)

        output = self.out_proj(context)

        return output, weights
```

---

# 19. 创建因果 Mask

```python
import torch


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    返回形状：
    [1, seq_len, seq_len]

    True 表示允许关注
    False 表示屏蔽未来
    """

    mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
        )
    )

    return mask.unsqueeze(0)
```

示例：

```python
mask = create_causal_mask(4)

print(mask[0].int())
```

输出：

```text
tensor([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
])
```

---

# 20. LSTM + Attention

LSTM 与 Attention 可以组合。

常见结构：

```text
输入
→ Embedding
→ BiLSTM
→ 每个时间步的隐藏状态
→ Attention Pooling
→ 分类器
```

BiLSTM 输出：

\[
h_1,h_2,\dots,h_T
\]

Attention 给每个位置计算分数：

\[
e_t=v^\top\tanh(Wh_t+b)
\]

归一化：

\[
\alpha_t=
\frac{\exp(e_t)}
{\sum_j\exp(e_j)}
\]

得到句子表示：

\[
c=\sum_t\alpha_t h_t
\]

这比只取最后一个隐藏状态更加灵活。

---

# 21. Attention 与 Transformer 的关系

Attention 是一种机制。

Transformer 是一种完整架构。

Transformer 通常包括：

```text
Multi-Head Attention
Residual Connection
Layer Normalization
Feed-Forward Network
Positional Encoding
Dropout
```

所以：

```text
Attention ≠ Transformer
```

更准确地说：

```text
Transformer 是以 Attention 为核心构件的神经网络架构。
```

---

# 22. Attention 的优势

## 22.1 长距离依赖

任意两个位置之间可以直接交互，不需要逐步传播。

## 22.2 并行计算

序列中的所有位置可以同时计算，不像 RNN 必须按时间步依次计算。

## 22.3 动态信息选择

不同 Query 可以得到不同的注意力权重。

## 22.4 表示能力强

通过多头机制，可以在多个表示子空间中建模关系。

---

# 23. Attention 的局限

## 23.1 标准复杂度较高

标准 Self-Attention 需要构造：

\[
n\times n
\]

的注意力矩阵。

时间和显存复杂度近似：

\[
O(n^2)
\]

长序列中开销较大。

## 23.2 不天然包含顺序

需要位置编码。

## 23.3 注意力权重不等于完整解释

高注意力权重不一定表示该 token 是最终预测的真正因果原因。

## 23.4 多头可能冗余

部分注意力头可能学习到相似关系。

---

# 24. 常见误区

## 24.1 Attention 是选择一个位置

不准确。

Attention 通常是对所有位置进行软加权：

\[
\sum_i\alpha_i v_i
\]

不是简单取最大值。

## 24.2 Attention 权重就是模型解释

不准确。

Attention 权重只反映：

```text
某一层、某一头、某一次计算中的信息聚合比例。
```

最终输出还受到其他注意力头、后续层、前馈网络、残差连接和 LayerNorm 等因素影响。

## 24.3 Q、K、V 是固定的

不准确。

它们通常由输入经过可学习线性层得到：

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

训练过程中会不断更新。

## 24.4 Attention 一定优于 LSTM

不一定。

LSTM 可能更适合：

- 小数据集；
- 短序列；
- 流式输入；
- 低延迟在线推理；
- 显存有限场景。

Attention 更适合：

- 大规模训练；
- 长距离依赖；
- 高度并行；
- 语言模型和多模态模型。

---

# 25. 学习路线

建议按以下顺序学习：

```text
基础向量与矩阵乘法
→ softmax
→ Query / Key / Value
→ Dot-Product Attention
→ Scaled Dot-Product Attention
→ Self-Attention
→ Mask
→ Multi-Head Attention
→ Positional Encoding
→ Transformer Encoder
→ Transformer Decoder
→ BERT / GPT
```

---

# 26. 必须掌握的问题

学完 Attention 后，应能回答：

1. Attention 为什么出现？
2. Query、Key、Value 分别是什么？
3. 为什么使用 \(QK^\top\)？
4. 为什么除以 \(\sqrt{d_k}\)？
5. softmax 沿哪个维度计算？
6. 为什么最终聚合的是 Value？
7. Self-Attention 的 Q、K、V 来自哪里？
8. 为什么 Self-Attention 需要位置编码？
9. causal mask 和 padding mask 有什么区别？
10. Multi-Head Attention 为什么有用？
11. Attention 与 Transformer 有什么区别？
12. 为什么标准 Attention 的复杂度是 \(O(n^2)\)？

---

# 27. 自测练习

## 练习 1：形状推导

给定：

```text
batch = 8
q_len = 10
k_len = 12
d_k = 64
d_v = 32
```

并且：

```text
Q: [8, 10, 64]
K: [8, 12, 64]
V: [8, 12, 32]
```

请推导：

1. \(K^\top\) 的形状；
2. \(QK^\top\) 的形状；
3. 注意力权重的形状；
4. 最终输出的形状。

答案：

```text
K^T:       [8, 64, 12]
QK^T:      [8, 10, 12]
weights:   [8, 10, 12]
output:    [8, 10, 32]
```

## 练习 2：概念判断

判断下列说法是否正确：

1. Attention 只会选择一个位置。
2. Self-Attention 的 Q、K、V 来自同一序列。
3. softmax 后每一行权重之和为 1。
4. causal mask 用于屏蔽 padding。
5. Attention 权重就是完整模型解释。
6. Transformer 只有 Attention 层。

答案：

```text
1. 错
2. 对
3. 对
4. 错
5. 错
6. 错
```

## 练习 3：代码练习

独立完成以下任务：

1. 手写 `scaled_dot_product_attention`；
2. 加入 padding mask；
3. 加入 causal mask；
4. 输出并可视化注意力矩阵；
5. 实现单头 Self-Attention；
6. 实现 Multi-Head Attention；
7. 与 `torch.nn.MultiheadAttention` 对比输出形状。

---

# 28. 最简记忆框架

记住公式：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
\]

对应四句话：

```text
Q 表示我想找什么。
K 表示我应该如何被匹配。
QKᵀ 计算相关性。
相关性经过 softmax 后，对 V 加权求和。
```

再记住一句：

```text
RNN 是把信息逐步传过来，
Attention 是需要时直接查过去。
```

---

# 29. 学习完成标准

当你能独立完成以下任务时，可以继续学习 Transformer：

- 不看资料解释 Query、Key、Value；
- 手算一个小型 Attention；
- 推导 Attention 中每个张量的形状；
- 手写 Scaled Dot-Product Attention；
- 解释 padding mask 与 causal mask；
- 解释 Self-Attention 为什么需要位置编码；
- 说明 Multi-Head Attention 的作用；
- 说明 Attention 与 RNN/LSTM 的区别。

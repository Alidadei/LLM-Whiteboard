# 当前主题：Self-Attention

## LLM讲解

### 一、这段代码在干什么？

Self-Attention 做一件事：**让序列中的每个 token 都去看一遍序列中所有 token（包括自己），然后根据"相关性"加权汇总信息**。

代码分三步走：

```
输入 x: (batch, seq_len, d_model)
    │
    ├── 线性投影 → Q, K, V（三个都是同样的形状）
    │
    ├── Q·Kᵀ / √d  →  scores: (batch, seq_len, seq_len)  ← 注意力分数矩阵
    │       │
    │       └── mask（可选）：把不该看到的位置填 -inf
    │
    ├── softmax(scores)  →  attn_weights: (batch, seq_len, seq_len)  ← 归一化权重
    │
    ├── weights · V  →  out: (batch, seq_len, d_model)  ← 加权求和
    │
    └── o_proj(out)  →  最终输出: (batch, seq_len, d_model)
```

### 二、逐行拆解 + 为什么这样设计

#### 1. 四个投影矩阵（Q/K/V/O）

```python
self.q_proj = nn.Linear(d_model, d_model, bias=False)
self.k_proj = nn.Linear(d_model, d_model, bias=False)
self.v_proj = nn.Linear(d_model, d_model, bias=False)
self.o_proj = nn.Linear(d_model, d_model, bias=False)
```

**为什么 Q/K/V 是三个独立的投影？**

- Q（Query）：当前 token 拿着"我想找什么信息"去提问
- K（Key）：每个 token 暴露"我有什么信息"供人匹配
- V（Value）：匹配成功后，实际传递的信息内容

如果 Q=K=V 用同一个投影，那每个 token 对自己的注意力永远最高（因为自己和自己点积最大），就失去了"在不同位置关注不同信息"的能力。分开投影让模型可以学到**不同的表示角色**。

**为什么 bias=False？**

在 Transformer 原始论文和大多数 LLM 实现中，QKV 投影不加偏置。原因：点积注意力本身是在做向量相似度，偏置项会让相似度计算多一个与输入无关的常数项，没有几何意义，还增加参数量。

**为什么需要 o_proj？**

attn_weights · V 的输出形状是 (batch, seq_len, d_model)，看起来和输入一样，为什么还要再过一层线性？

因为后面会接 Multi-Head Attention（第2章），多头拼接后维度会变（比如 h 个头拼接后变成 h·d_k），需要 o_proj 把它**投影回 d_model**，统一维度。单头场景下 o_proj 也可以学习一个额外的变换，保留它不亏。

#### 2. 注意力分数计算

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
```

- `Q @ Kᵀ`：每个 query 和每个 key 做点积，得到 seq_len × seq_len 的相似度矩阵
- `K.transpose(-2, -1)`：把 (batch, seq_len, d_model) 转成 (batch, d_model, seq_len)，这样矩阵乘法结果的最后两维是 (seq_len, seq_len)

**为什么要除以 √d？**

这是 **Scaled Dot-Product** 的核心。假设 Q 和 K 的每个元素独立、均值为 0、方差为 1，那点积结果的方差是 d_model。维度很大时（如 512、1024），点积的值会很大，softmax 输入一大会进入梯度极小区域（饱和区），梯度几乎为 0，训练不动。除以 √d 把方差拉回 1，保证 softmax 的输入在一个合理范围内。

#### 3. Mask

```python
if mask is not None:
    scores = scores.masked_fill(mask, float("-inf"))
```

**为什么用 `if mask is not None` 而不是 `if mask`？**

mask 是布尔型 tensor。Python 的 `if` 要求一个标量布尔值，但 tensor 里有很多元素，PyTorch 不知道你是想判断"全部为 True"还是"任意一个为 True"，所以直接报错。用 `is not None` 判断的是"有没有传 mask"，语义明确。

**mask 的上三角设计：**

```python
mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
```

以 seq_len=4 为例，生成：

```
F T T T
F F T T
F F F T
F F F F
```

T（True）= 屏蔽，意味着：第 0 个 token 不能看第 1/2/3 个 token（未来位置），第 1 个 token 不能看第 2/3 个，以此类推。这就是 **causal mask**（因果掩码），用于自回归生成，防止"偷看未来"。

diagonal=1 意味着主对角线（自己看自己）不被屏蔽，符合直觉——token 可以关注自身。

#### 4. Softmax + 加权求和

```python
attn_weights = F.softmax(scores, dim=-1)
out = torch.matmul(attn_weights, V)
```

- softmax 沿最后一维（dim=-1），对每个 query 来说，所有 key 的权重和为 1
- -inf 经过 softmax 变成 0，所以被 mask 的位置权重为 0，完全不参与计算

### 三、维度流转总览

```
x: (B, L, D)
  ↓ q/k/v_proj
Q, K, V: (B, L, D)
  ↓ Q @ Kᵀ
scores: (B, L, L)       ← 每行是一个 query 对所有 key 的相似度
  ↓ softmax
attn_weights: (B, L, L)  ← 每行和为 1 的概率分布
  ↓ @ V
out: (B, L, D)           ← 每个 token 得到一个 d_model 维的加权表示
  ↓ o_proj
output: (B, L, D)        ← 最终输出
```

## Open Questions

（请在上方内容旁直接批注提问，格式：`Q:你的问题`）

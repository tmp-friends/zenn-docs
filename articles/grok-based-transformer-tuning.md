---
title: "Grok-based Transformer の要素を取り入れて Two-Tower モデルを改善する"
emoji: "🐦‍🔥"
type: "tech"
topics: ["機械学習", "レコメンド", "PyTorch"]
published: false
publication_name: "dmmdata"
---

## はじめに

前回の記事[^1]で、X のレコメンドアルゴリズムの公開コードを読みました。

その中で、次のことを確認しました。
- Transformer を使ってユーザの行動履歴をエンコードしていること
- 最近の LLM で見られるような Transformer の構成要素が使われていること

| 要素 |（オリジナル）Transformer | Grok-based Transformer |
|---|---|---|
| Normalization | LayerNorm | RMSNorm |
| 位置エンコーディング | Sinusoidal | RoPE |
| FFN | 標準 FFN | SwiGLU |
| KV sharing | なし | GQA（今回は未検証） |

直近の取り組みにおいて、Two-Tower モデルの User Tower に Transformer を使っているので、上記の要素を参考に精度改善を試みることにしました。

## 各手法の詳細と実装

### RMSNorm（Root Mean Square Layer Normalization）

RMSNorm[^2] は、LayerNorm から平均の引き算（リセントリング）を省略し、**二乗平均平方根（RMS）のみで正規化（リスケーリング）** する手法です。

まず、標準的な LayerNorm は層の入力ベクトル $a \in \mathbb{R}^n$ に対して以下のように定義されます。

$$
\bar{a} = \frac{a - \mu}{\sigma} \odot g + b
$$

ここで $\odot$ は要素ごとの積、$g$ は学習可能なゲインパラメータ、$b$ はバイアスパラメータです。平均 $\mu$ と標準偏差 $\sigma$ は以下で計算されます。

$$
\mu = \frac{1}{n} \sum_{i=1}^{n} a_i, \quad \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (a_i - \mu)^2 + \epsilon}
$$

一方、RMSNorm は平均 $\mu$ を計算せず、RMS のみで正規化します。出力 $\bar{a}_i$ は以下の式で計算されます。

$$
\bar{a}_i = \frac{a_i}{\text{RMS}(a)} g_i, \quad \text{where} \quad \text{RMS}(a) = \sqrt{\frac{1}{n} \sum_{j=1}^{n} a_j^2 + \epsilon}
$$

平均 $\mu$ の計算と各要素からの減算が不要な分、計算コストが削減されます。
論文では、LayerNorm の効果の大部分はリスケーリングにあり、リセントリングを省略しても学習の安定性にはほとんど影響がないことが実験的に示されています。

PyTorch 2.4 以降では `nn.RMSNorm` が標準で利用できます。

:::details コード例
```python
# TransformerEncoderLayer で RMSNorm へ差し替える
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256,
    nhead=8,
    dim_feedforward=1024,
)
encoder_layer.norm1 = nn.RMSNorm(normalized_shape=256)
encoder_layer.norm2 = nn.RMSNorm(normalized_shape=256)
```
:::

### RoPE (Rotary Position Embedding)

RoPE[^3] は、トークンの位置情報を回転行列として Query と Key に埋め込む手法です。

RoPE の目標は、位置 $m$ の Query $q_m$ と位置 $n$ の Key $k_n$ の内積が**相対位置 $(m - n)$ のみに依存する**ようにすることです。

$$
\langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m - n)
$$

まず $d = 2$ の場合を考えます。ベクトルを複素数として表すと、位置 $m$ での回転は $e^{im\theta}$ の乗算になります。

$$
f_q(x_m, m) = q_m \, e^{im\theta}, \quad f_k(x_n, n) = k_n \, e^{in\theta}
$$

内積を計算すると $\text{Re}[q_m k_n^* e^{i(m-n)\theta}]$ となり、相対位置 $(m-n)$ のみに依存することがわかります。
これを実数の行列形式で書くと、おなじみの $2 \times 2$ 回転行列になります。

$$
f(x, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q^{(1)} \\ q^{(2)} \end{pmatrix}
$$

次に $d$ 次元への一般化を考えます。$d/2$ 個の2次元平面に分割し、それぞれ異なる回転角 $\theta_i = 10000^{-2(i-1)/d}$ を適用します。

実装上は、ブロック対角行列との積を要素ごとの演算に展開して効率的に計算します。

$$
\text{RoPE}(q, m) = \begin{pmatrix} q^{(1)} \\ q^{(2)} \\ \vdots \\ q^{(d-1)} \\ q^{(d)} \end{pmatrix} \odot \begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2} \\ \cos m\theta_{d/2} \end{pmatrix} + \begin{pmatrix} -q^{(2)} \\ q^{(1)} \\ \vdots \\ -q^{(d)} \\ q^{(d-1)} \end{pmatrix} \odot \begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2} \\ \sin m\theta_{d/2} \end{pmatrix}
$$

Sinusoidal PE が位置ベクトルを**加算**するのに対し、RoPE は回転行列を**乗算**する点が本質的に異なります。
行動履歴のように「何ステップ前のアクションか」が重要な系列に対して、相対的な順序関係を効率的にエンコードできます。

:::details コード例
```python
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 逆周波数を事前計算: [dim // 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        # キャッシュ
        self.register_buffer("cos_cached", emb.cos().view(1, 1, seq_len, -1), persistent=False)
        self.register_buffer("sin_cached", emb.sin().view(1, 1, seq_len, -1), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """位置に対応する cos/sin を返す。"""
        seq_len = x.size(2)
        if offset + seq_len > self.max_seq_len:
            self._build_cache(offset + seq_len)
        return (
            self.cos_cached[:, :, offset : offset + seq_len, :],
            self.sin_cached[:, :, offset : offset + seq_len, :],
        )


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE を Query と Key に適用する。"""
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    return q_rotated, k_rotated
```
:::

### SwiGLU（Swish-Gated Linear Unit）

SwiGLU[^4] は、Transformer の FFN をゲート機構で拡張した手法です。

標準的な FFN は、1つの線形変換に活性化関数を適用してから2つ目の線形変換を行います。

$$
\text{FFN}_{\text{ReLU}}(x, W_1, W_2) = \max(xW_1, 0) \, W_2
$$

GLU（Gated Linear Unit）[^5] は、入力に対して**2つの線形変換**を行い、片方を活性化関数に通したうえで要素ごとの積をとるゲート機構です。SwiGLU は活性化関数として $\text{Swish}_1(x) = x \cdot \sigma(x)$ を用いた GLU 変異体で、FFN に組み込むと以下のようになります。

$$
\text{FFN}_{\text{SwiGLU}}(x, W, V, W_2) = (\text{Swish}_1(xW) \otimes xV) \, W_2
$$

$\otimes$ は要素ごとの積です。ゲート機構により「どの情報を通すか」を学習でき、表現力が向上します。

:::details コード例
```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w = nn.Linear(d_model, d_ff, bias=False)   # W
        self.v = nn.Linear(d_model, d_ff, bias=False)    # V
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # W2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w(x)) * self.v(x))
```
:::

重み行列が $W, V, W_2$ の3つになるためパラメータ数が増加します。論文では $d_{ff}$ を従来の $\frac{2}{3}$ 倍に削減してパラメータ数を揃えています。

### GQA (Grouped Query Attention)

:::message
GQA は今回は未検証です。（時間がなかった。。）推論速度やメモリ効率の観点で効果が期待できるため、今後検証予定です。
:::


GQA[^6] は、Multi-Head Attention（MHA）と Multi-Query Attention（MQA）の中間に位置する手法です。

![KV sharing のパターン](/images/grok-based-transformer-tuning/gqa.png)

- MHA: Query と Key-Value が1対1で対応（6ヘッド → KV も6セット）
- GQA: Query をグループに分け、グループごとに Key-Value を共有（6ヘッド → KV は2セット）
- MQA: すべての Query が1つの Key-Value を共有（6ヘッド → KV は1セット）

GQA は MHA に比べて KV キャッシュのメモリ使用量を削減でき、推論速度が改善します。
MQA ほど極端に KV を減らさないため、精度への影響も抑えられます。


## 実験結果

### 評価指標

- **正例ユーザ数**: 正解アイテムが上位100件に1つでも含まれるユーザ数
- **Recall@100**: 正解アイテムのうち、上位100件に含まれるものの割合
- **nDCG@100**: 上位100件のランキング品質を測る指標。正解アイテムがより上位にランクされるほど高スコアになる

### 結果

各手法を段階的に適用した結果は以下のとおりです。

| variant | Δ 正例ユーザ数 | Δ Recall@100 | Δ nDCG@100 |
|---|---|---|---|
| baseline | — | — | — |
| w/ RMSNorm | +0.15% | ±0.00% | ±0.00% |
| w/ RMSNorm, RoPE | +1.16% | +8.33% | +3.06% |
| w/ RMSNorm, RoPE, SwiGLU | +1.87% | +8.33% | +3.06% |

### 考察

順当に精度が上がりました。

RMSNorm 単体では Recall と nDCG の数値上の差がほとんど見られませんでした。また、学習速度の向上も今回の実験では確認できませんでした。
LLM に比べモデル規模が小さいため、平均計算の省略による計算コスト削減の恩恵が顕在化しなかった可能性があります。

一方で、RoPE を追加した段階で Recall@100 と nDCG@100 がそれぞれ 8.33% と 3.06% 上昇しており、明確な改善が見られました。
用いている行動履歴の系列長が 256 と比較的長いこともあり、「何ステップ前のアクションか」という相対位置情報が効いていると考えられます。絶対位置エンコーディングでは捉えにくい、直近のアクションと過去のアクションの関係性をうまくモデリングできているのではないかと思います。

SwiGLU は RoPE との組み合わせで、直前の設定比で正例ユーザ数が +0.71% と微増しました。
FFN の表現力向上が、候補検索の精度にわずかに寄与しているようです。

## おわりに

Grok-based Transformer の構成要素を Two-Tower モデルに段階的に適用して、精度改善を確認することができました。
特に RoPE の効果が大きかったのは、行動履歴の相対的な順序関係をうまく捉えられるようになったからだと思います。

今回の検証では GQA は未実装ですが、推論レイテンシやメモリ効率の面で効果が期待できるため、今後検証していきたいです。
また、Grok-based Transformer 以外にも、最近の LLM で見られる構成要素（e.g. FlashAttention, MoE など）を取り入れてみるのも面白そうです。

## 参考文献

https://github.com/xai-org/grok-1


https://zenn.dev/shot4410/articles/5354ce65907e15


https://zenn.dev/lluminai_tech/articles/f488b0843efda3


[^1]: X のレコメンドアルゴリズムの実装を読む: https://zenn.dev/dmmdata/articles/x-recommendation-algorithm
[^2]: Root Mean Square Layer Normalization (Zhang & Sennrich, 2019): https://arxiv.org/abs/1910.07467
[^3]: RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021): https://arxiv.org/abs/2104.09864
[^4]: GLU Variants Improve Transformer (Shazeer, 2020): https://arxiv.org/abs/2002.05202
[^5]: Language Modeling with Gated Convolutional Networks (Dauphin et al., 2016): https://arxiv.org/abs/1612.08083
[^6]: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023): https://arxiv.org/abs/2305.13245

---
title: "Ruri-v3-30mを使ってテキストembeddingを取得し、類似度計算をやってみた"
emoji: "🔍"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["機械学習", "embedding", "sentencetransformers", "python"]
published: true
---

## はじめに

- 日本語に特化したSentence TransformerであるRuri-v3-30mモデルを試してみたかった
- テキストからembeddingを取得して類似度計算を実装してみたいと思った
- レコメンドシステムでの活用を想定した基本的な処理を学びたかった

## Ruri-v3-30mについて

Ruri-v3-30mは、名古屋大学が公開している日本語に特化したSentence Transformerモデルです。

- HuggingFace Hub: `cl-nagoya/ruri-v3-30m`
- 日本語テキストに対して高品質なembeddingを生成
- Sentence Transformersライブラリで簡単に使用可能

## 実装してみた

### 環境構築・ライブラリ導入

```bash
pip install -U "transformers>=4.48.0" sentence-transformers
pip install scann tensorflow
```

### モデルの読み込み

```python
import torch
from sentence_transformers import SentenceTransformer

# Ruri v3-30m を fp16 でロード（メモリ節約）
model = SentenceTransformer(
    "cl-nagoya/ruri-v3-30m",
    device="cuda",
    model_kwargs={"torch_dtype": torch.float16},
)
model.max_seq_length = 128  # 128 token で十分
```

### embeddingの取得

```python
import numpy as np
from tqdm.auto import tqdm

def encode_texts(text_list, batch_size=32):
    """SentenceTransformer で一括エンコードして np.ndarray を返す。"""
    chunks = range(0, len(text_list), batch_size)
    out = []
    
    with torch.no_grad():
        for st in tqdm(chunks, desc="encode", leave=False):
            emb = model.encode(
                text_list[st : st + batch_size],
                batch_size=batch_size,
                normalize_embeddings=True,  # L2 正規化
                convert_to_numpy=True,
            )
            out.append(emb.astype(np.float32))
    
    return np.concatenate(out, axis=0)

# サンプルテキスト
texts = [
    "機械学習について学んでいます",
    "深層学習の勉強をしています", 
    "今日は良い天気です",
    "プログラミングは楽しいです",
    "自然言語処理に興味があります"
]

# embeddingを取得
embeddings = encode_texts(texts)
print(f"Embedding shape: {embeddings.shape}")  # (5, 128)
```

### ScaNN indexを使った類似度検索

```python
import tensorflow as tf
from scann.scann_ops.py import scann_ops_pybind

# ScaNN indexを構築
def build_scann_index(embeddings, num_leaves=100, num_leaves_to_search=10):
    """ScaNN indexを構築してBruteForce検索を行う"""
    # TensorFlowテンソルに変換
    tf_embeddings = tf.constant(embeddings, dtype=tf.float32)
    
    # ScaNN builderを作成
    builder = scann_ops_pybind.builder(
        db=tf_embeddings, 
        num_neighbors=len(embeddings), 
        distance_measure="dot_product"
    ).tree(
        num_leaves=num_leaves,
        num_leaves_to_search=num_leaves_to_search,
        training_sample_size=len(embeddings)
    ).score_brute_force()
    
    return builder.build()

# ScaNN indexを構築
scann_index = build_scann_index(embeddings)

# 類似度検索を実行
query_idx = 0  # "機械学習について学んでいます"を検索
query_embedding = embeddings[query_idx:query_idx+1]

# 上位3件の類似アイテムを取得
neighbors, scores = scann_index.search_batched(
    queries=tf.constant(query_embedding, dtype=tf.float32),
    final_num_neighbors=3
)

print(f"クエリ: {texts[query_idx]}")
print("類似アイテム:")
for i, (neighbor_idx, score) in enumerate(zip(neighbors[0], scores[0])):
    neighbor_idx = int(neighbor_idx)
    print(f"{i+1}. {texts[neighbor_idx]} (スコア: {score:.4f})")
```

出力例:
```
クエリ: 機械学習について学んでいます
類似アイテム:
1. 機械学習について学んでいます (スコア: 1.0000)
2. 深層学習の勉強をしています (スコア: 0.8235)
3. 自然言語処理に興味があります (スコア: 0.7890)
```

## 工夫した点

### メモリ効率化
- `torch.float16`を使用してfp16で実行することでメモリ使用量を半減
- 大量のテキストを処理する際に重要な最適化

### バッチ処理
- 一度に全てのテキストを処理せず、バッチサイズごとに分割して処理
- メモリ不足を防ぎつつ効率的に処理

### 正規化の活用
- `normalize_embeddings=True`でL2正規化を適用
- コサイン類似度の計算が内積だけで済むため高速化

### ScaNN indexの活用
- GoogleのScaNN（Scalable Nearest Neighbors）を使用してBruteForce検索を実装
- 大規模データでも高速な類似度検索が可能
- `score_brute_force()`により正確な類似度スコアを取得

## 感想

- Ruri-v3-30mは日本語テキストに対して直感的に理解できる類似度を出力してくれた
- 「機械学習」と「深層学習」、「自然言語処理」の類似度が高く出るなど、期待通りの結果が得られた
- Sentence Transformersライブラリのおかげで、複雑な実装なしに高品質なembeddingが取得できることがわかった
- fp16での軽量化が想像以上に効果的で、メモリ制約がある環境でも使いやすかった
- ScaNN indexを使うことで、単純な行列計算よりも実用的な類似度検索システムが構築できた
- BruteForce検索により正確なスコアが得られ、レコメンドシステムの基盤として十分使えそうだと感じた

## おわりに

- 基本的なembedding取得と類似度計算を実装することができた
- 今後は複数種類のembeddingを組み合わせた加重合成なども試してみたい
- レコメンドシステムでの実用性を検証するため、より大規模なデータセットでの評価も行ってみたいと思う

## 参考

- [cl-nagoya/ruri-v3-30m - Hugging Face](https://huggingface.co/cl-nagoya/ruri-v3-30m)
- [sentence-transformers documentation](https://www.sbert.net/)
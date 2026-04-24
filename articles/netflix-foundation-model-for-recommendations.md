---
title: "Netflix のレコメンド基盤モデルまとめ"
emoji: "🎬"
type: "tech"
topics: ["機械学習", "レコメンド", "transformer", "llm"]
published: false
publication_name: "dmmdata"
---

## はじめに

Netflix は2025年3月から2026年1月にかけて、レコメンド向け Foundation Model（基盤モデル）に関するブログ記事を Netflix Tech Blog で3本公開しました。

1. Foundation Model for Personalized Recommendation（2025年3月）[^1]
2. Integrating Netflix's Foundation Model into Personalization applications（2025年11月）[^2]
3. Towards Generalizable and Efficient Large-Scale Generative Recommenders（2026年1月）[^3]

Netflix のホームページは「Continue Watching」「Today's Top Picks for You」など、多数の推薦面（surface）で構成されており、それぞれに専用のモデルが用意されています。しかし、共通のデータソースを用いているにもかかわらず、ほとんどのモデルが個別に学習されているため、以下の課題が顕在化していました。

- レコメンドシステム全体の運用コストが増大
- あるモデルでの改善を別のモデルへ転用するのが難しい
- 多くのモデルが直近の短い期間の行動履歴しか使えていない

この課題に対して Netflix は、NLP における LLM のパラダイムシフトに着想を得て、ユーザ嗜好の学習を一元化する基盤モデルを構築しました。NLP からの知見は大きく2つです。

1. **データ中心のアプローチ**: 特徴量エンジニアリングに依存するモデル中心の戦略から、大規模かつ高品質なデータを活用する方向へ
2. **半教師あり学習の活用**: LLM の Next-Token Prediction のように、ラベルのないデータを用いた大規模な事前学習

この記事では、3本のブログ記事の内容より、Netflix の基盤モデルのアーキテクチャ、スケーリングと効率化、3つの統合パターン、そしてコールドスタートへの対応についてまとめていきます。


## 基盤モデルのアーキテクチャ

基盤モデルの全体像は以下の通りです。

![基盤モデルのアーキテクチャ](/images/netflix-foundation-model-for-recommendations/model_arch.png)
*基盤モデルのアーキテクチャ*

### ユーザインタラクションのトークン化

Netflix では2024年末時点で3億人以上のユーザがおり、ブラウジングから映画視聴まで幅広いインタラクションが発生しています。これは数千億件に相当し、LLM のトークン量に匹敵する規模です。しかし LLM と同様に、量よりも質が重要であるため、**インタラクションのトークン化**によって意味のあるイベントを特定し、冗長性を最低限に抑えています。

トークン化のプロセスは、NLP における Byte Pair Encoding (BPE) に類似しています。隣接するアクションを結合して、新しい高レベルのトークンを形成するイメージです。たとえば、同一タイトルに対する複数のアクション（再生開始、一時停止、再開など）を1つのトークンにまとめ、視聴時間の合計やインタラクションの種類といった重要な情報は保持します。

![ユーザインタラクション履歴のトークン化](/images/netflix-foundation-model-for-recommendations/tokenization_of_user_interaction_history.webp)
*ユーザインタラクション履歴のトークン化*


### 長系列への対応

トークン化で圧縮したとしても、アクティブユーザのインタラクション系列は数千件を超えることがあり、標準的な Self Attention では処理できません。レコメンドシステムでは、ミリ秒単位のレイテンシが要求されるため、LLM のように数秒の推論時間は許容されません。

この課題に対して、Netflix は3つの解決策を導入しています。

- **Sparse Attention**: 低 rank 圧縮などの手法を活用し、計算効率を維持しつつ Context Window を数百件にまで拡張しています。これにより、長期的な嗜好を捉えることが可能になります。
- **Sliding Window Sampling**: 学習時に、系列全体から重なり合う Window をサンプリングします。モデルは複数の Epoch にわたってユーザインタラクション系列の異なるセグメントを学習するため、巨大な Context Window を必要とせずに系列全体から効率的に学習できます。
- **KV Cache**: 推論時に過去の計算結果（Key / Value）を再利用することで、長い履歴に対しても低レイテンシでの推論を実現しています。

### 目的関数の設計

基本的なアプローチは GPT と同様の自己回帰型 Next-Token Prediction (NTP) ですが、言語タスクとレコメンドタスクの違いを踏まえて、いくつかの重要な修正が加えられています。

まず、LLM の事前学習ではすべてのターゲットトークンが同等の weight で扱われますが、レコメンドにおいてはすべてのユーザインタラクションが同等の重要性を持つわけではありません。たとえば、5分間の予告編の再生と2時間の映画の視聴では、ユーザの嗜好に対する情報量がまったく異なります。

この課題に対して、以下の工夫が導入されています。

#### Multi-Token Prediction（MTP）

直後の1トークンだけでなく、次の $k$ トークンを予測する目的関数に拡張しています。これにより長期的な依存関係を捉えられるようになり、直後のインタラクションのみに焦点をあてるような近視眼的な予測を避けられます。

MTP の設計については、NTP が抱える実運用上の2つの課題が指摘されています。

1つ目は**レイテンシによるミスマッチ**です。実運用では推論結果がキャッシュされ、数分から数時間の遅延を経てユーザに提供されます。たとえば、ユーザの視聴履歴に基づいてタイトル A を推薦しても、キャッシュの遅延により、ユーザがその推薦を受け取る頃にはすでに A を視聴済みで、実際に必要なのはタイトル B の推薦である、という状況が発生します。

![レイテンシ条件別の MRR 劣化](/images/netflix-foundation-model-for-recommendations/mrr_degration.webp)
*レイテンシ条件別の MRR 劣化*

実際に、レイテンシ条件別の性能劣化を計測した結果、長期的なタスクはレイテンシの影響を受けにくい一方、短期的なタスクでは大幅な性能劣化が確認されています。

2つ目は**順列不変性**です。レコメンドでは、言語タスクと異なり、次のアイテムの順序が重要でないケースが多くあります。ユーザが A と B の両方を次に視聴する可能性がある場合、A → B でも B → A でも問題ありません。しかし、標準的な NTP では単一の順序にコミットするため、同等に妥当な他の系列にペナルティが課されてしまいます。

MTP ではこれらの課題に対処するため、時間窓内の将来アイテムをマルチラベル集合として予測し、指数的時間減衰で重み付けします。損失関数は以下のように定義されています。

$$
\mathcal{L} = \sum_{\text{label}} \text{Reward}(\text{label}) \cdot e^{-\beta \cdot \Delta t} \cdot \text{CrossEntropy}(\text{label})
$$

ここで $\text{Reward}(\text{label})$ は視聴時間や完了率、新規性などの効用シグナルを表し、$\beta$ はキャッシュの更新間隔に合わせて調整可能な半減期パラメータです。単一のデコーディングヘッドを維持しつつ、カタログ全体に対する logits を1パスで計算し、ラベル集合に対する重み付き cross-entropy で最適化します。

![MTP vs NTP の比較](/images/netflix-foundation-model-for-recommendations/mtp_comparison_ntp.webp)
*MTP vs NTP の性能比較*

MTP と NTP の比較では、高レイテンシ環境（48時間の遅延）ですべてのタスクにおいて大幅な改善が確認されています。オンライン環境（p95 レイテンシ < 1秒）でも、順列不変性の高いタスクでは改善が見られました。ただし、短期的な依存関係が強いタスクでは、イベントの順序が重要であるため、わずかな性能低下が生じる場合もあります。

#### 補助的な目的関数

元の系列に含まれるアイテムからジャンルを導出し、ジャンル系列の予測を補助的な目的関数として使用しています。アイテム ID の予測だけだとノイズが多くなるため、正則化効果によってこれを抑制できます。さらに、ユーザの意図や長期的なジャンル嗜好を捉えることで、結果としてアイテム ID の予測精度も向上します。


## スケーリングと効率化

### スケーリング則

Netflix では 50M パラメータから 1B パラメータへのスケーリング実験を行っています。LLM で知られるスケーリング則がレコメンド基盤モデルにも成り立つかを検証した結果、類似の傾向が確認されましたが、重要な違いとして**切片項 $P_0$** の存在が明らかになりました。

$$
P = P_0 - aN^{-b}
$$

ここで $N$ はモデルパラメータ数、$P_0$ は各タスクにおける達成可能な性能上限を表します。LLM ではモデルサイズを大きくすれば損失は理論上ゼロに近づきますが、レコメンドタスクではユーザ行動に本質的な予測不可能性があるため、性能には上限が存在します。

たとえば、あるタスクでは $P_0 = 0.311$ であり、どれだけスケールしても MRR（Mean Reciprocal Rank）が 0.311 を超えることはないことを示しています。一方、別のタスクでは $P_0 = 1.075$ であり、十分にスケールすれば理論上ほぼ完璧な精度に到達できることを示しています（MRR の上限は1なので、1を超える部分は推定誤差です）。

この切片項 $P_0$ はタスクの難易度を定量化する指標として有用であり、リソース配分の判断にも活用できます。

![タスク別のスケーリング則](/images/netflix-foundation-model-for-recommendations/scaling_law.webp)
*タスク別の MRR スケーリング則*

:::message
評価指標には MRR を使用しています。Hit Rate や NDCG などの非連続的な指標ではモデル間の差が出にくいため、漸進的な改善を捉えやすい連続的な指標として MRR が選ばれています。
:::

### 学習・推論の効率化

レコメンドシステムにおける効率化の重要性は、LLM のそれとは性質が異なります。LLM は一度学習すれば長期間使い続けられますが、レコメンドモデルはユーザ嗜好やコンテンツカタログの変化に追従するために頻繁な再学習が必要です。Netflix のカタログサイズは GPT-3 の語彙の約40倍にあたり、2兆トークンを定期的に処理する必要があります。

特にデコーディングのコストが問題になります。LLM の語彙サイズは約10万トークンですが、レコメンドシステムでは数百万から数十億のエンティティを扱う必要があり、語彙サイズの増大に伴って学習 FLOPs が急増します。

この課題に対して、以下の2つの手法が導入されています。

- **Sampled Softmax**: 学習時に語彙全体ではなく、サンプリングしたサブセットに対してのみ logits を計算します。これにより、実効的な語彙サイズを削減できます。
- **Projected Head**: 出力次元を縮小する射影層（活性化関数 + 線形射影）を追加し、penultimate layer の出力をより小さな次元に変換してから logits を計算します。

![語彙サイズと Training FLOPs の関係](/images/netflix-foundation-model-for-recommendations/flops.webp)
*語彙サイズと Training FLOPs の関係*

この2つを組み合わせることで、学習コストを1〜2桁削減できることが報告されています。現行のインフラでは、80台の A100 GPU（80GB）で240時間/サイクルの学習を定期的に実施し、頻繁な fine-tuning も行っています。

### Embedding の安定化

基盤モデルを再学習するたびに、ランダム初期化の影響で embedding 空間がまったく異なるものになります。また、日次の fine-tuning でも、前日のモデルから warm-start しているにもかかわらず embedding はドリフトしていきます。

下流モデルが基盤モデルの embedding を特徴量として消費する場合、再学習のたびに互換性が崩れるのは運用上大きな問題です。この課題に対して、Netflix は**直交低 rank 変換**を適用し、毎日生成される embedding を同一の embedding 空間にマッピングしています[^4]。これにより、embedding の次元の意味が一貫して保たれ、下流モデルが再学習なしで embedding を消費し続けられるようになっています。


## 3つの統合パターン

基盤モデルをどのように下流タスクへ組み込むかについて、Netflix は3つのアプローチを実験し、いずれも本番環境で運用しています。重要なのは、この3つは単なる代替案ではなく、**制約に応じて選べる統合パターン**として設計されている点です。チームごとにレイテンシ要件、技術スタックの制約、基盤モデルの活用度合いが異なるため、それぞれに適したパターンを選択できるようになっています。

### Embeddings パターン

基盤モデルからユーザ embedding（最後のユーザインタラクションの hidden state）とアイテム embedding（アイテムタワーの重み）をバッチ生成し、Embedding Store に保存して各アプリケーションから利用する方式です。

更新サイクルは以下のようになっています。

1. 月次で基盤モデルを事前学習
2. 日次で最新データに対して fine-tuning（新タイトルのエンティティ ID も追加）
3. 日次の fine-tuning 後にバッチ推論で embedding を更新・公開

![Embedding Store 統合フロー](/images/netflix-foundation-model-for-recommendations/embedding_store_integration.webp)
*Embedding Store 統合フロー*

Embedding Store はバージョニングやタイムスタンプの管理を担い、オフライン・オンラインの両方からアクセスできるインターフェースを提供しています。

この方式は導入コストが低く、既存のパイプラインに組み込みやすいのが大きな利点です。一方で、embedding がバッチ更新であるため鮮度に課題があり、リアルタイムな適応が求められるユースケースには向きません。

Netflix はこの課題に対処するため、ニアリアルタイム embedding 生成フレームワークの構築にも投資しており、セッション中のユーザアクションに基づいて embedding を更新できる仕組みを整備しています。

### Subgraph パターン

基盤モデルのデコーダスタックを、下流モデルの計算グラフの中にサブグラフとして組み込む方式です。埋め込みを事前計算して使うのではなく、リクエスト時に基盤モデルのサブグラフを実行して、その出力を ranking などの下流モデルへ渡します。

![Subgraph 統合](/images/netflix-foundation-model-for-recommendations/subgraph_integration.webp)
*Subgraph 統合*

この方式では、基盤モデルの推論と下流モデルの推論の間にタイムラグがないため、最新のユーザ行動を即座に反映できます。また、最終的な embedding だけでなく中間層の表現も活用できるため、アプリケーション固有の改善余地が大きくなります。

一方で、基盤モデルの一部を下流推論に持ち込むため、特徴生成や推論パイプラインの設計が複雑になり、推論コストも増加します。Netflix では、サブグラフをプロファイルごとに1回だけ実行し、リクエスト内の全アイテムで共有することで推論パフォーマンスを最適化しています。

### Fine-tuning パターン

基盤モデルを推薦面固有のデータと目的関数で fine-tune し、その面専用のモデルとして直接利用する方式です。たとえば、「Trending Now」の行では最近のトレンドタイトルへのインタラクションを重視し、別の面では長期的な嗜好を重視する、といった調整が可能です。

![Fine-tuned FM 統合](/images/netflix-foundation-model-for-recommendations/fine_tuned_integration.webp)
*Fine-tuned FM 統合*

全パラメータの fine-tuning だけでなく、一部のレイヤーを凍結したり、異なる出力ヘッドを追加して別の目的関数で最適化したりすることもできます。新しい推薦面を立ち上げる際にも、ゼロからモデルを設計するのではなく、基盤モデルを fine-tune することで強力なベースラインが得られます。

ただし、面ごとに fine-tuned モデルや運用パイプラインが増えるため、保守・運用コストが高くなる点には注意が必要です。

### 3パターンの比較

| パターン | 導入コスト | 鮮度 | 表現力 | 運用負荷 |
|---|---|---|---|---|
| Embeddings | 低い | バッチ更新のため低い | 最終 embedding のみ | 低い |
| Subgraph | 中程度 | リアルタイム | 中間層も活用可能 | 中程度 |
| Fine-tuning | 高い | 目的に応じて調整可能 | 最も高い | 高い |


## コールドスタートへの対応

Netflix では新しいタイトルが頻繁にカタログに追加されるため、まだ誰もエンゲージしていない新タイトルに対してもユーザの嗜好を推定する必要があります。このコールドスタート問題に対して、複数のアプローチが組み合わされています。

### Incremental Training

基盤モデルは全ユーザの行動履歴を使って学習するため、頻繁にゼロから再学習するのは現実的ではありません。そこで、前回のモデルのパラメータを再利用し、新タイトルのパラメータのみを初期化する warm-start 方式を採用しています。新タイトルの embedding は、既存の平均 embedding にわずかなノイズを加えたり、メタデータが類似するタイトルの embedding を加重平均したりして初期化されます。

### メタデータ Embedding の活用

タイトルにはジャンル、ストーリーライン、トーンなど多様なメタデータが紐づいています。各メタデータの embedding を平均し、それらを結合してメタデータベースの embedding を構成します。最終的なタイトル embedding は、この metadata embedding と学習可能な ID embedding を混合レイヤーで結合して生成されます。

ここで重要なのは、単純に加算するのではなく、エンティティの**経過時間**（公開からどれだけ経ったか）に基づく Attention 機構を使って混合する点です。インタラクションデータが少ない新タイトルではメタデータの重みを大きくし、十分なデータが蓄積されたタイトルでは ID embedding の重みを大きくします。

![メタデータ Embedding の構成](/images/netflix-foundation-model-for-recommendations/metadata_embedding.webp)
*メタデータ Embedding の構成*

さらに、学習時にランダムにID embedding をマスクすることで、メタデータのみから推論する能力を強化しています。

### Multi-modal Semantic Tower

さらに発展した取り組みとして、視覚（ボックスアート）、言語（あらすじ）、ナレッジグラフの特徴量を統合する **Multi-modal Semantic Tower** が導入されています。

![Semantic Item Tower](/images/netflix-foundation-model-for-recommendations/semantic_item_tower.webp)
*Semantic Item Tower のアーキテクチャ*

学習可能な embedding とセマンティックタワーからの出力を加重和で統合し、encoding と decoding の両方にセマンティック情報を注入します。学習時には、collaborative embedding（学習可能な embedding）をランダムに OOV embedding でマスクすることで、セマンティック情報のみからアイテムの特性を推論する能力をモデルに獲得させています。マスクの確率は、学習とサービング間の時間差で発生するコールドスタートアイテムの出現確率に合わせて設定されています。


## おわりに

Netflix の基盤モデルに関する3本のブログ記事を通して、レコメンドにおける大規模 Transformer 活用の実践的なテクニックを学ぶことができました。

まとめると、以下の通りです。

1. ユーザインタラクションのトークン化により、大規模かつ多様な行動データを統一的に扱える
2. Sparse Attention と Sliding Window Sampling により、長い行動履歴を効率的に学習できる
3. Multi-Token Prediction と補助目的関数により、ノイズの多いレコメンドタスクに適した学習が可能になる
4. LLM と同様のスケーリング則が確認されたが、タスク固有の性能上限（切片項 $P_0$）が存在する
5. 3つの統合パターン（Embeddings / Subgraph / Fine-tuning）は代替案ではなく、制約に応じた使い分けとして設計されている

個人的には、Sparse Attention、Sliding Window Sampling、Multi-Token Prediction あたりのテクニックは実務でも試してみたいと思いました。特に Sliding Window Sampling は、長い行動履歴を持つアクティブユーザへの対応として実装しやすそうです。

スケーリング則に切片項が存在するのは面白い結果だと思います。ユーザ行動には本質的な予測不可能性があり、どれだけモデルを大きくしても超えられない壁がある、というのは直感的にも納得感があります。

また、Netflix は映像コンテンツのみを提供しているため、Cross-Domain（たとえば EC と動画を横断するような推薦）については考慮する必要がありません。マルチドメインのプラットフォームで同様の基盤モデルを構築する場合は、ドメイン間の表現統合が必要になると思われるので、このあたりは別途調査していきたいです。



[^1]: Foundation Model for Personalized Recommendation: https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39
[^2]: Integrating Netflix's Foundation Model into Personalization applications: https://netflixtechblog.medium.com/integrating-netflixs-foundation-model-into-personalization-applications-cf176b5860eb
[^3]: Towards Generalizable and Efficient Large-Scale Generative Recommenders: https://netflixtechblog.medium.com/towards-generalizable-and-efficient-large-scale-generative-recommenders-a7db648aa257
[^4]: Orthogonal Low Rank Embedding Stabilization: https://dl.acm.org/doi/full/10.1145/3705328.3748141

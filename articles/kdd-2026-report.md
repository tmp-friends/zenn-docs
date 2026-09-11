---
title: "KDD 2026 参加レポート: Generative Recommendation の最新動向"
emoji: "🏝️"
type: "tech"
topics: ["AI", "機械学習", "レコメンド", "llm"]
published: true
publication_name: "dmmdata"
---

## はじめに

DMM.com のデータサイエンス&AIグループで、レコメンドシステムの開発を担当している寺井です。

2026/8/9 ~ 2026/8/13 にて、韓国・済州島で開催された KDD 2026 (The 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining)[^1] に参加してきました。

今回の KDD では次の2つのテーマに注目して聴講しました。

- Generative Recommendation を中心としたレコメンドシステムの動向
- AI Agent × Information Retrieval / Data Science

本記事では特に印象に残った Generative Recommendation を中心に、Tutorial で整理されていた基本的な考え方と、本会議で印象に残ったその周辺の研究について紹介します。

## KDD 2026 の概要

KDD はデータマイニング、機械学習、レコメンド、検索、データサイエンスに関する研究を幅広く扱う国際会議です。
研究機関による最新の研究成果だけでなく、企業における大規模な機械学習システムの実運用事例も数多く発表されることが特徴です。

KDD 2026 は、2026/8/9 ~ 2026/8/13 にて、韓国・済州島の International Convention Center Jeju (ICC Jeju) で開催されました。
済州島は「アジアのハワイ」とも呼ばれるリゾート地で、海や山など豊かな自然に囲まれています。日本からも直行便で約2〜3時間とアクセスしやすいです。

前半2日間（8月9日, 8月10日）は Workshop や Tutorial が中心に開催され、後半3日間（8月11日〜8月13日）は Keynote、Research Track、Applied Data Science（ADS）Track などの本会議セッションが行われました。

![ICC Jeju](/images/kdd-2026-report/icc_jeju.jpg)
*ICC Jeju*

## 印象に残った発表
### Tutorial on Generative Recommendation: Foundations and Frontiers [^2]
#### 概要

従来のレコメンドシステムでは、Retrieval、Ranking、Reranking を組み合わせた多段構成が一般的です。大規模なカタログを低レイテンシで扱える一方、前段で落とした候補を後段で拾えないことや、各ステージが個別に最適化されることで、システム全体の目的とずれが生じることがあります。

Generative Recommendation では、ユーザの行動系列などを入力として、次に推薦するアイテムを生成する問題として捉えます。これにより、レコメンドタスクをより End-to-End に最適化しやすくなるほか、Transformer を中心としたアーキテクチャによって、LLM で発展してきた Scaling の知見を取り込みやすくなることが期待されています。
![Generative Recommendation へのパラダイムシフト](/images/kdd-2026-report/kdd_genrec_tutorial.png)
*Generative Recommendation へのパラダイムシフト*

今回の Tutorial では、Generative Recommendation の設計を次の3つの観点から整理していました。

- Tokenization
- Architecture
- Optimization

まず Tokenization では、Item ID をそのままトークンとして扱う方法から、アイテムの意味的な情報を反映した Semantic ID まで、さまざまな表現方法が紹介されていました。

実サービスではカタログが数百万〜数千万アイテム規模になることがあり、Item ID をそのままトークンとして扱うと Vocabulary Size もカタログサイズに応じて大きくなります。一方、一般的な LLM の Vocabulary は数万〜十数万程度です。

そこで用いられる方法の一つが Semantic ID です。1アイテムを1トークンで表現するのではなく、複数の小さな符号の組み合わせとして表現します。

```text
Item A -> <12> <48> <05>
Item B -> <12> <48> <19>
Item C -> <37> <04> <82>
```

例えば Item A と Item B は一部の符号を共有しており、意味的に近いアイテムであることを表現できます。小さな Vocabulary の組み合わせで大きなカタログを表現できるため、Vocabulary Size を抑えながらアイテム間の関係もモデルに取り込める点が特徴です。

Architecture では、Encoder–Decoder、Decoder-only、Diffusion-based など複数の構成が紹介されていました。特に Decoder-only は LLM と近い構成で、ユーザの行動履歴とアイテムを一つの系列として扱えるため、LLM で蓄積されてきたモデル構造や Scaling の知見を活用しやすい点が特徴です。

Optimization については、ユーザの行動系列から次のアイテムを予測する Next-token Prediction が基本になります。さらに近年は、過去の行動を再現するだけでなく、複数のユーザ選好やサービス側の目的へ直接 Align するため、Reinforcement Learning を用いた Preference Alignment へと発展していることが紹介されていました。

#### 感想・考察

特に面白いと感じたのは、LLM で見られてきた Scaling の考え方を、レコメンドモデルにも適用しようとしている点です。計算量を大規模化することで推薦精度をどこまで伸ばせるのかは、Generative Recommendation の今後の発展を考える上で興味深いテーマだと感じました。

また、実際に取り入れていくとしたら、既存の多段構成をすべて生成モデルへ置き換えることが必ずしも出発点ではなく、Semantic ID を用いた Retrieval や Generative Reranking など、一部のコンポーネントから段階的に検証するアプローチも現実的だと感じました。

この Tutorial を通して、Generative Recommendation はレコメンドモデルを Transformer に置き換える技術というより、アイテム表現から最適化まで含めてレコメンドシステム全体を再設計するアプローチなのだと理解できました。

### PinRec: Unified Generative Retrieval for Pinterest Recommender Systems [^3]
#### 概要

PinRec は、Pinterest の Home Feed、Search、Related Pins といった複数の面 (Surface) を、一つの Generative Retrieval の枠組みで扱う研究です。

従来の大規模なレコメンドシステムでは、Two-Tower などの Retrieval Model を用いてユーザとアイテムを Embedding 空間へ写像し、近傍探索によって推薦候補を取得する構成が広く使われています。

一方 PinRec では、ユーザの行動を系列として Transformer に入力し、「次にユーザが興味を持つアイテム」を表す Embedding を自己回帰的に生成します。生成された Embedding を Query として既存の ANN Index を検索することで、最終的な推薦候補を取得します。
Generative Recommendation でありながら、既存の Embedding / ANN ベースの Retrieval Infrastructure をそのまま利用できる点が特徴です。

PinRec の主なポイントは、以下の3つです。

1. 複数 Surface を横断した Pre-training と Surface-specific Fine-tuning
Home Feed、Search、Related Pins など、Pinterest 内の複数 Surface で発生したユーザ行動を時系列にまとめ、Next-item Prediction によって共通モデルを Pre-training します。
その後、各 Surface の Impression Log を利用して Fine-tuning します。これにより、複数 Surface のデータからユーザの興味を共通して学習しつつ、Home Feed と Search のように異なる推薦面へ適応します。
Fine-tuning では、実際にユーザが Action したアイテムだけでなく、表示されたものの Action されなかったアイテムも Negative として利用します。
Surface-specific Fine-tuning によって、オフラインの Recall@10 は 2.0〜4.5% 改善しています。
![PinRec Training-phase](/images/kdd-2026-report/kdd_pinrec_training.png)
*PinRec Training-phase*

2. Outcome-Conditioned Generation
推薦 Surface によって、期待するユーザ行動は異なります。例えば Home Feed では Save、Search では商品サイトへの Outbound Click など、重視したい Outcome が異なる場合があります。
PinRec では、「どの Outcome を期待するか」を Conditioning Signal としてモデルへ入力します。
これにより、同じユーザの行動履歴に対しても、「Save されやすい候補」や「Outbound Click されやすい候補」のように、目的に応じて異なる推薦候補を生成できます。
さらに推論時には、それぞれの Outcome に割り当てる Retrieval Budget を変更できます。モデルを再学習することなく、各 Outcome から取得する候補数を調整することで、複数の事業指標のバランスを制御できます。
![PinRec Inference-phase](/images/kdd-2026-report/kdd_pinrec_inference.png)
*PinRec Inference-phase*

3. Dense Embedding の自己回帰生成
TIGER[^4] などの Generative Recommendation では、アイテムを Semantic ID と呼ばれる離散 Token へ変換し、その Token Sequence を生成する方法が提案されています。
一方 PinRec では Semantic ID を生成するのではなく、連続値の Item Embedding を直接生成します。
Pinterest のような非常に大きな Item Catalog では、異なるアイテムが同じ Semantic ID に割り当てられる衝突が増え、情報が失われる Representational Collapse が問題になると報告されています。
そこで PinRec では、Transformer が生成した Dense Embedding を Query として Faiss による ANN 検索を行い、実在するアイテムを取得します。
この構成では Item ID 自体を生成しないため、存在しないアイテムを生成する Hallucination の問題も避けられます。

工夫として挙げられるのが、PinRec が「次の1アイテム」だけを予測するのではなく、複数の Item Embedding を自己回帰的に生成している点です。
一つ前に生成した Embedding を次の生成ステップへ入力しながら候補を生成することで、同じユーザ表現から独立に候補を取得するのではなく、それまでに生成した候補を考慮しながら次の候補を生成します。
1-step Generation と比較したオフライン実験では、16-step の自己回帰生成によって Related Pins の Recall が 71.4% 改善しています。

また、実運用を意識した Serving についても検証されています。
CUDA Graph や KV Cache を利用して自己回帰生成を高速化しており、Outcome-Conditioned PinRec のレイテンシは 80 QPS 時に p50 40ms、p90 65ms と報告されています。
Two-Tower と比較するとモデル単体の推論コストは増加しますが、既存の Retrieval Source と並列実行することで、End-to-End のレイテンシ増加は 1% 未満に抑えています。

オンライン A/B テストでも複数の Surface で改善が確認されています。Search では Search Fulfillment Rate が 2.24%、Save が 3.88%、Share が 5.30% 改善しました。Home Feed では Grid Click が 4.01% 増加し、Related Pins でも Fulfilled Sessions や Time Spent などが改善しています。

#### 感想・考察

Generative Recommendation というと Semantic ID の生成を想起しやすいですが、PinRec が生成するのは従来の Retrieval System でも使われてきた Dense Embedding です。つまり、Transformer でユーザ行動系列をモデリングし、Embedding + ANN で検索します。
この構成なら、既存の ANN Index や後段の Ranking System を維持したまま、候補生成のみを Generative Model に置き換えられるため、実サービスにも導入しやすいと感じました。

また、複数 Surface のデータをまとめて Pre-training した後に、Surface ごとのデータで Fine-tuning する構成も参考になりました。
「複数の推薦面に共通するユーザの興味を学習する部分」と「各推薦面の目的へ適応する部分」を分けることで、横断的なデータ活用と Surface 固有の最適化を両立しています。

さらに Outcome Conditioning によって、Save や Click といった目的ごとに異なる候補を生成し、推論時の Retrieval Budget によってそのバランスを調整できる点も、重要だと思います。

PinRec を見ると、Generative Recommendation の価値は単に Two-Tower より高い Recall を実現することだけではなく、長いユーザ行動系列、推薦 Surface、最適化したい Outcome といった複数の情報を一つの Sequence Model に取り込み、候補生成自体を柔軟に制御できることにあると感じました。

### OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System [^5]

#### 概要

OnePiece は、Shopee の検索システムを対象に、LLM で発展してきた Context Engineering と Reasoning の考え方を、既存の大規模な多段 Ranking System に取り入れた研究です。

一般的な Transformer ベースのレコメンドモデルでは、モデルアーキテクチャの改善に注目が集まりがちです。一方 OnePiece では、LLM の性能を支えている要素を「どのような Context を与えるか」「その Context をどのように段階的に処理するか」という観点から捉え、既存の Retrieval / Ranking の構成を維持したままモデルを拡張しています。

提案手法は、大きく以下の3つから構成されます。

1. Context Engineering
ユーザの行動履歴である Interaction History に加え、Preference Anchors と Situational Descriptors を Context として入力します。
Preference Anchors は、現在のユーザや検索クエリに関連する「よくクリックされる商品」「よく購入される商品」などを補助的なアイテム系列として与える仕組みです。Situational Descriptors にはユーザ属性や検索クエリなど、その時点の状況を表す情報が含まれます。
Ranking ではさらに複数の候補アイテムをまとめて入力し、候補同士を見比べながらスコアリングできるようにしています。

2. Block-wise Latent Reasoning
Transformer を複数の Reasoning Block に分割し、前段で得られた内部表現を次の Block へ渡しながら、予測に必要な表現を段階的に更新します。
ここでいう Reasoning は、LLM の Chain-of-Thought のように自然言語の思考過程を生成するものではなく、モデル内部の潜在表現上で行われる Reasoning です。

3. Progressive Multi-task Training
ユーザ行動を Click → Add-to-Cart → Order のような Funnel として捉え、前段の Block では量の多い弱いシグナルを、後段ではより強いシグナルを学習します。これにより、Block を進むにつれてより購買意図の強い行動を考慮した表現へ更新していきます。

![OnePiece Overview](/images/kdd-2026-report/kdd_onepiece.png)
*OnePiece Overview*

Shopee でのオンライン A/B テストでは、Retrieval に OnePiece を導入した場合にユーザあたりの GMV が 1.08% 改善しました。また Ranking への導入では、ユーザあたりの GMV が 1.12%、広告収益が 2.90% 改善したと報告されています。

#### 感想・考察

本論文で特に印象的だったのは、モデルそのものを大きく変更するのではなく、「モデルへ何を Context として与えるか」を性能改善の主要な論点として扱っていた点です。

ユーザの行動履歴だけでは、その時点で何を探しているのかを十分に表現できない場合があります。検索クエリや現在のセッション、候補アイテム、周囲で人気のアイテムなどを Context として組み合わせることで、ユーザの長期的な嗜好と、その時の意図の双方を捉えられる可能性があります。

Generative Recommendation においても、モデルサイズや学習データを Scaling するだけでなく、「推論時にどの情報を Context として構成するか」は重要な論点になると感じました。

一方、利用する Context を増やすほど、特徴量生成やオンライン Serving は複雑になります。OnePiece のような仕組みを実サービスへ取り入れる場合には、Context を増やすこと自体を目的とせず、それぞれの Context がどの程度性能へ寄与しているのかを Ablation しながら選択していくことが重要だと思います。

### Sharpness-aware Model Merging with Salience Recovery for LLM-based Cross-Domain Sequential Recommendation [^6]

#### 概要

SharpRec は、ドメインごとに学習した LLM ベースのレコメンドモデルを Model Merging によって統合し、Cross-domain Sequential Recommendation を実現する研究です。

共通の LLM Backbone を固定したうえで、Book、Movie、Sports などの各ドメインについて個別に LoRA Adapter を学習し、最後にそれらの Adapter Parameter を統合します。この構成であれば、すべてのドメインのデータをまとめて再学習することなく、ドメインごとに獲得した知識を組み合わせられます。

一方、著者らは単純な Model Merging には大きく2つの課題があることを示しています。

- Negative Transfer
  性質の異なるドメインの Parameter を統合すると互いの知識が干渉し、単一ドメインのモデルより性能が低下する場合があります。特に類似度の低いドメインでは、統合 Weight を調整するだけでは解消できないケースが確認されています。

- Performance Saturation
  統合するドメインを増やしても性能向上が途中で頭打ちになります。分析では、複数モデルの Parameter を線形に平均することで、各ドメインで重要だった特徴的な Parameter が平滑化され、Parameter Distribution が均質化してしまうことが原因の一つとして示されています。

![Model Merging Challenges](/images/kdd-2026-report/kdd_sharprec_model_merging_challenges.png)
*Model Merging Challenges*

これらに対して SharpRec では、以下の2つの仕組みを導入しています。

1. Sharpness-aware Geometric Alignment（SGA）
   ドメインごとの LoRA を学習する際、Parameter が多少変化しても Loss が大きく悪化しない平坦な領域を探索します。Model Merging は異なるモデルの Parameter 間を補間する操作とみなせるため、それぞれを平坦で互いに統合しやすい領域へ学習することで、統合時の Parameter Interference を抑えます。

2. Preference Salience Activation（PSA）
   Model Merging 後に、平均化によって弱まった Parameter の特徴を再び強調します。単純な平均化によって Gaussian に近づいた Parameter Distribution を Heavy-tailed な分布へ変換することで、各ドメインに固有の重要な信号を復元します。

Amazon Review 2023 の7ドメインを利用した実験では、Book ↔ Movie、Kitchen ↔ Food、Sports ↔ Toy などの Cross-domain Recommendation において既存手法を上回りました。また既存の Model Merging 手法では3〜4ドメイン程度から性能が飽和するのに対し、SharpRec は統合するドメインを増やした場合にも継続的な改善を示しています。

#### 感想・考察

複数のサービスやドメインを横断してレコメンドモデルを構築する場合、利用できるデータを増やせば増やすほど性能が上がるとは限らないことを改めて示している研究でした。

ドメインによってユーザ行動の意味、アイテムの分布、推薦時に重視すべき目的は異なります。そのため、一方のドメインで有効だった知識が、別のドメインでもそのまま有効とは限りません。

Generative Recommendation や Recommendation Foundation Model のように、より多くのサービス・ドメイン・タスクを一つのモデルへ集約する方向へ進むほど、「どの知識を共有し、どの知識をドメイン固有として残すのか」は重要な設計要素になると思います。

また、Model Merging はドメインごとのモデルを独立して更新できるという運用上のメリットもあります。一方で、統合後の平均性能だけを確認すると、一部のドメインで生じている Negative Transfer を見落とす可能性があります。Cross-domain Model を評価する際には、全体指標だけでなくドメインごとの改善・悪化を確認することが重要だと感じました。

### A Unified Model for Personalization: Language-Steerable Generative Recommendation, Search, and User Understanding [^7]

#### 概要

Spotify による、レコメンド・検索・ユーザ理解を単一の Generative Model で扱う研究です。

NEO と呼ばれるモデルを用いて、行動履歴からのアイテム推薦だけでなく、自然言語による検索、推薦理由の生成、ユーザの興味の要約など、従来は別々のシステムで扱われていたタスクを共通のモデルで実行します。

NEO の中心となるのが、自然言語とアイテムを同じ Token Sequence として扱う仕組みです。

アイテムは、コンテンツ Embedding を量子化して得られる Semantic ID を用いて Token Sequence に変換します。これにより、

- ユーザの行動履歴を表す Semantic ID
- ユーザからの自然言語による指示
- モデルが出力する Semantic ID や自然言語

を、一つの Decoder-only Language Model 上で統一的に扱えるようになります。

![NEO Overview](/images/kdd-2026-report/kdd_neo.png)
*NEO Overview*

学習は主に3段階で行われます。

1. Semantic Foundation
   アイテムのコンテンツ Embedding を量子化し、各アイテムを Semantic ID として表現します。

2. Domain Grounding
   Semantic ID と自然言語の意味を対応付けます。この段階では Language Model 本体を固定し、Semantic ID 用に追加した Embedding を中心に学習することで、元の言語能力を維持しながらアイテムを Language Model の語彙へ取り込みます。

3. Capability Induction
   Recommendation、Natural-language Retrieval、推薦理由の生成、ユーザ興味の Profiling などを Instruction Tuning し、一つのモデルから複数の Personalization Task を実行できるようにします。

![NEO training-phase](/images/kdd-2026-report/kdd_neo_training.png)
*NEO Training-phase*

アイテムを生成するときには Prefix Trie を用いた Constrained Decoding を行い、Spotify のカタログに実在する Semantic ID のみを出力できるようにしています。一方、推薦理由などの自然言語部分については通常の Language Model と同様に自由に生成できます。

1,000万件以上のアイテムを対象としたオフライン評価では、単一の Multi-task NEO が複数の Task-specific Baseline を上回りました。例えば Episode Recommendation では HR@10 が 58%、NDCG@10 が 80%、Audiobook Recommendation では HR@10 が 46%、NDCG@10 が 97% 改善しています。

#### 感想・考察

今回紹介する研究の中でも、Generative Recommendation が単なる「次のアイテムを生成するモデル」から、どのように機能を広げていくのかをイメージしやすい研究でした。

特に興味深かったのは、過去の行動から推定する嗜好と、自然言語でユーザが明示する意図を同じモデルへ入力できる点です。例えば普段の視聴傾向とは異なるコンテンツを一時的に探したい場合でも、「今日はこういうものを聴きたい」と自然言語で補足できれば、行動履歴だけでは捉えにくい一時的な意図を推薦へ反映できます。

また、レコメンドと検索が Semantic ID と自然言語という共通の表現を介してつながっている点も興味深いです。今後 Generative Recommendation が発展すると、レコメンド・検索・ユーザ理解を個別のシステムとして最適化するのではなく、一つの Personalization Model の異なるタスクとして扱う方向へ進む可能性があることがわかりました。

### The Pitfall of Scaling Up: Uncovering and Mitigating Popularity Bias Amplification in Scaling Transformer-based Recommenders [^8]

#### 概要

Transformer ベースのレコメンドモデルを Scaling したとき、推薦精度だけでなく Popularity Bias がどのように変化するかを分析した研究です。

モデルを大規模化すると一般に Recall や NDCG は改善しますが、本研究では、その一方で人気アイテムへの推薦集中が強まる場合があることを示しています。
特に Transformer の Depth を増やした場合、推薦精度が改善する一方でロングテールアイテムの露出が継続的に低下しました。また単純な Parameter 数との関係は単調ではなく、中規模までは Fairness が改善するケースもあるものの、さらに Scaling すると再び Popularity Bias が強まることが確認されています。

![Scaling と Popularity Bias の変化](/images/kdd-2026-report/kdd_pitfall.png)
*Scaling と Popularity Bias の変化*

モデルが深くなるにつれて、予測スコア行列を分解したときの情報が少数の成分に集中し、特定の成分が推薦結果を強く左右するようになります。著者らは、このように予測が少数の方向へ偏る現象を Spectral Collapse と呼んでいます。
さらに、その中で最も支配的な成分がアイテムの人気度と強く対応していることから、Spectral Collapse が Popularity Bias の増幅につながると分析しています。

原因として、Transformer の主要な2つの構成要素が挙げられています。
- Self-Attention では、人気アイテムに Attention が集まりやすく、Layer を重ねることでその偏りが蓄積される
- Feed-forward Network では、Depth が増えるほどロングテールアイテムに対応する特徴を学習しにくくなる

これに対して、著者らは SPRINT という正則化手法を提案しています。
SPRINT では、Self-Attention に対して一部のアイテムへの Attention の集中を抑える制約を加えるとともに、Feed-forward Network の Weight に対して Spectral Norm を抑える制約を加えます。これにより、Scaling による推薦精度の改善を維持しつつ、ロングテールアイテムの露出を確保することを狙います。

MovieLens や Amazon Review など6データセットを利用した実験では、既存の Debiasing 手法と比較して、平均で Accuracy 指標を 15.70%、Fairness 指標を 7.12% 改善しました。また SASRec 系のモデルだけでなく、Semantic ID を生成する TIGER や LETTER[^9] といった Generative Recommendation に対しても同様の改善が確認されています。

#### 感想・考察

Generative Recommendation では、モデルサイズ、Depth、学習データ量、ユーザの行動系列長など、さまざまな方向で Scaling が進んでいくと考えられます。
その際に、「モデルを大きくして NDCG が上がった」という評価だけでは不十分であることを示している点が印象的でした。
精度改善の一部が、すでに人気のあるアイテムをより強く推薦することで得られているのであれば、ユーザごとの嗜好をより正確に捉えられるようになったとは限りません。特に大規模モデルでは、精度指標だけを見ると改善しているため、このような変化を見落としやすい点には注意が必要です。

私たちのチームでも、すでに Transformer Encoder をベースとしたモデルを利用しているため、今後 Scaling していく際には、精度指標の改善だけで判断せず、こうした Popularity Bias の増幅が起きていないかにも気をつけたいと思います。

## KDD 2026 に参加して

KDD 2026 では、Research Track の最新研究だけでなく、ADS Track や Sponsor Invited Talk など、企業による実サービスでの取り組みを数多く聞くことができました。モデルの性能だけでなく、大規模なシステムでどう運用しているのか、どのような課題に直面しているのかまで知ることができ、今後のレコメンドシステムの方向性を考えるうえでも非常に参考になりました。

また、複数のセッションが同時並行で開催されているため、バイキングのような感覚で、その時間帯に気になった発表を選んで聞けるのも KDD の面白さだと感じました。特に今回は、どの時間帯を見てもレコメンド関連のセッションが一つはあるような印象で、その中でも Generative Recommendation に関する発表が非常に多く、この分野への注目度の高さを実感しました。

今回は Generative Recommendation を中心に紹介しましたが、AI Agent に関する発表も印象に残っています。データ分析やモデル開発といった開発者側の業務を効率化するだけでなく、レコメンド・検索といったユーザ向けの体験そのものをどのように変えていくのか、AI Agent の可能性を双方の観点から考えるきっかけになりました。

## おわりに

私たちのチームでは、DMM の多様なサービス・行動データを活用しながら、大規模なレコメンドシステムの研究開発とサービス導入に取り組んでいます。

今回紹介した Generative Recommendation をはじめ、レコメンド、検索、機械学習基盤、生成 AI などのテーマに興味をお持ちの方は、ぜひ DMM のデータサイエンス・AI 領域の取り組みもご覧いただけますと幸いです。

[^1]: KDD 2026: https://www.kdd.org/kdd2026/
[^2]: Tutorial on Generative Recommendation: Foundations and Frontiers: https://applied-machine-learning-lab.github.io/KDD2026_GenRec_Tutorial/
[^3]: PinRec: Unified Generative Retrieval for Pinterest Recommender Systems: https://arxiv.org/abs/2504.10507
[^4]: Recommender Systems with Generative Retrieval: https://arxiv.org/abs/2305.05065
[^5]: OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System: https://arxiv.org/abs/2509.18091
[^6]: Sharpness-aware Model Merging with Salience Recovery for LLM-based Cross-Domain Sequential Recommendation: https://arxiv.org/abs/2607.25366
[^7]: A Unified Model for Personalization: Language-Steerable Generative Recommendation, Search, and User Understanding: https://research.atspotify.com/publications/unified-model-personalization-language-steerable-generative-recommendation-search-user-understanding
[^8]: The Pitfall of Scaling Up: Uncovering and Mitigating Popularity Bias Amplification in Scaling Transformer-based Recommenders: https://arxiv.org/abs/2606.21911
[^9]: Learnable Item Tokenization for Generative Recommendation: https://arxiv.org/abs/2405.07314

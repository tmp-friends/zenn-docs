---
title: "RecSys Challenge 2026 参加記：コンペ参加から論文発表まで"
emoji: "🎵"
type: "tech"
topics: ["AI", "機械学習", "レコメンド", "llm"]
published: false
publication_name: "dmmdata"
---

## はじめに

DMM.com のデータサイエンス&AIグループで、レコメンドシステムの開発を担当している寺井です。

RecSys Challenge 2026[^1] に参加し、総合16位、Industry Track では8位という結果になりました。また、コンペ中に行ったデータセットと評価指標の分析を論文[^2] にまとめ、2026/10/2 に開催された RecSys Challenge Workshop で発表してきました。

https://prtimes.jp/main/html/rd/p/000005136.000002581.html


今年のコンペの題材は、ユーザとの対話を通じて、その時点の要求に合った楽曲を推薦する「対話型音楽推薦」です。楽曲のランキングに加えて、推薦理由を伝える自然言語の応答も生成します。

本記事では、コンペの概要とデータセット、上位解法を紹介したうえで、自身の取り組みと論文での分析内容を紹介します。

## コンペ概要

RecSys Challenge は、レコメンドシステムの国際会議 ACM RecSys に併設されるコンペです。
2026年のタスクでは、各ターンで次の情報を入力として利用し、楽曲ランキングと自然言語の応答文を生成します[^3]。

**入力（利用可能な補助データを含む）**

- 現在のユーザ発話までの会話履歴：ユーザ発話、過去に推薦された楽曲、システムの応答文
- ユーザプロフィール：年齢層・性別・国。一部の評価セッションでは非公開
- 楽曲メタデータ：曲名・アーティスト・アルバム・タグなど
- 楽曲の Embedding：音響・アルバム画像・タグ・歌詞・メタデータ・協調フィルタリングの6種類
- ユーザの Embedding：協調フィルタリングに基づく表現

**出力**

- 47,071曲のカタログから選ぶ Top 20 の楽曲ランキング
- 会話の文脈やユーザの嗜好を踏まえ、推薦理由を説明する自然言語の応答文

ユーザ発話には、曲名を直接指定するものもあれば、「もう少し落ち着いた曲で」のように、直前の推薦を受けて条件を変えるものもあります。固有名詞を正確に照合するだけでなく、対話を通じて変化する意図を捉える必要があります。

評価では、与えられた会話履歴に対して、そのターンのランキングと応答文を生成します。生成した応答にユーザが返答し、その後の会話が変化していくような評価ではなく、各ターンの入力はあらかじめ決まっています。

### 評価指標

最終順位は、以下の4つの指標を重み付きで合計した Composite Score によって決まります。

| 指標 | 重み | 評価内容 |
| --- | ---: | --- |
| nDCG@20 | 50% | 正解曲をランキングの上位に出せているか |
| Catalog Diversity | 10% | 提出全体で、どれだけ多くの異なる曲を推薦しているか |
| Lexical Diversity | 10% | 応答文の表現がどれだけ多様か |
| LLM-as-a-Judge | 30% | 応答文の Personalization と Explanation Quality |

各ターンには1曲の正解曲があり、その曲を上位に出すほど nDCG@20 が高くなります。Catalog Diversity は、カタログ全体に対して、提出したランキングに含まれる異なる曲が占める割合です。

Lexical Diversity には、応答全体に含まれる単語の 2-gram の多様性を測る Distinct-2 が使われます。LLM-as-a-Judge の評点は1〜5で、`(評点 − 1) / 4` によって0〜1へ正規化してから加算されます。

この評価では、曲を推薦する能力と、選んだ曲について応答する能力が一つのスコアにまとめられています。

## データセット

コンペで用いられたのは、TalkPlayData 2 を基にコンペ用に提供された合成対話データセット、TalkPlayData-Challenge です。学習用には15,199セッション、開発用には1,000セッションがあり、それぞれ8回の推薦ターンを含みます。

基になった TalkPlayData 2[^4] では、実際のリスニング履歴や楽曲情報を土台として、役割の異なる LLM が会話を生成します。
![TalkPlayData 2 の生成フロー](/images/recsys-challenge-2026/talkplaydata2_pipeline.png)
*TalkPlayData 2 の生成フロー*

中心となるのは、ユーザ役の Listener LLM と、レコメンドシステム役の Recsys LLM です。
各会話には、ユーザが達成したい Conversation Goal が設定されます。Listener LLM はその目標を知ったうえで発話しますが、Recsys LLM には目標が直接渡されません。Recsys LLM は、対話を通じて要求を理解し、楽曲を選びます。

公開データには、会話履歴とユーザプロフィールに加えて、会話の目標（Conversation Goal）や、目標への進み具合を表すラベル（Goal Progress）も含まれています[^5]。また、補助データとして、楽曲メタデータ、楽曲の Embedding、ユーザの Embedding も提供されています。

:::details 会話履歴データのサンプル

会話の流れが分かるように、公開学習データから最初の2ターンを抜粋します。ここでは、`session_id` と、各発話の `role`・`turn_number`・`content` を示しています。

```json
{
  "session_id": "b9bbaba3-c014-4435-95be-e6b92b239e37",
  "conversations": [
    {
      "role": "user",
      "turn_number": 1,
      "content": "Play 'Madness' by Muse."
    },
    {
      "role": "music",
      "turn_number": 1,
      "content": "b87391bb-0e53-4333-83eb-19d5b9662599"
    },
    {
      "role": "assistant",
      "turn_number": 1,
      "content": "No problem at all! Here's \"Madness\" by Muse for you. Hope you enjoy it!"
    },
    {
      "role": "user",
      "turn_number": 2,
      "content": "Perfect! That's exactly what I wanted. Now, can you play something else by Muse with a similar epic sound?"
    },
    {
      "role": "music",
      "turn_number": 2,
      "content": "11516f27-704f-450a-8039-b36fd1611451"
    },
    {
      "role": "assistant",
      "turn_number": 2,
      "content": "Awesome, glad you liked \"Madness\"! If you're looking for another epic Muse track, you definitely need to hear \"Supremacy.\" It's got that huge, powerful sound you're after."
    }
  ]
}
```

1ターン目のユーザ発話は、「Muse の『Madness』を流して」という、曲名とアーティストを指定した要求です。`role: "music"` の `content` には楽曲 ID が入り、楽曲メタデータと結び付けることで、推薦された曲の情報を確認できます。`role: "assistant"` の `content` は、その曲を案内する自然言語の応答文です。

2ターン目では、ユーザが推薦を受け入れたうえで、「同じような壮大なサウンドの Muse の別の曲」を求めています。それに対して、応答文では『Supremacy』を案内しています。このように、後続ターンには、直前の推薦への反応と追加の条件が含まれます。

:::

## 上位解法

上位チームの構成を、以下の観点に分けて整理します。
- 候補を集める **Retriever**
- 候補を並べ替える **Reranker**
- 自然言語の応答文を作る **Response Generator**
- **その他の工夫**

| チーム・総合順位 | Retriever | Reranker | Response Generator | その他の工夫 |
| --- | --- | --- | --- | --- |
| hallucinated・1位※ | BM25・TF-IDF。<br>Qwen3-Embedding。<br>マルチモーダル Two-Tower。<br>BERT4Rec ベースの系列モデル。 | 重み付き RRF（各 Retriever の上位200件）。<br>XGBoost アンサンブル（38特徴量）。<br>履歴の連続性などに基づく先頭曲の選択。 | Gemma-4-26B-A4B-it（8-bit）。<br>要約→生成→圧縮→多様化。<br>提出ランキングから独立した応答内の選曲。 | SWAG による疑似アンサンブル。<br>初回ターン専用 RRF。<br>評価データの学習利用（下記注記）。[^6] |
| volart・2位 | BM25、text-embedding-3-large、LLM 楽曲説明の密ベクトル Retrieval、共起。<br>LLM によるクエリ化・エンティティ抽出。 | RRF（500候補）。<br>LightGBM LambdaMART（69特徴量）。<br>欠損時も次元固定の特徴量。 | Gemini-3.1-Flash-Lite による上位3曲の説明。<br>独立した gpt-4o-mini の Critic による修正。<br>定型句の削減。 | CPU と外部 API のみの構成。<br>情報の限界の診断に基づく応答品質・語彙多様性の改善。[^7] |
| niwatori・3位 | 語彙・タグ・完全一致・履歴・共起・遷移などの14ソース。<br>LoRA 調整済み Qwen3-Embedding-0.6B の Two-Tower。 | 全体の件数上限なしの候補統合。<br>LightGBM LambdaRank。<br>各 Retriever での出現有無・順位・スコアと会話・楽曲特徴量。 | Qwen3.6-27B による上位3曲の説明。<br>各ケース10候補から、提出全体の Distinct-2 の代理指標に基づく選択。 | 5-fold の OOF 特徴量。<br>正解曲が候補内にあるケースでの学習。<br>Retrieval での候補漏れ・Top 20 除外・順位損失への誤り分解。[^8] |
| swyoo・4位 | BM25＋固定 Qwen3-Embedding-0.6B のハイブリッド Retrieval。<br>QLoRA 調整済み Qwen3-Embedding-8B の会話に基づく Retrieval。 | 各プールの上位100件の統合。<br>LightGBM LambdaRank。<br>プール別順位・スコアとプール間 RRF、ターン・要求特徴量。 | GLM-5.2 による Propose–Assign–Select（PAS）。<br>上位20曲と説明根拠の対応付け。<br>18個の応答例からの選択。 | カタログの96.6%に外部情報を1項目以上補完。<br>セッション単位の5-fold 特徴量。<br>曲名表記・引用ルールの検証。[^9] |
| PoliBaJukebox・5位 | BM25、Qwen3・CF-BPR・CLAP の埋め込み、ALS、調整済み BGE などの10ソース。 | 重み付き RRF（300候補）。<br>LightGBM LambdaRank＋CatBoost YetiRank（37特徴量）。<br>スコア差によるモデル選択と Z スコアブレンド。 | 1位の曲のメタデータに基づく2段階生成。<br>Gemini-3.1-Pro-Preview で下書き。<br>Gemini-3.5-Flash で批評・修正。 | Adversarial Validation による開発・Blind 間の分布差の推定。<br>重要度重み付けによる学習分布の補正。[^10] |

※総合1位の hallucinated について、主催者の総括論文は、**Blind B の会話ターンを重み調整や Reranker の学習に利用した点**を注記しています。非公開の正解曲を利用したという意味ではありませんが、未見の会話への汎化という評価条件が異なるため、性能の単純比較には注意が必要です。

### Retriever

候補生成の段階では、異種候補ソースの組み合わせと、会話の文脈を活かした入力の作り方が参考になりました。上位チームは、語彙・意味・共起・遷移などを手掛かりに、候補を幅広く集めています。会話を入力に使う際には、volart は LLM で複数ターンをクエリにまとめ、swyoo は再生履歴・要約・現在の要求を構造化してエンコーダへ渡しています。

### Reranker

順位付けでは、候補を統合した後も、各 Retriever での順位やスコアを残すことがポイントのようでした。niwatori や swyoo は、これらの情報と候補の出現有無、会話と楽曲の関係を表す特徴量を使って順位付けします。さらに、hallucinated は先頭曲をルールで選び、PoliBaJukebox は上位曲のスコア差に応じてモデルを切り替えるなど、最終順位の決め方も調整しています。

### Response Generator

応答生成では、説明を楽曲の情報に基づかせることに加え、提出全体の表現の多様性も改善の対象になっています。volart・niwatori・PoliBaJukebox は上位曲のメタデータを根拠に応答を生成し、swyoo は PAS で曲と説明根拠を対応付けます。volart は Critic による説明の修正と定型句の削減を行い、niwatori は提出全体の語彙の多様性を考慮して応答候補を選びます。

ランキングと応答内の推薦曲が対応しているかも、確認したい点です。hallucinated は応答内の曲をランキングから独立して選ぶため、両者が一致するとは限りません。

### 評価条件への対応と指標の限界

開発データで得られた改善を最終評価でも再現するために、各チームはデータの欠損や分布の違いに対応しています。volart は、一部の情報が欠けていても特徴量の数が変わらないよう、欠損値をゼロなどで補っています。PoliBaJukebox は、開発データと評価データの分布の違いを考慮し、学習例ごとの重みを調整しています。また、niwatori や swyoo は OOF 特徴量を使い、特徴量を作る際に学習対象の正解情報が混入するのを防いでいます。

スコアが伸び悩む原因を、会話に含まれる情報や評価方法から考える視点も参考になりました。volart は、会話からユーザの好みに合う曲をある程度絞り込めても、その中から正解の1曲を特定するには情報が足りないと考え、応答文の改善に力を入れていました。主催者の総括論文でも、ユーザの要求に合う曲が複数あるにもかかわらず、1曲だけを正解として評価することの限界が指摘されています。

## 自身の取り組みと論文の内容

コンペに取り組んでいたのですが、途中で nDCG の挙動が不自然な挙動をとることに気づきました 🤔
例えば、Qwen-Embedding を会話文で Fine-tuning し、意味的に合う楽曲を候補生成へ加えた際に、出力結果を見て定性的に良くなっていそうなのに、スコアはあまり改善しませんでした。
また、「会話に登場したアーティストの別の曲を推薦する」というロジックが、今回の nDCG 評価において、とても有効そうなことに違和感を覚えました。

本コンペでは、LLM が生成した合成対話データセットを使っており、生成時に Recsys LLM が選んだ曲が正解曲になります。そこで、モデルの改善に加え、この正解曲がユーザの要求や会話の目標に沿っているかを調べることにしました。

具体的には、合成データに記録された正解曲との一致（Item Relevance）、直前の推薦に対する目標への進捗（Goal Progress）、応答文の質（Response Quality）がどう対応しているかを、次の2つの観点から検証しました。

### RQ1: Item Relevance と Goal Progress の整合性

Goal Progress は、Listener LLM が直前の推薦について、会話の目標に近づいたかを評価したラベルです。2ターン目以降のターン `t` に付くラベルは、ターン `t−1` の推薦に対する評価であり、人間の満足度を表すものではありません。

データセットには、別のアーティストを要求しても、同じアーティストの曲が正解になっている会話がありました。次は、その一例です。

| 会話内の位置 | 内容 |
| --- | --- |
| ターン2の推薦曲 | Express Yourself — Madonna |
| ターン3のユーザ発話・一部抜粋 | “… not Madonna, but someone else with cool songs from that time?” |
| ターン3の次の正解曲 | Burning Up — Madonna |

応答文も『Burning Up』を案内しており、曲と説明は対応していますが、「Madonna ではなく別のアーティスト」という要求には従っていません。

この傾向を調べるため、明示的なアーティスト変更要求をルールで抽出し、その直後の正解曲を集計しました。会話の後半ほど既出アーティストが増える影響を抑えるため、会話位置の分布を揃えて比較し、セッション単位の Bootstrap で不確実性も評価しました。

| ケース | 次の正解曲が、会話内の既出アーティストを繰り返す割合 |
| --- | ---: |
| 明示的なアーティスト変更要求あり | 76.0% |
| 変更要求なし | 63.1% |

変更要求があるケースでも、約4分の3の正解曲が既出アーティストの曲でした。

さらに、次の3つの条件をすべて満たすケースに絞って調べました。

- ユーザが、別のアーティストへの変更を明示的に求めている
- 直前の推薦曲のアーティストが、それ以前にも会話内で推薦されている
- Listener LLM が、その直前の推薦を「目標に近づいていない」（`DOES_NOT_MOVE_TOWARD_GOAL`）と評価している

この5,278件のうち、次の正解曲も直前と同じアーティストだったのは4,529件（85.8%）でした。つまり、ユーザが変更を求め、直前の推薦も目標に近づいていないと評価されているケースでも、正解曲は同じアーティストを繰り返すことが多くありました。

上記の分析により、正解曲を当てることと、ユーザの要求や会話の目標に沿うことの食い違いが見えてきました。

### RQ2: ランキングを固定したときの応答文の影響

Blind A の Top 20 の楽曲と順序を固定し、応答文だけを変更して公式評価へ提出しました。

| 応答の構成 | Lexical Diversity | LLM Judge | Composite Score |
| --- | ---: | ---: | ---: |
| 初期応答 | 0.5925 | 4.05 | 0.5086 |
| Qwen 4B・プロンプト修正版 | 0.6605 | 4.20 | 0.5266 |
| Claude・短めの応答 | **0.8133** | 4.30 | 0.5494 |
| Claude・メタデータに基づく長めの応答 | 0.7983 | **4.90** | **0.5929** |
| Gemini Pro・メタデータに基づく応答 | 0.7204 | 4.30 | 0.5401 |

初期応答と最も高い構成では、Composite Score が0.0843上昇しました。nDCG@20 の重みは 50% なので、評価式上は nDCG@20 を約0.17改善するのと同程度の寄与であることがわかりました。
応答文の改善を評価に反映すること自体は妥当です。ただし、Composite Score の上昇だけでは、選曲の精度、説明の質、表現の多様性のどこが改善したかは分かりません。応答文が高く評価されても、推薦曲がユーザの要求に合っているか、説明が実際の曲の情報と一致しているかは、別に確認する必要があります。

### 評価への提案

上述の結果を踏まえ、論文では nDCG や LLM-as-a-Judge による評価を補うため、次の3点を提案しました。

- **Goal Adherence の個別報告**：「別のアーティストを」など、アーティスト・ジャンル・年代・テンポ・新規性に関する明示的な変更要求に推薦が従っているかを評価する。今回の検出ルールでは、アーティスト変更を扱いました。
- **Ranking–Response Consistency の評価**：推薦曲のメタデータを Judge に渡し、説明がその曲の情報と一致しているかを、文章の質とは別に評価する。
- **Judge の校正と条件別の結果の公開**：応答の長短、ユーザ情報の有無、推薦曲と説明の一致・不一致などの条件を網羅した人手評価で Judge の妥当性を確認する。

『Madonna』 の例のように、曲と説明が対応していても、要求に従っているとは限りません。正解曲との一致、要求への適合、応答文の質、曲と説明の整合性を分けて見る必要があると考えています。

## 論文採択と現地発表

今回の分析をまとめた論文は、RecSys Challenge Workshop に採択されました。2026/10/2 のプログラムでは、正解ラベルと要求の関係や指標の整合性を扱う「Auditing the benchmark」のセッションで発表しました。

https://x.com/tmp_friends/status/2106013197530431948

Workshop では、参加チームの論文発表に加えて2つの Keynote がありましたので、併せて講演内容を紹介します。

### Spotify：対話型音楽推薦エージェントの構築と改善

Spotify の Enrico Palumbo 氏による Keynote は、「From Language to Items: Building Conversational Recommendation Experiences at Spotify」でした。Spotify では、会話を通じて選曲や追加の条件指定ができる「Talk to Spotify」[^11] や、自分の嗜好がどう解釈されているかを確認し、自然言語で Home の推薦を調整できる「Taste Profile」[^12] が提供されています。

講演では、アイテムの識別子を直接生成する GenRec と、既存の Retrieval・推薦機能をツールとして呼び出す AgentRec の両方が詳しく紹介されました。

GenRec は、KDD 2026 で発表された Spotify の NEO に基づく構成です。NEO は、アイテムを表す Semantic ID と自然言語を同じ系列で扱い、推薦・検索・推薦理由の生成・ユーザの興味の要約を単一のモデルで実行します。自然言語の指示でタスクや出力形式を指定でき、アイテムの生成時には Constrained Decoding によってカタログに実在する ID のみを出力する仕組みです[^13]。

AgentRec については、ReAct をベースにした構成や、ユーザの意図に応じて複数の Retrieval・推薦ツールを並列に呼び出す Parallel Fusion Router が説明されました。後者は、RecSys 2025 の「You Say Search, I Say Recs」で紹介された構成です[^14]。

AgentRec の評価・改善で今回のコンペと共通していたのは、合成会話と LLM-as-a-Judge を用いる点です。RecSys 2026 の関連論文では、合成会話100件を5人のアノテータが確認し、Judge の判定が人間評価と整合していると評価された割合は97%でした。さらに、成功・失敗した実行を比較し、コーディングエージェントでプロンプトやツールの説明を修正する Self-Improvement Loop により、十分に手動調整したプロンプトから pass@1 が相対8%改善しています。改善例には、会話途中の意図の解釈や、曖昧なジャンル名の解消が挙げられていました[^15]。

### Netflix Prize：20年後の再挑戦と AI による実験

もう一つは、Taboola の Domonkos Tikk 氏と Istvan Pilaszy 氏による「Netflix Prize revisited: a hard problem even after 20 years」です。Netflix Prize に参加した元 Gravity チームが、現在の計算環境や深層学習、AI による開発支援を使って当時の課題に取り組み直した研究でした[^16]。

講演では、比較的シンプルなモデルが高い精度を示し、深層学習のモデルでは AutoEncoder が最もよかったと紹介されました。実サービスで使える豊富な特徴量や多様な行動ログが Netflix Prize では利用できない点も、モデルの性能を考えるうえでの背景として挙げられていました。

AI の活用例としては、Claude Code による105個の特徴量の実装や、人間が改善の方向性を決め、AI が183件の実験を進める取り組みも紹介されていました。実験用の Rust コードも公開されています[^17]。

## おわりに

RecSys Challenge 2026 では、上位解法を通して、異なる Retriever を組み合わせることや、候補の出どころを Reranker まで残すこと、応答生成を独立して改善することの重要性を学びました。

コンペでは、与えられた評価指標をどう改善するかに集中することが多いと思います。今回は、モデルを改善する中で感じた違和感を出発点として、「その指標で改善と見なされることは、ユーザにとってのよい推薦とどう関係するのか」を調べる機会になりました。
データを読み直し、評価の前提を確認し、論文として発表するところまで取り組めたことが、今回のコンペで得られた大きな学びでした。

今後のレコメンドシステム開発でも、スコアをどう上げるかに加えて、そのスコアが何を表しているのか、どの側面が改善したのかを確認していきたいと思います。

最後に、コンペ終了後、急遽論文執筆から現地発表へと進む中で、渡航費の手配をはじめ、発表の機会を後押ししてくださった上司に感謝します。また、業務の調整や不在中のフォローに柔軟に対応してくれたチームメンバーにも感謝します。皆さんの支えのおかげで、安心して論文執筆と現地での発表に取り組むことができました。

[^1]: [RecSys Challenge 2026 公式サイト・Workshop プログラム](https://www.recsyschallenge.com/2026/)
[^2]: [Auditing Alignment among Item Relevance, Goal Progress, and LLM-Judged Responses in Music-CRS](https://dl.acm.org/doi/10.1145/3842413.3842423)
[^3]: [Overview and Analysis of the RecSys Challenge 2026: Conversational Music Recommendation](https://arxiv.org/abs/2609.33045)
[^4]: [TalkPlayData 2: An Agentic Synthetic Data Pipeline for Multimodal Conversational Music Recommendation](https://arxiv.org/abs/2509.09685)
[^5]: [TalkPlayData-Challenge-Dataset（公開データ・Hugging Face）](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Dataset)
[^6]: [Talk Less, Rank Better: A Decoupled Pipeline for Conversational Music Recommendation（hallucinated）](https://dl.acm.org/doi/10.1145/3842413.3842429)
[^7]: [Frozen Embeddings, LLM Retrieval Lanes, Critic-Guided Replies: Conversational Music Recommendation at an Information-Limited Ceiling（volart）](https://dl.acm.org/doi/10.1145/3842413.3842424)
[^8]: [A Practical Multi-Source Pipeline for Conversational Music Recommendation in the RecSys Challenge 2026（niwatori・論文原稿）](https://raw.githubusercontent.com/ryowk/recsys2026-niwatori/main/paper/main.tex)
[^9]: [Two Views, One Voice: Evidence-Grounded Conversational Music Recommendation（swyoo）](https://arxiv.org/abs/2607.24846v1)
[^10]: [Distribution-Robust Reranking for Conversational Music Recommendation: PoliBaJukebox at the ACM RecSys Challenge 2026](https://dl.acm.org/doi/10.1145/3842413.3842418)
[^11]: [Just Say the Word: A More Personal Way to Ask, Discover, and Listen（Spotify）](https://newsroom.spotify.com/2026-07-14/talk-to-spotify-announcement-beta/)
[^12]: [How to Use Our Taste Profile Feature to Shape Your Spotify Homepage（Spotify）](https://newsroom.spotify.com/2026-09-23/taste-profile-shape-home-feed/)
[^13]: [A Unified Model for Personalization: Language-Steerable Generative Recommendation, Search, and User Understanding（KDD 2026）](https://research.atspotify.com/publications/unified-model-personalization-language-steerable-generative-recommendation-search-user-understanding)
[^14]: [You Say Search, I Say Recs: A Scalable Agentic Approach to Query Understanding and Exploratory Search at Spotify](https://research.atspotify.com/2025/9/you-say-search-i-say-recs-a-scalable-agentic-approach-to-query-understanding)
[^15]: [Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops（RecSys 2026）](https://arxiv.org/abs/2609.30297)
[^16]: [Netflix Prize, 20 Years Later: Take on the challenge with our new codebase（発表者による研究紹介）](https://www.meetup.com/taboola-budapest-tech-meetup/events/316564131/)
[^17]: [Netflix Prize 再挑戦の公開コード（gtakacs/netflix_prize）](https://github.com/gtakacs/netflix_prize)

---
title: "Vertex AI Workbench の STOCKOUT エラーを自動リトライで乗り切る"
emoji: "🧪"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["gcp", "vertexai", "shell", "bash"]
published: true
publication_name: "dmmdata"
---

## はじめに

普段 Vertex AI Workbench を PoC 環境として利用しています。GPU に `A100 80GB` を割り当てた構成です。

しかし、インスタンスを起動しようとしたところ、以下のような `STOCKOUT` エラーが発生することが度々あります。

```
The zone 'projects/my-project-id/zones/us-central1-a' does not have enough
resources available to fulfill the request. 'NULL:0/NULL:0/NULL:0
(state:STOCKOUT, sub-state:STOCKOUT, resource type:compute)'
```

`STOCKOUT` は、指定したゾーンで要求したマシンタイプ（特に GPU）の空きが物理的に枯渇していることを示すエラーです。自分のプロジェクトの割り当て（Quota）超過ではないため、Quota を引き上げても解消しません。[^1] `A100 80GB` のような希少なアクセラレータを積んだ構成では特に起こりやすく、利用者側の選択肢は、

- 空きが出るまで待つ
- マシンタイプを変える

のどちらかになります。

インスタンスの起動は、Google Cloud コンソール上の「開始」ボタンのほか、`gcloud workbench instances start` コマンドでも行えます。[^2] ただしどちらの方法でも、リソースが空くまでボタンを押下し続けたりコマンドを再実行し続けたりすることになり、手間がかかります。

今回は PoC 環境を `A100 80GB` のまま使い続けたかったので、マシンタイプは変えずに起動できるようになるまで待つことにしました。とはいえ手動でリトライし続けるのは非効率なので、シェルスクリプトで自動リトライすることにしました。


## スクリプト

このスクリプトは `gcloud workbench instances start` コマンドを定期的に実行し、起動に成功するまでリトライし続けます。コマンドの出力を見て `STOCKOUT` エラーかどうかを判定し、判定結果をログに出力します。起動に成功した時点で自動的に終了します。


```bash:auto_start_workbench.sh
#!/bin/bash
# Script to periodically attempt to start a Vertex AI Workbench instance
# until it succeeds (e.g., when resources become available after a STOCKOUT).

INSTANCE_NAME="my-workbench-instance"
LOCATION="us-central1-a"
PROJECT="my-project-id"
INTERVAL=120  # Retry interval in seconds (default: 2 minutes)

echo "=========================================="
echo " Vertex AI Workbench Auto-Start Script"
echo "=========================================="
echo " Instance : ${INSTANCE_NAME}"
echo " Location : ${LOCATION}"
echo " Project  : ${PROJECT}"
echo " Interval : ${INTERVAL}s"
echo "=========================================="

attempt=1

while true; do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempt #${attempt}: Starting instance '${INSTANCE_NAME}'..."

    output=$(gcloud workbench instances start "${INSTANCE_NAME}" \
        --location="${LOCATION}" \
        --project="${PROJECT}" 2>&1)
    exit_code=$?

    echo "${output}"

    if [ ${exit_code} -eq 0 ]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Instance '${INSTANCE_NAME}' started successfully!"
        exit 0
    fi

    # Check if the error is a STOCKOUT / resource unavailability error
    if echo "${output}" | grep -qiE "STOCKOUT|does not have enough resources|resource type:compute"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resource unavailable (STOCKOUT). Retrying in ${INTERVAL} seconds..."
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Unexpected error occurred. Retrying in ${INTERVAL} seconds..."
    fi

    attempt=$((attempt + 1))
    sleep "${INTERVAL}"
done
```

## スクリプトの解説

### 設定パラメータ

スクリプト上部で対象インスタンスの情報とリトライ間隔を定義しています。

- `INSTANCE_NAME`: 起動対象の Workbench インスタンス名
- `LOCATION`: インスタンスのゾーン
- `PROJECT`: Google Cloud のプロジェクト ID
- `INTERVAL`: リトライ間隔（秒）。デフォルトは 120 秒（2 分）

### エラー判定

起動の成否は `gcloud` コマンドの終了コードで判定しています。出力内容は、後述するログの出し分けに使います。

```bash
output=$(gcloud workbench instances start "${INSTANCE_NAME}" \
    --location="${LOCATION}" \
    --project="${PROJECT}" 2>&1)
exit_code=$?
```

`2>&1` で stderr を stdout にリダイレクトし、出力全体を `output` 変数に格納しています。終了コードが 0 であれば成功と判断して終了し、0 以外であればリトライに進みます。

### STOCKOUTエラーの検出

```bash
if echo "${output}" | grep -qiE "STOCKOUT|does not have enough resources|resource type:compute"; then
```

出力に `STOCKOUT` や `does not have enough resources` などのキーワードが含まれる場合は、リソース不足と判定します。ただしこの判定はログメッセージの出し分けに使っているだけで、リトライするかどうかには影響しません。終了コードが 0 以外であれば、エラーの種類にかかわらずリトライを続けます。

## 使い方

スクリプトを実行可能にして実行するだけです。

```bash
chmod +x auto_start_workbench.sh
./auto_start_workbench.sh
```

実行すると以下のようなログが出力されます。

```
==========================================
 Vertex AI Workbench Auto-Start Script
==========================================
 Instance : my-workbench-instance
 Location : us-central1-a
 Project  : my-project-id
 Interval : 120s
==========================================

[2026-03-10 10:00:00] Attempt #1: Starting instance 'my-workbench-instance'...
...
[2026-03-10 10:00:05] Resource unavailable (STOCKOUT). Retrying in 120 seconds...

[2026-03-10 10:02:05] Attempt #2: Starting instance 'my-workbench-instance'...
...
[2026-03-10 10:02:12] Instance 'my-workbench-instance' started successfully!
```

`Ctrl+C` で途中終了することもできます。

## おわりに

STOCKOUT エラーは発生するタイミングが読みにくく、手動でリトライし続けるのは意外と手間がかかります。GPU 構成を変えれば回避できることもありますが、それでは PoC で使いたいリソースそのものが変わってしまいますし、別のゾーンに移るなら新しいインスタンスの作成と環境移行が必要になり、運用コストも小さくありません。

このスクリプトを走らせておけば、リソースが空き次第自動で起動してくれるため、待ち時間を他の作業に充てられます。`INTERVAL` を変えればリトライ頻度もコントロールできるので、利用状況に応じて調整してみてください。

[^1]: Google Cloud, "Troubleshooting resource availability errors"（"Resource errors aren't related to your Compute Engine quota."）
  https://cloud.google.com/compute/docs/troubleshooting/troubleshooting-resource-availability

[^2]: Google Cloud SDK, "`gcloud workbench instances start`"
  https://cloud.google.com/sdk/gcloud/reference/workbench/instances/start

---
title: "Vertex AI WorkbenchのSTOCKOUTエラーが解消されるまで自動リトライするシェルスクリプト"
emoji: "🔄"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["gcp", "vertexai", "shell", "bash"]
published: false
publication_name: "dmmdata"
---

## はじめに

普段 Vertex AI Workbench を PoC 環境として利用しています。

Vertex AI Workbench でインスタンスを起動しようとしたところ、以下のような `STOCKOUT` エラーが発生することが度々あります。

```
The zone 'projects/ai-mlops-dev/zones/us-central1-a' does not have enough
resources available to fulfill the request. 'NULL:0/NULL:0/NULL:0
(state:STOCKOUT, sub-state:STOCKOUT, resource type:compute)'
```

これはリソース不足によるエラーなので、解消されるまで一定時間待ってから再試行する必要があります。インスタンスの起動は `gcloud workbench instances start` コマンドでも行えますが、[^1] リソースが空くまで Google Cloud コンソール上の「開始」ボタンを押下したり、コマンドを手動で再実行したりするのは手間がかかります。

また、Workbench の仕様上、既存インスタンスをそのまま別の region に切り替えて回避することもできません。別の region を試すには、新しく Workbench を作成し、データや設定を移す必要があります。[^2][^3]

そこで、まずは現在の構成のまま起動に成功するまで待てるように、シェルスクリプトで自動リトライすることにしました。


## スクリプト

このスクリプトは `gcloud workbench instances start` コマンドを定期的に実行し、起動に成功するまでリトライし続けます。コマンドの出力を見て `STOCKOUT` エラーかどうかを判定し、その内容をログに出力します。起動に成功した時点で自動的に終了します。


```bash:auto_start_workbench.sh
#!/bin/bash
# Script to periodically attempt to start a Vertex AI Workbench instance
# until it succeeds (e.g., when resources become available after a STOCKOUT).

INSTANCE_NAME="terai-tomoya-a100-80gb"
LOCATION="us-central1-a"
PROJECT="ai-mlops-dev"
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
- `PROJECT`: GCPプロジェクトID
- `INTERVAL`: リトライ間隔（秒）。デフォルトは120秒（2分）

### エラー判定

`gcloud` コマンドの終了コードと出力内容を合わせて判定しています。

```bash
output=$(gcloud workbench instances start "${INSTANCE_NAME}" \
    --location="${LOCATION}" \
    --project="${PROJECT}" 2>&1)
exit_code=$?
```

`2>&1` で stderr を stdout にリダイレクトし、出力全体を `output` 変数に格納しています。終了コードが 0 であれば成功と判断して終了します。

### STOCKOUTエラーの検出

```bash
if echo "${output}" | grep -qiE "STOCKOUT|does not have enough resources|resource type:compute"; then
```

出力に `STOCKOUT` や `does not have enough resources` などのキーワードが含まれる場合は、リソース不足と判定します。それ以外のエラーも予期しないエラーとしてリトライ対象にしているため、リソース不足が解消されれば自動的に起動できます。

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
 Instance : terai-tomoya-a100-80gb
 Location : us-central1-a
 Project  : ai-mlops-dev
 Interval : 120s
==========================================

[2026-03-10 10:00:00] Attempt #1: Starting instance 'terai-tomoya-a100-80gb'...
...
[2026-03-10 10:00:05] Resource unavailable (STOCKOUT). Retrying in 120 seconds...

[2026-03-10 10:02:05] Attempt #2: Starting instance 'terai-tomoya-a100-80gb'...
...
[2026-03-10 10:02:12] Instance 'terai-tomoya-a100-80gb' started successfully!
```

`Ctrl+C` で途中終了することもできます。

## おわりに

STOCKOUT エラーは発生するタイミングが読みにくく、手動でリトライし続けるのは意外と手間がかかります。特に今回のように `A100 80GB` を使う構成では、別の region を試すにも新しい Workbench の作成や環境移行が必要になり、運用コストも小さくありません。

このスクリプトを走らせておけば、リソースが空き次第自動で起動してくれるため、待ち時間を他の作業に充てられます。`INTERVAL` を調整すればリトライ頻度もコントロールできるので、利用状況に応じて調整するとよさそうです。

[^1]: Google Cloud SDK, "`gcloud workbench instances start`"
  https://cloud.google.com/sdk/gcloud/reference/workbench/instances/start

[^2]: Google Cloud, "Create a Vertex AI Workbench instance"
  https://cloud.google.com/vertex-ai/docs/workbench/instances/create

[^3]: Google Cloud, "Migrate to Vertex AI Workbench instances"
  https://cloud.google.com/vertex-ai/docs/workbench/instances/migrate

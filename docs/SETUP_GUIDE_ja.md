# セットアップガイド

このガイドでは、ポートフォリオプロジェクトの環境構築から実行まで、詳細に説明します。

## 目次

- [システム要件](#システム要件)
- [インストール手順](#インストール手順)
- [データの準備](#データの準備)
- [モデルのトレーニング](#モデルのトレーニング)
- [ダッシュボードの起動](#ダッシュボードの起動)
- [トラブルシューティング](#トラブルシューティング)

## システム要件

### 必須環境
- **Python**: 3.12以上
- **メモリ**: 最低4GB（推奨8GB以上）
- **ディスク空間**: 2GB以上

### 対応OS
- Linux (Ubuntu 20.04+, Debian 11+)
- macOS (12.0+)
- Windows 10/11 (WSL2推奨)

### 推奨ツール
- **uv**: Pythonパッケージマネージャー（高速）
- **Git**: バージョン管理

## インストール手順

### ステップ1: Python環境の確認

Pythonのバージョンを確認します：

```bash
python --version
# または
python3 --version
```

**出力例**: `Python 3.12.3`

3.12未満の場合は、Pythonをアップデートしてください：

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

**macOS (Homebrew):**
```bash
brew install python@3.12
```

**Windows:**
[Python公式サイト](https://www.python.org/downloads/)からインストーラーをダウンロードしてインストール

### ステップ2: uvのインストール（推奨）

uvは高速なPythonパッケージマネージャーです：

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール確認：
```bash
uv --version
```

> **Note**: uvがインストールできない場合は、pipでも動作します（後述）

### ステップ3: リポジトリのクローン

```bash
# GitHubからクローン（URLは実際のリポジトリに置き換え）
git clone https://github.com/your-username/portfolio.git
cd portfolio
```

または、ZIPファイルをダウンロードして解凍：

```bash
unzip portfolio-main.zip
cd portfolio-main
```

### ステップ4: 依存関係のインストール

#### uvを使用する場合（推奨）

```bash
uv sync
```

これにより、以下が自動的に行われます：
- 仮想環境の作成 (`.venv/`)
- すべての依存パッケージのインストール
- 依存関係のロック (`uv.lock`)

#### pipを使用する場合

```bash
# 仮想環境の作成
python3 -m venv .venv

# 仮想環境の有効化
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 依存関係のインストール
pip install -e .
```

### ステップ5: 環境変数の設定（オプション）

`.env.example`をコピーして`.env`を作成：

```bash
cp .env.example .env
```

`.env`を編集して必要な設定を記入：

```bash
# モデル設定
RANDOM_STATE=42
TEST_SIZE=0.2

# パス設定
DATA_DIR=./data
MODELS_DIR=./models
```

> **Note**: 現時点では外部APIやデータベースを使用しないため、このステップはオプションです。

## データの準備

### サンプルデータの生成

プロジェクトに含まれるスクリプトで、3つのサンプルデータセットを生成します：

```bash
uv run python main.py --generate-data
```

または：

```bash
uv run python scripts/generate_sample_data.py
```

**生成されるデータ**:

1. **customer_churn.csv** (5,000件)
   - 顧客ID、年齢、性別、契約情報、利用サービス、課金情報、解約フラグ

2. **sales_data.csv** (1,095日 = 3年分)
   - 日付、総売上、カテゴリ別売上、トランザクション数

3. **market_basket.csv** (10,000トランザクション)
   - トランザクションID、商品名、数量

**出力例**:
```
Generating sample datasets...

1. Generating customer churn dataset...
   Saved to /home/tsune/portfolio/data/raw/customer_churn.csv
   Shape: (5000, 14)
   Churn rate: 23.88%

2. Generating sales time series dataset...
   Saved to /home/tsune/portfolio/data/raw/sales_data.csv
   Shape: (1095, 8)
   Date range: 2021-01-01 to 2023-12-31

3. Generating market basket dataset...
   Saved to /home/tsune/portfolio/data/raw/market_basket.csv
   Shape: (48057, 3)
   Unique transactions: 10000

✅ All datasets generated successfully!
```

### データの確認

生成されたデータを確認：

```bash
ls -lh data/raw/
```

**出力例**:
```
total 2.4M
-rw-r--r-- 1 user user 423K Jan 11 10:55 customer_churn.csv
-rw-r--r-- 1 user user 1.8M Jan 11 10:55 market_basket.csv
-rw-r--r-- 1 user user 178K Jan 11 10:55 sales_data.csv
```

## モデルのトレーニング

### 顧客チャーン予測モデルのトレーニング

```bash
uv run python main.py --train-model
```

または：

```bash
uv run python scripts/train_churn_model.py
```

**実行内容**:

1. データの読み込み
2. データ品質の検証
3. 前処理（エンコーディング、スケーリング、特徴量生成）
4. Train/Test分割
5. 複数モデルの比較（XGBoost、LightGBM、Random Forest）
6. 最終モデルの学習
7. モデル評価
8. モデルと前処理パイプラインの保存

**実行時間**: 約2〜3分（マシンスペックによる）

**出力例**:
```
======================================================================
CUSTOMER CHURN PREDICTION - TRAINING PIPELINE
======================================================================

📊 Step 1: Loading data...
Loaded 5000 records with 14 features

🔍 Step 2: Validating data quality...
Quality Score: 100.0/100

⚙️  Step 3: Preprocessing data...
Encoding 7 categorical features...
Scaling 5 numerical features...
Creating interaction features...
Final feature matrix: (5000, 16)

🔀 Step 4: Splitting data...
Training set: (4000, 16)
Test set: (1000, 16)

🤖 Step 5: Comparing models...
XGBoost: 0.7473 (+/- 0.0214)
LightGBM: 0.7461 (+/- 0.0195)
Random Forest: 0.7533 (+/- 0.0199)

🎯 Step 6: Training final model (XGBoost)...
Training completed!

📈 Step 7: Evaluating model...
Accuracy: 0.7540
Precision: 0.7160
Recall: 0.7540
F1_score: 0.7225
ROC AUC: 0.7550

💾 Step 8: Saving model...
Model saved to /path/to/portfolio/models/customer_churn_model.joblib

✅ TRAINING COMPLETED SUCCESSFULLY!
```

### 保存されるファイル

トレーニング完了後、以下のファイルが生成されます：

```
models/
├── customer_churn_model.joblib      # 訓練済みモデル
├── customer_churn_model.json        # モデルメタデータ
└── churn_preprocessor.joblib        # 前処理パイプライン
```

## ダッシュボードの起動

### Streamlitダッシュボードの起動

```bash
uv run python main.py --dashboard
```

または：

```bash
uv run streamlit run dashboards/app.py
```

**出力例**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

### ダッシュボードへのアクセス

ブラウザで以下のURLを開きます：

```
http://localhost:8501
```

**ダッシュボードの内容**:

1. **Overview** (概要ページ)
   - プロジェクト説明
   - 使用技術スタック
   - プロジェクト一覧

2. **Customer Churn** (顧客チャーン分析)
   - KPIメトリクス
   - 契約タイプ別解約率
   - 課金額分布
   - モデル性能指標

3. **Sales Forecasting** (売上予測)
   - 時系列トレンド
   - カテゴリ別売上
   - 月次集計

4. **Market Basket** (マーケットバスケット分析)
   - 人気商品ランキング
   - バスケットサイズ分布
   - 商品購買頻度

### ダッシュボードの停止

ターミナルで `Ctrl + C` を押すと停止します。

## テストの実行

### 単体テストの実行

```bash
uv run python main.py --test
```

または：

```bash
uv run pytest tests/ -v
```

**出力例**:
```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.2
collected 9 items

tests/test_data_loader.py::test_data_loader_initialization PASSED        [ 11%]
tests/test_data_loader.py::test_load_csv PASSED                          [ 22%]
tests/test_data_loader.py::test_load_csv_file_not_found PASSED           [ 33%]
tests/test_data_loader.py::test_save_processed PASSED                    [ 44%]
tests/test_preprocessor.py::test_handle_missing_values PASSED            [ 55%]
tests/test_preprocessor.py::test_encode_categorical PASSED               [ 66%]
tests/test_preprocessor.py::test_scale_features PASSED                   [ 77%]
tests/test_preprocessor.py::test_create_time_features PASSED             [ 88%]
tests/test_preprocessor.py::test_create_interaction_features PASSED      [100%]

============================== 9 passed in 1.39s ================================
```

### カバレッジレポート（オプション）

テストカバレッジを確認：

```bash
uv run pytest tests/ --cov=src --cov-report=html
```

HTMLレポートが `htmlcov/index.html` に生成されます。

## トラブルシューティング

### 問題1: Python バージョンエラー

**エラーメッセージ**:
```
RuntimeError: This project requires Python 3.12 or higher
```

**解決方法**:
1. Pythonのバージョンを確認: `python --version`
2. Python 3.12以上をインストール
3. 仮想環境を再作成

### 問題2: パッケージインストールエラー

**エラーメッセージ**:
```
ERROR: Could not find a version that satisfies the requirement ...
```

**解決方法**:

#### uvの場合:
```bash
# キャッシュをクリア
uv cache clean

# 再インストール
uv sync --refresh
```

#### pipの場合:
```bash
# pipをアップグレード
pip install --upgrade pip

# 再インストール
pip install -e . --no-cache-dir
```

### 問題3: データが見つからないエラー

**エラーメッセージ**:
```
FileNotFoundError: File not found: data/raw/customer_churn.csv
```

**解決方法**:
```bash
# サンプルデータを生成
uv run python scripts/generate_sample_data.py
```

### 問題4: Streamlit起動エラー

**エラーメッセージ**:
```
Address already in use: Port 8501 is already in use
```

**解決方法**:

#### ポート番号を変更:
```bash
uv run streamlit run dashboards/app.py --server.port 8502
```

#### または既存のプロセスを終了:
```bash
# Linuxの場合
lsof -ti:8501 | xargs kill -9

# Windowsの場合
netstat -ano | findstr :8501
taskkill /PID <プロセスID> /F
```

### 問題5: メモリ不足エラー

**エラーメッセージ**:
```
MemoryError: Unable to allocate array
```

**解決方法**:
1. データサイズを縮小（`generate_sample_data.py`の`n_samples`を減らす）
2. Polarsを使用（大規模データ処理用）
3. メモリを増やす（最低8GB推奨）

### 問題6: モデルロードエラー

**エラーメッセージ**:
```
FileNotFoundError: Model file not found
```

**解決方法**:
```bash
# モデルを再トレーニング
uv run python scripts/train_churn_model.py
```

## その他のコマンド

### コードの整形

```bash
uv run black src/ tests/ scripts/
```

### リント実行

```bash
uv run ruff check src/ tests/ scripts/
```

### 依存関係の更新

```bash
uv lock --upgrade
uv sync
```

### 仮想環境のクリーンアップ

```bash
# 仮想環境を削除
rm -rf .venv

# 再作成
uv sync
```

## 開発モード

### Jupyterの起動

Jupyter Labを起動して分析を行う：

```bash
uv run jupyter lab
```

ブラウザで `http://localhost:8888` が開きます。

### インタラクティブシェル

Pythonシェルでモジュールをテスト：

```bash
uv run python

>>> from src.data.loader import DataLoader
>>> loader = DataLoader()
>>> df = loader.load_csv("customer_churn.csv")
>>> print(df.head())
```

## 次のステップ

セットアップが完了したら：

1. [README_ja.md](../README_ja.md) でプロジェクト概要を確認
2. [ARCHITECTURE_ja.md](./ARCHITECTURE_ja.md) で技術設計を理解
3. [PORTFOLIO_GUIDE_ja.md](./PORTFOLIO_GUIDE_ja.md) でポートフォリオの見どころを確認（面接対策）

## サポート

問題が解決しない場合は、以下を確認してください：

- [GitHub Issues](https://github.com/your-username/portfolio/issues)
- [CONTRIBUTING.md](./CONTRIBUTING.md)

---

**最終更新**: 2026年1月
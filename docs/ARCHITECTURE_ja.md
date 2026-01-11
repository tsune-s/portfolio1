# システムアーキテクチャ設計書

このドキュメントでは、データ分析ポートフォリオの技術アーキテクチャと設計思想を説明します。

## 目次

- [全体アーキテクチャ](#全体アーキテクチャ)
- [モジュール設計](#モジュール設計)
- [データフロー](#データフロー)
- [技術選定の理由](#技術選定の理由)
- [拡張性への配慮](#拡張性への配慮)

## 全体アーキテクチャ

### レイヤー構造

本プロジェクトは、**3層アーキテクチャ**を採用しています：

```
┌─────────────────────────────────────────┐
│     プレゼンテーション層                    │
│  (Streamlit Dashboard, CLI)            │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     ビジネスロジック層                      │
│  (Models, Preprocessing, Analysis)     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     データアクセス層                        │
│  (Loaders, Validators, Storage)        │
└─────────────────────────────────────────┘
```

### コンポーネント構成

```
┌────────────────────────────────────────────────────┐
│                   ユーザー                          │
└──────────┬─────────────────────────┬───────────────┘
           │                         │
    ┌──────▼──────┐          ┌──────▼──────┐
    │  CLI (main) │          │  Dashboard  │
    │             │          │  (Streamlit)│
    └──────┬──────┘          └──────┬──────┘
           │                        │
    ┌──────▼────────────────────────▼──────┐
    │         Scripts Layer                │
    │  - generate_sample_data.py          │
    │  - train_churn_model.py             │
    └──────┬──────────────────────────────┘
           │
    ┌──────▼──────────────────────────────┐
    │       Business Logic Layer          │
    │                                     │
    │  ┌──────────┐  ┌────────────┐     │
    │  │  Models  │  │Preprocessor│     │
    │  │          │  │            │     │
    │  └────┬─────┘  └─────┬──────┘     │
    │       │              │             │
    │  ┌────▼──────────────▼──────┐     │
    │  │   Visualization          │     │
    │  └──────────────────────────┘     │
    └──────┬──────────────────────────────┘
           │
    ┌──────▼──────────────────────────────┐
    │      Data Access Layer              │
    │                                     │
    │  ┌──────────┐  ┌────────────┐     │
    │  │  Loader  │  │ Validator  │     │
    │  └────┬─────┘  └─────┬──────┘     │
    └───────┼───────────────┼────────────┘
            │               │
    ┌───────▼───────────────▼────────────┐
    │         Data Storage               │
    │  - CSV Files                       │
    │  - Serialized Models               │
    │  - Preprocessors                   │
    └────────────────────────────────────┘
```

## モジュール設計

### 1. データアクセス層 (`src/data/`)

#### `loader.py` - データローダー

**責務**: 複数のデータソースからデータを読み込む

**設計パターン**: Factory Pattern

```python
DataLoader
├── load_csv()           # CSV読み込み
├── load_from_url()      # Web APIからのデータ取得
├── load_from_sql()      # SQLデータベース接続
├── scrape_table()       # Webスクレイピング
└── save_processed()     # 処理済みデータの保存
```

**設計思想**:
- 単一責任の原則: 各メソッドは1つのデータソースのみを扱う
- 開放閉鎖の原則: 新しいデータソースは既存コードを変更せずに追加可能
- Pandas/Polars両対応で、データ量に応じて使い分け可能

#### `preprocessor.py` - データ前処理

**責務**: データのクリーニングと変換

```python
DataPreprocessor
├── handle_missing_values()    # 欠損値処理
├── remove_outliers()          # 外れ値除去
├── scale_features()           # 特徴量スケーリング
├── encode_categorical()       # カテゴリカル変数エンコーディング
├── create_time_features()     # 時系列特徴量生成
└── create_interaction_features() # 相互作用特徴量生成
```

**設計思想**:
- **Stateful設計**: Scalerやエンコーダーを内部に保持し、train/testで一貫性を保つ
- **Pipeline互換**: Scikit-learnのPipelineと統合可能な設計
- **再現性**: 同じパラメータで常に同じ結果を保証

#### `validator.py` - データ品質検証

**責務**: データの妥当性チェックとレポート生成

```python
DataValidator
├── validate_schema()          # スキーマ検証
├── check_missing_values()     # 欠損値チェック
├── check_duplicates()         # 重複チェック
├── check_outliers()           # 外れ値検出
└── generate_quality_report()  # 品質レポート生成
```

**設計思想**:
- **Pydantic統合**: 型安全なデータ検証
- **早期エラー検出**: データパイプラインの入口で問題を検出
- **詳細レポート**: 何が問題かを明確に報告

### 2. ビジネスロジック層 (`src/models/`)

#### `base_model.py` - 抽象基底クラス

**設計パターン**: Template Method Pattern

```python
BaseModel (ABC)
├── train()      # 抽象メソッド: サブクラスで実装
├── predict()    # 抽象メソッド: サブクラスで実装
├── evaluate()   # 抽象メソッド: サブクラスで実装
├── save_model() # 共通実装: モデル保存
└── load_model() # 共通実装: モデル読み込み
```

**設計思想**:
- **抽象化**: すべてのモデルが実装すべきインターフェースを定義
- **再利用性**: 保存・読み込み処理は共通実装で重複排除
- **拡張性**: 新しいモデルタイプの追加が容易

#### `classifier.py` - 分類モデル

**設計パターン**: Strategy Pattern

```python
ClassifierModel
├── SUPPORTED_ALGORITHMS    # 対応アルゴリズム辞書
├── train()                # 学習
├── predict()              # 予測
├── predict_proba()        # 確率予測
├── evaluate()             # 評価
└── get_feature_importance() # 特徴量重要度
```

**設計思想**:
- **アルゴリズム切り替え**: 辞書でアルゴリズムを管理し、動的に切り替え
- **統一インターフェース**: すべてのアルゴリズムを同じ方法で扱える
- **詳細メトリクス**: 複数の評価指標を自動計算

#### `trainer.py` - モデルトレーニング

**責務**: モデル最適化とハイパーパラメータチューニング

```python
ModelTrainer
├── train_test_split()     # データ分割
├── cross_validate()       # 交差検証
├── grid_search()          # グリッドサーチ
├── random_search()        # ランダムサーチ
└── compare_models()       # モデル比較
```

**設計思想**:
- **最適化の分離**: モデル定義と最適化ロジックを分離
- **実験管理**: 各種最適化手法を統一的に扱える
- **再現性**: random_stateで結果を再現可能

### 3. プレゼンテーション層 (`src/visualization/`, `dashboards/`)

#### `plots.py` - 可視化ユーティリティ

```python
DataVisualizer
├── plot_distribution()         # 分布プロット
├── plot_correlation_matrix()   # 相関行列
├── plot_feature_importance()   # 特徴量重要度
├── plot_confusion_matrix()     # 混同行列
├── create_interactive_scatter() # インタラクティブ散布図
└── create_interactive_line()    # インタラクティブ折れ線
```

**設計思想**:
- **静的・動的の両対応**: Matplotlib/SeabornとPlotlyを使い分け
- **一貫したインターフェース**: すべてのプロット関数が同様の引数を取る
- **カスタマイズ可能**: デフォルト設定と詳細設定の両立

#### `app.py` - Streamlitダッシュボード

**アーキテクチャ**: Single Page Application (SPA)

```python
Main Application
├── show_overview()         # 概要ページ
├── show_churn_analysis()   # 顧客チャーン分析
├── show_sales_analysis()   # 売上分析
└── show_basket_analysis()  # マーケットバスケット分析
```

**設計思想**:
- **ページベース設計**: 各分析を独立したページとして実装
- **レスポンシブレイアウト**: カラムレイアウトで画面サイズに対応
- **インタラクティブ**: Plotlyによるリアルタイム操作

## データフロー

### 1. トレーニングフロー

```
┌──────────────┐
│  Raw Data    │
│  (CSV)       │
└──────┬───────┘
       │
┌──────▼───────┐
│ DataLoader   │
│  .load_csv() │
└──────┬───────┘
       │
┌──────▼───────────┐
│ DataValidator    │
│  .validate()     │
└──────┬───────────┘
       │
┌──────▼──────────────┐
│ DataPreprocessor    │
│  - handle_missing() │
│  - scale()          │
│  - encode()         │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│ ModelTrainer        │
│  - split()          │
│  - compare()        │
│  - optimize()       │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│ ClassifierModel     │
│  .train()           │
│  .evaluate()        │
└──────┬──────────────┘
       │
┌──────▼──────────────┐
│ Save Model          │
│  .joblib            │
└─────────────────────┘
```

### 2. 推論フロー

```
┌──────────────┐
│  New Data    │
└──────┬───────┘
       │
┌──────▼───────────────┐
│ Load Preprocessor    │
│  (from .joblib)      │
└──────┬───────────────┘
       │
┌──────▼───────────────┐
│ Apply Transform      │
│  - scale()           │
│  - encode()          │
└──────┬───────────────┘
       │
┌──────▼───────────────┐
│ Load Model           │
│  (from .joblib)      │
└──────┬───────────────┘
       │
┌──────▼───────────────┐
│ Predict              │
│  .predict()          │
│  .predict_proba()    │
└──────┬───────────────┘
       │
┌──────▼───────────────┐
│ Results              │
└──────────────────────┘
```

### 3. ダッシュボードフロー

```
┌────────────┐
│ User       │
│ Browser    │
└──────┬─────┘
       │
┌──────▼──────────┐
│ Streamlit App   │
│  - Navigation   │
└──────┬──────────┘
       │
┌──────▼──────────┐
│ Load Data       │
│ (DataLoader)    │
└──────┬──────────┘
       │
┌──────▼───────────────┐
│ Process & Aggregate  │
│ (Pandas operations)  │
└──────┬───────────────┘
       │
┌──────▼──────────┐
│ Visualize       │
│ (Plotly charts) │
└──────┬──────────┘
       │
┌──────▼──────────┐
│ Render          │
│ (Browser)       │
└─────────────────┘
```

## 技術選定の理由

### Python 3.12+
- **型ヒント**: より厳密な型チェックが可能
- **パフォーマンス**: 3.11/3.12の高速化恩恵
- **最新機能**: match文、Structural Pattern Matchingなど

### Pandas + Polars
- **Pandas**: 豊富なエコシステム、成熟したAPI
- **Polars**: 大規模データ処理時の高速化オプション
- **両対応**: データサイズに応じて使い分け

### XGBoost & LightGBM
- **高性能**: 勾配ブースティングの最先端実装
- **産業標準**: Kaggleコンペなどで実績豊富
- **解釈性**: 特徴量重要度、SHAP値対応

### Streamlit
- **高速開発**: Pythonのみでダッシュボード作成
- **インタラクティブ**: ウィジェットが豊富
- **デプロイ容易**: Streamlit Cloudで簡単デプロイ

### uv (パッケージマネージャー)
- **高速**: Rustベースで超高速インストール
- **再現性**: uv.lockで依存関係を固定
- **シンプル**: pyproject.toml一元管理

## 拡張性への配慮

### 1. モジュール独立性

各モジュールは疎結合で、個別にテスト・置き換えが可能：

```python
# DataLoaderは他のモジュールに依存しない
loader = DataLoader()
df = loader.load_csv("data.csv")

# Preprocessorは単独で使用可能
preprocessor = DataPreprocessor()
df_clean = preprocessor.handle_missing_values(df)

# モデルもデータソースに依存しない
model = ClassifierModel(algorithm='xgboost')
model.train(X, y)
```

### 2. 新しいモデルの追加

`BaseModel`を継承するだけで新しいモデルタイプを追加可能：

```python
class RegressorModel(BaseModel):
    def train(self, X, y, **kwargs):
        # 回帰モデルの実装
        pass

    def predict(self, X):
        # 予測実装
        pass

    def evaluate(self, X, y):
        # 評価実装（RMSE, MAEなど）
        pass
```

### 3. 新しいデータソースの追加

`DataLoader`に新しいメソッドを追加：

```python
def load_from_api(self, endpoint: str, headers: dict) -> pd.DataFrame:
    """REST APIからデータを取得"""
    response = requests.get(endpoint, headers=headers)
    return pd.DataFrame(response.json())
```

### 4. カスタム前処理の追加

`DataPreprocessor`を拡張または継承：

```python
class CustomPreprocessor(DataPreprocessor):
    def apply_domain_specific_transform(self, df):
        """業界特有の前処理"""
        # カスタム処理
        return df
```

## セキュリティ考慮事項

### 1. データ保護
- 機密情報は`.env`ファイルで管理（`.gitignore`対象）
- API キーはハードコードしない
- データベース接続情報は環境変数から読み込み

### 2. 入力検証
- Pydanticによる型検証
- SQLインジェクション対策（SQLAlchemy ORM使用）
- ファイルパス検証

### 3. 依存関係管理
- `uv.lock`で依存バージョンを固定
- 定期的な脆弱性スキャン推奨

## パフォーマンス最適化

### 1. データ処理
- **Polars**: 大規模データ（100万行以上）はPolarsで処理
- **チャンク処理**: メモリに載らないデータは分割処理
- **並列処理**: n_jobs=-1で全コア活用

### 2. モデル学習
- **早期停止**: eval_setで過学習を防止
- **GPUサポート**: XGBoost/LightGBMはGPU利用可能
- **キャッシュ**: 前処理結果をキャッシュして再利用

### 3. ダッシュボード
- **データキャッシュ**: `@st.cache_data`で再読み込み回避
- **遅延読み込み**: 必要なデータのみロード
- **最適化済みプロット**: Plotlyの軽量モード使用

---

このアーキテクチャは、**保守性・拡張性・テスト容易性**を重視した設計となっています。
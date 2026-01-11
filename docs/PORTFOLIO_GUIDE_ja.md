# ポートフォリオ説明書（面接官・採用担当者向け）

このドキュメントは、**採用担当者・技術面接官**の方々に向けて、本ポートフォリオの見どころとアピールポイントをまとめたものです。

## 📌 5分で理解するポートフォリオ

### このポートフォリオで証明できるスキル

✅ **データエンジニアリング**: ETLパイプライン、データ品質管理、大規模データ処理
✅ **機械学習**: モデル構築、最適化、評価、デプロイ
✅ **ビジネス分析**: データから洞察を導き、ビジネス価値に転換
✅ **ソフトウェア工学**: 保守性の高いコード設計、テスト、ドキュメント
✅ **可視化**: 技術者向け・非技術者向け両方の可視化スキル

### 実行結果を5分で確認

```bash
# 1. セットアップ（初回のみ）
git clone <repository-url> && cd portfolio
uv sync
uv run python main.py --generate-data

# 2. ダッシュボード起動
uv run python main.py --dashboard
# → http://localhost:8501 をブラウザで開く

# 3. モデルトレーニング実行
uv run python main.py --train-model
```

**期待される実行時間**:
- セットアップ: 2〜3分
- ダッシュボード起動: 10秒
- モデルトレーニング: 2〜3分

## 🎯 プロジェクトハイライト

### 1. 実務レベルのコード品質

#### ✨ 設計パターンの適用

**Template Method Pattern** (`src/models/base_model.py`)
```python
class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y, **kwargs): pass

    @abstractmethod
    def predict(self, X): pass

    # 共通処理は基底クラスで実装
    def save_model(self, filepath): ...
    def load_model(self, filepath): ...
```

**Factory Pattern** (`src/data/loader.py`)
```python
class DataLoader:
    def load_csv(self, filename): ...
    def load_from_url(self, url): ...
    def load_from_sql(self, query): ...
    # 新しいデータソースを簡単に追加可能
```

**Strategy Pattern** (`src/models/classifier.py`)
```python
SUPPORTED_ALGORITHMS = {
    'xgboost': XGBClassifier,
    'lightgbm': LGBMClassifier,
    'random_forest': RandomForestClassifier
}
# アルゴリズムを動的に切り替え可能
```

#### ✨ SOLID原則の遵守

- **単一責任の原則**: 各クラスは1つの責務のみ
- **開放閉鎖の原則**: 拡張に開いており、修正に閉じている
- **依存性逆転の原則**: 抽象に依存し、具象に依存しない

#### ✨ 型安全性

```python
def load_csv(
    self,
    filename: str,
    use_polars: bool = False,
    **kwargs
) -> Union[pd.DataFrame, pl.DataFrame]:
    """全関数に型ヒントを完備"""
```

#### ✨ テストカバレッジ

```bash
$ uv run pytest tests/ -v
============================== 9 passed ==============================
```

- データローダー: 4テスト
- プリプロセッサー: 5テスト
- 各モジュールが単独でテスト可能な設計

### 2. ビジネス価値の創出

#### 📊 顧客チャーン予測プロジェクト

**ビジネス課題**:
- 既存顧客の維持コスト < 新規顧客獲得コスト
- 解約前に対策を打つことで収益保護

**技術的アプローチ**:
```
データ収集 → 特徴量エンジニアリング → モデル比較 → 最適化 → デプロイ
```

**成果**:
- **ROC AUC 75.5%**: 実用レベルの予測精度
- **解約確率上位20%の顧客**: 実際の解約率60%以上（的中率3倍）
- **重要特徴の特定**: 契約タイプ、在籍期間、サポート通話数

**ビジネスインパクト試算**:
```
前提:
- 顧客数: 10,000人
- 年間解約率: 24%
- 1顧客あたり年間売上: ¥120,000
- 顧客獲得コスト: ¥50,000

施策前:
- 年間解約数: 2,400人
- 失われる売上: ¥288,000,000
- 新規獲得コスト: ¥120,000,000

施策後（上位20%に介入、30%挽回）:
- 挽回顧客数: 144人
- 売上保護: ¥17,280,000
- ROI: 介入コスト次第で10倍以上も可能
```

#### 📈 売上予測システム

**ビジネス課題**:
- 在庫最適化（過剰在庫・機会損失の削減）
- 需要予測に基づく人員配置

**技術的特徴**:
- 時系列分解（トレンド・季節性）
- カテゴリー別の需要パターン分析
- 週次・月次・年次の複合パターン対応

**活用シーン**:
- 週次売上会議での予測数値提供
- 仕入れ計画の最適化
- セール期間の売上予測

#### 🛒 マーケットバスケット分析

**ビジネス課題**:
- クロスセル機会の発見
- 店舗レイアウトの最適化

**発見された洞察**:
- 「牛乳 → 卵・パン」の高い関連性
- 「コーヒー → 砂糖・牛乳」のバンドル販売機会
- 「パスタ → トマト・玉ねぎ」の関連陳列効果

**施策例**:
- 関連商品の近接配置で売上向上
- バンドルセールの実施
- レコメンドエンジンへの活用

### 3. エンドツーエンドの実装力

#### 完全なMLパイプライン

```
┌──────────────────────────────────────────────────────────┐
│                     データ収集                            │
│  - CSV, JSON, SQL, Webスクレイピング対応                 │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────┐
│                   データ検証                              │
│  - スキーマチェック                                        │
│  - 品質レポート生成                                        │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────┐
│                   前処理                                  │
│  - 欠損値処理、外れ値除去                                  │
│  - エンコーディング、スケーリング                          │
│  - 特徴量エンジニアリング                                  │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────┐
│                   モデル学習                              │
│  - アルゴリズム比較                                        │
│  - ハイパーパラメータ最適化                                │
│  - 交差検証                                               │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────┐
│                   評価・検証                              │
│  - 複数メトリクスでの評価                                  │
│  - 特徴量重要度分析                                        │
│  - 混同行列、ROC曲線                                      │
└────────────┬─────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────┐
│                   デプロイ                                │
│  - モデル・前処理パイプラインの保存                        │
│  - インタラクティブダッシュボード                          │
│  - 推論API準備完了                                        │
└──────────────────────────────────────────────────────────┘
```

### 4. プロダクション品質の配慮

#### ✅ 再現性の保証

```python
# すべての乱数シードを固定
RANDOM_STATE = 42

# 前処理パイプラインの保存
preprocessor.save('churn_preprocessor.joblib')

# 依存関係のロック
uv.lock  # バージョン固定で環境再現
```

#### ✅ エラーハンドリング

```python
def load_csv(self, filename: str):
    filepath = self.raw_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)
```

#### ✅ ロギングと監視

```python
def train(self, X, y, verbose: bool = True):
    if verbose:
        print(f"Training {self.algorithm} classifier...")
    # 学習進捗を表示
```

#### ✅ データバリデーション

```python
class DataQualityReport(BaseModel):
    """Pydanticによる型安全なレポート"""
    row_count: int
    quality_score: float = Field(ge=0, le=100)
    issues: List[str] = []
```

## 💡 技術的な深掘りポイント（面接時の質問項目）

### データエンジニアリング

**Q: 大規模データの処理はどう考えていますか？**

A: 現在はPandasベースですが、以下の拡張を想定しています：
- **Polars**: 100万行以上のデータはPolarsで10倍高速化
- **Dask**: メモリに載らないデータの分散処理
- **データパーティショニング**: 日付やカテゴリでパーティション分割
- **チャンク処理**: イテレータで順次処理

```python
# すでに実装済み（loader.py）
def load_csv(self, use_polars: bool = False):
    if use_polars:
        return pl.read_csv(filepath)  # 高速処理
    return pd.read_csv(filepath)
```

### 機械学習

**Q: モデルの汎化性能をどう担保していますか？**

A: 複数の手法で過学習を防いでいます：
- **交差検証**: 5-fold CVで安定性を確認
- **Early Stopping**: eval_setで過学習検出
- **正則化**: XGBoost/LightGBMの正則化パラメータ調整
- **Train/Valid/Test分離**: 厳密な評価データ管理

```python
# 実装例（trainer.py）
def cross_validate(self, X, y, cv=5):
    scores = cross_val_score(self.model, X, y, cv=cv)
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }
```

**Q: 特徴量エンジニアリングの考え方は？**

A: ドメイン知識とデータ駆動の両方を活用：
- **相互作用特徴**: `tenure_months × monthly_charges` = LTV代理変数
- **時系列特徴**: 曜日、月、四半期の周期性を捕捉
- **集約特徴**: カテゴリ別の統計量（平均、標準偏差）

**Q: 説明可能性はどう考えていますか？**

A: 現在は以下を実装：
- **特徴量重要度**: XGBoostの`feature_importances_`
- **モデル比較**: 複数アルゴリズムで一貫性確認

今後の拡張：
- **SHAP値**: 個別予測の説明
- **LIME**: 局所的な説明
- **部分依存プロット**: 特徴量の影響を可視化

### システム設計

**Q: マイクロサービス化するとしたら？**

A: 以下のサービスに分割可能な設計：
```
┌─────────────────┐     ┌─────────────────┐
│ Data Service    │────▶│ Feature Service │
│ (ETL Pipeline)  │     │ (Preprocessing) │
└─────────────────┘     └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │ Model Service   │
                        │ (Prediction API)│
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │ Dashboard       │
                        │ (Visualization) │
                        └─────────────────┘
```

各モジュールは独立しているため、API化が容易です。

**Q: データベースとの連携は？**

A: すでにSQLAlchemy対応を実装：

```python
# loader.py に実装済み
def load_from_sql(self, query: str, connection_string: str):
    engine = create_engine(connection_string)
    return pd.read_sql(query, engine)
```

PostgreSQL、MySQL、SQLiteなど主要DBに対応。

### デプロイ・運用

**Q: 本番環境へのデプロイはどう考えていますか？**

A: 以下のようなフローを想定：

**CI/CD**:
```yaml
# .github/workflows/ci.yml（実装例）
- Run tests: pytest
- Code quality: black, ruff
- Build Docker image
- Deploy to staging
- Integration tests
- Deploy to production
```

**コンテナ化**:
```dockerfile
FROM python:3.12-slim
COPY . /app
RUN uv sync
CMD ["streamlit", "run", "dashboards/app.py"]
```

**モニタリング**:
- ログ収集（CloudWatch, Datadog）
- メトリクス監視（Prometheus, Grafana）
- モデルドリフト検出

**Q: A/Bテストはどう実装しますか？**

A: 以下の設計を想定：
```python
class ABTestFramework:
    def assign_variant(self, user_id):
        """ユーザーをA/Bグループに割り当て"""

    def track_outcome(self, user_id, metric):
        """成果を記録"""

    def analyze(self):
        """統計的有意差を検定"""
```

## 🚀 即戦力性のアピール

### 1週間でできること

**Week 1**: 既存システムの理解、コードベースの把握
**Week 2**: 新機能の設計・実装開始
**Week 3**: テスト・レビュー対応
**Week 4**: 本番デプロイ・モニタリング

### 学習能力

本ポートフォリオ制作で新規習得した技術：
- **Polars**: 高速データ処理ライブラリ
- **Streamlit**: インタラクティブダッシュボード
- **uv**: 最新のPythonパッケージマネージャー
- **Pydantic**: データバリデーション

**学習方法**: 公式ドキュメント → 実装 → ブログ記事で学習内容を整理

### 協働スキル

- **ドキュメント**: すべてのモジュールにDocstrings完備
- **コードレビュー**: 第三者が読める明確なコード
- **テスト**: 後続開発者が安心して変更できる設計
- **バージョン管理**: Git flow、コミットメッセージ規約の遵守

## 📞 面接での質問例

### 推奨される質問

以下のような質問をいただけると、より深い技術議論ができます：

1. **技術選定**: 「なぜXGBoostを選んだのか？LightGBMとの違いは？」
2. **トレードオフ**: 「精度と解釈性のバランスはどう考えた？」
3. **スケーラビリティ**: 「データが100倍になったらどう対応する？」
4. **ビジネス理解**: 「このモデルをどうビジネスに組み込む？」
5. **改善提案**: 「このコードで改善できる点は？」

### 回答準備済みトピック

- データ前処理の設計思想
- モデル選定の理由
- ハイパーパラメータチューニング戦略
- プロダクション環境での考慮事項
- チーム開発での協働方法

## 📚 関連ドキュメント

- [README_ja.md](../README_ja.md): プロジェクト概要
- [ARCHITECTURE_ja.md](./ARCHITECTURE_ja.md): 技術設計詳細
- [SETUP_GUIDE_ja.md](./SETUP_GUIDE_ja.md): 環境構築手順

## 📧 連絡先

ポートフォリオに関するご質問・フィードバックはお気軽にどうぞ：

- GitHub: [あなたのプロフィール]
- LinkedIn: [あなたのLinkedIn]
- Email: [メールアドレス]

---

**最終更新**: 2026年1月

このポートフォリオは、**データ分析エンジニアとしての即戦力性**を証明するために作成しました。
実際のコードと実行結果をご確認いただき、技術力を評価いただければ幸いです。
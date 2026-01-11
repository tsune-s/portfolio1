"""Train customer churn prediction model."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator
from src.models.classifier import ClassifierModel
from src.models.trainer import ModelTrainer
from src.visualization.plots import DataVisualizer


def main():
    """Main training pipeline for customer churn prediction."""
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("=" * 70)

    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    validator = DataValidator()
    visualizer = DataVisualizer()

    # Step 1: Load data
    print("\n📊 Step 1: Loading data...")
    df = data_loader.load_csv("customer_churn.csv")
    print(f"Loaded {len(df)} records with {len(df.columns)} features")

    # Step 2: Data validation
    print("\n🔍 Step 2: Validating data quality...")
    quality_report = validator.generate_quality_report(df, "Customer Churn Dataset")
    validator.print_quality_report(quality_report)

    # Step 3: Data preprocessing
    print("\n⚙️  Step 3: Preprocessing data...")

    # Remove customer_id as it's not a feature
    df_processed = df.drop('customer_id', axis=1)

    # Encode categorical variables
    categorical_cols = ['gender', 'contract_type', 'internet_service',
                       'online_security', 'tech_support', 'streaming_tv', 'payment_method']
    print(f"Encoding {len(categorical_cols)} categorical features...")
    df_processed = preprocessor.encode_categorical(df_processed, categorical_cols, method='label')

    # Scale numerical features
    numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'num_support_calls']
    print(f"Scaling {len(numerical_cols)} numerical features...")
    df_processed = preprocessor.scale_features(df_processed, numerical_cols, method='standard')

    # Create interaction features
    print("Creating interaction features...")
    df_processed = preprocessor.create_interaction_features(
        df_processed,
        [('tenure_months', 'monthly_charges'), ('age', 'tenure_months')]
    )

    # Prepare features and target
    X = df_processed.drop('churn', axis=1)
    y = df_processed['churn']

    print(f"Final feature matrix: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Step 4: Train-test split
    print("\n🔀 Step 4: Splitting data...")
    trainer = ModelTrainer(None)
    X_train, X_test, y_train, y_test = trainer.train_test_split(X, y, test_size=0.2)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Step 5: Model comparison
    print("\n🤖 Step 5: Comparing models...")

    models_to_compare = {}

    # XGBoost
    xgb_model = ClassifierModel(
        algorithm='xgboost',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    models_to_compare['XGBoost'] = xgb_model.model

    # LightGBM
    lgbm_model = ClassifierModel(
        algorithm='lightgbm',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    models_to_compare['LightGBM'] = lgbm_model.model

    # Random Forest
    rf_model = ClassifierModel(
        algorithm='random_forest',
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    models_to_compare['Random Forest'] = rf_model.model

    comparison_trainer = ModelTrainer(None)
    results = comparison_trainer.compare_models(
        models_to_compare,
        X_train,
        y_train,
        cv=5,
        scoring='roc_auc'
    )

    # Step 6: Train best model (XGBoost typically performs well)
    print("\n🎯 Step 6: Training final model (XGBoost)...")
    final_model = ClassifierModel(
        algorithm='xgboost',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )

    final_model.train(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        verbose=True
    )

    # Step 7: Evaluate model
    print("\n📈 Step 7: Evaluating model...")
    metrics = final_model.evaluate(X_test, y_test, verbose=True)

    # Step 8: Feature importance
    print("\n🔑 Step 8: Analyzing feature importance...")
    feature_importance = final_model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

    # Step 9: Save model
    print("\n💾 Step 9: Saving model...")
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "customer_churn_model"
    final_model.save_model(model_path)

    # Step 10: Save preprocessing objects
    print("\n💾 Step 10: Saving preprocessing objects...")
    import joblib
    preprocessor_path = models_dir / "churn_preprocessor.joblib"
    joblib.dump({
        'scalers': preprocessor.scalers,
        'encoders': preprocessor.encoders,
        'feature_names': X.columns.tolist()
    }, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nFinal Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    print(f"\nModel saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

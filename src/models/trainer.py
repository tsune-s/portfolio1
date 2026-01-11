"""Model training utilities with hyperparameter optimization."""

from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split


class ModelTrainer:
    """Advanced model training utilities."""

    def __init__(self, model, random_state: int = 42):
        """Initialize trainer.

        Args:
            model: Model instance to train
            random_state: Random state for reproducibility
        """
        self.model = model
        self.random_state = random_state
        self.best_params = None
        self.cv_results = None

    def train_test_split(
        self,
        X,
        y,
        test_size: float = 0.2,
        stratify: bool = True
    ) -> tuple:
        """Split data into train and test sets.

        Args:
            X: Features
            y: Labels
            test_size: Proportion of data for testing
            stratify: Whether to stratify split

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        stratify_param = y if stratify else None

        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

    def cross_validate(
        self,
        X,
        y,
        cv: int = 5,
        scoring: str = 'accuracy',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Perform cross-validation.

        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
            verbose: Whether to print results

        Returns:
            Dictionary with cross-validation results
        """
        scores = cross_val_score(
            self.model.model,
            X, y,
            cv=cv,
            scoring=scoring
        )

        results = {
            'scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max()
        }

        if verbose:
            print(f"\nCross-Validation Results ({cv} folds):")
            print("=" * 50)
            print(f"Scores: {scores}")
            print(f"Mean {scoring}: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
            print(f"Min: {results['min_score']:.4f}, Max: {results['max_score']:.4f}")
            print("=" * 50)

        self.cv_results = results
        return results

    def grid_search(
        self,
        X,
        y,
        param_grid: Dict[str, List],
        cv: int = 5,
        scoring: str = 'accuracy',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Perform grid search for hyperparameter tuning.

        Args:
            X: Features
            y: Labels
            param_grid: Dictionary with parameter names as keys and lists of values
            cv: Number of cross-validation folds
            scoring: Scoring metric
            verbose: Whether to print progress

        Returns:
            Dictionary with best parameters and scores
        """
        if verbose:
            print(f"Starting Grid Search with {cv}-fold CV...")
            print(f"Parameter grid: {param_grid}")

        grid_search = GridSearchCV(
            self.model.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            verbose=1 if verbose else 0,
            n_jobs=-1
        )

        grid_search.fit(X, y)

        self.best_params = grid_search.best_params_
        self.model.model = grid_search.best_estimator_
        self.model.is_trained = True

        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

        if verbose:
            print("\nGrid Search Results:")
            print("=" * 50)
            print(f"Best parameters: {results['best_params']}")
            print(f"Best {scoring}: {results['best_score']:.4f}")
            print("=" * 50)

        return results

    def random_search(
        self,
        X,
        y,
        param_distributions: Dict[str, Any],
        n_iter: int = 50,
        cv: int = 5,
        scoring: str = 'accuracy',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Perform randomized search for hyperparameter tuning.

        Args:
            X: Features
            y: Labels
            param_distributions: Dictionary with parameter distributions
            n_iter: Number of parameter settings to sample
            cv: Number of cross-validation folds
            scoring: Scoring metric
            verbose: Whether to print progress

        Returns:
            Dictionary with best parameters and scores
        """
        if verbose:
            print(f"Starting Randomized Search with {n_iter} iterations...")

        random_search = RandomizedSearchCV(
            self.model.model,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            verbose=1 if verbose else 0,
            n_jobs=-1,
            random_state=self.random_state
        )

        random_search.fit(X, y)

        self.best_params = random_search.best_params_
        self.model.model = random_search.best_estimator_
        self.model.is_trained = True

        results = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }

        if verbose:
            print("\nRandomized Search Results:")
            print("=" * 50)
            print(f"Best parameters: {results['best_params']}")
            print(f"Best {scoring}: {results['best_score']:.4f}")
            print("=" * 50)

        return results

    def compare_models(
        self,
        models: Dict[str, Any],
        X,
        y,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models using cross-validation.

        Args:
            models: Dictionary mapping model names to model instances
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Dictionary with results for each model
        """
        results = {}

        print(f"\nComparing {len(models)} models using {cv}-fold CV...")
        print("=" * 60)

        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }

            print(f"{name}: {results[name]['mean_score']:.4f} (+/- {results[name]['std_score']:.4f})")

        print("=" * 60)

        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['mean_score'])
        print(f"\nBest model: {best_model[0]} with score {best_model[1]['mean_score']:.4f}")

        return results

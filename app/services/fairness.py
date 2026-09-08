# Fairness evaluation, model comparison, and mitigation
# for AI-powered candidate screening.

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    selection_rate,
)
from fairlearn.reductions import (
    DemographicParity,
    ExponentiatedGradient,
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# Predictive features
# ============================================================

# IMPORTANT:
# protected_group is deliberately NOT included here.
#
# The model uses job-relevant candidate features only.
# protected_group is isolated for fairness evaluation
# and fairness mitigation.

FEATURE_COLUMNS = [
    "required_skill_score",
    "experience_score",
    "education_score",
    "semantic_score",
    "preferred_skill_score",
]

TARGET_COLUMN = "selected"
PROTECTED_COLUMN = "protected_group"

TEST_SIZE = 0.40
RANDOM_STATE = 42


# ============================================================
# Model configuration
# ============================================================

KNN_NEIGHBORS = 5


# ============================================================
# Fairness Model
# ============================================================

class FairnessModel:
    """
    Candidate screening model with:

    1. Logistic Regression baseline
    2. K-Nearest Neighbors comparison model
    3. Fairness-aware ExponentiatedGradient model

    The protected attribute is never used as a predictive
    feature.

    Models:
        Logistic Regression
            Main baseline classifier.

        KNN
            Non-linear distance-based comparison classifier.

        ExponentiatedGradient + DemographicParity
            Fairness-aware classifier.
    """

    def __init__(self):

        # ----------------------------------------------------
        # Feature scaler
        # ----------------------------------------------------

        self.scaler = StandardScaler()

        # ----------------------------------------------------
        # Logistic Regression baseline
        # ----------------------------------------------------

        self.model = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
        )

        # ----------------------------------------------------
        # KNN comparison model
        # ----------------------------------------------------

        self.knn_model = KNeighborsClassifier(
            n_neighbors=KNN_NEIGHBORS,
            weights="distance",
        )

        # ----------------------------------------------------
        # Fairness-aware model
        # ----------------------------------------------------

        self.fair_model = ExponentiatedGradient(
            estimator=LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
            ),
            constraints=DemographicParity(),
        )

        # ----------------------------------------------------
        # Stored test data
        # ----------------------------------------------------

        self.X_test = None
        self.y_test = None
        self.sensitive_test = None

        self.is_trained = False


    # ========================================================
    # Training
    # ========================================================

    def train(self, dataframe: pd.DataFrame) -> dict:
        """
        Train:

        - Logistic Regression
        - KNN
        - Fairness-aware ExponentiatedGradient

        Returns baseline Logistic Regression metrics.
        """

        self._validate_dataframe(dataframe)

        X = dataframe[FEATURE_COLUMNS].copy()
        y = dataframe[TARGET_COLUMN].copy()
        sensitive_features = dataframe[PROTECTED_COLUMN].copy()

        # ----------------------------------------------------
        # Train/test split
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test,
            sensitive_train,
            sensitive_test,
        ) = train_test_split(
            X,
            y,
            sensitive_features,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        # ----------------------------------------------------
        # Scale numerical features
        # ----------------------------------------------------

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_scaled_df = pd.DataFrame(
            X_train_scaled,
            columns=FEATURE_COLUMNS,
        )

        X_test_scaled_df = pd.DataFrame(
            X_test_scaled,
            columns=FEATURE_COLUMNS,
        )

        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------

        self.model.fit(
            X_train_scaled_df,
            y_train,
        )

        # ----------------------------------------------------
        # KNN
        # ----------------------------------------------------

        self.knn_model.fit(
            X_train_scaled_df,
            y_train,
        )

        # ----------------------------------------------------
        # Fairness-aware model
        # ----------------------------------------------------

        # protected_group is supplied separately to Fairlearn.
        # It is NOT part of the predictive feature matrix.

        self.fair_model.fit(
            X_train_scaled_df,
            y_train,
            sensitive_features=sensitive_train,
        )

        # ----------------------------------------------------
        # Store held-out test data
        # ----------------------------------------------------

        self.X_test = X_test_scaled_df
        self.y_test = y_test.reset_index(drop=True)
        self.sensitive_test = sensitive_test.reset_index(drop=True)

        self.is_trained = True

        # ----------------------------------------------------
        # Baseline evaluation
        # ----------------------------------------------------

        predictions = self.model.predict(
            self.X_test
        )

        probabilities = self.model.predict_proba(
            self.X_test
        )[:, 1]

        return self._performance_metrics(
            self.y_test,
            predictions,
            probabilities,
        )


    # ========================================================
    # Baseline evaluation
    # ========================================================

    def evaluate_baseline(self) -> dict:
        """
        Evaluate Logistic Regression performance
        and fairness on the held-out test set.
        """

        self._ensure_trained()

        predictions = self.model.predict(
            self.X_test
        )

        probabilities = self.model.predict_proba(
            self.X_test
        )[:, 1]

        return {
            "performance": self._performance_metrics(
                self.y_test,
                predictions,
                probabilities,
            ),
            "fairness": self._fairness_metrics(
                self.y_test,
                predictions,
                self.sensitive_test,
            ),
        }


    # ========================================================
    # KNN evaluation
    # ========================================================

    def evaluate_knn(self) -> dict:
        """
        Evaluate KNN performance and fairness.
        """

        self._ensure_trained()

        predictions = self.knn_model.predict(
            self.X_test
        )

        probabilities = self._predict_knn_probabilities(
            self.X_test
        )

        return {
            "performance": self._performance_metrics(
                self.y_test,
                predictions,
                probabilities,
            ),
            "fairness": self._fairness_metrics(
                self.y_test,
                predictions,
                self.sensitive_test,
            ),
        }


    # ========================================================
    # Backwards-compatible baseline fairness method
    # ========================================================

    def evaluate_fairness(
        self,
        dataframe: pd.DataFrame | None = None,
    ) -> dict:
        """
        Evaluate fairness of the Logistic Regression baseline.

        If dataframe is None:
            Uses held-out test data.

        Otherwise:
            Uses the supplied complete evaluation dataset.
        """

        if dataframe is None:

            self._ensure_trained()

            predictions = self.model.predict(
                self.X_test
            )

            return self._fairness_metrics(
                self.y_test,
                predictions,
                self.sensitive_test,
            )

        self._validate_dataframe(dataframe)

        X = dataframe[FEATURE_COLUMNS]
        y = dataframe[TARGET_COLUMN]
        sensitive_features = dataframe[PROTECTED_COLUMN]

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=FEATURE_COLUMNS,
        )

        predictions = self.model.predict(
            X_scaled
        )

        return self._fairness_metrics(
            y,
            predictions,
            sensitive_features,
        )


    # ========================================================
    # Fairness-aware evaluation
    # ========================================================

    def evaluate_fairness_aware_model(self) -> dict:
        """
        Evaluate the fairness-aware model.
        """

        self._ensure_trained()

        predictions = self.fair_model.predict(
            self.X_test
        )

        probabilities = self._predict_fair_probabilities(
            self.X_test
        )

        return {
            "performance": self._performance_metrics(
                self.y_test,
                predictions,
                probabilities,
            ),
            "fairness": self._fairness_metrics(
                self.y_test,
                predictions,
                self.sensitive_test,
            ),
        }


    # ========================================================
    # Model comparison
    # ========================================================

    def compare_models(self) -> dict:
        """
        Compare:

        1. Logistic Regression
        2. KNN
        3. Fairness-aware ExponentiatedGradient

        This provides the main model comparison for the
        assignment.
        """

        self._ensure_trained()

        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------

        baseline_predictions = self.model.predict(
            self.X_test
        )

        baseline_probabilities = self.model.predict_proba(
            self.X_test
        )[:, 1]

        baseline_performance = self._performance_metrics(
            self.y_test,
            baseline_predictions,
            baseline_probabilities,
        )

        baseline_fairness = self._fairness_metrics(
            self.y_test,
            baseline_predictions,
            self.sensitive_test,
        )

        # ----------------------------------------------------
        # KNN
        # ----------------------------------------------------

        knn_predictions = self.knn_model.predict(
            self.X_test
        )

        knn_probabilities = self._predict_knn_probabilities(
            self.X_test
        )

        knn_performance = self._performance_metrics(
            self.y_test,
            knn_predictions,
            knn_probabilities,
        )

        knn_fairness = self._fairness_metrics(
            self.y_test,
            knn_predictions,
            self.sensitive_test,
        )

        # ----------------------------------------------------
        # Fairness-aware model
        # ----------------------------------------------------

        fair_predictions = self.fair_model.predict(
            self.X_test
        )

        fair_probabilities = self._predict_fair_probabilities(
            self.X_test
        )

        fair_performance = self._performance_metrics(
            self.y_test,
            fair_predictions,
            fair_probabilities,
        )

        fair_fairness = self._fairness_metrics(
            self.y_test,
            fair_predictions,
            self.sensitive_test,
        )

        # ----------------------------------------------------
        # Return complete comparison
        # ----------------------------------------------------

        return {
            "logistic_regression": {
                "performance": baseline_performance,
                "fairness": baseline_fairness,
            },
            "knn": {
                "configuration": {
                    "n_neighbors": KNN_NEIGHBORS,
                    "weights": "distance",
                },
                "performance": knn_performance,
                "fairness": knn_fairness,
            },
            "fairness_aware": {
                "performance": fair_performance,
                "fairness": fair_fairness,
            },
        }


    # ========================================================
    # Save comparison
    # ========================================================

    def save_comparison(
        self,
        path: str = "data/evaluation/fairness_results.json",
    ) -> dict:
        """
        Save model comparison results to JSON.
        """

        results = self.compare_models()

        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                results,
                file,
                indent=2,
            )

        return results


    # ========================================================
    # Prediction
    # ========================================================

    def predict(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Generate Logistic Regression predictions
        for new candidates.
        """

        self._ensure_model_ready()

        self._validate_features(dataframe)

        X = dataframe[FEATURE_COLUMNS]

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=FEATURE_COLUMNS,
        )

        return self.model.predict(
            X_scaled
        )


    def predict_knn(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Generate KNN predictions for new candidates.
        """

        self._ensure_model_ready()

        self._validate_features(dataframe)

        X = dataframe[FEATURE_COLUMNS]

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=FEATURE_COLUMNS,
        )

        return self.knn_model.predict(
            X_scaled
        )


    def predict_fair(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Generate predictions using the fairness-aware model.
        """

        self._ensure_model_ready()

        self._validate_features(dataframe)

        X = dataframe[FEATURE_COLUMNS]

        X_scaled = self.scaler.transform(X)

        return self.fair_model.predict(
            X_scaled
        )


    # ========================================================
    # Save models
    # ========================================================

    def save_models(
        self,
        directory: str = "models",
    ) -> dict:
        """
        Save all trained models and the scaler.
        """

        self._ensure_trained()

        model_dir = Path(directory)

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        baseline_path = (
            model_dir / "baseline_model.joblib"
        )

        knn_path = (
            model_dir / "knn_model.joblib"
        )

        fair_path = (
            model_dir / "fairness_model.joblib"
        )

        scaler_path = (
            model_dir / "scaler.joblib"
        )

        joblib.dump(
            self.model,
            baseline_path,
        )

        joblib.dump(
            self.knn_model,
            knn_path,
        )

        joblib.dump(
            self.fair_model,
            fair_path,
        )

        joblib.dump(
            self.scaler,
            scaler_path,
        )

        return {
            "baseline_model": str(baseline_path),
            "knn_model": str(knn_path),
            "fairness_model": str(fair_path),
            "scaler": str(scaler_path),
        }


    # ========================================================
    # Load models
    # ========================================================

    def load_models(
        self,
        directory: str = "models",
    ):
        """
        Load trained models and scaler.

        The KNN model is optional during loading so that
        previously saved model sets remain backwards compatible.

        If KNN is absent, the model instance is still usable for
        existing Logistic Regression screening, but KNN comparison
        requires retraining.
        """

        model_dir = Path(directory)

        baseline_path = (
            model_dir / "baseline_model.joblib"
        )

        knn_path = (
            model_dir / "knn_model.joblib"
        )

        fair_path = (
            model_dir / "fairness_model.joblib"
        )

        scaler_path = (
            model_dir / "scaler.joblib"
        )

        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline model not found: {baseline_path}"
            )

        if not fair_path.exists():
            raise FileNotFoundError(
                f"Fairness-aware model not found: {fair_path}"
            )

        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found: {scaler_path}"
            )

        self.model = joblib.load(
            baseline_path
        )

        self.fair_model = joblib.load(
            fair_path
        )

        self.scaler = joblib.load(
            scaler_path
        )

        # ----------------------------------------------------
        # KNN model
        # ----------------------------------------------------

        if knn_path.exists():

            self.knn_model = joblib.load(
                knn_path
            )

        else:

            # Keep a valid object for compatibility.
            # It is not trained until train() is called.
            self.knn_model = KNeighborsClassifier(
                n_neighbors=KNN_NEIGHBORS,
                weights="distance",
            )

        self.is_trained = True

        return {
            "status": "loaded",
            "baseline_model": str(baseline_path),
            "knn_model": str(knn_path),
            "fairness_model": str(fair_path),
            "scaler": str(scaler_path),
            "knn_available": knn_path.exists(),
        }


    # ========================================================
    # Performance metrics
    # ========================================================

    @staticmethod
    def _performance_metrics(
        y_true,
        predictions,
        probabilities=None,
    ) -> dict:
        """
        Calculate standard classification metrics.
        """

        metrics = {
            "accuracy": round(
                accuracy_score(
                    y_true,
                    predictions,
                ),
                4,
            ),

            "precision": round(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                4,
            ),

            "recall": round(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                4,
            ),

            "f1": round(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                4,
            ),
        }

        # ROC-AUC requires both classes.
        if (
            probabilities is not None
            and len(set(y_true)) == 2
        ):

            metrics["roc_auc"] = round(
                roc_auc_score(
                    y_true,
                    probabilities,
                ),
                4,
            )

        else:

            metrics["roc_auc"] = None

        return metrics


    # ========================================================
    # Fairness metrics
    # ========================================================

    @staticmethod
    def _fairness_metrics(
        y_true,
        predictions,
        sensitive_features,
    ) -> dict:
        """
        Calculate:

        - selection rate by protected group
        - demographic parity difference
        """

        metric_frame = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
            },
            y_true=y_true,
            y_pred=predictions,
            sensitive_features=sensitive_features,
        )

        dp_difference = demographic_parity_difference(
            y_true=y_true,
            y_pred=predictions,
            sensitive_features=sensitive_features,
        )

        by_group = metric_frame.by_group

        selection_rates = {
            str(group): round(
                float(rate),
                4,
            )
            for group, rate
            in by_group["selection_rate"].items()
        }

        return {
            "selection_rate_by_group": selection_rates,
            "demographic_parity_difference": round(
                float(dp_difference),
                4,
            ),
        }


    # ========================================================
    # KNN probability helper
    # ========================================================

    def _predict_knn_probabilities(
        self,
        X,
    ):
        """
        Return probability of the positive class for KNN.
        """

        try:

            probabilities = (
                self.knn_model.predict_proba(X)
            )

            if probabilities.ndim == 2:
                return probabilities[:, 1]

            return probabilities

        except (
            AttributeError,
            ValueError,
        ):

            return None


    # ========================================================
    # Fair model probability helper
    # ========================================================

    def _predict_fair_probabilities(
        self,
        X,
    ):
        """
        ExponentiatedGradient may not expose probabilities
        consistently.

        Return None when unavailable so ROC-AUC is omitted.
        """

        try:

            probabilities = (
                self.fair_model.predict_proba(X)
            )

            if probabilities.ndim == 2:
                return probabilities[:, 1]

            return probabilities

        except (
            AttributeError,
            ValueError,
        ):

            return None


    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ):
        """
        Validate the complete evaluation dataset.
        """

        if (
            dataframe is None
            or dataframe.empty
        ):

            raise ValueError(
                "Evaluation dataset is empty."
            )

        required_columns = (
            FEATURE_COLUMNS
            + [
                PROTECTED_COLUMN,
                TARGET_COLUMN,
            ]
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        # ----------------------------------------------------
        # Numerical feature validation
        # ----------------------------------------------------

        for column in FEATURE_COLUMNS:

            if dataframe[column].isna().any():

                raise ValueError(
                    f"Feature column '{column}' "
                    "contains missing values."
                )

        # ----------------------------------------------------
        # Protected attribute validation
        # ----------------------------------------------------

        if dataframe[PROTECTED_COLUMN].isna().any():

            raise ValueError(
                "protected_group contains missing values."
            )

        # ----------------------------------------------------
        # Target validation
        # ----------------------------------------------------

        if dataframe[TARGET_COLUMN].isna().any():

            raise ValueError(
                "selected contains missing values."
            )

        unique_targets = set(
            dataframe[TARGET_COLUMN].unique()
        )

        if not unique_targets.issubset({0, 1}):

            raise ValueError(
                "selected must contain only 0 and 1."
            )

        if len(unique_targets) < 2:

            raise ValueError(
                "selected must contain both classes: 0 and 1."
            )

        # ----------------------------------------------------
        # Protected group validation
        # ----------------------------------------------------

        if dataframe[PROTECTED_COLUMN].nunique() < 2:

            raise ValueError(
                "protected_group must contain at least "
                "two groups for fairness evaluation."
            )


    @staticmethod
    def _validate_features(
        dataframe: pd.DataFrame,
    ):
        """
        Validate feature columns used for prediction.
        """

        if (
            dataframe is None
            or dataframe.empty
        ):

            raise ValueError(
                "Prediction dataset is empty."
            )

        missing_columns = [
            column
            for column in FEATURE_COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing feature columns: "
                + ", ".join(missing_columns)
            )

        for column in FEATURE_COLUMNS:

            if dataframe[column].isna().any():

                raise ValueError(
                    f"Feature column '{column}' "
                    "contains missing values."
                )


    # ========================================================
    # State validation
    # ========================================================

    def _ensure_model_ready(self):
        """
        Ensure persisted/trained models are available.
        """

        if (
            not self.is_trained
            or self.scaler is None
            or self.model is None
            or self.fair_model is None
        ):

            raise RuntimeError(
                "Models are not available. "
                "Train or load the models first."
            )


    def _ensure_trained(self):
        """
        Ensure models were trained and evaluation data exists.
        """

        self._ensure_model_ready()

        if (
            self.X_test is None
            or self.y_test is None
            or self.sensitive_test is None
        ):

            raise RuntimeError(
                "Evaluation data is not available. "
                "Train the model before running evaluation."
            )


# ============================================================
# Dataset loader
# ============================================================

def load_evaluation_data(
    path: str = "data/evaluation/candidates.csv",
) -> pd.DataFrame:
    """
    Load candidate evaluation dataset.
    """

    dataframe = pd.read_csv(path)

    return dataframe


# ============================================================
# Shared model instance
# ============================================================

fairness_model = FairnessModel()

try:

    fairness_model.load_models()

except FileNotFoundError:

    pass
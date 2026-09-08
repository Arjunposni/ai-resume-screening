from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from app.services.fairness import (
    FEATURE_COLUMNS,
    FairnessModel,
)


class SHAPExplainer:
    """
    Explain candidate screening predictions using SHAP.

    The Logistic Regression model is explained using
    SHAP LinearExplainer.

    Protected attributes are deliberately excluded from
    the explanation because they are not predictive features.
    """

    def __init__(
        self,
        fairness_model: FairnessModel,
    ):
        self.fairness_model = fairness_model

        self._ensure_model_ready()

        # SHAP LinearExplainer is appropriate for
        # Logistic Regression.
        self.explainer = shap.LinearExplainer(
            self.fairness_model.model,
            self.fairness_model.X_test,
        )

    # ========================================================
    # Explain one candidate
    # ========================================================

    def explain_candidate(
        self,
        features: dict | pd.DataFrame,
    ) -> dict:
        """
        Generate a SHAP explanation for one candidate.

        Args:
            features:
                Dictionary or one-row DataFrame containing
                the five screening features.

        Returns:
            JSON-serializable explanation.
        """

        dataframe = self._prepare_features(
            features
        )

        # Apply the SAME scaler used during model training.
        scaled = self.fairness_model.scaler.transform(
            dataframe[FEATURE_COLUMNS]
        )

        scaled_df = pd.DataFrame(
            scaled,
            columns=FEATURE_COLUMNS,
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        prediction = int(
            self.fairness_model.model.predict(
                scaled_df
            )[0]
        )

        probability = float(
            self.fairness_model.model.predict_proba(
                scaled_df
            )[0, 1]
        )

        # ----------------------------------------------------
        # SHAP values
        # ----------------------------------------------------

        shap_values = self.explainer(
            scaled_df
        )

        values = np.asarray(
            shap_values.values
        )

        # Binary classification normally produces
        # one SHAP value per feature for this explainer.
        if values.ndim == 2:
            feature_contributions = values[0]
        else:
            feature_contributions = values

        base_value = np.asarray(
            shap_values.base_values
        ).reshape(-1)[0]

        # ----------------------------------------------------
        # Build feature contribution output
        # ----------------------------------------------------

        contributions = []

        for index, feature in enumerate(
            FEATURE_COLUMNS
        ):

            contribution = float(
                feature_contributions[index]
            )

            contributions.append(
                {
                    "feature": feature,
                    "value": float(
                        dataframe.iloc[
                            0
                        ][feature]
                    ),
                    "shap_value": round(
                        contribution,
                        6,
                    ),
                    "impact": (
                        "positive"
                        if contribution > 0
                        else "negative"
                        if contribution < 0
                        else "neutral"
                    ),
                }
            )

        # Sort by absolute contribution so
        # the strongest factors appear first.
        contributions.sort(
            key=lambda item: abs(
                item["shap_value"]
            ),
            reverse=True,
        )

        return {
            "model": "logistic_regression",
            "prediction": prediction,
            "prediction_label": (
                "selected"
                if prediction == 1
                else "not_selected"
            ),
            "selection_probability": round(
                probability,
                6,
            ),
            "base_value": round(
                float(base_value),
                6,
            ),
            "feature_contributions": contributions,
        }

    # ========================================================
    # Explain multiple candidates
    # ========================================================

    def explain_candidates(
        self,
        dataframe: pd.DataFrame,
    ) -> list[dict]:
        """
        Generate SHAP explanations for multiple candidates.
        """

        if dataframe.empty:
            return []

        results = []

        for _, row in dataframe.iterrows():

            features = {
                feature: row[feature]
                for feature in FEATURE_COLUMNS
            }

            results.append(
                self.explain_candidate(
                    features
                )
            )

        return results

    # ========================================================
    # Save explanation
    # ========================================================

    def save_explanation(
        self,
        explanation: dict,
        path: str = (
            "data/evaluation/"
            "shap_explanation.json"
        ),
    ) -> dict:
        """
        Save a SHAP explanation to JSON.
        """

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
                explanation,
                file,
                indent=2,
            )

        return {
            "path": str(output_path),
            "status": "saved",
        }

    # ========================================================
    # Prepare feature input
    # ========================================================

    @staticmethod
    def _prepare_features(
        features: dict | pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert input into a validated one-row DataFrame.
        """

        if isinstance(
            features,
            dict,
        ):

            dataframe = pd.DataFrame(
                [features]
            )

        elif isinstance(
            features,
            pd.DataFrame,
        ):

            dataframe = features.copy()

        else:

            raise TypeError(
                "features must be a dict or pandas DataFrame."
            )

        missing = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in dataframe.columns
        ]

        if missing:

            raise ValueError(
                "Missing required features: "
                + ", ".join(missing)
            )

        dataframe = dataframe[
            FEATURE_COLUMNS
        ].copy()

        if dataframe.empty:

            raise ValueError(
                "Feature dataframe is empty."
            )

        if dataframe[
            FEATURE_COLUMNS
        ].isna().any().any():

            raise ValueError(
                "Feature values cannot contain NaN."
            )

        return dataframe

    # ========================================================
    # Model validation
    # ========================================================

    def _ensure_model_ready(self):
        """
        Ensure the trained Logistic Regression model,
        scaler, and reference data are available.
        """

        if not self.fairness_model.is_trained:

            raise RuntimeError(
                "FairnessModel is not trained."
            )

        if self.fairness_model.X_test is None:

            raise RuntimeError(
                "FairnessModel test data is unavailable."
            )

        if self.fairness_model.model is None:

            raise RuntimeError(
                "Logistic Regression model is unavailable."
            )

        if self.fairness_model.scaler is None:

            raise RuntimeError(
                "Feature scaler is unavailable."
            )
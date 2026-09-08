from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FEATURE_COLUMNS = [
    "required_skill_score",
    "experience_score",
    "education_score",
    "semantic_score",
    "preferred_skill_score",
]

TARGET_COLUMN = "selected"
PROTECTED_COLUMN = "protected_group"


class HiringEDA:
    """
    Exploratory Data Analysis for candidate screening data.

    Generates:
        - Class distribution
        - Feature distributions
        - Feature comparison by hiring outcome
        - Protected-group selection rates
        - Correlation matrix
        - JSON summary
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        output_dir: str = "data/evaluation/eda",
    ):
        self.dataframe = dataframe.copy()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._validate_dataframe()

    # ========================================================
    # Main analysis
    # ========================================================

    def run(self) -> dict:
        """
        Run the complete EDA pipeline.
        """

        summary = {
            "dataset": self.dataset_summary(),
            "class_distribution": self.class_distribution(),
            "feature_summary": self.feature_summary(),
            "selection_analysis": self.selection_analysis(),
            "protected_group_analysis": self.protected_group_analysis(),
            "correlation_analysis": self.correlation_analysis(),
            "generated_plots": [],
        }

        # Generate plots
        summary["generated_plots"].append(
            self.plot_class_distribution()
        )

        summary["generated_plots"].append(
            self.plot_feature_distributions()
        )

        summary["generated_plots"].append(
            self.plot_feature_vs_selection()
        )

        summary["generated_plots"].append(
            self.plot_protected_group_selection()
        )

        summary["generated_plots"].append(
            self.plot_correlation_matrix()
        )

        # Save summary
        summary_path = self.output_dir / "eda_summary.json"

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
            )

        summary["summary_file"] = str(summary_path)

        return summary

    # ========================================================
    # Dataset summary
    # ========================================================

    def dataset_summary(self) -> dict:
        """
        Return basic dataset statistics.
        """

        return {
            "total_candidates": int(len(self.dataframe)),
            "total_features": len(FEATURE_COLUMNS),
            "selected_candidates": int(
                (self.dataframe[TARGET_COLUMN] == 1).sum()
            ),
            "not_selected_candidates": int(
                (self.dataframe[TARGET_COLUMN] == 0).sum()
            ),
            "protected_groups": [
                str(group)
                for group in sorted(
                    self.dataframe[
                        PROTECTED_COLUMN
                    ].unique()
                )
            ],
        }

    # ========================================================
    # Class distribution
    # ========================================================

    def class_distribution(self) -> dict:
        """
        Calculate selected vs non-selected distribution.
        """

        counts = (
            self.dataframe[TARGET_COLUMN]
            .value_counts()
            .sort_index()
        )

        total = len(self.dataframe)

        return {
            "not_selected": int(
                counts.get(0, 0)
            ),
            "selected": int(
                counts.get(1, 0)
            ),
            "not_selected_percentage": round(
                (counts.get(0, 0) / total) * 100,
                2,
            ),
            "selected_percentage": round(
                (counts.get(1, 0) / total) * 100,
                2,
            ),
        }

    # ========================================================
    # Feature summary
    # ========================================================

    def feature_summary(self) -> dict:
        """
        Calculate descriptive statistics for screening features.
        """

        statistics = {}

        for feature in FEATURE_COLUMNS:

            series = self.dataframe[feature]

            statistics[feature] = {
                "mean": round(
                    float(series.mean()),
                    2,
                ),
                "median": round(
                    float(series.median()),
                    2,
                ),
                "minimum": round(
                    float(series.min()),
                    2,
                ),
                "maximum": round(
                    float(series.max()),
                    2,
                ),
                "standard_deviation": round(
                    float(series.std()),
                    2,
                ),
            }

        return statistics

    # ========================================================
    # Selection analysis
    # ========================================================

    def selection_analysis(self) -> dict:
        """
        Compare average feature scores between selected
        and non-selected candidates.
        """

        grouped = (
            self.dataframe
            .groupby(TARGET_COLUMN)[FEATURE_COLUMNS]
            .mean()
        )

        result = {}

        for target_value in [0, 1]:

            label = (
                "selected"
                if target_value == 1
                else "not_selected"
            )

            if target_value in grouped.index:

                result[label] = {
                    feature: round(
                        float(
                            grouped.loc[
                                target_value,
                                feature,
                            ]
                        ),
                        2,
                    )
                    for feature in FEATURE_COLUMNS
                }

        # Calculate difference
        if 0 in grouped.index and 1 in grouped.index:

            result["selected_minus_not_selected"] = {
                feature: round(
                    float(
                        grouped.loc[1, feature]
                        - grouped.loc[0, feature]
                    ),
                    2,
                )
                for feature in FEATURE_COLUMNS
            }

        return result

    # ========================================================
    # Protected group analysis
    # ========================================================

    def protected_group_analysis(self) -> dict:
        """
        Calculate candidate counts and selection rates
        for each protected group.
        """

        result = {}

        for group, group_df in (
            self.dataframe.groupby(
                PROTECTED_COLUMN
            )
        ):

            total = len(group_df)

            selected = int(
                (
                    group_df[TARGET_COLUMN] == 1
                ).sum()
            )

            selection_rate = (
                selected / total
                if total > 0
                else 0
            )

            result[str(group)] = {
                "candidate_count": total,
                "selected_count": selected,
                "not_selected_count": total - selected,
                "selection_rate": round(
                    selection_rate,
                    4,
                ),
                "selection_rate_percentage": round(
                    selection_rate * 100,
                    2,
                ),
            }

        # Calculate observed selection-rate difference
        rates = [
            values["selection_rate"]
            for values in result.values()
        ]

        if len(rates) >= 2:

            result["selection_rate_difference"] = round(
                max(rates) - min(rates),
                4,
            )

        return result

    # ========================================================
    # Correlation analysis
    # ========================================================

    def correlation_analysis(self) -> dict:
        """
        Calculate Pearson correlations between features
        and the selection outcome.
        """

        columns = FEATURE_COLUMNS + [
            TARGET_COLUMN
        ]

        correlation_matrix = (
            self.dataframe[columns]
            .corr()
            .round(4)
        )

        feature_target_correlations = {
            feature: round(
                float(
                    correlation_matrix.loc[
                        feature,
                        TARGET_COLUMN,
                    ]
                ),
                4,
            )
            for feature in FEATURE_COLUMNS
        }

        return {
            "feature_target_correlation": (
                feature_target_correlations
            ),
            "matrix": correlation_matrix.to_dict(),
        }

    # ========================================================
    # Plot: class distribution
    # ========================================================

    def plot_class_distribution(self) -> str:
        """
        Generate selected vs non-selected bar chart.
        """

        counts = [
            int(
                (
                    self.dataframe[TARGET_COLUMN] == 0
                ).sum()
            ),
            int(
                (
                    self.dataframe[TARGET_COLUMN] == 1
                ).sum()
            ),
        ]

        labels = [
            "Not Selected",
            "Selected",
        ]

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        bars = ax.bar(
            labels,
            counts,
        )

        ax.set_title(
            "Candidate Selection Distribution"
        )

        ax.set_xlabel(
            "Hiring Outcome"
        )

        ax.set_ylabel(
            "Number of Candidates"
        )

        for bar, count in zip(
            bars,
            counts,
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                str(count),
                ha="center",
                va="bottom",
            )

        fig.tight_layout()

        path = (
            self.output_dir
            / "class_distribution.png"
        )

        fig.savefig(
            path,
            dpi=150,
        )

        plt.close(fig)

        return str(path)

    # ========================================================
    # Plot: feature distributions
    # ========================================================

    def plot_feature_distributions(self) -> str:
        """
        Generate distribution plots for all screening features.
        """

        fig, axes = plt.subplots(
            3,
            2,
            figsize=(12, 12),
        )

        axes = axes.flatten()

        for index, feature in enumerate(
            FEATURE_COLUMNS
        ):

            axes[index].hist(
                self.dataframe[feature],
                bins=10,
            )

            axes[index].set_title(
                feature.replace(
                    "_",
                    " ",
                ).title()
            )

            axes[index].set_xlabel(
                "Score"
            )

            axes[index].set_ylabel(
                "Candidate Count"
            )

        # Hide unused axis if necessary
        for index in range(
            len(FEATURE_COLUMNS),
            len(axes),
        ):
            axes[index].set_visible(False)

        fig.suptitle(
            "Screening Feature Distributions",
            fontsize=14,
        )

        fig.tight_layout()

        path = (
            self.output_dir
            / "feature_distributions.png"
        )

        fig.savefig(
            path,
            dpi=150,
        )

        plt.close(fig)

        return str(path)

    # ========================================================
    # Plot: features vs selection
    # ========================================================

    def plot_feature_vs_selection(self) -> str:
        """
        Compare average feature scores for selected
        and non-selected candidates.
        """

        grouped = (
            self.dataframe
            .groupby(TARGET_COLUMN)[FEATURE_COLUMNS]
            .mean()
        )

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        x = range(
            len(FEATURE_COLUMNS)
        )

        width = 0.35

        not_selected = [
            grouped.loc[0, feature]
            for feature in FEATURE_COLUMNS
        ]

        selected = [
            grouped.loc[1, feature]
            for feature in FEATURE_COLUMNS
        ]

        x_not_selected = [
            value - width / 2
            for value in x
        ]

        x_selected = [
            value + width / 2
            for value in x
        ]

        ax.bar(
            x_not_selected,
            not_selected,
            width,
            label="Not Selected",
        )

        ax.bar(
            x_selected,
            selected,
            width,
            label="Selected",
        )

        ax.set_xticks(
            list(x)
        )

        ax.set_xticklabels(
            [
                feature.replace(
                    "_",
                    " ",
                ).title()
                for feature in FEATURE_COLUMNS
            ],
            rotation=20,
            ha="right",
        )

        ax.set_ylabel(
            "Average Score"
        )

        ax.set_title(
            "Average Screening Scores by Hiring Outcome"
        )

        ax.legend()

        fig.tight_layout()

        path = (
            self.output_dir
            / "feature_vs_selection.png"
        )

        fig.savefig(
            path,
            dpi=150,
        )

        plt.close(fig)

        return str(path)

    # ========================================================
    # Plot: protected group selection
    # ========================================================

    def plot_protected_group_selection(self) -> str:
        """
        Compare selection rates between protected groups.
        """

        grouped = (
            self.dataframe
            .groupby(PROTECTED_COLUMN)[
                TARGET_COLUMN
            ]
            .mean()
        )

        groups = [
            str(group)
            for group in grouped.index
        ]

        rates = [
            float(rate) * 100
            for rate in grouped.values
        ]

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        bars = ax.bar(
            groups,
            rates,
        )

        ax.set_title(
            "Selection Rate by Protected Group"
        )

        ax.set_xlabel(
            "Protected Group"
        )

        ax.set_ylabel(
            "Selection Rate (%)"
        )

        ax.set_ylim(
            0,
            100,
        )

        for bar, rate in zip(
            bars,
            rates,
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
            )

        fig.tight_layout()

        path = (
            self.output_dir
            / "protected_group_selection.png"
        )

        fig.savefig(
            path,
            dpi=150,
        )

        plt.close(fig)

        return str(path)

    # ========================================================
    # Plot: correlation matrix
    # ========================================================

    def plot_correlation_matrix(self) -> str:
        """
        Generate a correlation matrix visualization.
        """

        columns = FEATURE_COLUMNS + [
            TARGET_COLUMN
        ]

        correlation = (
            self.dataframe[columns]
            .corr()
        )

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        image = ax.imshow(
            correlation,
            aspect="auto",
        )

        ax.set_xticks(
            range(len(columns))
        )

        ax.set_yticks(
            range(len(columns))
        )

        ax.set_xticklabels(
            [
                column.replace(
                    "_",
                    " ",
                ).title()
                for column in columns
            ],
            rotation=45,
            ha="right",
        )

        ax.set_yticklabels(
            [
                column.replace(
                    "_",
                    " ",
                ).title()
                for column in columns
            ]
        )

        # Add numerical values
        for row in range(
            len(columns)
        ):

            for column in range(
                len(columns)
            ):

                ax.text(
                    column,
                    row,
                    f"{correlation.iloc[row, column]:.2f}",
                    ha="center",
                    va="center",
                )

        fig.colorbar(
            image,
            ax=ax,
            label="Correlation",
        )

        ax.set_title(
            "Feature Correlation Matrix"
        )

        fig.tight_layout()

        path = (
            self.output_dir
            / "correlation_matrix.png"
        )

        fig.savefig(
            path,
            dpi=150,
        )

        plt.close(fig)

        return str(path)

    # ========================================================
    # Validation
    # ========================================================

    def _validate_dataframe(self):
        """
        Validate the input evaluation dataset.
        """

        if (
            self.dataframe is None
            or self.dataframe.empty
        ):

            raise ValueError(
                "EDA dataset is empty."
            )

        required_columns = (
            FEATURE_COLUMNS
            + [
                TARGET_COLUMN,
                PROTECTED_COLUMN,
            ]
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in self.dataframe.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        for feature in FEATURE_COLUMNS:

            if self.dataframe[
                feature
            ].isna().any():

                raise ValueError(
                    f"Feature '{feature}' "
                    "contains missing values."
                )


# ============================================================
# Convenience function
# ============================================================

def run_eda(
    path: str = "data/evaluation/candidates.csv",
    output_dir: str = "data/evaluation/eda",
) -> dict:
    """
    Load evaluation data and run complete EDA.
    """

    dataframe = pd.read_csv(path)

    analyzer = HiringEDA(
        dataframe=dataframe,
        output_dir=output_dir,
    )

    return analyzer.run()
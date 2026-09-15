from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

VALID_SEEDS = [7, 21, 42]

def load_predictions(model_name, seed):
    prediction_path = (
        RESULTS_DIR
        / f"{model_name}_predictions_seed{seed}.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_path}"
        )

    predictions = pd.read_csv(prediction_path)

    required_columns = {
        "row_id",
        "actual",
        "predicted_probability",
        "predicted_class",
    }

    missing_columns = (
        required_columns - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{model_name} predictions for seed {seed} "
            f"are missing columns: {missing_columns}"
        )

    return predictions

def validate_same_test_rows(
    logistic_predictions,
    xgb_predictions,
    seed,
):
    logistic_ids = set(
        logistic_predictions["row_id"]
    )

    xgb_ids = set(
        xgb_predictions["row_id"]
    )

    if logistic_ids != xgb_ids:
        raise ValueError(
            f"Logistic and XGBoost do not use "
            f"the same test rows for seed {seed}."
        )

    comparison = logistic_predictions[
        ["row_id", "actual"]
    ].merge(
        xgb_predictions[
            ["row_id", "actual"]
        ],
        on="row_id",
        suffixes=("_logistic", "_xgb"),
    )

    mismatched_targets = comparison[
        comparison["actual_logistic"]
        != comparison["actual_xgb"]
    ]

    if not mismatched_targets.empty:
        raise ValueError(
            f"Target mismatch detected for seed {seed}."
        )

def calculate_metrics(predictions):
    y_true = predictions["actual"]
    probabilities = predictions[
        "predicted_probability"
    ]
    predicted_class = predictions[
        "predicted_class"
    ]

    metrics = {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "accuracy": accuracy_score(
            y_true,
            predicted_class,
        ),
        "precision": precision_score(
            y_true,
            predicted_class,
        ),
        "recall": recall_score(
            y_true,
            predicted_class,
        ),
        "f1": f1_score(
            y_true,
            predicted_class,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
    }

    return metrics

def compare_models_for_seed(seed):
    logistic_predictions = load_predictions(
        "logistic",
        seed,
    )

    xgb_predictions = load_predictions(
        "xgb",
        seed,
    )

    validate_same_test_rows(
        logistic_predictions,
        xgb_predictions,
        seed,
    )

    logistic_metrics = calculate_metrics(
        logistic_predictions
    )

    xgb_metrics = calculate_metrics(
        xgb_predictions
    )

    comparison = {
        "seed": seed,

        "logistic_auc":
            logistic_metrics["roc_auc"],

        "xgb_auc":
            xgb_metrics["roc_auc"],

        "auc_delta":
            xgb_metrics["roc_auc"]
            - logistic_metrics["roc_auc"],

        "logistic_accuracy":
            logistic_metrics["accuracy"],

        "xgb_accuracy":
            xgb_metrics["accuracy"],

        "accuracy_delta":
            xgb_metrics["accuracy"]
            - logistic_metrics["accuracy"],

        "logistic_f1":
            logistic_metrics["f1"],

        "xgb_f1":
            xgb_metrics["f1"],

        "logistic_brier":
            logistic_metrics["brier_score"],

        "xgb_brier":
            xgb_metrics["brier_score"],

        "brier_delta":
            xgb_metrics["brier_score"]
            - logistic_metrics["brier_score"],
    }

    return comparison
#THE LOWER BRIER SCORE, THE BETTER MODEL

def compare_all_seeds():
    results = []

    for seed in VALID_SEEDS:
        print(f"Comparing models for seed {seed}")

        seed_result = compare_models_for_seed(
            seed
        )

        results.append(seed_result)

    comparison_df = pd.DataFrame(results)

    output_path = (
        RESULTS_DIR
        / "model_comparison_summary.csv"
    )

    comparison_df.to_csv(
        output_path,
        index=False,
    )

    print("\nModel comparison:")
    print(comparison_df)

    print(
        f"\nComparison saved to: {output_path}"
    )

    return comparison_df

def summarize_comparison(comparison_df):
    summary = {
        "mean_logistic_auc":
            comparison_df["logistic_auc"].mean(),

        "mean_xgb_auc":
            comparison_df["xgb_auc"].mean(),

        "mean_auc_delta":
            comparison_df["auc_delta"].mean(),

        "std_auc_delta":
            comparison_df["auc_delta"].std(),

        "min_auc_delta":
            comparison_df["auc_delta"].min(),

        "max_auc_delta":
            comparison_df["auc_delta"].max(),

        "mean_logistic_brier":
            comparison_df["logistic_brier"].mean(),

        "mean_xgb_brier":
            comparison_df["xgb_brier"].mean(),
    }

    summary_df = pd.DataFrame([summary])

    output_path = (
        RESULTS_DIR
        / "model_comparison_robustness_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    print("\nAcross-seed summary:")
    print(summary_df)

    print(
        f"\nRobustness summary saved to: {output_path}"
    )

    return summary_df

def save_probability_disagreements(seed):
    logistic_predictions = load_predictions(
        "logistic",
        seed,
    )

    xgb_predictions = load_predictions(
        "xgb",
        seed,
    )

    validate_same_test_rows(
        logistic_predictions,
        xgb_predictions,
        seed,
    )

    comparison = logistic_predictions[
        [
            "row_id",
            "actual",
            "predicted_probability",
            "predicted_class",
        ]
    ].merge(
        xgb_predictions[
            [
                "row_id",
                "predicted_probability",
                "predicted_class",
            ]
        ],
        on="row_id",
        suffixes=("_logistic", "_xgb"),
    )

    comparison["probability_difference"] = (
        comparison["predicted_probability_xgb"]
        - comparison["predicted_probability_logistic"]
    )

    comparison["absolute_probability_difference"] = (
        comparison["probability_difference"].abs()
    )

    comparison["models_disagree_on_class"] = (
        comparison["predicted_class_logistic"]
        != comparison["predicted_class_xgb"]
    )

    comparison = comparison.sort_values(
        "absolute_probability_difference",
        ascending=False,
    )

    output_path = (
        RESULTS_DIR
        / f"model_probability_disagreements_seed{seed}.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Probability disagreement table saved to: "
        f"{output_path}"
    )

    return comparison

def run_full_comparison():
    comparison_df = compare_all_seeds()

    summarize_comparison(
        comparison_df
    )

    save_probability_disagreements(
        seed=42
    )

    return comparison_df

def run_full_comparison():
    comparison_df = compare_all_seeds()

    summarize_comparison(
        comparison_df
    )

    save_probability_disagreements(
        seed=42
    )

    save_auc_comparison_plot(
        comparison_df
    )

    save_auc_delta_plot(
        comparison_df
    )

    return comparison_df

def save_auc_comparison_plot(comparison_df):
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = comparison_df[
        [
            "seed",
            "logistic_auc",
            "xgb_auc",
        ]
    ].copy()

    plot_data = plot_data.set_index("seed")

    ax = plot_data.plot(
        kind="bar",
        figsize=(8, 5),
    )

    ax.set_title(
        "ROC-AUC: Logistic Regression vs XGBoost"
    )

    ax.set_xlabel("Train/Test Split Seed")
    ax.set_ylabel("ROC-AUC")

    ax.set_ylim(0.5, 1.0)

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "auc_model_comparison.png"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"AUC comparison plot saved to: "
        f"{output_path}"
    )

def save_auc_delta_plot(comparison_df):
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        comparison_df["seed"].astype(str),
        comparison_df["auc_delta"],
    )

    plt.axhline(
        y=0,
        linewidth=1,
    )

    plt.title(
        "XGBoost Predictive Advantage Across Splits"
    )

    plt.xlabel(
        "Train/Test Split Seed"
    )

    plt.ylabel(
        "AUC Difference (XGBoost - Logistic)"
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "auc_delta_across_seeds.png"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"AUC delta plot saved to: "
        f"{output_path}"
    )


def main():
    print(
        "Model comparison code is ready. "
        "Waiting for logistic and XGBoost "
        "prediction files."
    )


if __name__ == "__main__":
    main()
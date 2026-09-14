from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

TARGET = "Two_yr_Recidivism"

PRIMARY_FEATURES = [
    "Number_of_Priors",
    "Age_Above_FourtyFive",
    "Age_Below_TwentyFive",
    "Female",
    "Misdemeanor",
]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "propublica_data_for_fairml.csv"
)

SPLITS_DIR = PROJECT_ROOT / "splits"

def load_xgboost_model(seed):
    model_path = RESULTS_DIR / f"xgb_model_seed{seed}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Final XGBoost model not found: {model_path}"
        )

    model = XGBClassifier()
    model.load_model(model_path)

    return model

def load_test_data(seed):
    test_path = SPLITS_DIR / f"test_seed{seed}.csv"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test split not found: {test_path}"
        )

    data = pd.read_csv(DATA_PATH)
    test_split = pd.read_csv(test_path)

    if "row_id" not in test_split.columns:
        raise ValueError(
            "Test split must contain a row_id column."
        )

    test_ids = list(test_split["row_id"])

    if "row_id" in data.columns:
        indexed_data = data.set_index("row_id")
    else:
        indexed_data = data.copy()

    missing_ids = set(test_ids) - set(indexed_data.index)

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} test row IDs are missing from the dataset."
        )

    test_data = indexed_data.loc[test_ids].copy()

    X_test = test_data[PRIMARY_FEATURES].copy()
    y_test = test_data[TARGET].copy()

    return X_test, y_test, test_split

def load_xgb_predictions(seed):
    prediction_path = (
        RESULTS_DIR
        / f"xgb_predictions_seed{seed}.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"XGBoost predictions not found: {prediction_path}"
        )

    return pd.read_csv(prediction_path)

def load_logistic_predictions(seed):
    prediction_path = (
        RESULTS_DIR
        / f"logistic_predictions_seed{seed}.csv"
    )

    if not prediction_path.exists():
        return None

    return pd.read_csv(prediction_path)

def load_logistic_predictions(seed):
    prediction_path = (
        RESULTS_DIR
        / f"logistic_predictions_seed{seed}.csv"
    )

    if not prediction_path.exists():
        return None

    return pd.read_csv(prediction_path)

def compute_shap_values(model, X_test):
    explainer = shap.TreeExplainer(model) #creates the SHAP explainer specifically designed for tree models such as XGBoost.

    shap_values = explainer(
        X_test,
        check_additivity=True,
    ) #SHAP will produce an explanation for each person across those features

    return explainer, shap_values

def save_global_shap_outputs(shap_values, X_test):
    #make sure folders exist:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    importance_df = pd.DataFrame({ #creates a table with two columns: one for the feature names and another for the mean absolute SHAP values.
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap,
    })

    importance_df = importance_df.sort_values( #sorts the table from the most important feature to the least important feature.
        by="mean_abs_shap",
        ascending=False,
    )

    importance_path = RESULTS_DIR / "shap_global_importance.csv"

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    shap.plots.bar(
        shap_values,
        show=False,
    )

    plt.tight_layout()

    bar_path = FIGURES_DIR / "shap_global_importance.png"

    plt.savefig(
        bar_path,
        bbox_inches="tight",
    )

    plt.close()

    shap.plots.beeswarm(
        shap_values,
        show=False,
    )

    plt.tight_layout()

    beeswarm_path = FIGURES_DIR / "shap_beeswarm.png" #How do different values of those variables push different individual predictions?

    plt.savefig(
        beeswarm_path,
        bbox_inches="tight",
    )

    plt.close()

    print(f"SHAP importance saved to: {importance_path}")
    print(f"SHAP bar plot saved to: {bar_path}")
    print(f"SHAP beeswarm saved to: {beeswarm_path}")

def select_local_cases(xgb_predictions, logistic_predictions=None):
    cases = {}

    true_positives = xgb_predictions[ #filters the prediction table so it contains only true positives
        (xgb_predictions["actual"] == 1)
        & (xgb_predictions["predicted_class"] == 1)
    ]

    if not true_positives.empty:
        cases["confident_true_positive"] = (
            true_positives
            .sort_values("predicted_probability", ascending=False)
            .iloc[0]["row_id"]
        )

    true_negatives = xgb_predictions[
        (xgb_predictions["actual"] == 0)
        & (xgb_predictions["predicted_class"] == 0)
    ]

    if not true_negatives.empty:
        cases["confident_true_negative"] = (
            true_negatives
            .sort_values("predicted_probability", ascending=True)
            .iloc[0]["row_id"]
        )

    errors = xgb_predictions[
        xgb_predictions["actual"]
        != xgb_predictions["predicted_class"]
    ].copy()

    if not errors.empty:
        errors["wrong_class_confidence"] = np.where(
            errors["predicted_class"] == 1,
            errors["predicted_probability"],
            1 - errors["predicted_probability"],
        )

        cases["confident_xgb_error"] = (
            errors
            .sort_values("wrong_class_confidence", ascending=False)
            .iloc[0]["row_id"]
        )

    if logistic_predictions is not None:
        comparison = xgb_predictions[
            ["row_id", "predicted_probability"]
        ].merge(
            logistic_predictions[
                ["row_id", "predicted_probability"]
            ],
            on="row_id",
            suffixes=("_xgb", "_logistic"),
        )

        comparison["probability_difference"] = (
            comparison["predicted_probability_xgb"]
            - comparison["predicted_probability_logistic"]
        ).abs()

        cases["largest_model_disagreement"] = (
            comparison
            .sort_values("probability_difference", ascending=False)
            .iloc[0]["row_id"]
        )

    return cases

def save_local_shap_plots(
    shap_values,
    X_test,
    selected_cases,
):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for case_name, row_id in selected_cases.items():

        matching_positions = np.where(
            X_test.index.to_numpy() == row_id
        )[0]

        if len(matching_positions) != 1:
            raise ValueError(
                f"Expected exactly one test row for row_id {row_id}, "
                f"found {len(matching_positions)}."
            )

        position = matching_positions[0]

        shap.plots.waterfall(
            shap_values[position],
            show=False,
        )

        plt.tight_layout()

        output_path = (
            FIGURES_DIR
            / f"shap_local_{case_name}.png"
        )

        plt.savefig(
            output_path,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Local SHAP plot saved to: {output_path}"
        )

def save_priors_dependence_plot(shap_values, X_test):
    feature_name = "Number_of_Priors"

    if feature_name not in X_test.columns:
        raise ValueError(
            f"{feature_name} is missing from X_test."
        )

    feature_index = X_test.columns.get_loc(feature_name)

    dependence_data = pd.DataFrame({
        "row_id": X_test.index,
        feature_name: X_test[feature_name].values,
        "shap_value": shap_values.values[:, feature_index],
    })

    dependence_path = (
        RESULTS_DIR / "shap_priors_dependence_data.csv"
    )

    dependence_data.to_csv(
        dependence_path,
        index=False,
    )

    shap.plots.scatter(
        shap_values[:, feature_index],
        show=False,
    )

    plt.tight_layout()

    figure_path = (
        FIGURES_DIR / "shap_priors_dependence.png"
    )

    plt.savefig(
        figure_path,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"SHAP dependence data saved to: {dependence_path}"
    )
    print(
        f"SHAP dependence plot saved to: {figure_path}"
    )

def run_shap_analysis(seed):
    model = load_xgboost_model(seed)

    X_test, y_test, test_split = load_test_data(seed)

    print(f"Running TreeSHAP for seed {seed}")
    print(f"Test observations: {len(X_test)}")
    print(f"Features: {list(X_test.columns)}")

    explainer, shap_values = compute_shap_values(
        model,
        X_test,
    )

    print(
        f"SHAP matrix shape: {shap_values.values.shape}"
    )

    expected_shape = (
        len(X_test),
        len(PRIMARY_FEATURES),
    )

    if shap_values.values.shape != expected_shape:
        raise ValueError(
            f"Unexpected SHAP shape. "
            f"Expected {expected_shape}, "
            f"got {shap_values.values.shape}."
        )

    save_global_shap_outputs(
        shap_values,
        X_test,
    )

    save_priors_dependence_plot(
        shap_values,
        X_test,
    )

    xgb_predictions = load_xgb_predictions(seed)

    logistic_predictions = load_logistic_predictions(seed)

    selected_cases = select_local_cases(
        xgb_predictions,
        logistic_predictions,
    )

    print("\nSelected local cases:")

    for case_name, row_id in selected_cases.items():
        print(f"{case_name}: row_id {row_id}")

    save_local_shap_plots(
        shap_values,
        X_test,
        selected_cases,
    )

    return explainer, shap_values

def main():
    print(f"SHAP version: {shap.__version__}")
    print(
        "SHAP analysis code is ready. "
        "Waiting for the final XGBoost model."
    )


if __name__ == "__main__":
    main()
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

# capital lettered variables are configuration values that should not change while the program runs
TARGET = "Two_yr_Recidivism"

PRIMARY_FEATURES = [
    "Number_of_Priors",
    "Age_Above_FourtyFive",
    "Age_Below_TwentyFive",
    "Female",
    "Misdemeanor",
]

SCALE_FEATURES = ["Number_of_Priors"]  # the only non-binary feature

C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3]  # half-decade steps from 10^-2 to 10^0.5

CHOSEN_C = 0.03  # highest CV AUC and the strongest regularization keeping all 5 features

PRIMARY_SEED = 42  # C is tuned on this seed; the other seeds reuse the frozen value

VALID_SEEDS = [7, 21, 42]  # frozen train/test splits created in 01_data_preparation.py

CV_FOLDS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # compas-xai

CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_compas.csv"
SPLITS_DIR = PROJECT_ROOT / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_clean_data():
    data = pd.read_csv(CLEAN_PATH)

    if "row_id" not in data.columns:
        raise ValueError("Clean dataset must contain a 'row_id' column.")

    if TARGET not in data.columns:
        raise ValueError(f"Clean dataset must contain the target '{TARGET}'.")

    return data


def validate_features(X):  # enforces the non-negotiable exclusion rules
    if TARGET in X.columns:
        raise ValueError("Target leaked into the feature matrix.")

    unexpected = [column for column in X.columns if column not in PRIMARY_FEATURES]

    if unexpected:
        raise ValueError(f"Unexpected columns in X: {unexpected}")


def load_split(seed):
    if seed not in VALID_SEEDS:
        raise ValueError(f"Seed {seed} is not one of the frozen seeds {VALID_SEEDS}.")

    return pd.read_csv(SPLITS_DIR / f"split_seed{seed}.csv")


def build_train_test_sets(data, split):  # (clean dataset, split table for one seed)
    merged = data.merge(split, on="row_id", validate="one_to_one")

    train = merged[merged["split"] == "train"]
    test = merged[merged["split"] == "test"]

    # Safety check: no person/row should appear in both sets
    overlap = set(train["row_id"]).intersection(test["row_id"])

    if overlap:
        raise ValueError(f"Train/test overlap detected: {len(overlap)} shared rows.")

    X_train = train[PRIMARY_FEATURES]
    X_test = test[PRIMARY_FEATURES]

    validate_features(X_train)
    validate_features(X_test)

    return X_train, X_test, train[TARGET], test[TARGET], test["row_id"]


def build_pipeline(C):  # (inverse regularization strength; smaller = stronger penalty)
    preprocessor = ColumnTransformer(
        transformers=[("scale_priors", StandardScaler(), SCALE_FEATURES)],
        remainder="passthrough",
    )

    # random_state stays fixed across seeds so that seed-to-seed differences come
    # only from the split, not from the solver
    model = LogisticRegression(
        l1_ratio=1,  # pure L1 the penalty; can force some coefficients to 0 -> sparse model
        solver="liblinear",
        C=C, # inverse regularization strength; smaller = stronger penalty = 1/lambda
        max_iter=1000,
        random_state=PRIMARY_SEED,
    )

    return Pipeline([("preprocess", preprocessor), ("logreg", model)])


def tune_C(X_train, y_train):  # training-only cross-validation over C_GRID
    cross_validator = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=PRIMARY_SEED,
    )

    rows = []

    for C in C_GRID:
        pipeline = build_pipeline(C)

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cross_validator,
            scoring="roc_auc",
        )

        # refit on the whole training split to count surviving coefficients
        pipeline.fit(X_train, y_train)

        rows.append({
            "C": C,
            "cv_auc_mean": scores.mean(),
            "cv_auc_std": scores.std(),
            "non_zero_coefs": int((pipeline.named_steps["logreg"].coef_[0] != 0).sum()),
        })

    return pd.DataFrame(rows)


def train_final_model(X_train, y_train, C):
    pipeline = build_pipeline(C)
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    probabilities = pipeline.predict_proba(X_test)[:, 1]  # column 1 = positive class
    predictions = pipeline.predict(X_test) # this funcion automatically uses a threshold of 0.5 to convert probabilities into binary predictions

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    metrics = {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "brier_score": brier_score_loss(y_test, probabilities),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return probabilities, predictions, metrics


def extract_coefficients(pipeline):
    # ColumnTransformer reorders columns, so names must come from the fitted transformer
    feature_names = [
        name.split("__", 1)[1]
        for name in pipeline.named_steps["preprocess"].get_feature_names_out()
    ]

    coefficients = pipeline.named_steps["logreg"].coef_[0]

    return pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "odds_ratio": np.exp(coefficients), #e to the power of coefficient; so a 1 unit increase in the feature x multiplies the predicted odds by approx e^coefficient
        "non_zero": coefficients != 0,
    })


def get_priors_scale(pipeline):  # SD used to standardize Number_of_Priors
    scaler = pipeline.named_steps["preprocess"].named_transformers_["scale_priors"]

    return scaler.scale_[0]


def save_table(table, filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / filename
    table.to_csv(output_path, index=False)

    return output_path


def print_tuning_table(tuning_results):
    print("\nC               = regularization strength; SMALLER = stronger penalty = simpler model")
    print("CV AUC mean     = average ROC-AUC over 5 training folds")
    print("CV AUC std      = spread across those folds; gaps smaller than this are noise, not signal")
    print("non-zero coefs  = how many of the 5 features survived L1 shrinkage") #l1 is a penalty for large coefficients; it can force some coef to be 0 -> the feature disappears from the prediction

    print(f"\n{'C':>6} {'CV AUC mean':>12} {'CV AUC std':>11} {'non-zero coefs':>15}")

    for row in tuning_results.itertuples(index=False):
        print(f"{row.C:>6} {row.cv_auc_mean:>12.4f} {row.cv_auc_std:>11.4f} {row.non_zero_coefs:>15}")


def run_primary_experiment(seed, C):  # P1-4 to P1-8: tune, fit, evaluate, export
    data = load_clean_data()
    split = load_split(seed)

    X_train, X_test, y_train, y_test, test_ids = build_train_test_sets(data, split)

    print(f"seed {seed} training rows: {len(X_train)}")

    tuning_results = tune_C(X_train, y_train)
    print_tuning_table(tuning_results)

    # the test set is scored only after C is frozen
    pipeline = train_final_model(X_train, y_train, C)
    probabilities, predictions, metrics = evaluate_model(pipeline, X_test, y_test)

    metrics_row = {
        "seed": seed,
        "C": C,
        "n_train": len(X_train),
        "n_test": len(X_test),
        **metrics,
    }

    save_table(pd.DataFrame([metrics_row]), f"logistic_metrics_seed{seed}.csv")

    save_table(
        pd.DataFrame({
            "row_id": test_ids.values,
            "actual": y_test.values,
            "predicted_probability": probabilities,
            "predicted_class": predictions,
        }),
        f"logistic_predictions_seed{seed}.csv",
    )

    coefficients = extract_coefficients(pipeline)
    save_table(coefficients, "logistic_coefficients.csv")

    print(f"\n=== test metrics (seed {seed}, C={C}) ===")
    for name, value in metrics_row.items():
        print(f"  {name:16} {value}")

    print(f"\n=== coefficients (C={C}) ===")
    print(coefficients.to_string(index=False))
    print(f"\nintercept: {pipeline.named_steps['logreg'].intercept_[0]:.4f}")
    print(f"priors SD used for scaling: {get_priors_scale(pipeline):.4f}")

    return pipeline, metrics_row, coefficients


def run_robustness_experiment(C):  # P1-9: repeat fixed model logic across all seeds
    data = load_clean_data()

    seed_rows = []
    coefficient_rows = []

    for seed in VALID_SEEDS:
        X_train, X_test, y_train, y_test, test_ids = build_train_test_sets(data, load_split(seed))

        pipeline = train_final_model(X_train, y_train, C)
        probabilities, predictions, metrics = evaluate_model(pipeline, X_test, y_test)

        # the paired comparison needs predictions for every seed, not just the primary one
        save_table(
            pd.DataFrame({
                "row_id": test_ids.values,
                "actual": y_test.values,
                "predicted_probability": probabilities,
                "predicted_class": predictions,
            }),
            f"logistic_predictions_seed{seed}.csv",
        )

        seed_rows.append({"seed": seed, "C": C, **metrics})

        for row in extract_coefficients(pipeline).itertuples(index=False):
            coefficient_rows.append({
                "seed": seed,
                "feature": row.feature,
                "coefficient": row.coefficient,
            })

    robustness = pd.DataFrame(seed_rows)
    save_table(robustness, "logistic_robustness_summary.csv")

    coefficients_long = pd.DataFrame(coefficient_rows)

    stability = (
        coefficients_long.groupby("feature")["coefficient"]
        .agg(
            mean="mean",
            std="std",
            min="min",
            max="max",
            seeds_non_zero=lambda values: int((values != 0).sum()),
            sign_stable=lambda values: len(set(np.sign(values[values != 0]))) <= 1,
        )
        .reset_index()
        .sort_values("mean", key=abs, ascending=False)
    )

    save_table(stability, "logistic_coefficient_stability.csv")

    print("\n=== robustness across seeds ===")
    print(robustness.to_string(index=False))

    print("\n=== coefficient stability ===")
    print(stability.to_string(index=False))

    print(f"\nAUC spread across seeds: "
          f"{robustness['roc_auc'].max() - robustness['roc_auc'].min():.4f}")

    return robustness, stability


def main():
    run_primary_experiment(PRIMARY_SEED, CHOSEN_C)
    run_robustness_experiment(CHOSEN_C)


# ensure run case of main()
if __name__ == "__main__":
    main()

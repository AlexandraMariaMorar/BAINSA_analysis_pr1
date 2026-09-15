import pandas as pd
import numpy as np
from pathlib import Path
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_compas.csv"
SPLITS_DIR = PROJECT_ROOT / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"


PRIMARY_SEED = 42  # 1 st i will tune C on this seed and then i ll refit the chosen C on all three seeds 

TARGET = "Two_yr_Recidivism"
PRIMARY_FEATURES = [
    "Number_of_Priors",
    "Age_Above_FourtyFive",
    "Age_Below_TwentyFive",
    "Female",
    "Misdemeanor",
]
SCALE_FEATURES = ["Number_of_Priors"]  # the only non-binary feature

# load the primary seed training rows from the split 

clean = pd.read_csv(CLEAN_PATH)
split = pd.read_csv(SPLITS_DIR / f"split_seed{PRIMARY_SEED}.csv")

merged = clean.merge(split, on="row_id", validate="one_to_one")

train = merged[merged["split"] == "train"]

X_train = train[PRIMARY_FEATURES]
y_train = train[TARGET]

print(f"seed {PRIMARY_SEED} training rows: {len(X_train)}")

preprocessor = ColumnTransformer( # reorders collums ; issue rn 
    transformers=[("scale_priors", StandardScaler(), SCALE_FEATURES)],
    remainder="passthrough",
)

model = LogisticRegression(
    l1_ratio = 1,  # pure l1
    solver="liblinear",
    max_iter=1000,
    random_state=PRIMARY_SEED,
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("logreg", model),
])

C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3] # they are 6 because number of points = (range in decades = 2.5 ÷ step size = 0.5) + 1 = 6 
# 0.01 was chosen as a lower bound because below it every coefficient goes to 0 
# 3 was chosen as an upper bound because above it the coefficients stop changing substantially  


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=PRIMARY_SEED)

print("\nC               = regularization strength; SMALLER = stronger penalty = simpler model")
print("CV AUC mean     = average ROC-AUC over 5 training folds")
print("CV AUC std      = spread across those folds; gaps smaller than this are noise, not signal")
print("non-zero coefs  = how many of the 5 features survived L1 shrinkage")

print(f"\n{'C':>6} {'CV AUC mean':>12} {'CV AUC std':>11} {'non-zero coefs':>15}")


for C in C_GRID:
    pipeline.set_params(logreg__C=C)

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

    pipeline.fit(X_train, y_train)
    coefs = pipeline.named_steps["logreg"].coef_[0]
    n_nonzero = int((coefs != 0).sum())

    print(f"{C:>6} {cv_scores.mean():>12.4f} {cv_scores.std():>11.4f} {n_nonzero:>15}")

# FINAL MODEL 


CHOSEN_C = 0.03  # highest CV AUC and the strongest regularization keeping all 5 features

test = merged[merged["split"] == "test"]

X_test = test[PRIMARY_FEATURES]
y_test = test[TARGET]

pipeline.set_params(logreg__C=CHOSEN_C)
pipeline.fit(X_train, y_train)

test_prob = pipeline.predict_proba(X_test)[:, 1]
test_pred = pipeline.predict(X_test)


# TEST METRICS 


metrics = {
    "seed": PRIMARY_SEED,
    "C": CHOSEN_C,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "roc_auc": roc_auc_score(y_test, test_prob),
    "accuracy": accuracy_score(y_test, test_pred),
    "precision": precision_score(y_test, test_pred),
    "recall": recall_score(y_test, test_pred),
    "f1": f1_score(y_test, test_pred),
    "brier_score": brier_score_loss(y_test, test_prob),
}

true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(y_test, test_pred).ravel()
metrics.update({"true_negatives": int(true_negatives), "false_positives": int(false_positives), "false_negatives": int(false_negatives), "true_positives": int(true_positives)})

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame([metrics]).to_csv(
    RESULTS_DIR / f"logistic_metrics_seed{PRIMARY_SEED}.csv", index=False
)

print(f"\n=== test metrics (seed {PRIMARY_SEED}, C={CHOSEN_C}) ===")
for name, value in metrics.items():
    print(f"  {name:10} {value}")


# PREDICTIONS FILE 


predictions = pd.DataFrame({
    "row_id": test["row_id"].values,
    "actual": y_test.values,
    "predicted_probability": test_prob,
    "predicted_class": test_pred,
})
predictions.to_csv(
    RESULTS_DIR / f"logistic_predictions_seed{PRIMARY_SEED}.csv", index=False
)


# COEFFICIENTS + ODDS RATIOS (P1-8)


feature_names = [
    name.split("__", 1)[1]
    for name in pipeline.named_steps["preprocess"].get_feature_names_out()
]
coefs = pipeline.named_steps["logreg"].coef_[0]

coefficients = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefs,
    "odds_ratio": np.exp(coefs),
    "non_zero": coefs != 0,
})
coefficients.to_csv(RESULTS_DIR / "logistic_coefficients.csv", index=False)

print(f"\n=== coefficients (C={CHOSEN_C}) ===")
print(coefficients.to_string(index=False))
print(f"\nintercept: {pipeline.named_steps['logreg'].intercept_[0]:.4f}")
print(f"priors SD used for scaling: "
      f"{pipeline.named_steps['preprocess'].named_transformers_['scale_priors'].scale_[0]:.4f}")


# ROBUSTNESS ACROSS SEEDS 

ALL_SEEDS = [7, 21, 42]

seed_rows = []
coef_rows = []

for seed in ALL_SEEDS:
    seed_split = pd.read_csv(SPLITS_DIR / f"split_seed{seed}.csv")
    seed_merged = clean.merge(seed_split, on="row_id", validate="one_to_one")

    seed_train = seed_merged[seed_merged["split"] == "train"]
    seed_test = seed_merged[seed_merged["split"] == "test"]

    pipeline.set_params(logreg__C=CHOSEN_C)
    pipeline.fit(seed_train[PRIMARY_FEATURES], seed_train[TARGET])

    seed_y = seed_test[TARGET]
    seed_prob = pipeline.predict_proba(seed_test[PRIMARY_FEATURES])[:, 1]
    seed_pred = pipeline.predict(seed_test[PRIMARY_FEATURES])

    s_tn, s_fp, s_fn, s_tp = confusion_matrix(seed_y, seed_pred).ravel()

    seed_rows.append({
        "seed": seed,
        "C": CHOSEN_C, #C stays pinned at 0.03; no re-tuning for each seed because we want to see how the same model performs across different splits
        "roc_auc": roc_auc_score(seed_y, seed_prob),
        "accuracy": accuracy_score(seed_y, seed_pred),
        "precision": precision_score(seed_y, seed_pred),
        "recall": recall_score(seed_y, seed_pred),
        "f1": f1_score(seed_y, seed_pred),
        "brier_score": brier_score_loss(seed_y, seed_prob),
        "true_negatives": int(s_tn),
        "false_positives": int(s_fp),
        "false_negatives": int(s_fn),
        "true_positives": int(s_tp),
    })

    for name, value in zip(feature_names, pipeline.named_steps["logreg"].coef_[0]):
        coef_rows.append({"seed": seed, "feature": name, "coefficient": value})

robustness = pd.DataFrame(seed_rows)
robustness.to_csv(RESULTS_DIR / "logistic_robustness_summary.csv", index=False)

coef_long = pd.DataFrame(coef_rows)
stability = (
    coef_long.groupby("feature")["coefficient"]
    .agg(
        mean="mean",
        std="std",
        min="min",
        max="max",
        seeds_non_zero=lambda s: int((s != 0).sum()),
        sign_stable=lambda s: len(set(np.sign(s[s != 0]))) <= 1,#checks whether a coef ever flips direction across seeds
    )#a feature that looks protective in one split and risk-increasing in another => the model can't pin it down => shouldnt be interpreted
    .reset_index()
    .sort_values("mean", key=abs, ascending=False)
)
stability.to_csv(RESULTS_DIR / "logistic_coefficient_stability.csv", index=False)

print("\n=== robustness across seeds ===")
print(robustness.to_string(index=False))
print("\n=== coefficient stability ===")
print(stability.to_string(index=False))
print(f"\nAUC spread across seeds: "
      f"{robustness['roc_auc'].max() - robustness['roc_auc'].min():.4f}")

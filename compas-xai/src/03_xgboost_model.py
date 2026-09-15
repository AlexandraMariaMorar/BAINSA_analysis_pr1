from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

#capital lettered variables are used intentionally for configuration values that should not normally change while the program is running.
TARGET = "Two_yr_Recidivism"

PRIMARY_FEATURES = [
    "Number_of_Priors",
    "Age_Above_FourtyFive",
    "Age_Below_TwentyFive",
    "Female", #dataset encodes gender as a binary variable
    "Misdemeanor",
]

FORBIDDEN_FEATURES = [
    "Two_yr_Recidivism", #prevent leakage of the target variable into the features
    "score_factor", #prohibitied because it contains information about the COMPAS score, not available yet at prediction time
    "African_American",
    "Asian",
    "Hispanic",
    "Native_American",
    "Other", #race variables are excluded because race is reserved for the secondary sensitivity analysis

]

VALID_SEEDS = [7, 21, 42] #predefined random seeds, frozen train/test splits created using these predefined seeds


PROJECT_ROOT = Path(__file__).resolve().parents[1] #compas-xai

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "propublica_data_for_fairml.csv"
SPLITS_DIR = PROJECT_ROOT / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_clean_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {DATA_PATH}. "
            "Person 1 must create and provide this file."
        )

    return pd.read_csv(DATA_PATH)


def validate_data(data):
    required_columns = set(PRIMARY_FEATURES + [TARGET])

    missing_columns = required_columns - set(data.columns)

    if missing_columns: 
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    for feature in FORBIDDEN_FEATURES:
        if feature in PRIMARY_FEATURES:
            raise ValueError(
                f"Forbidden feature found in PRIMARY_FEATURES: {feature}"
            )


def create_feature_target(data):
    X = data[PRIMARY_FEATURES].copy() #inputs of model
    y = data[TARGET].copy() #correct outputs of model, the answer that XGBoost tries to learn to predict

    assert TARGET not in X.columns #make sure we did not accidentally give the true outcome to the model
    assert "score_factor" not in X.columns #make sure we did not accidentally give the model information about the COMPAS score, which is not available at prediction time
    assert set(X.columns) == set(PRIMARY_FEATURES)

    return X, y


def load_split(seed):
    if seed not in VALID_SEEDS:
        raise ValueError(
            f"Seed must be one of {VALID_SEEDS}"
        )

    train_path = SPLITS_DIR / f"train_seed{seed}.csv"
    test_path = SPLITS_DIR / f"test_seed{seed}.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training split not found: {train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test split not found: {test_path}"
        )

    #Load the train/test splits from CSV files
    #80% of the data is used for training, and 20% is used for testing

    train_split = pd.read_csv(train_path) 
    test_split = pd.read_csv(test_path)

    return train_split, test_split

#make sure no overlapping happens between two datasets

def validate_split(train_split, test_split):
    if "row_id" not in train_split.columns:
        raise ValueError(
            "Training split must contain row_id"
        )

    if "row_id" not in test_split.columns:
        raise ValueError(
            "Test split must contain row_id"
        )

    train_ids = set(train_split["row_id"])
    test_ids = set(test_split["row_id"])

    overlap = train_ids.intersection(test_ids)

    if overlap:
        raise ValueError(
            f"Train/test leakage detected: "
            f"{len(overlap)} overlapping IDs"
        )

def build_train_test_sets(data, train_split, test_split): #(cleaned dataset, training split, test split)
    id_column = "row_id"

    if id_column not in train_split.columns:
        raise ValueError(
            f"Train split must contain a '{id_column}' column."
        )

    if id_column not in test_split.columns:
        raise ValueError(
            f"Test split must contain a '{id_column}' column."
        )

    train_ids = list(train_split[id_column])
    test_ids = list(test_split[id_column])

    # Safety check: no person/row should appear in both sets
    overlap = set(train_ids).intersection(test_ids)

    if overlap: #(for all of the project this notation is being used to check if variable is nonzero)
        raise ValueError(
            f"Train/test overlap detected: {len(overlap)} shared rows."
        )

    # If Person 1 adds an explicit row_id column to the dataset
    if id_column in data.columns:
        indexed_data = data.set_index(id_column)

    # Otherwise assume row_id corresponds to the dataframe row index
    else:
        indexed_data = data.copy()

    # Check that every requested row actually exists
    all_ids = set(train_ids + test_ids)
    available_ids = set(indexed_data.index)

    missing_ids = all_ids - available_ids

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} split row IDs are missing from the dataset."
        )

    train_data = indexed_data.loc[train_ids].copy()
    test_data = indexed_data.loc[test_ids].copy()

    X_train = train_data[PRIMARY_FEATURES].copy()
    y_train = train_data[TARGET].copy()

    X_test = test_data[PRIMARY_FEATURES].copy()
    y_test = test_data[TARGET].copy()

    return X_train, X_test, y_train, y_test    

def build_xgboost_model(seed, max_depth=3, n_estimators=200):  #(randomness, maximum depth of each tree, number of trees in the ensemble)
    model = XGBClassifier(
        objective="binary:logistic", #task definition: binary classification with logistic loss, model produces a probability score between 0 and 1 for each instance
        eval_metric="logloss", #internal metric, our tuning metric is AUC, but logloss is used for early stopping and internal evaluation
        n_estimators=n_estimators, 
        max_depth=max_depth,
        learning_rate=0.05, #how much each tree contributes to the final prediction
        subsample=0.8, #each tree is trained using 80% of the training observations
        colsample_bytree=1.0, #what fraction of the features are available to each tree.
        random_state=seed,
        n_jobs=-1,
    )

    return model

def tune_xgboost(X_train, y_train, seed):
    results = []

    cv = StratifiedKFold( #we split the training data into several sections called folds, each containing a similar proportion of the target variable
        n_splits=5,
        shuffle=True,
        random_state=seed,
    ) #we repeat the training and evaluation process 5 times, each time using a different fold as the validation set and the remaining folds as the training set

    for max_depth in [2, 3]:
        for n_estimators in [150, 250]:

            model = build_xgboost_model(
                seed=seed,
                max_depth=max_depth,
                n_estimators=n_estimators,
            )

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                scoring="roc_auc",
                cv=cv,
            )

            results.append({
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "mean_cv_auc": scores.mean(),
                "std_cv_auc": scores.std(),
            })

    return pd.DataFrame(results)

def choose_best_params(tuning_results): #takes the table produced by tune_xgboost and selects the best hyperparameters based on mean cross-validation AUC
    sorted_results = tuning_results.sort_values(
        by=["mean_cv_auc", "max_depth", "n_estimators"],
        ascending=[False, True, True], #descending order for mean_cv_auc, ascending for others
    )

    best_row = sorted_results.iloc[0]

    best_params = {
        "max_depth": int(best_row["max_depth"]),
        "n_estimators": int(best_row["n_estimators"]),
    }

    return best_params

def train_final_xgboost(X_train, y_train, seed, best_params): #training one final model using all of the training data and the best hyperparameters found during tuning
    model = build_xgboost_model(
        seed=seed,
        max_depth=best_params["max_depth"],
        n_estimators=best_params["n_estimators"],
    )

    model.fit(X_train, y_train)

    return model

def evaluate_xgboost(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1] #turns probabilities into predicted classes using the fixed 0.5 threshold

    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    brier = brier_score_loss(y_test, probabilities)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "roc_auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return metrics, predictions, probabilities

def train_final_xgboost(X_train, y_train, seed, best_params):
    model = build_xgboost_model(
        seed=seed,
        max_depth=best_params["max_depth"],
        n_estimators=best_params["n_estimators"],
    )

    model.fit(X_train, y_train)

    return model

def evaluate_xgboost(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    brier = brier_score_loss(y_test, probabilities)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "roc_auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return metrics, predictions, probabilities

def save_xgboost_outputs(
    test_split,
    y_test,
    predictions,
    probabilities,
    metrics,
    seed,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prediction_results = pd.DataFrame({
        "row_id": list(test_split["row_id"]),
        "actual": list(y_test),
        "predicted_probability": probabilities,
        "predicted_class": predictions,
    })

    prediction_path = RESULTS_DIR / f"xgb_predictions_seed{seed}.csv"

    prediction_results.to_csv(
        prediction_path,
        index=False,
    )

    metrics_results = pd.DataFrame([metrics])
    metrics_results.insert(0, "seed", seed)

    metrics_path = RESULTS_DIR / f"xgb_metrics_seed{seed}.csv"

    metrics_results.to_csv(
        metrics_path,
        index=False,
    )

    print(f"Predictions saved to: {prediction_path}")
    print(f"Metrics saved to: {metrics_path}")

def save_xgboost_model(model, seed):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RESULTS_DIR / f"xgb_model_seed{seed}.json"

    model.save_model(model_path)

    print(f"Model saved to: {model_path}")

def run_seed_experiment(seed):
    data = load_clean_data()
    validate_data(data)

    train_split, test_split = load_split(seed)
    validate_split(train_split, test_split)

    X_train, X_test, y_train, y_test = build_train_test_sets(
        data,
        train_split,
        test_split,
    )

    print(f"\nRunning XGBoost experiment for seed {seed}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Training prevalence: {y_train.mean():.3f}")
    print(f"Test prevalence: {y_test.mean():.3f}")

    tuning_results = tune_xgboost(
        X_train,
        y_train,
        seed,
    )

    print("\nCross-validation tuning results:")
    print(
        tuning_results.sort_values(
            "mean_cv_auc",
            ascending=False,
        )
    )

    best_params = choose_best_params(
        tuning_results
    )

    print(f"\nSelected parameters: {best_params}")

    final_model = train_final_xgboost(
        X_train,
        y_train,
        seed,
        best_params,
    )

    metrics, predictions, probabilities = evaluate_xgboost(
        final_model,
        X_test,
        y_test,
    )

    print("\nHeld-out test metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")

    save_xgboost_outputs(
        test_split,
        y_test,
        predictions,
        probabilities,
        metrics,
        seed,
    )

    save_xgboost_model(
        final_model,
        seed,
    )

    return {
        "model": final_model,
        "metrics": metrics,
        "tuning_results": tuning_results,
        "best_params": best_params,
    }

def main():
    data = load_clean_data()

    validate_data(data)

    X, y = create_feature_target(data)

    print("Dataset loaded successfully.")
    print(f"Rows: {len(data)}")
    print(f"Features: {list(X.columns)}")
    print(f"Target: {TARGET}")
    print(f"Target prevalence: {y.mean():.3f}")

#ensure run case of main()
if __name__ == "__main__":
    main()
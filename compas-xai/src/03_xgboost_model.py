from pathlib import Path
import pandas as pd

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

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_compas.csv"
SPLITS_DIR = PROJECT_ROOT / "splits"


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
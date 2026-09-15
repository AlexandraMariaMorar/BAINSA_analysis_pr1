import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split #does the splitting


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "propublica_data_for_fairml.csv"
)

CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "clean_compas.csv"
SPLITS_DIR = PROJECT_ROOT / "splits"


df = pd.read_csv(DATA_PATH) # creating a pandas data frame from the csv file; store it in a variable df (data frame)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# --------------------------------------------------
# 1. BASIC DATA AUDIT
# --------------------------------------------------

print("\nMissing values:")
print(df.isna().sum()) # counts no of missing values in cells in each column of the data frame
print("\nDuplicate rows:")
print(df.duplicated().sum()) # optional here because of the nature of our data, but good practice so i added it 

print("\nTarget counts:")
print(df["Two_yr_Recidivism"].value_counts()) # two year recidivism is our target variable so we count how many times each possible value occurs , in this case how many times 0 occurs and 1 occurs

print("\nTarget proportions:")# shows how the data set is balanced
print(df["Two_yr_Recidivism"].value_counts(normalize=True)) # gives proportions of each value in the target variable instead of countsn

print("\nSummary statistics:")
print(df.describe().T) # used the transpose for easier reading of the summary statistics

print("\nUnique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

TARGET = "Two_yr_Recidivism"

PRIMARY_FEATURES = [
    "Number_of_Priors",
    "Age_Above_FourtyFive",
    "Age_Below_TwentyFive",
    "Female",
    "Misdemeanor"
]
RACE_FEATURES = [
    "African_American",
    "Asian",
    "Hispanic",
    "Native_American",
    "Other"
]


clean = df.drop(columns="score_factor").copy()
clean.insert(0, "row_id", range(len(clean))) #constructing the clean data frame by dropping the score factor column and adding a row id column at the beginning of the data frame

CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
clean.to_csv(CLEAN_PATH, index=False)

print("\nClean dataset written to:")
print(CLEAN_PATH.relative_to(PROJECT_ROOT))
print(f"rows: {len(clean)}, columns: {list(clean.columns)}")

X = clean[PRIMARY_FEATURES].copy()
y = clean[TARGET].copy()

assert TARGET not in X.columns

assert not any(col in X.columns for col in RACE_FEATURES)


# score_factor:
# excluded because it is derived from the COMPAS risk score

# race variables:
# excluded from the primary analysis;
# will be added later in a sensitivity analysis


# FROZEN TRAIN/TEST SPLITS (seeds 7, 21, 42)


SEEDS = [7, 21, 42] # i chose 3 because a single 80/20 split can make performance look better or worse by chance
TEST_SIZE = 0.2
# If results change substantially across seeds, report that instability as a finding rather than cherry-picking the best split.
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

for seed in SEEDS:
    train_ids, test_ids = train_test_split(
        clean["row_id"],
        test_size=TEST_SIZE,
        stratify=y,          # keeps the recidivism rate the same in both halves
        random_state=seed,
    )

    split_table = pd.DataFrame(
        {
            "row_id": pd.concat([train_ids, test_ids]),
            "split": ["train"] * len(train_ids) + ["test"] * len(test_ids),
        }
    ).sort_values("row_id")

    assert not set(train_ids) & set(test_ids), f"seed {seed}: train/test overlap"
    assert len(split_table) == len(clean), f"seed {seed}: rows lost or duplicated"

    split_path = SPLITS_DIR / f"split_seed{seed}.csv"
    split_table.to_csv(split_path, index=False)

    train_rate = y[train_ids.index].mean()
    test_rate = y[test_ids.index].mean()
    print(
        f"\nseed {seed}: {len(train_ids)} train / {len(test_ids)} test"
        f" -> {split_path.relative_to(PROJECT_ROOT)}"
        f"\n  target prevalence: train {train_rate:.4f}, test {test_rate:.4f}"
    )

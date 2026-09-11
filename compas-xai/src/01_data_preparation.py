import pandas as pd

DATA_PATH = "/Users/alexandramariamorar/Documents/Projects/InterpretableVsExplainble/BAINSA_analysis_pr1/compas-xai/data/processed/propublica_data_for_fairml.csv"

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
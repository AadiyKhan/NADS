import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

# --- Configuration ---
# NOTE: Update this path to where your CIC-IDS-2017 CSV files are located
DATA_PATH = Path('./dataset/') 
OUTPUT_FILENAME = 'cleaned_merged_cicids2017.csv'
TARGET_COLUMN = 'Label' # The column containing 'BENIGN' or attack names

print(f"Starting data merging and cleaning from: {DATA_PATH}")

# 1. FIND AND MERGE ALL CSV FILES
# Use glob to find all CSV files in the data directory
all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

# Use a generator expression for memory efficiency with large files
df_list = (pd.read_csv(f) for f in all_files)

# Concatenate all DataFrames
df = pd.concat(df_list, ignore_index=True)

print(f"Merged DataFrame shape: {df.shape}")

# 2. CLEAN COLUMN NAMES
# Remove leading/trailing spaces and replace spaces with underscores for Python-friendly names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('/', '_')

# Fix a known issue where 'Flow_ID' can cause issues, we will drop it later anyway
df = df.rename(columns={'Flow_ID': 'Flow_ID_str'})


# 3. HANDLE INFINITY AND MISSING VALUES
# The CIC-IDS 2017 dataset often generates Infinity values (inf) from division by zero, 
# which can break ML algorithms. We must handle them explicitly.

# Replace all np.inf (positive and negative infinity) with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Check for and report all NaN values
print("\n--- NaN/Missing Value Report ---")
nan_count = df.isnull().sum().sum()
print(f"Total NaN values found: {nan_count}")

# For a huge dataset like this, the simplest and often best approach is to drop rows with any NaN.
# Since the total count is usually a tiny fraction (<1%) of the millions of rows, this doesn't lose much information.
df.dropna(inplace=True)
print(f"DataFrame shape after dropping NaN rows: {df.shape}")


# 4. REMOVE DUPLICATE ROWS
# Duplicate rows are common in this dataset and can lead to overfitting.
df.drop_duplicates(inplace=True)
print(f"DataFrame shape after dropping duplicate rows: {df.shape}")


# 5. TARGET LABEL PREPARATION (Binary Classification)
# Map all attacks to 'Anomaly' (1) and 'BENIGN' to 'Normal' (0)
# This simplifies the problem from multi-class to binary anomaly detection.
df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip() # Clean whitespace in labels
df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 0 if x == 'BENIGN' else 1)

print("\n--- Final Label Counts (Anomaly vs. Normal) ---")
print(df[TARGET_COLUMN].value_counts())


# 6. SAVE THE CLEANED DATASET
df.to_csv(OUTPUT_FILENAME, index=False)
print(f"\nSuccessfully cleaned and saved data to: {OUTPUT_FILENAME}")
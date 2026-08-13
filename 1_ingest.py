"""
1_ingest.py 
Author: Jacob Erasmus
Project: Honours Research Project
Purpose: Executes the inital data ingestion, memory downcasting, and stratified dataset spilliting to establish isolated training and test sets.
Alignment with Methodology: 
    - Section 4.2.1: Memory Optimisation and Datatype Downcasting.
    - Section 4.2.2: 80/20 Stratified Dataset Partitioning.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import gc # Garbage collector

##########################
# STEP 1.1: SAFE INGESTION
##########################
print("--- Step 1: Initial Ingestion Commencing ---")
file_path = "logs.csv"

try:
    df_raw = pd.read_csv(
        file_path, 
        low_memory=False  # Prevents Pandas from wasting overhead guessing mixed types
    )
    print(f"Dataset successfully loaded. Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
except MemoryError:
    print("CRITICAL: Out of Memory Error during initial load. Switch to chunking.")
    sys.exit()

############################# 
# STEP 1.2: DEEP MEMORY AUDIT
#############################
print("\n--- Deep Memory Audit ---")

# Calculate total memory usage in Megabytes (MB)
total_memory_bytes = df_raw.memory_usage(deep=True).sum()
total_memory_mb = total_memory_bytes / (1024 ** 2)
print(f"Total DataFrame Memory Footprint: {total_memory_mb:.2f} MB")

# Print the standard info to verify non-null counts and default types
print("\nDataFrame Info:")
df_raw.info(memory_usage='deep')

##################################################
# STEP 1.3: TYPE PROFILING (IDENTIFYING THE BLOAT)
##################################################
print("\n--- Datatype Distribution ---")
# Count how many columns belong to each datatype
type_counts = df_raw.dtypes.value_counts()
print(type_counts)

print("\n--- Top 10 Most Memory-Intensive Columns ---")
# Calculate memory per column in MB, sort descending, and show the top 10
memory_by_col = df_raw.memory_usage(deep=True) / (1024 ** 2)
worst_offenders = memory_by_col.sort_values(ascending=False).head(10)
print(worst_offenders)

#############################################################
# STEP 2: MEMORY OPTIMISATION AND DOWNCASTING (Section 4.2.1)
#############################################################
print("--- Step 2: Downcasting Protocol Commencing ---")

# 2.1. Drop redundant string labels to remove text bloat (if both 'Label' and 'LabelEnc' exist, we keep the encoded version)
if 'Label' in df_raw.columns and 'LabelEnc' in df_raw.columns:
    print("Dropping redundant string 'Label' column (relying on 'LabelEnc')...")
    df_raw = df_raw.drop(columns=['Label'])

# 2.2. Downcast Floats (64-bit to 32-bit of the continous network telemetry)
float_cols = df_raw.select_dtypes(include=['float64']).columns
df_raw[float_cols] = df_raw[float_cols].astype('float32')
print(f"Downcasted {len(float_cols)} float columns to float32.")

# 2.3. Downcast Integers (Pandas will automatically find the smallest safe int size: 8, 16, or 32)
int_cols = df_raw.select_dtypes(include=['int64']).columns
df_raw[int_cols] = df_raw[int_cols].apply(pd.to_numeric, downcast='integer')
print(f"Downcasted {len(int_cols)} integer columns.")

# 2.4. Convert any remaining object/string columns to categorical
obj_cols = df_raw.select_dtypes(include=['object', 'string']).columns
for col in obj_cols:
    df_raw[col] = df_raw[col].astype('category')
print(f"Converted {len(obj_cols)} string columns to categorical.")

# 2.5. Final Memory Check: Verifying the matematical reduction in RAM usage
new_memory_bytes = df_raw.memory_usage(deep=True).sum()
new_memory_mb = new_memory_bytes / (1024 ** 2)
print(f"\nOptimization Complete.")
print(f"New Memory Footprint: {new_memory_mb:.2f} MB")

##############################################
# STEP 3: STRATIFIED ISOLATION (Section 4.2.2)
##############################################
print("--- Step 3: Enforcing 80/20 Split ---")
# Stratify the split to ensure the severe class imbalance of zero-day attacks is perfectly proportional in both the training and testing sets.

# Define features (X) and target (y)
X_full = df_raw.drop(columns=['LabelEnc'])
y_full = df_raw['LabelEnc']

# Perform the stratified split (stratified to maintain your severe class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, 
    test_size=0.20, 
    stratify=y_full, 
    random_state=42
)

print(f"Training Fold (80%): {X_train.shape[0]} rows")
print(f"Test Vault (20%): {X_test.shape[0]} rows")

###################################
# STEP 4: EXPORT THE DATASET VAULTS
###################################
# Export Test Set (only used again at final inference (ensemble stage))
df_test_vault = pd.concat([X_test, y_test], axis=1)
df_test_vault.to_csv("logs_20percent.csv", index=False)
print("Test set safely exported to 'logs_20percent.csv'.")

# Export Training Set (Used for EDA, Model Selection, and Tuning)
df_train_export = pd.concat([X_train, y_train], axis=1)
df_train_export.to_csv("logs_80percent.csv", index=False)
print("Training set safely exported to 'logs_80percent.csv'.")

# DELETE the variables and original df to free up memory
del X_test
del y_test
del df_test_vault
del X_train
del y_train
del df_train_export
del df_raw
gc.collect()

print("1_ingest.py complete: All data safely exported and purged from active memory. Ready for next stage.")

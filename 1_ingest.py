"""
File: 1_ingest.py 
Author: Jacob Erasmus
Project: BSc(Hons) Computer Science Research Project (University of Pretoria)
Purpose: Executes initial data ingestion, class distribution and duplicate auditing, memory optimisation via datatype downcasting, and 80/20 stratified partitioning.
Alignment with Methodology: 
    - Chapter 3 (Step 1): Data Ingestion and Exploratory Profiling
    - Chapter 3 (Step 2): Memory Optimisation and Dataype Downcasting
    - Chapter 3 (Step 3): Stratified 80/20 Train-Test Partitioning
"""

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import gc # Garbage collector

#################################################
# STEP 1.1: DATASET INGESTION & INITIAL PROFILING
#################################################
print("=====================================================")
print(" STEP 1: Dataset Ingestion and Exploratory Profiling ")
print("=====================================================")

file_path = "logs.csv"

try:
    df_raw = pd.read_csv(
        file_path, 
        low_memory=False  # Prevents mixed type inference overhead during ingestion
    )
    print(f"Dataset successfully loaded. Dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
except FileNotFoundError:
    print(f"[!] ERROR: '{file_path}' not found in the root directory.")
    sys.exit(1)
except MemoryError:
    print("[!] CRITICAL: Insufficient memory during initial load. Chunked ingestion required.")
    sys.exit(1)

print("\n--- Dataset Composition & Class Distribution ---")
if 'Label' in df_raw.columns:
    class_counts = df_raw['Label'].value_counts()
    for attack_class, count in class_counts.items():
        percentage = (count / df_raw.shape[0]) * 100
        print(f"- {attack_class}: {count:,} ({percentage:.2f}%)")

print("\n--- Synthetic Artefact & Duplicate Audit ---")
duplicates = df_raw.duplicated().sum()
duplicate_percent = (duplicates / df_raw.shape[0]) * 100
print(f"Exact Duplicate Records: {duplicates:,} ({duplicate_percent:.2f}%)")

#######################################
# STEP 1.2: BASELINE MEMORY USAGE AUDIT
#######################################
print("\n--- Baseline Memory Usage Audit ---")

# Calculate total memory usage in Megabytes (MB)
total_memory_bytes = df_raw.memory_usage(deep=True).sum()
total_memory_mb = total_memory_bytes / (1024 ** 2)
print(f"Initial DataFrame Memory Footprint: {total_memory_mb:.2f} MB")

print("\nDataFrame Schema Summary:")
df_raw.info(memory_usage='deep')

########################################
# STEP 1.3: ATTRIBUTE DATATYPE PROFILING
########################################
print("\n--- Datatype Distribution ---")
# Count how many columns belong to each datatype
type_counts = df_raw.dtypes.value_counts()
print(type_counts)

print("\n--- Top 10 Most Memory-Intensive Attributes (MB) ---")
# Calculate memory per column in MB, sort descending, and show the top 10
memory_by_col = df_raw.memory_usage(deep=True) / (1024 ** 2)
worst_offenders = memory_by_col.sort_values(ascending=False).head(10)
print(worst_offenders)

#############################################
# STEP 2: MEMORY OPTIMISATION AND DOWNCASTING
#############################################
print("\n=========================================================")
print(" STEP 2: MEMORY OPTIMISATION AND DATATYPE DOWNCASTING    ")
print("=========================================================")

# 2.1. Remove redundant string target column ('Label') while retaining encoded target ('LabelEnc')
if 'Label' in df_raw.columns and 'LabelEnc' in df_raw.columns:
    print("Dropping redundant string 'Label' column (retaining integer-encoded 'LabelEnc')...")
    df_raw = df_raw.drop(columns=['Label'])

# 2.2. Downcast 64-bit continuous floating-point features to 32-bit floats (float32)
float_cols = df_raw.select_dtypes(include=['float64']).columns
df_raw[float_cols] = df_raw[float_cols].astype('float32')
print(f"Downcasted {len(float_cols)} float64 columns to float32.")

# 2.3. Downcast 64-bit integer columns to minimal safe integer representations (int8/int16/int32)
int_cols = df_raw.select_dtypes(include=['int64']).columns
df_raw[int_cols] = df_raw[int_cols].apply(pd.to_numeric, downcast='integer')
print(f"Downcasted {len(int_cols)} int64 columns to minimal integer representations.")

# 2.4. Convert any remaining object/string columns to categorical types
obj_cols = df_raw.select_dtypes(include=['object', 'string']).columns
for col in obj_cols:
    df_raw[col] = df_raw[col].astype('category')
print(f"[+] Converted {len(obj_cols)} object/string columns to categorical.")

# 2.5. Verify post-downcasting memory footprint reduction
new_memory_bytes = df_raw.memory_usage(deep=True).sum()
new_memory_mb = new_memory_bytes / (1024 ** 2)
reduction_pct = ((total_memory_mb - new_memory_mb) / total_memory_mb) * 100
print(f"\n[+] Memory Optimisation Complete.")
print(f"    Optimised Memory Footprint: {new_memory_mb:.2f} MB ({reduction_pct:.2f}% reduction)")

##################################################
# STEP 3: STRATIFIED 80/20 TRAIN/TEST PARTITIONING
##################################################
print("\n==============================================")
print("STEP 3: STRATIFIED 80/20 DATASET PARTITIONING ")
print("==============================================")

# Separate input features (X) and target class labels (y)
X_full = df_raw.drop(columns=['LabelEnc'])
y_full = df_raw['LabelEnc']

# Perform stratified split to preserve exact class proportions across both partitions
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, 
    test_size=0.20, 
    stratify=y_full, 
    random_state=42
)

print(f"[+] Training Partition (80%): {X_train.shape[0]:,} records")
print(f"[+] Testing Partition (20%): {X_test.shape[0]:,} records")

#######################################################
# STEP 4: EXPORT ISOLATED PARTITIONS AND RELEASE MEMORY
#######################################################
print("\n[+] Exporting partitioned datasets to disk...")

df_test_vault = pd.concat([X_test, y_test], axis=1)
df_test_vault.to_csv("logs_20percent.csv", index=False)
print("     -> Unseen test partition exported to 'logs_20percent.csv'.")

df_train_export = pd.concat([X_train, y_train], axis=1)
df_train_export.to_csv("logs_80percent.csv", index=False)
print("     -> Training partition exported to 'logs_80percent.csv'.")

# Release memory objects and trigger garbage collection
del X_test
del y_test
del df_test_vault
del X_train
del y_train
del df_train_export
del df_raw
gc.collect()

print("\n[+] 1_ingest.py completed successfully. Active memory cleared.")

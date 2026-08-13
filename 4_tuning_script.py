"""
4_tuning_script.py
Author: Jacob Erasmus
Project: Honours Research Project 
Purpose: Executes the discovery phase to lock the Golden Features (the best features to use) and optimal hyperparameters. 
         This script is run once to establish the mathematical constants used in the final ensemble. 
Alignment with Methodology:
    - Section 4.2.1: Memory Optimisation and Dataype Downcasting.
    - Section 4.2.2: Stratified 5-Fold Cross-Validation.
    - Section 4.2.3: Information Gain dimensionality reduction.
    - Section 4.2.4: Recursive Feature Elimination (RFE)
    - Section 4.2.5: Imblearn Pipeline explicitly prevents SMOTE data leakage by generating synthetic data inside the CV folds.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import warnings
from scipy.stats import uniform, randint
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

print("==================================================")
print("--- FULL DATASET DISCOVERY AND TUNING ---")
print("==================================================")

###############################################
# 1. LOAD AND DOWNCAST THE DATA (Section 4.2.1)
###############################################
print("[+] Loading 80%Training Vault...")
df_train = pd.read_csv("logs_80percent.csv")

print("    -> Executing Memory Optimisation & Downcasting (Sec 4.2.1)...")
# Downcast continuous floats to 32-bit to prevent RAM exhaustion during RFE
float_cols = df_train.select_dtypes(include=['float64']).columns
df_train[float_cols] = df_train[float_cols].astype('float32')

# Downcast binary flags and counters to minimal integer states
int_cols = df_train.select_dtypes(include=['int64']).columns
for col in int_cols:
    if df_train[col].max() <= 127 and df_train[col].min() >= -128:
        df_train[col] = df_train[col].astype('int8')
    else:
        df_train[col] = df_train[col].astype('int32')

# Drop the column
X_train = df_train.drop(columns=['LabelEnc'])
y_train = df_train['LabelEnc']

print(f"    [>] Ingested full training vault: {X_train.shape[0]} rows, {X_train.shape[1]} features.")

########################################################
# 2. OPTIMIsED FEATURE SELECTION (Section 4.2.3 & 4.2.4)
########################################################
print("\n[+] Executing Global Feature Selection (IG -> RFE)...")

# Step 2A: Information Gain (Entropy) (section 4.2.3)
print("    -> Calculating IG entropy...")
ig_scores = mutual_info_classif(X_train, y_train, random_state=42)
ig_series = pd.Series(ig_scores, index=X_train.columns)
# Prune completely noisy features to accelrate the processing
top_50_features = ig_series.sort_values(ascending=False).head(50).index.tolist()

X_train_ig = X_train[top_50_features]

# Step 2B: Recursive Feature Elimination (section 4.2.4)
print("    -> Executing Recursive Feature Elimination with 5-Fold CV")
rf_estimator = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Execute RFECV. 'scoring=f1_macro' ensures the Random Forest heavily peanlises feature sets that ignore minority zero-day attacks during the pruning process.
# Step by 2 to accelerate the loop, enforcing a minimum of 10 features
rfecv = RFECV(
    estimator=rf_estimator,
    step=2, 
    cv=cv,
    scoring='f1_macro',
    min_features_to_select=10,
    n_jobs=-1
)
rfecv.fit(X_train_ig, y_train)

optimal_num = rfecv.n_features_
golden_features = X_train_ig.columns[rfecv.support_].tolist()
X_train_final = X_train_ig[golden_features]

print(f"\n    [>] RFE Cross-Validation Complete. Mathematical optimum found at: {optimal_num} features.")
print(f"    [>] GOLDEN FEATURE LIST TO COPY TO ENSEMBLE SCRIPT:")
print(f"        {golden_features}")

# Step 2C: Generate the Ablation Study Graph for Result write up
plt.figure(figsize=(10, 6))
x_axis = range(10, len(rfecv.cv_results_['mean_test_score']) * 2 + 10, 2)
plt.plot(x_axis, rfecv.cv_results_['mean_test_score'], marker='o', linestyle='-', color='b')
plt.title('RFE with 5-Fold CV: Feature Dimensionality vs. F1-Score')
plt.xlabel('Number of Features Selected')
plt.ylabel('Macro F1-Score (Cross-Validation)')
plt.grid(True)
plt.tight_layout()
plt.savefig('rfe_curve.png')
plt.close()
print("    [>] Feature elimination curve saved as 'rfe_curve.png'.")
# -------------------------------------------------------------
# 3. LAYER 1: UNSUPERVISED (OCSVM) JUSTIFICATION (section 4.3)
# -------------------------------------------------------------
print("\n[+] Layer 1 (OCSVM) Parameter Evaluation...")
print("    -> 'nu' parameter locked at 0.20 (Derived mathematically from EDA 19.7% anomaly rate).")
print("    -> 'n_components' locked at 300 (Hardware/Time-Complexity ceiling).")
print("    [>] Layer 1 parameters empirically optimised. Skipping automated CV.")

# ------------------------------------------------------
# 4. TUNE LAYER 2: SUPERVISED HYPERPARAMETERS (XGBoost)
# ------------------------------------------------------
print("\n[+] Initiating RandomizedSearchCV for Layer 2 (XGBoost)...")

# Define target minority threshold for SMOTE based on the full dataset
class_counts = y_train.value_counts()
majority_count = class_counts.max()
target_minority = int(majority_count * 0.10)

smote_strategy = {}
for cls, count in class_counts.items():
    if cls == 0:
        smote_strategy[cls] = majority_count
    elif count > 1: # Safety check to prevent SMOTE from crashing on 1-row classes
        smote_strategy[cls] = count if count >= target_minority else target_minority

smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=1, random_state=42)
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)

# Methodology Compliance (section 4.2.5): Wrap SMOTE and XGBoost in a pipeline so SMOTE only 
# applies to the training folds inside the CV loop. This prevents synthetic signatures from leaking into the validation data.
tuning_pipeline = ImbPipeline([
    ('smote', smote),
    ('xgb', xgb_model)
])

# The Parameter Grid for XGBoost ('xgb__' targets the model inside the pipeline)
param_dist_xgb = {
    'xgb__max_depth': randint(3, 15),
    'xgb__learning_rate': uniform(0.01, 0.29),
    'xgb__n_estimators': randint(50, 200),
    'xgb__subsample': uniform(0.6, 0.4) # Prevents overfitting by sampling data rows per tree
}

macro_f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)

random_search_xgb = RandomizedSearchCV(
    tuning_pipeline, 
    param_distributions=param_dist_xgb, 
    n_iter=10, 
    cv=cv, 
    scoring=macro_f1_scorer,
    random_state=42,
    n_jobs=-1
)

print("    -> Searching for optimal hyperparameters...")
random_search_xgb.fit(X_train_final, y_train)

# Strip the 'xgb__' prefix for clean printing
best_params_clean = {k.replace('xgb__', ''): v for k, v in random_search_xgb.best_params_.items()}
print(f"\n    [>] Layer 2 Golden Parameters: {best_params_clean}")
print(f"    [>] Layer 2 Best Tuning F1-Score: {random_search_xgb.best_score_:.4f}")

print("\n==================================================")
print(" TUNING COMPLETE. READY FOR ENSEMBLE INTEGRATION. ")
print("==================================================")

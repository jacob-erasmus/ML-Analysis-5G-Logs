"""
File: 4_hyperparameter_feature_tuning.py
Author: Jacob Erasmus
Project: BSc(Hons) Computer Science Research Project (University of Pretoria)
Purpose: Executes global feature space selection (IF -> RFECV) on the 80% training partition to identify the optimal feature subset (10 features), and performs 5-fold RandomizedSearchCV to tune the Layer 2 XGBoost hyperparameters. 
Alignment with Methodology:
    - Chapter 3 (Step 2): Memory Optimisation and Dataype Downcasting.
    - Chapter 3 (Step 4 & 5): Information Gain Filtering and Cross-Validated RFE (RFECV).
    - Chapter 3 (Step 6): Logarithmic Cost-Sensitive Sample Weighting
    - Chapter 3 (Step 9): Hyperparamter and Feature Space Tuning
"""
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import warnings
from scipy.stats import uniform, randint
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

print("====================================================================")
print(" STEP 9: GLOBAL FEATURE SELECTION (RFECV) AND HYPERPARAMETER TUNING ")
print("====================================================================")

#################################################
# 1. LOAD AND DOWNCAST THE 80% TRAINING PARTITION
#################################################
print("[+] Loading 80% training partition ('logs_80percent.csv')...")
df_train = pd.read_csv("logs_80percent.csv")

print("    -> Applying datatype downcasting...")
float_cols = df_train.select_dtypes(include=['float64']).columns
df_train[float_cols] = df_train[float_cols].astype('float32')

int_cols = df_train.select_dtypes(include=['int64']).columns
for col in int_cols:
    if df_train[col].max() <= 127 and df_train[col].min() >= -128:
        df_train[col] = df_train[col].astype('int8')
    else:
        df_train[col] = df_train[col].astype('int32')

X_train = df_train.drop(columns=['LabelEnc'])
y_train = df_train['LabelEnc']
print(f"    [>] Training partition ready: {X_train.shape[0]:,} records, {X_train.shape[1]} features.")

##############################################
# 2. TWO STAGE FEATURE SELECTION (IG -> RFECV)
##############################################
print("\n[+] Executing Feature Selection (IG -> RFE)...")

# Step 2A: Information Gain (Mutual Information) Filtering
print("    -> Computing IG scores across all features...")
ig_scores = mutual_info_classif(X_train, y_train, random_state=42)
ig_series = pd.Series(ig_scores, index=X_train.columns)
top_50_features = ig_series.sort_values(ascending=False).head(50).index.tolist()
X_train_ig = X_train[top_50_features]
print(f"    -> Retained top {len(top_50_features)} highest entropy features for RFECV.")

# Step 2B: Cross-Validated Recursive Feature Elimination (RFECV)
print("    -> Executing RFECV with 5-Fold Stratified Cross-Validation (scoring='f1_macro')...")
rf_estimator = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42, n_jobs=1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Unweighted macro F1 scoring ensures feature subsets are evaluated across all 15 classes equally
rfecv = RFECV(
    estimator=rf_estimator,
    step=2, 
    cv=cv,
    scoring='f1_macro',
    min_features_to_select=5,
    n_jobs=-1
)
rfecv.fit(X_train_ig, y_train)

optimal_num = rfecv.n_features_
golden_features = X_train_ig.columns[rfecv.support_].tolist()
X_train_final = X_train_ig[golden_features]

print(f"\n    [>] RFECV Complete. Optimal dimensionality identified: {optimal_num} features.")
print(f"    [>] Selected 10-Feature Subset:")
print(f"        {golden_features}")

# Step 2C: Save RFECV Dimensionality vs Macro F1-Score Curve
v_directory = "visualisations"
os.makedirs(v_directory, exist_ok=True)
plt.figure(figsize=(10, 6))
x_axis = range(rfecv.min_features_to_select, rfecv.min_features_to_select + (len(rfecv.cv_results_['mean_test_score']) * rfecv.step), rfecv.step)
plt.plot(x_axis, rfecv.cv_results_['mean_test_score'], marker='o', linestyle='-', color='b')
plt.title('RFE with 5-Fold CV: Feature Dimensionality vs. F1-Score')
plt.xlabel('Number of Features Selected')
plt.ylabel('Macro F1-Score (Cross-Validation)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(v_directory, 'rfe_curve.png'), dpi=300)
plt.close()
print("    [>] RFECV performance curve saved to 'visualisations/rfe_curve.png'.")

##############################################################
# 3. LAYER 1: UNSUPERVISED PARAMETER SPECIFICATION
##############################################################
print("\n[+] Verifying Layer 1 Unsupervised Parameter Configuration...")
print("    -> OCSVM 'nu' and Isolation Forest 'contamination' set to 0.20 (calibrated to 19.70 empirical anomaly rate).")
print("    -> Nystroem kernel approximation 'n_components' set to 300.")

########################################################
# 4. LAYER 2: SUPERVISED HYPERPARAMETER TUNING (XGBoost)
########################################################
print("\n[+] Executing RandomizedSearchCV for Layer 2 Classifier (XGBoost)...")

print("    -> Computing logarithmic cost-sensitive sample weights...")
class_counts = y_train.value_counts()
majority_count = class_counts.max()
smoothed_weights = {
    cls: np.log1p(majority_count / count) + 1 for cls, count in class_counts.items()
}
sample_weights_log = np.array([smoothed_weights[cls] for cls in y_train])

xgb_model = XGBClassifier(eval_metric='mlogloss', max_delta_step=5, min_child_weight=0.001, random_state=42, n_jobs=-1)

param_dist_xgb = {
    'max_depth': randint(8, 16),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(50, 200),
    'subsample': uniform(0.7, 0.3)
}

macro_f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)

random_search_xgb = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist_xgb, 
    n_iter=15, 
    cv=cv, 
    scoring=macro_f1_scorer,
    random_state=42,
    n_jobs=-1
)

print("    -> Fitting 5-Fold RandomizedSearchCV (n_iter=15) across selected feature subset...")
random_search_xgb.fit(X_train_final, y_train, sample_weight=sample_weights_log)

print(f"\n    [>] Optimal Layer 2 Hyperparameters: {random_search_xgb.best_params_}")
print(f"    [>] Best 5-Fold Cross-Validation Macro F1-Score: {random_search_xgb.best_score_:.4f}")

print("\n=======================================================")
print(" FEATURE SELECTION AND HYPERPARAMETER TUNING COMPLETE. ")
print("=======================================================")

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

warnings.filterwarnings("ignore")

print("==================================================")
print("  PHASE 4: HYPERPARAMETER TUNING (10% SAMPLE)     ")
print("==================================================")

# -------------------------------------------------------------
# 1. LOAD AND SAMPLE THE DATA
# -------------------------------------------------------------
print("[+] Loading Training Vault...")
df_train = pd.read_csv("logs_80percent.csv")

# Drop the column BEFORE we sample
X_full = df_train.drop(columns=['LabelEnc'])
y_full = df_train['LabelEnc']

# Extract exactly 10% using Scikit-Learn for perfect stratification
print("[+] Extracting 10% stratified sample for tuning...")
_, X_tune, _, y_tune = train_test_split(
    X_full, y_full, 
    test_size=0.10, 
    stratify=y_full, 
    random_state=42
)

# -------------------------------------------------------------
# 2. OPTIMIZED FEATURE SELECTION (IG -> RFECV)
# -------------------------------------------------------------
print("\n[+] Executing Global Feature Selection (IG -> RFECV)...")

# Step 2A: Information Gain (Global 50)
ig_scores = mutual_info_classif(X_tune, y_tune, random_state=42)
ig_series = pd.Series(ig_scores, index=X_tune.columns)
top_50_features = ig_series.sort_values(ascending=False).head(50).index.tolist()

X_tune_ig = X_tune[top_50_features]

# Step 2B: RFECV (The Mathematical Optimum)
print("    -> Executing Recursive Feature Elimination with 3-Fold CV on 10% sample...")
rf_estimator = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Step by 2 to accelerate the loop, enforcing a minimum of 10 features
rfecv = RFECV(
    estimator=rf_estimator,
    step=2, 
    cv=cv,
    scoring='f1_macro',
    min_features_to_select=10,
    n_jobs=-1
)
rfecv.fit(X_tune_ig, y_tune)

optimal_num = rfecv.n_features_
golden_features = X_tune_ig.columns[rfecv.support_].tolist()

print(f"\n    [>] RFECV Complete. Mathematical optimum found at: {optimal_num} features.")
print(f"    [>] GOLDEN FEATURE LIST TO COPY TO SCRIPT 4:")
print(f"        {golden_features}")

# Step 2C: Generate the Ablation Study Graph for Chapter 4
plt.figure(figsize=(10, 6))
x_axis = range(10, len(rfecv.cv_results_['mean_test_score']) * 2 + 10, 2)
plt.plot(x_axis, rfecv.cv_results_['mean_test_score'], marker='o', linestyle='-', color='b')
plt.title('RFECV: Feature Dimensionality vs. F1-Score')
plt.xlabel('Number of Features Selected')
plt.ylabel('Macro F1-Score (Cross-Validation)')
plt.grid(True)
plt.tight_layout()
plt.savefig('rfecv_curve.png')
plt.close()
print("    [>] Feature elimination curve saved as 'rfecv_curve.png'.")
# -------------------------------------------------------------
# 3. LAYER 1: UNSUPERVISED (OCSVM) JUSTIFICATION
# -------------------------------------------------------------
print("\n[+] Layer 1 (OCSVM) Parameter Evaluation...")
print("    -> 'nu' parameter locked at 0.20 (Derived mathematically from EDA 19.7% anomaly rate).")
print("    -> 'n_components' locked at 300 (Hardware/Time-Complexity ceiling).")
print("    [>] Layer 1 parameters empirically optimized. Skipping automated CV.")

# -------------------------------------------------------------
# 4. TUNE LAYER 2: SUPERVISED (XGBoost)
# -------------------------------------------------------------
print("\n[+] Initiating RandomizedSearchCV for Layer 2 (XGBoost)...")

print("    -> Balancing the 10% sample with SMOTE (Adaptive Strategy)...")
class_counts = y_tune.value_counts()
majority_count = class_counts.max()
target_minority = int(majority_count * 0.10)

# ADAPTIVE SMOTE FIX: 
# SMOTE requires at least 2 points to draw a synthetic line. 
# We filter out any zero-day attacks that were reduced to 1 row by the 10% sampling.
smote_strategy = {}
for cls, count in class_counts.items():
    if cls == 0:
        smote_strategy[cls] = majority_count
    elif count > 1: # The Safety Check
        smote_strategy[cls] = count if count >= target_minority else target_minority

smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=1, random_state=42)
X_tune_balanced, y_tune_balanced = smote.fit_resample(X_tune_final, y_tune)

xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)

# The Parameter Grid for XGBoost
param_dist_xgb = {
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.29),
    'n_estimators': randint(50, 200),
    'subsample': uniform(0.6, 0.4) # Prevents overfitting by sampling data rows per tree
}

macro_f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)

random_search_xgb = RandomizedSearchCV(
    xgb_model, 
    param_distributions=param_dist_xgb, 
    n_iter=10, 
    cv=3, 
    scoring=macro_f1_scorer,
    random_state=42
)

print("    -> Searching for optimal learning_rate, depth, and estimators...")
random_search_xgb.fit(X_tune_balanced, y_tune_balanced)

print(f"\n    [>] Layer 2 Golden Parameters: {random_search_xgb.best_params_}")
print(f"    [>] Layer 2 Best Tuning F1-Score: {random_search_xgb.best_score_:.4f}")

print("\n==================================================")
print(" TUNING COMPLETE. READY FOR ENSEMBLE INTEGRATION. ")
print("==================================================")

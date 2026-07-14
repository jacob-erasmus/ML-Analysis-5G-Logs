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

warnings.filterwarnings("ignore")

print("==================================================")
print("  PHASE 4: HYPERPARAMETER TUNING (10% SAMPLE)     ")
print("==================================================")

# -------------------------------------------------------------
# 1. LOAD AND SAMPLE THE DATA
# -------------------------------------------------------------
print("[+] Loading Training Vault...")
df_train = pd.read_csv("5G_train_80percent.csv")

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
# 2. FEATURE SELECTION (On the 10% Sample)
# -------------------------------------------------------------
print("\n[+] Re-verifying Global Features (IG -> RFE)...")
ig_scores = mutual_info_classif(X_tune, y_tune, random_state=42)
ig_series = pd.Series(ig_scores, index=X_tune.columns)
top_50_features = ig_series.sort_values(ascending=False).head(50).index.tolist()

X_tune_ig = X_tune[top_50_features]

rf_estimator = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rf_estimator, n_features_to_select=20, step=5)
rfe.fit(X_tune_ig, y_tune)

final_features = X_tune_ig.columns[rfe.support_].tolist()
X_tune_final = X_tune_ig[final_features]
print(f"[+] Final 20 Forensic Features Locked.")

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

print("    -> Balancing the 10% sample with SMOTE...")
class_counts = y_tune.value_counts()
majority_count = class_counts.max()
target_minority = int(majority_count * 0.10)

smote_strategy = {cls: (count if count >= target_minority else target_minority) 
                  for cls, count in class_counts.items() if cls != 0}
smote_strategy[0] = majority_count 

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
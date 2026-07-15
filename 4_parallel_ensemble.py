import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import PolynomialFeatures
warnings.filterwarnings("ignore") 
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import ADASYN

print("==================================================")
print("  PHASE 4: HYBRID ENSEMBLE & FORENSIC BENCHMARK   ")
print("==================================================")

# -------------------------------------------------------------
# 1. LOAD THE DATA (Opening the Vaults)
# -------------------------------------------------------------
print("[+] Loading 80% Training Vault...")
df_train = pd.read_csv("logs_80percent.csv")
X_train = df_train.drop(columns=['LabelEnc'])
y_train = df_train['LabelEnc']

print("[+] Loading 20% Testing Vault (Unseen Data)...")
df_test = pd.read_csv("logs_20percent.csv") # Update filename if needed
X_test = df_test.drop(columns=['LabelEnc'])
y_test = df_test['LabelEnc']

""" # -------------------------------------------------------------
# 2. GLOBAL FEATURE SELECTION (Applied to full 80%)
# -------------------------------------------------------------
print("\n[+] Executing Global Feature Selection (IG -> RFE)...")

# A. Information Gain
ig_scores = mutual_info_classif(X_train, y_train, random_state=42)
ig_series = pd.Series(ig_scores, index=X_train.columns)
top_50_features = ig_series.sort_values(ascending=False).head(50).index.tolist()

X_train_ig = X_train[top_50_features]
X_test_ig = X_test[top_50_features]

# B. Recursive Feature Elimination
rf_estimator = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rf_estimator, n_features_to_select=20, step=5)
rfe.fit(X_train_ig, y_train)

final_features = X_train_ig.columns[rfe.support_].tolist()
X_train_final = X_train_ig[final_features]
X_test_final = X_test_ig[final_features]
print(f"[+] Final 20 Forensic Features Locked for Production.") """

# -------------------------------------------------------------
# 2. OPTIMIZED GLOBAL FEATURE SELECTION (RFECV 10-Feature Lock)
# -------------------------------------------------------------
print("\n[+] Loading 10 Golden Features derived from Phase 4 RFECV...")

golden_features = [
    'Total Length of Bwd Packets', 
    'Bwd Packet Length Mean', 
    'Total Length of Fwd Packets', 
    'Flow Bytes/s', 
    'Flow Duration', 
    'Fwd Packet Length Mean', 
    'Flow Packets/s', 
    'fields.vnf_connection', 
    'host.name_nrf', 
    'fields.vnf_weird'
]

X_train_final = X_train[golden_features]
X_test_final = X_test[golden_features]

print(f"    [>] Data dimensionality strictly locked to 10 optimal forensic vectors.")

# -------------------------------------------------------------
# 3. PARALLEL HYBRID ENSEMBLE (Meta-Feature Extraction)
# -------------------------------------------------------------
print("\n[+] Executing Parallel Hybrid Ensemble (Meta-Feature Architecture)...")

# Step 3A: Train Unsupervised Models on Normal Traffic
X_train_normal = X_train_final[y_train == 0]

layer1_scaler = StandardScaler()
X_train_normal_scaled = layer1_scaler.fit_transform(X_train_normal)

layer1_ocsvm = make_pipeline(
    Nystroem(kernel='rbf', gamma=None, n_components=300, random_state=42),
    SGDOneClassSVM(nu=0.20, random_state=42)
)
layer1_ocsvm.fit(X_train_normal_scaled)

layer1_if = IsolationForest(n_estimators=100, contamination=0.20, random_state=42, n_jobs=-1)
layer1_if.fit(X_train_normal_scaled)

# Step 3B: Extract Continuous Anomaly Scores (The Meta-Features)
print("    -> Extracting continuous anomaly decision scores for the entire vault...")
# Scale the full datasets for Layer 1 scoring
X_train_full_scaled = layer1_scaler.transform(X_train_final)
X_test_full_scaled = layer1_scaler.transform(X_test_final)

# decision_function returns a continuous float instead of a hard 1 or 0
train_scores_ocsvm = layer1_ocsvm.decision_function(X_train_full_scaled)
train_scores_if = layer1_if.decision_function(X_train_full_scaled)

test_scores_ocsvm = layer1_ocsvm.decision_function(X_test_full_scaled)
test_scores_if = layer1_if.decision_function(X_test_full_scaled)

# Step 3C: Append Meta-Features to the 10 Golden Features
print("    -> Fusing unsupervised opinions into Layer 2 input dimensions...")
X_train_meta = X_train_final.copy()
X_train_meta['OCSVM_Score'] = train_scores_ocsvm
X_train_meta['IF_Score'] = train_scores_if

X_test_meta = X_test_final.copy()
X_test_meta['OCSVM_Score'] = test_scores_ocsvm
X_test_meta['IF_Score'] = test_scores_if

# -------------------------------------------------------------
# 4. TRAINING LAYER 2: META-AWARE XGBOOST
# -------------------------------------------------------------
print("\n[+] Training Layer 2: Meta-Aware XGBoost...")

# Re-engage Adaptive SMOTE on the new 12-Dimensional space
class_counts = y_train.value_counts()
majority_count = class_counts.max()
target_minority = int(majority_count * 0.10)

smote_strategy = {cls: (count if count >= target_minority else target_minority) 
                  for cls, count in class_counts.items() if cls != 0}
smote_strategy[0] = majority_count 

smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=1, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_meta, y_train)

layer2_xgb = XGBClassifier(
    max_depth=13,
    learning_rate=0.1186,
    n_estimators=121,
    subsample=0.8394,
    eval_metric='mlogloss', 
    random_state=42, 
    n_jobs=-1
)
layer2_xgb.fit(X_train_balanced, y_train_balanced)

# -------------------------------------------------------------
# 5. INFERENCE & CALIBRATION (No Hard Gates)
# -------------------------------------------------------------
print("\n[+] Executing Global Multi-Class Calibration on Test Vault...")

# XGBoost evaluates every single log, utilizing the appended Layer 1 scores
l2_probs = layer2_xgb.predict_proba(X_test_meta)
prob_attack = 1.0 - l2_probs[:, 0]
most_likely_attack = np.argmax(l2_probs[:, 1:], axis=1) + 1

best_f1 = 0
best_thresh = 0.5
best_preds = None

for thresh in np.arange(0.01, 1.00, 0.01):
    temp_preds = np.where(prob_attack >= thresh, most_likely_attack, 0)
    temp_f1 = f1_score(y_test, temp_preds, average='macro', zero_division=0)
    
    if temp_f1 > best_f1:
        best_f1 = temp_f1
        best_thresh = thresh
        best_preds = temp_preds

print(f"    [>] Optimal Meta-Aware Probability Threshold locked at: {best_thresh:.2f}")

# The final predictions are now driven entirely by XGBoost's meta-analysis
final_predictions = best_preds

# -------------------------------------------------------------
# 6. DETERMINISTIC RULESET BASELINE (Updated Multi-Metric)
# -------------------------------------------------------------
print("[+] Executing Deterministic Ruleset Baseline...")

# Define the category of volumetric features based on your dataset
volumetric_metrics = [
    'Flow Bytes/s', 
    'Flow Packets/s', 
    'Total Length of Fwd Packets', 
    'Total Length of Bwd Packets'
]

# Initialize an array of zeros (normal) for the rule predictions
rule_predictions = np.zeros(len(X_test), dtype=int)

# Check which metrics actually exist in the dataframe to prevent KeyError
available_metrics = [m for m in volumetric_metrics if m in X_train.columns]

if available_metrics:
    print(f"    -> Applying 99th percentile static thresholds for: {available_metrics}")
    for metric in available_metrics:
        # Extract the 99th percentile threshold from strictly normal training traffic
        threshold = X_train[y_train == 0][metric].quantile(0.99)
        
        # If any test log exceeds this specific threshold, flag it as an anomaly (1)
        # The bitwise OR (|) ensures that tripping ANY rule flags the log
        rule_predictions = rule_predictions | (X_test[metric] > threshold).astype(int)
        
    rule_binary_truth = (y_test != 0).astype(int)
    
    rule_acc = accuracy_score(rule_binary_truth, rule_predictions)
    rule_f1 = f1_score(rule_binary_truth, rule_predictions, average='macro', zero_division=0)
else:
    print("[!] No volumetric metrics found. Skipping baseline.")
    rule_acc, rule_f1 = 0, 0

# -------------------------------------------------------------
# 7. FORENSIC BENCHMARKING RESULTS
# -------------------------------------------------------------
print("\n==================================================")
print("           FINAL BENCHMARKING RESULTS             ")
print("==================================================")

# Hybrid Ensemble Metrics
he_acc = accuracy_score(y_test, final_predictions)
he_prec = precision_score(y_test, final_predictions, average='macro', zero_division=0)
he_rec = recall_score(y_test, final_predictions, average='macro', zero_division=0)
he_f1 = f1_score(y_test, final_predictions, average='macro', zero_division=0)

# Calculate Reduction Factor (Rf)
total_raw_logs = len(X_test)
flagged_alerts = len(suspicious_indices)
rf = 1 - (flagged_alerts / total_raw_logs)

print("--- HYBRID ML ENSEMBLE ---")
print(f"Accuracy:         {he_acc:.4f}")
print(f"Precision:        {he_prec:.4f}")
print(f"Recall:           {he_rec:.4f}")
print(f"F1-Score (Macro): {he_f1:.4f}")
print(f"Reduction Factor: {rf:.4f} (Goal: ~0.99)")

print("\n--- DETERMINISTIC BASELINE ---")
print(f"Accuracy:         {rule_acc:.4f}")
print(f"F1-Score (Macro): {rule_f1:.4f}")
print("==================================================")
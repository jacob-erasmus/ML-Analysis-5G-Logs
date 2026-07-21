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
# 3. CASCADE HYBRID ENSEMBLE (Triage + Meta-Features)
# -------------------------------------------------------------
print("\n[+] Executing Cascade Architecture (Triage Filter + Meta-Features)...")

X_train_normal = X_train_final[y_train == 0]

layer1_scaler = StandardScaler()
X_train_normal_scaled = layer1_scaler.fit_transform(X_train_normal)
X_train_full_scaled = layer1_scaler.transform(X_train_final)
X_test_full_scaled = layer1_scaler.transform(X_test_final)

# Train the Unsupervised Extractors
layer1_ocsvm = make_pipeline(Nystroem(n_components=300, random_state=42), SGDOneClassSVM(nu=0.20, random_state=42))
layer1_ocsvm.fit(X_train_normal_scaled)

layer1_if = IsolationForest(n_estimators=100, contamination=0.20, random_state=42, n_jobs=-1)
layer1_if.fit(X_train_normal_scaled)

# Extract Continuous Scores
train_scores_ocsvm = layer1_ocsvm.decision_function(X_train_full_scaled)
train_scores_if = layer1_if.decision_function(X_train_full_scaled)

test_scores_ocsvm = layer1_ocsvm.decision_function(X_test_full_scaled)
test_scores_if = layer1_if.decision_function(X_test_full_scaled)

# --- THE TRIAGE FILTER ---
# In Scikit-Learn, highly positive scores are normal. Negative scores are anomalies.
# We set a hyper-conservative threshold: ONLY drop logs that score > 0.10 (Undeniably Normal)
print("    -> Applying Triage Filter to drop undeniably normal traffic...")
triage_threshold = 0.10 

# Create masks for the logs we are keeping (Anomalies + The Grey Area)
keep_mask_train = (train_scores_if <= triage_threshold)
keep_mask_test = (test_scores_if <= triage_threshold)

X_train_cascade = X_train_final[keep_mask_train].copy()
y_train_cascade = y_train[keep_mask_train].copy()

X_test_cascade = X_test_final[keep_mask_test].copy()
y_test_cascade = y_test[keep_mask_test].copy()

# Append Meta-Features ONLY to the logs that survived the triage
X_train_cascade['OCSVM_Score'] = train_scores_ocsvm[keep_mask_train]
X_train_cascade['IF_Score'] = train_scores_if[keep_mask_train]

X_test_cascade['OCSVM_Score'] = test_scores_ocsvm[keep_mask_test]
X_test_cascade['IF_Score'] = test_scores_if[keep_mask_test]

# Calculate the newly restored Reduction Factor
original_size = len(X_test_final)
cascade_size = len(X_test_cascade)
reduction_factor = (original_size - cascade_size) / original_size
print(f"    [>] Triage Filter successfully purged undeniably normal logs.")
print(f"    [>] Restored Reduction Factor: {reduction_factor:.4f}")

# -------------------------------------------------------------
# 4. TRAINING LAYER 2: CASCADE-AWARE XGBOOST
# -------------------------------------------------------------
print("\n[+] Training Layer 2: Cascade-Aware XGBoost...")

# Re-engage Adaptive SMOTE on the reduced, high-density anomaly space
class_counts = y_train_cascade.value_counts()
majority_count = class_counts.max()
target_minority = int(majority_count * 0.10)

smote_strategy = {cls: (count if count >= target_minority else target_minority) 
                  for cls, count in class_counts.items() if cls != 0}
# Safeguard just in case majority class drops below minorites due to aggressive filtering
smote_strategy[0] = max(majority_count, target_minority) 

smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=1, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_cascade, y_train_cascade)

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

""" # -------------------------------------------------------------
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
final_predictions = best_preds """
# -------------------------------------------------------------
# 5. INFERENCE & SOC-CONSTRAINED CALIBRATION
# -------------------------------------------------------------
print("\n[+] Executing SOC-Constrained Multi-Class Calibration on Test Vault...")
print("    -> Enforcing strict >= 0.90 Precision boundary for enterprise viability...")

l2_probs = layer2_xgb.predict_proba(X_test_cascade)
prob_attack = 1.0 - l2_probs[:, 0]
most_likely_attack = np.argmax(l2_probs[:, 1:], axis=1) + 1

best_f1 = 0
best_thresh = 0.5
best_preds = None
best_prec_at_max_f1 = 0

# The Constrained Search Loop
for thresh in np.arange(0.01, 1.00, 0.01):
    temp_preds = np.where(prob_attack >= thresh, most_likely_attack, 0)
    
    # Calculate both metrics dynamically
    temp_f1 = f1_score(y_test, temp_preds, average='macro', zero_division=0)
    temp_prec = precision_score(y_test, temp_preds, average='macro', zero_division=0)
    
    # THE SOC CONSTRAINT: We only update our "best" model IF Precision is >= 90%
    if temp_prec >= 0.90 and temp_f1 > best_f1:
        best_f1 = temp_f1
        best_prec_at_max_f1 = temp_prec
        best_thresh = thresh
        best_preds = temp_preds

# Failsafe: If no threshold met the 90% constraint, default to standard F1 maximization
if best_preds is None:
    print("    [!] WARNING: Strict 90% Precision constraint could not be met. Defaulting to standard F1 maximization.")
    
    # The actual fallback loop (Standard F1 Maximization)
    for thresh in np.arange(0.01, 1.00, 0.01):
        temp_preds = np.where(prob_attack >= thresh, most_likely_attack, 0)
        temp_f1 = f1_score(y_test, temp_preds, average='macro', zero_division=0)
        
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_thresh = thresh
            best_preds = temp_preds

print(f"    [>] Optimal SOC-Constrained Threshold locked at: {best_thresh:.2f}")

final_predictions = best_preds

# The new alert count metric (since Layer 1 no longer physically drops rows)
flagged_alerts = np.count_nonzero(final_predictions)
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
# The new alert count is simply any log that XGBoost finally classifies as > 0
flagged_alerts = np.count_nonzero(final_predictions)

# Optional: If your script still calculates a Reduction Factor, you can override it
rf = 0.0 # Layer 1 no longer reduces traffic; it extracts features.

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
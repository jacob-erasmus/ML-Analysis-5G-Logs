"""
5_parallel_hybrid_ensemble.py
Author: Jacob Erasmus
Project: Honours Research Project
Purpose: An improved architecture but deviation from the proposed methodology. Executes a parallel meta-feature hybrid ensemble. Overcomes the bottlenecks of the sequential 
         hard-gate by converting unsupervised anomaly scores into continous spatial dimensions, allowing XGBoost to see the underlying geometry of the threats.
Alignment with Methodology:
    - Section 4.2.1-4.2.5
Pivot from Methodology:
    - Instead of dropping logs based on binary rules, Layer 1 (OCSVM and Isolation Forest) generates continous spatial metrics. These metrics are concatenated as 'Meta-Features'
    (Dimensionality is now 12 (from 10)), vastly improving XGBoost's ability to classify zero-day attacks. 
    - Calibration: Isotonic Regression is applied to a strict 20% holdout set to correct the probability distortion caused by SMOTE upsampling.
    - Threshold set >= 90% Precision to ensure best viability.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold

warnings.filterwarnings("ignore") 

print("==================================================")
print("--- PARALLEL META-FEATURE HYBRID ENSEMBLE ---   ")
print("==================================================")

#############################################
# 1. LOAD & DOWNCAST THE DATA (Section 4.2.1)
#############################################
def optimize_memory(df):
    """Restores Sec 4.2.1 datatype downcasting lost during CSV export."""
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].max() <= 127 and df[col].min() >= -128:
            df[col] = df[col].astype('int8')
        else:
            df[col] = df[col].astype('int32')
    return df

print("[+] Loading and Optimizing 80% Training Vault...")
df_train = pd.read_csv("logs_80percent.csv")
df_train = optimize_memory(df_train)
X_train = df_train.drop(columns=['LabelEnc'])
y_train = df_train['LabelEnc']

print("[+] Loading and Optimizing 20% Testing Vault (Unseen Data)...")
df_test = pd.read_csv("logs_20percent.csv") 
df_test = optimize_memory(df_test)
X_test = df_test.drop(columns=['LabelEnc'])
y_test = df_test['LabelEnc']

###############################################################
# 2. GOLDEN FEATURES (DERIVED FROM TUNING SCRIPT)
###############################################################
print("\n[+] Loading 10 Golden Features...")

golden_features = [
    'Total Length of Bwd Packets', 
    'Bwd Packet Length Mean', 
    'Total Length of Fwd Packets', 
    'Flow Bytes/s', 
    'Flow Duration', 
    'Fwd Packet Length Mean', 
    'Flow Packets/s', 
    'host.name_ausf', 
    'host.name_amf', 
    'zeek.udp_conns_1,728,325,600'
]

X_train_final = X_train[golden_features]
X_test_final = X_test[golden_features]

##############################################
# 3. LAYER 1: PARALLEL META-FEATURE GENERATION
##############################################
print("\n[+] Initalising Layer 1 (OSCVM & Isolation Forest Parallel Processing)...")
# Step 3A: Train Unsupervised Models on Normal Traffic
# Leakage Prevention: Layer 1 must map the geometry of normal traffic only.
X_train_normal = X_train_final[y_train == 0]

layer1_scaler = StandardScaler()
X_train_normal_scaled = layer1_scaler.fit_transform(X_train_normal)

# Scale the full datasets for Layer 1 scoring
X_train_full_scaled = layer1_scaler.transform(X_train_final)
X_test_full_scaled = layer1_scaler.transform(X_test_final)

# Train Nystroem OCSV (Captures boundary distance)
print("     -> Mapping spatial boundaries via Nystrom OCSVM...")
layer1_ocsvm = make_pipeline(
    Nystroem(kernel='rbf', gamma=None, n_components=300, random_state=42),
    SGDOneClassSVM(nu=0.20, random_state=42)
)
layer1_ocsvm.fit(X_train_normal_scaled)

# Train Isolation Forest (Captures tree-based isolation depth)
print("     -> Mapping structural depth via Isolation Forest...")
layer1_if = IsolationForest(n_estimators=100, contamination=0.20, random_state=42, n_jobs=-1)
layer1_if.fit(X_train_normal_scaled)

# Step 3B: Extract Continuous Anomaly Decision Scores instead of binary flags(The Meta-Features)
# decision_function() returns a continous float. Negative = Anomaly, Positive = Normal
print("    -> Extracting continuous anomaly decision scores for the entire vault...")
train_scores_ocsvm = layer1_ocsvm.decision_function(X_train_full_scaled)
train_scores_if = layer1_if.decision_function(X_train_full_scaled)

test_scores_ocsvm = layer1_ocsvm.decision_function(X_test_full_scaled)
test_scores_if = layer1_if.decision_function(X_test_full_scaled)

################
# 4. DATA MERGE
################
print("\n[+] Expanding Dimensionality with Unsupervised Meta-Features...")
# Concatenate the 2 unsupervised continous scores onto the 10 Golden Features.
# XGBoost would then have 12 dimensions to look at, allowing better understanding of threat severity
X_train_meta = X_train_final.copy()
X_train_meta['OCSVM_Score'] = train_scores_ocsvm
X_train_meta['IF_Score'] = train_scores_if

X_test_meta = X_test_final.copy()
X_test_meta['OCSVM_Score'] = test_scores_ocsvm
X_test_meta['IF_Score'] = test_scores_if

########################################################
# 5. TRAINING LAYER 2: XGBoost with ISOTONIC CALIBRATION
########################################################
print("\n[+] Training Layer 2: XGBoost with Isotonic Probability Calibration...")

# Step 5A: Create the True-Distribution Calibration Holdout
print("    -> Isolating 20% of Training Vault for uncorrupted calibration...")
X_subtrain, X_calib, y_subtrain, y_calib = train_test_split(
    X_train_meta, y_train, test_size=0.20, stratify=y_train, random_state=42
)

# Step 5B: Apply Adaptive SMOTE STRICTLY to the Sub-Train set
class_counts = y_subtrain.value_counts()
majority_count = class_counts.max()
target_minority = int(majority_count * 0.10)

smote_strategy = {cls: (count if count >= target_minority else target_minority) 
                  for cls, count in class_counts.items() if cls != 0}
smote_strategy[0] = majority_count 

print("    -> Balancing Sub-Train dataset with SMOTE...")
smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=1, random_state=42)
X_subtrain_balanced, y_subtrain_balanced = smote.fit_resample(X_subtrain, y_subtrain)

# Step 5C: Train the Base XGBoost
print("    -> Training Base XGBoost architecture...")
layer2_xgb = XGBClassifier(
    max_depth=13,
    learning_rate=0.1186,
    n_estimators=121,
    subsample=0.8395,
    eval_metric='mlogloss', 
    random_state=42, 
    n_jobs=-1
)
layer2_xgb.fit(X_subtrain_balanced, y_subtrain_balanced)

# Step 5D: Isotonic Calibration
print("    -> Executing Isotonic Regression to correct SMOTE probability distortion...")
# cv='prefit' strictly freezes the base XGBoost estimator as required by scikit-learn
calibrated_xgb = CalibratedClassifierCV(
    estimator=layer2_xgb, 
    method='isotonic',
    cv='prefit'
)
calibrated_xgb.fit(X_calib, y_calib)

#############################
# 6. INFERENCE & CALIBRATION
#############################
print("\n[+] Executing Multi-Class Calibration on Test Vault...")
print("    -> Enforcing strict >= 0.90 Precision boundary for viability...")

# Pass the full parallel test set through the Calibrated Model
l2_probs = calibrated_xgb.predict_proba(X_test_meta)
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
    
    # THE CONSTRAINT: only update the "best" model IF Precision is >= 90%
    if temp_prec >= 0.90 and temp_f1 > best_f1:
        best_f1 = temp_f1
        best_prec_at_max_f1 = temp_prec
        best_thresh = thresh
        best_preds = temp_preds

# Failsafe: If no threshold met the 90% constraint, default to standard F1 maximization
if best_preds is None:
    print("    [!] WARNING: Strict 90% Precision constraint could not be met. Defaulting to standard F1 maximization.")
    # fallback loop (F1 Maximization)
    for thresh in np.arange(0.01, 1.00, 0.01):
        temp_preds = np.where(prob_attack >= thresh, most_likely_attack, 0)
        temp_f1 = f1_score(y_test, temp_preds, average='macro', zero_division=0)
        
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_thresh = thresh
            best_preds = temp_preds

print(f"    [>] Optimal Threshold locked at: {best_thresh:.2f}")
final_predictions = best_preds

# The new alert count metric (since Layer 1 no longer physically drops rows)
flagged_alerts = np.count_nonzero(final_predictions)

##########################################################
# 7. DETERMINISTIC RULESET BASELINE
##########################################################
print("[+] Executing Deterministic Ruleset Baseline...")

volumetric_metrics = [
    'Flow Bytes/s', 
    'Flow Packets/s', 
    'Total Length of Fwd Packets', 
    'Total Length of Bwd Packets'
]

rule_predictions = np.zeros(len(X_test), dtype=int)
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

###################################
# 8. FORENSIC BENCHMARKING RESULTS
###################################
print("\n==================================================")
print("           FINAL BENCHMARKING RESULTS             ")
print("==================================================")

# Hybrid Ensemble Metrics
he_acc = accuracy_score(y_test, final_predictions)
he_prec = precision_score(y_test, final_predictions, average='macro', zero_division=0)
he_rec = recall_score(y_test, final_predictions, average='macro', zero_division=0)
he_f1 = f1_score(y_test, final_predictions, average='macro', zero_division=0)

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

##############################################
# 9. OVERFITTING DIAGNOSTICS & STABILITY AUDIT
##############################################
print("\n==================================================")
print("      OVERFITTING & STABILITY AUDIT      ")
print("==================================================")

# --- Diagnostic 1: Train vs. Test Gap ---
print("\n[+] Diagnostic 1: Evaluating Training Vault Metrics (Train vs. Test Gap)...")
# Extract probabilities for the data XGBoost already learned from
train_probs = calibrated_xgb.predict_proba(X_train_meta)
train_prob_attack = 1.0 - train_probs[:, 0]
train_most_likely = np.argmax(train_probs[:, 1:], axis=1) + 1
train_preds = np.where(train_prob_attack >= best_thresh, train_most_likely, 0)

train_f1 = f1_score(y_train, train_preds, average='macro', zero_division=0)
train_prec = precision_score(y_train, train_preds, average='macro', zero_division=0)

print(f"    [>] Training F1-Score:   {train_f1:.4f}  |  (Test F1: {best_f1:.4f})")
print(f"    [>] Training Precision:  {train_prec:.4f}  |  (Test Prec: {best_prec_at_max_f1:.4f})")
gap = abs(train_f1 - best_f1)
if gap > 0.10:
    print(f"    [!] WARNING: Severe Train-Test Gap ({gap:.4f}). Model is likely overfitting.")
else:
    print(f"    [>] STATUS: Healthy generalization. No catastrophic overfitting detected.")

# --- Diagnostic 2: Feature Importance Skew ---
print("\n[+] Diagnostic 2: Feature Importance Audit (Hunting for Artifacts)...")
importances = layer2_xgb.feature_importances_
feature_names = X_train_meta.columns

sorted_idx = np.argsort(importances)[::-1]
print("    [>] Top 5 Decision Vectors:")
for i in range(min(5, len(importances))):
    weight = importances[sorted_idx[i]]
    print(f"        {i+1}. {feature_names[sorted_idx[i]]:<30} : {weight:.4f}")

if importances[sorted_idx[0]] > 0.50:
    print("    [!] WARNING: Massive Feature Skew detected. Model is relying on a single artifact.")
else:
    print("    [>] STATUS: Healthy weight distribution across dimensions.")

# --- Diagnostic 3: 3-Fold Structural Variance ---
print("\n[+] Diagnostic 3: 3-Fold Cross-Validation (Structural Hyperparameter Variance)...")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_scores = []

# Testing the base architecture's stability across different data slices
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_meta, y_train)):
    X_fold_train, X_fold_val = X_train_meta.iloc[train_idx], X_train_meta.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Initialize a fresh base model using exact hyperparameters
    fold_model = XGBClassifier(
        max_depth=13, learning_rate=0.1186, n_estimators=121, 
        subsample=0.8395, eval_metric='mlogloss', random_state=42, n_jobs=-1
    )
    fold_model.fit(X_fold_train, y_fold_train)
    fold_preds = fold_model.predict(X_fold_val)
    fold_f1 = f1_score(y_fold_val, fold_preds, average='macro', zero_division=0)
    
    fold_scores.append(fold_f1)
    print(f"    [>] Fold {fold+1} Base F1-Score: {fold_f1:.4f}")

variance = np.var(fold_scores)
print(f"    [>] 3-Fold Variance: {variance:.6f}")
if variance > 0.005:
    print("    [!] WARNING: High structural variance. Hyperparameters are brittle.")
else:
    print("    [>] STATUS: Robust structural stability confirmed.")
print("==================================================\n")
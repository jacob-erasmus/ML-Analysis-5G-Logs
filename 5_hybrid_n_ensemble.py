"""
5_hybrid_n_ensemble.py
Author: Jacob Erasmus
Project: Honours Research Project
Purpose: An improved architecture that uses a 
Alignment with Methodology:
    - Section 4.2.1-4.2.3
Pivot from Methodology:
    - 

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
from sklearn.frozen import FrozenEstimator
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore") 

print("==================================================")
print("--- PARALLEL META-FEATURE HYBRID ENSEMBLE v2---   ")
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
print("\n[+] Loading 10 Golden Features + 3 Class 4 Targets...")

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
    'zeek.udp_conns_1,728,325,600',
    # Additions: Class 4 Key Features from Script
    'fields.vnf_connection',
    'host.name_nrf',
    'fields.vnf_weird'
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

# -------------------------------------------------------------
# 5. SINGLE-STAGE ENGINE (PURE LOG-SMOOTHING)
# -------------------------------------------------------------
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from xgboost import XGBClassifier

print("\n[+] Re-initiating Champion Normalization Architecture...")

# 1. Strict Validation Split for Leakage-Free Threshold Discovery
X_subtrain, X_valid, y_subtrain, y_valid = train_test_split(
    X_train_meta, y_train, test_size=0.20, stratify=y_train, random_state=42
)

# 2. Pure Log-Smoothed Weights (Zero Manual Heuristics)
print("    -> Calculating pure logarithmic weights to anchor class boundaries...")
class_counts = y_subtrain.value_counts()
majority_count = class_counts.max()
smoothed_weights = {
    cls: np.log1p(majority_count / count) + 1 for cls, count in class_counts.items()
}
custom_weights = np.array([smoothed_weights[cls] for cls in y_subtrain])

print("    -> Training Baseline XGBoost Engine...")
layer2_xgb = XGBClassifier(
    max_depth=13,
    learning_rate=0.1186,
    n_estimators=121,
    subsample=0.8395,
    max_delta_step=5,       
    min_child_weight=0.001, 
    eval_metric='mlogloss', 
    random_state=42, 
    n_jobs=-1
)
layer2_xgb.fit(X_subtrain, y_subtrain, sample_weight=custom_weights)
# -------------------------------------------------------------
# 6. AUTONOMOUS RECALL-CONSTRAINED GRID SEARCH
# -------------------------------------------------------------
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np

print("\n[+] Initiating Recall-Constrained Hyperparameter Grid Search...")
print("    -> Objective: Maximize Macro Recall while strictly bounding Precision >= 81%...")

y_valid_probs = layer2_xgb.predict_proba(X_valid)

# 1. Establish the pure geometric boundaries for Continuous Beta
attack_volumes = [(y_valid == cls).sum() for cls in range(1, 15)]
valid_volumes = [v for v in attack_volumes if v > 0]
V_max = max(valid_volumes)
V_min = min(valid_volumes)

severity_indices = {}
for cls in range(1, 15):
    vol = (y_valid == cls).sum()
    if vol > 0:
        severity_indices[cls] = (np.log(V_max) - np.log(vol)) / (np.log(V_max) - np.log(V_min))
    else:
        severity_indices[cls] = 1.0

# Micro-Resolution Array captures sub-1% probabilities
fine_thresholds = np.concatenate([
    np.arange(0.001, 0.01, 0.001), 
    np.arange(0.01, 1.00, 0.01)
])

# The Search Space for the Severity Scalar
severity_scalars = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0]

best_global_scalar = 0.0
best_valid_macro_recall = 0.0
final_optimal_thresholds = {}

for scalar in severity_scalars:
    temp_thresholds = {}
    
    # A. Calculate optimal thresholds for this specific scalar
    for cls in range(1, 15):
        y_valid_binary = (y_valid == cls).astype(int)
        if y_valid_binary.sum() == 0:
            temp_thresholds[cls] = 0.50
            continue
            
        dynamic_beta = 1.0 + (severity_indices[cls] * scalar)
        best_metric = 0
        best_thresh = 0.5 
        
        for thresh in fine_thresholds:
            temp_preds = (y_valid_probs[:, cls] >= thresh).astype(int)
            temp_score = fbeta_score(y_valid_binary, temp_preds, beta=dynamic_beta, zero_division=0)
            
            if temp_score > best_metric:
                best_metric = temp_score
                best_thresh = thresh
        temp_thresholds[cls] = best_thresh

    # B. Simulate Pure Normalized Inference on Validation Holdout
    base_ratios_valid = np.zeros_like(y_valid_probs)
    for cls in range(1, 15):
        base_ratios_valid[:, cls] = y_valid_probs[:, cls] / temp_thresholds[cls]
        
    best_attack_valid = np.argmax(base_ratios_valid[:, 1:], axis=1) + 1
    row_idx_valid = np.arange(len(y_valid))
    winning_ratios_valid = base_ratios_valid[row_idx_valid, best_attack_valid]
    
    valid_predictions = np.where(winning_ratios_valid >= 1.0, best_attack_valid, 0)
    
    # C. Evaluate against the Thesis Mandate (Recall > 80, Precision > 81)
    temp_macro_prec = precision_score(y_valid, valid_predictions, average='macro', zero_division=0)
    temp_macro_rec = recall_score(y_valid, valid_predictions, average='macro', zero_division=0)
    
    print(f"    -> Tested Scalar: {scalar:.2f} | Valid Precision: {temp_macro_prec:.4f} | Valid Recall: {temp_macro_rec:.4f}")
    
    # Strictly enforce the Precision boundary while hunting for maximum Recall
    if temp_macro_prec >= 0.8100 and temp_macro_rec > best_valid_macro_recall:
        best_valid_macro_recall = temp_macro_rec
        best_global_scalar = scalar
        final_optimal_thresholds = temp_thresholds

# Failsafe: If no scalar maintained 81% precision, default to scalar 0.0 (Pure F1)
if best_valid_macro_recall == 0.0:
    print("    [!] Warning: Precision constraint could not be met. Defaulting to Scalar 0.0.")
    best_global_scalar = 0.0
    # (Re-run scalar 0.0 logic to lock thresholds...)
    for cls in range(1, 15):
        y_valid_binary = (y_valid == cls).astype(int)
        best_metric, best_thresh = 0, 0.5
        if y_valid_binary.sum() > 0:
            for thresh in fine_thresholds:
                temp_preds = (y_valid_probs[:, cls] >= thresh).astype(int)
                temp_score = f1_score(y_valid_binary, temp_preds, zero_division=0)
                if temp_score > best_metric:
                    best_metric = temp_score
                    best_thresh = thresh
        final_optimal_thresholds[cls] = best_thresh

print(f"\n    [>] Optimal Severity Scalar mathematically locked at: {best_global_scalar:.2f}")

# -------------------------------------------------------------
# 7. NORMALIZED INFERENCE (UNSEEN TEST VAULT)
# -------------------------------------------------------------
print("\n[+] Executing Pure Normalized Inference on Unseen Test Vault...")
y_test_probs = layer2_xgb.predict_proba(X_test_meta)

base_ratios = np.zeros_like(y_test_probs)

for cls in range(1, 15):
    # Pure Normalization (No Tie-Breaker Multiplier)
    base_ratios[:, cls] = y_test_probs[:, cls] / final_optimal_thresholds[cls]

best_attack_classes = np.argmax(base_ratios[:, 1:], axis=1) + 1
row_indices = np.arange(len(y_test))
winning_base_ratios = base_ratios[row_indices, best_attack_classes]

final_predictions = np.where(winning_base_ratios >= 1.0, best_attack_classes, 0)

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
# with weighted metrics
he_w_prec = precision_score(y_test, final_predictions, average='weighted', zero_division=0)
he_w_rec = recall_score(y_test, final_predictions, average='weighted', zero_division=0)
he_w_f1 = f1_score(y_test, final_predictions, average='weighted', zero_division=0)
# with macro metrics
he_m_prec = precision_score(y_test, final_predictions, average='macro', zero_division=0)
he_m_rec = recall_score(y_test, final_predictions, average='macro', zero_division=0)
he_m_f1 = f1_score(y_test, final_predictions, average='macro', zero_division=0)

rf = 0.0 # Layer 1 no longer reduces traffic; it extracts features.

print("--- Hybrid w/Normalisation ML ENSEMBLE (WEIGHTED AVERAGE) ---")
print(f"Accuracy:         {he_acc:.4f}")
print(f"Precision:        {he_w_prec:.4f}")
print(f"Recall:           {he_w_rec:.4f}")
print(f"F1-Score: {he_w_f1:.4f}")
print(f"Reduction Factor: {rf:.4f} (Goal: ~0.99)")
print("--- Hybrid w/Normalisatio ML ENSEMBLE (MACRO AVERAGE) ---")
print(f"Accuracy:         {he_acc:.4f}")
print(f"Precision:        {he_m_prec:.4f}")
print(f"Recall:           {he_m_rec:.4f}")
print(f"F1-Score (Macro): {he_m_f1:.4f}")
print(f"Reduction Factor: {rf:.4f} (Goal: ~0.99)")

print("\n--- DETERMINISTIC BASELINE ---")
print(f"Accuracy:         {rule_acc:.4f}")
print(f"F1-Score (Macro): {rule_f1:.4f}")
print("==================================================")

##############################
# 9. EXPORT DEPLOYED FRAMEWORK 
##############################
import joblib
import os
export_dir = "alt_hn_deployed_framework"
print("\n[+] Exporting Hybrid w/Normalisatio ML Ensemble Model for Deployed Framework which will be used for Scalability Validation to '{export_dir}/' directory...")
joblib.dump(layer1_scaler, os.path.join(export_dir, 'hn_layer1_scaler.pkl'))
joblib.dump(layer1_ocsvm, os.path.join(export_dir, 'hn_layer1_ocsvm.pkl'))
joblib.dump(layer1_if, os.path.join(export_dir, 'hn_layer1_if.pkl'))
joblib.dump(layer2_xgb, os.path.join(export_dir, 'hn_layer2_xgb.pkl'))
joblib.dump(final_optimal_thresholds, os.path.join(export_dir, 'hn_thresholds.pkl'))
#with open(os.path.join(export_dir, 'p_threshold.txt'), 'w') as f:
  #  f.write(str(best_thresh))
print("     [>] Scaler, Layer 1, and Layer 2 succuessfully saved to disk.")

################################
# 10. Confusion Matrix generation
################################
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

v_directory = "visualisations"
os.makedirs(v_directory, exist_ok=True)

print(f"\f[+] Generating Confusion Matrix")
cfm = confusion_matrix(y_test, final_predictions)
plt.figure(figsize=(12, 10))
sns.heatmap(cfm, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black')
plt.title('Hybrid w/Normalisation Ensemble - Confusion Matrix', fontsize=14, pad=15)
plt.ylabel('True Network State', fontsize=12)
plt.xlabel('Predicted Network State', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(v_directory, 'hn_confusion_matrix.png'), dpi=300)
plt.close()
print("     [>] 'hn_confusion_matrix.png saved.")




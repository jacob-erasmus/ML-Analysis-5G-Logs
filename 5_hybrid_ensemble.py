"""
5_hybrid_ensemble.py
Author: Jacob Erasmus
Project: Honours Research Project
Purpose: An improved architecture but deviation from the proposed methodology. Executes a parallel meta-feature hybrid ensemble. Overcomes the bottlenecks of the sequential 
         hard-gate by converting unsupervised anomaly scores into continous spatial dimensions, allowing XGBoost to see the underlying geometry of the threats.
Alignment with Methodology:
    - Section 4.2.1-4.2.5
Pivot from Methodology:
    - Instead of dropping logs based on binary rules, Layer 1 (OCSVM and Isolation Forest) generates continous spatial metrics. These metrics are concatenated as 'Meta-Features'
    (Dimensionality is now 12 (from 10)), vastly improving XGBoost's ability to classify zero-day attacks. 
    - Abandoned synthethic oversampling
    - Added dynmaic Power-Law smoothing loop on 20% holdout to calculate optimal penalties without data leakage.

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
# 5. ALGORITHMIC WEIGHT DISCOVERY (POWER-LAW SMOOTHING)
# -------------------------------------------------------------
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n[+] Initiating Algorithmic Weight Discovery...")
print("    -> Sweeping Power-Law exponents to find the mathematical sweet spot...")

# 1. Create a strict validation split to discover the weights without leakage
X_subtrain, X_valid, y_subtrain, y_valid = train_test_split(
    X_train_meta, y_train, test_size=0.20, stratify=y_train, random_state=42
)

class_counts = y_subtrain.value_counts()
total_logs = len(y_subtrain)
k = len(class_counts)

best_p = 0.0
best_macro_f1 = 0.0

# 2. The Discovery Loop (Testing exponents from severe dampening to near-balanced)
for p in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
    # Calculate Power-Law weights for this exponent
    raw_weights = {
        cls: np.power(total_logs / (k * count), p) for cls, count in class_counts.items()
    }
    
    # Normalize weights so the mean penalty is 1.0 (protects learning rate stability)
    weight_sum = sum(raw_weights.values())
    normalized_weights = {c: (w / weight_sum) * k for c, w in raw_weights.items()}
    
    custom_weights = np.array([normalized_weights[cls] for cls in y_subtrain])
    
    # Train a lightweight, fast version of XGBoost to test the weights
    temp_xgb = XGBClassifier(
        max_depth=13, learning_rate=0.1186, n_estimators=60, # 60 trees for speed
        subsample=0.8395, max_delta_step=5, min_child_weight=0.001, 
        eval_metric='mlogloss', random_state=42, n_jobs=-1
    )
    temp_xgb.fit(X_subtrain, y_subtrain, sample_weight=custom_weights)
    
    temp_preds = temp_xgb.predict(X_valid)
    temp_f1 = f1_score(y_valid, temp_preds, average='macro', zero_division=0)
    
    print(f"    -> Tested p={p:.2f} | Validation Macro F1: {temp_f1:.4f}")
    
    if temp_f1 > best_macro_f1:
        best_macro_f1 = temp_f1
        best_p = p

print(f"\n    [>] Optimal Power-Law Exponent Discovered: p={best_p:.2f}")

# 3. Train the FINAL Champion Model on 100% of the Training Vault
print("\n[+] Training Final Champion Engine using Optimal Weight Distribution...")

full_class_counts = y_train.value_counts()
full_total_logs = len(y_train)

# Recalculate the winning formula on the entire dataset
final_raw_weights = {
    cls: np.power(full_total_logs / (k * count), best_p) for cls, count in full_class_counts.items()
}
final_weight_sum = sum(final_raw_weights.values())
final_normalized_weights = {c: (w / final_weight_sum) * k for c, w in final_raw_weights.items()}
final_custom_weights = np.array([final_normalized_weights[cls] for cls in y_train])

layer2_xgb = XGBClassifier(
    max_depth=13,
    learning_rate=0.1186,
    n_estimators=121, # Full tree depth
    subsample=0.8395,
    max_delta_step=5,
    min_child_weight=0.001, 
    # Removed gamma to allow the perfected weights to build micro-leaves naturally
    eval_metric='mlogloss', 
    random_state=42, 
    n_jobs=-1
)
layer2_xgb.fit(X_train_meta, y_train, sample_weight=final_custom_weights)
print("\n[+] Executing Final Inference on Unseen Test Set...")
# Native predict() completely eliminates the "Phantom Attack" false positive trap
final_predictions = layer2_xgb.predict(X_test_meta)
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

print("--- HYBRID ML ENSEMBLE (WEIGHTED AVERAGE) ---")
print(f"Accuracy:         {he_acc:.4f}")
print(f"Precision:        {he_w_prec:.4f}")
print(f"Recall:           {he_w_rec:.4f}")
print(f"F1-Score: {he_w_f1:.4f}")
print(f"Reduction Factor: {rf:.4f} (Goal: ~0.99)")
print("--- HYBRID ML ENSEMBLE (MACRO AVERAGE) ---")
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
export_dir = "alt_deployed_framework"
print("\n[+] Exporting Hybrid Ensemble Model for Deployed Framework which will be used for Scalability Validation to '{export_dir}/' directory...")
joblib.dump(layer1_scaler, os.path.join(export_dir, 'he_layer1_scaler.pkl'))
joblib.dump(layer1_ocsvm, os.path.join(export_dir, 'he_layer1_ocsvm.pkl'))
joblib.dump(layer1_if, os.path.join(export_dir, 'he_layer1_if.pkl'))
joblib.dump(layer2_xgb, os.path.join(export_dir, 'he_layer2_xgb_calibrated.pkl'))
#with open(os.path.join(export_dir, 'p_threshold.txt'), 'w') as f:
#    f.write(str(best_thresh))
print("     [>] Scaler, Layer 1, and Layer 2 and best threshold succuessfully saved to disk.")

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
plt.figure(figsize=(10, 8))
sns.heatmap(cfm, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black')
plt.title('Parallel Hybrid Ensemble - Confusion Matrix', fontsize=14, pad=15)
plt.ylabel('True Network State', fontsize=12)
plt.xlabel('Predicted Network State', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(v_directory, 'he_confusion_matrix.png'), dpi=300)
plt.close()
print("     [>] 'he_confusion_matrix.png saved.")




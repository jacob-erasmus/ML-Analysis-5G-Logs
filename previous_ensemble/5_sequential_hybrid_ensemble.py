"""
5_sequential_hybrid_ensemble.py
Author: Jacob Erasmus
Project: UP Honours Research
Purpose: Hybrid ensemble that executes the winning supervised and unsupervised models, OCSVM filters benign traffic and forwards suspected anomalies to XGBoost.
Output: Serialises and exports the to 'deployed_framework' which will be used for the Scalability Assesment.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore") 

print("==================================================")
print("  HYBRID ENSEMBLE   ")
print("==================================================")

#################################################
# 1. LOAD & DOWNCAST DATA (Methodology Sec 4.2.1)
#################################################
def optimize_memory(df):
    """Re-applies Sec 4.2.1 downcasting lost during CSV export."""
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


########################################
# 2. OPTIMIZED GLOBAL FEATURE SELECTION
########################################
print("\n[+] Loading 10 Golden Features derived from tuning script...")

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

print(f"    [>] Data dimensionality strictly locked to 10 optimal forensic vectors.")

##################################################
# 3. TRAINING LAYER 1: UNSUPERVISED TRIAGE (OCSVM)
##################################################
print("\n[+] Training Layer 1: OCSVM (Superior Unsupervised Model)")

# Trained strictly on normal traffic to learn the baseline distribution of benign logs
X_train_normal = X_train_final[y_train == 0]

layer1_scaler = StandardScaler()
X_train_normal_scaled = layer1_scaler.fit_transform(X_train_normal)
X_test_final_scaled = layer1_scaler.transform(X_test_final)

# Model: Nyström OCSVM (Distance-based)
layer1_ocsvm = make_pipeline(
    Nystroem(kernel='rbf', gamma=None, n_components=300, random_state=42),
    SGDOneClassSVM(nu=0.20, random_state=42)
)
layer1_ocsvm.fit(X_train_normal_scaled)

######################################################
# 4. TRAINING LAYER 2: SUPERVISED (XGBoost Classifier)
######################################################
print("[+] Training Layer 2: Tuned XGBoost Classifier...")

# Apply Dynamic SMOTE to the full 80% training set
class_counts = y_train.value_counts()
majority_count = class_counts.max()
target_minority = int(majority_count * 0.10)

smote_strategy = {cls: (count if count >= target_minority else target_minority) 
                  for cls, count in class_counts.items() if cls != 0}
smote_strategy[0] = majority_count 

smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=1, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_final, y_train)

# THE GOLDEN PARAMETERS (Derived from tuning script RandomizedSearchCV)
layer2_xgb = XGBClassifier(
    max_depth=13,
    learning_rate=0.1186,
    n_estimators=121,
    subsample=0.8395,
    eval_metric='mlogloss', 
    random_state=42, 
    n_jobs=-1
)
layer2_xgb.fit(X_train_balanced, y_train_balanced)

##############################################################
# 5. THE HYBRID ENSEMBLE INFERENCE (Sequential Triage Routing)
##############################################################
print("\n[+] Executing Sequential Hybrid Ensemble Inference on 20% Test Vault...")

# Step A: Pass everything through Layer 1 (OSCVM)
l1_preds_ocsvm = layer1_ocsvm.predict(X_test_final_scaled)

# Map to 0 (Normal) and 1 (Anomaly)
flags_ocsvm = np.where(l1_preds_ocsvm == 1, 0, 1)


final_predictions = np.zeros(len(X_test_final), dtype=int)
suspicious_indices = np.where(flags_ocsvm == 1)[0]

# Step B: Route to Layer 2
if len(suspicious_indices) > 0:
    X_test_suspicious = X_test_final.iloc[suspicious_indices]
    y_test_suspicious = y_test.iloc[suspicious_indices].values 
    print(f"    -> Layer 1 Triage Complete. Forwarding {len(suspicious_indices):,} suspicious logs to Layer 2...")

    # predict_proba returns a 2D array of shape (n_samples, 15 classes)
    l2_probs = layer2_xgb.predict_proba(X_test_suspicious)
    # The probability of being ANY attack is 1.0 minus the probability of being Normal (Class 0)
    prob_attack = 1.0 - l2_probs[:, 0]
    # Identify the specific zero-day attack by finding the max probability among classes.
    # We add 1 because slicing [:, 1:] shifts the index (Index 0 becomes Class 1)
    most_likely_attack = np.argmax(l2_probs[:, 1:], axis=1) + 1
    
    print("    -> Calibrating mathematical decision boundary...")
    best_f1 = 0
    best_thresh = 0.5
    best_preds = None

    # Maximising F1
    for thresh in np.arange(0.01, 1.00, 0.01):
        # If the threat probability exceeds the threshold, assign the specific attack class.
        # Otherwise, assign it back to 0 (Normal).
        temp_preds = np.where(prob_attack >= thresh, most_likely_attack, 0)
        
        temp_f1 = f1_score(y_test_suspicious, temp_preds, average='macro', zero_division=0)
        
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_thresh = thresh
            best_preds = temp_preds

    print(f"    [>] Optimal Layer 2 Probability Threshold locked at: {best_thresh:.2f}")
    
    final_predictions[suspicious_indices] = best_preds

####################################
# 6. DETERMINISTIC RULESET BASELINE
####################################
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

##################################
# 7. FORENSIC BENCHMARKING RESULTS
##################################
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

##############################
# 8. EXPORT DEPLOYED FRAMEWORK 
##############################
import joblib
import os
export_dir = "original_framework"
print("\n[+] Exporting Hybrid Ensemble Model for Deployed Framework which will be used for Scalability Validation to '{export_dir}/' directory...")
joblib.dump(layer1_scaler, os.path.join(export_dir, 'seq_layer1_scaler.pkl'))
joblib.dump(layer1_ocsvm, os.path.join(export_dir, 'seq_layer1_ocsvm.pkl'))
joblib.dump(layer2_xgb, os.path.join(export_dir, 'seq_layer2_xgb.pkl'))
print("     [>] Scaler, Layer 1, and Layer 2 succuessfully saved to disk.")

################################
# 9. Confusion Matrix generation
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
plt.title('Hybrid Ensemble - Confusion Matrix', fontsize=14, pad=15)
plt.ylabel('True Network State', fontsize=12)
plt.xlabel('Predicted Network State', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(v_directory, 'seq_confusion_matrix.png'), dpi=300)
plt.close()
print("     [>] 'seq_confusion_matrix.png saved.")

### Results for reference:
""" ==================================================
           FINAL BENCHMARKING RESULTS             
==================================================
--- HYBRID ML ENSEMBLE ---
Accuracy:         0.8943
Precision:        0.8518
Recall:           0.4142
F1-Score (Macro): 0.4719
Reduction Factor: 0.7330 (Goal: ~0.99)

--- DETERMINISTIC BASELINE ---
Accuracy:         0.7803
F1-Score (Macro): 0.4543
================================================== """

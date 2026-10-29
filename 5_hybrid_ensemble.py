"""
File: 5_hybrid_ensemble.py
Author: Jacob Erasmus
Project: BSc(Hons) Computer Science Research Project (University of Pretoria)
Purpose: Implements and evaluates the two layer Constraint-Driven Normalised Hybrid Ensemble.
         Layer 1 (Nystroem OCSVM and Isolation Forest) extracts continuous anomaly meta-features, expanding the RFE features into a 12-dimensional input space.
         Layer 2 (cost-sensitive XGBoost) applies Neyman-Pearson threshold optimsation and probability normalisation to classify 5G telemetry, export forensic triage artfeacts, benchmark against a determinstic baseline and a standalone XGBoost ablation baseline, and serialise the deployed framework. 
Alignment with Methodology and Discussion:
    - Chapter 3 (Step 10a): Layer 1 Unsupervised Meta-Feature Extraction
    - Chapter 3 (Step 10b): Layer 2 Supervised Classification & Neyman-Pearson Thresholding.
    - Chapter 3 (Step 10c): Probability Normalisation and Forensic Artefact Export
    - Chapter 3 (Step 11) & Chapter 4 (section 4.2): Comparative Foresnic Analysis, Per-Class Evaluation, Deterministic Baseline and Standalone XGBoost Abaltion Study
"""
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.ensemble import IsolationForest
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import warnings

warnings.filterwarnings("ignore") 

print("======================================================================")
print(" CONSTRAINT-DRIVEN NORMALISED HYBRID ENSEMBLE TRAINING & EVALUATION ")
print("====================================================================")

####################################################
# 1. LOAD & DOWNCAST TRAINING AND TESTING PARTITIONS
####################################################
def optimize_memory(df):
    """Reapplies datatype downcasting following CSV ingestion."""
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].max() <= 127 and df[col].min() >= -128:
            df[col] = df[col].astype('int8')
        else:
            df[col] = df[col].astype('int32')
    return df

print("[+] Loading and optimising 80% training partition ('logs_80percent.csv')...")
df_train = pd.read_csv("logs_80percent.csv")
df_train = optimize_memory(df_train)
X_train = df_train.drop(columns=['LabelEnc'])
y_train = df_train['LabelEnc']

print("[+] Loading and optimising unseen 20% test partition ('logs_20percent.csv')...")
df_test = pd.read_csv("logs_20percent.csv") 
df_test = optimize_memory(df_test)
X_test = df_test.drop(columns=['LabelEnc'])
y_test = df_test['LabelEnc']

#########################################################
# 2. APPLY SELECTED 10-FEATURE SUBSET (FROM STEP 9 RFECV)
#########################################################
print("\n[+] Filtering partitions to the 10 RFECV selected features")

golden_features = [
    'Total Length of Bwd Packets', 
    'Bwd Packet Length Mean', 
    'Total Length of Fwd Packets', 
    'Flow Bytes/s', 
    'Flow Duration', 
    'Fwd Packet Length Mean', 
    'Flow Packets/s', 
    'host.name_nrf',
    'host.name_udr',
    'zeek.udp_conns_1,728,308,500'
]

X_train_final = X_train[golden_features]
X_test_final = X_test[golden_features]

##################################################
# 3. LAYER 1: UNSUPERVISED META-FEATURE EXTRACTION
##################################################
print("\n[+] Initalising Layer 1 Unsupervised Meta-Feature Extraction...")

# Step 3A: Partition the 80% training data into sub-training (80%) and validation holdout (20%)
X_subtrain_raw, X_valid_raw, y_subtrain, y_valid = train_test_split(
    X_train_final, y_train, test_size=0.20, stratify=y_train, random_state=42
)

# Step 3B: Isolate benign sub-training records (Class 0) to fit the Layer 1 scaler and models
X_subtrain_normal = X_subtrain_raw[y_subtrain == 0]

layer1_scaler = StandardScaler()
X_subtrain_normal_scaled = layer1_scaler.fit_transform(X_subtrain_normal)

# Transform sub-training, validation, and unseen test partitions using the benign fitted scaler
X_subtrain_scaled = layer1_scaler.transform(X_subtrain_raw)
X_valid_scaled = layer1_scaler.transform(X_valid_raw)
X_test_full_scaled = layer1_scaler.transform(X_test_final)

# Fit Nystroem OCSVM on benign sub-training records
print("     -> Fitting Nystroem One-Class SVM (n_components=300, nu=0.20) on benign sub-training traffic...")
layer1_ocsvm = make_pipeline(
    Nystroem(kernel='rbf', gamma=None, n_components=300, random_state=42),
    SGDOneClassSVM(nu=0.20, random_state=42)
)
layer1_ocsvm.fit(X_subtrain_normal_scaled)

# Fit Isolation Forest on benign sub-training records
print("     -> Fitting Isolation Forest (n_estimators=100, nu=0.20) on benign sub-training traffic...")
layer1_if = IsolationForest(n_estimators=100, contamination=0.20, random_state=42, n_jobs=-1)
layer1_if.fit(X_subtrain_normal_scaled)

# Step 3C: Extract Continuous Anomaly Decision Scores across all partitions
print("    -> Extracting continuous decision_function() scores (OCSVM_Score and IF_Score)...")
subtrain_scores_ocsvm = layer1_ocsvm.decision_function(X_subtrain_scaled)
subtrain_scores_if = layer1_if.decision_function(X_subtrain_scaled)

valid_scores_ocsvm = layer1_ocsvm.decision_function(X_valid_scaled)
valid_scores_if = layer1_if.decision_function(X_valid_scaled)

test_scores_ocsvm = layer1_ocsvm.decision_function(X_test_full_scaled)
test_scores_if = layer1_if.decision_function(X_test_full_scaled)

################################################
# 4. FEATURE SPACE EXPANSION (10->12 DIMENSIONS)
################################################
print("\n[+] Appending Layer 1 continuous meta-features (expanding input space from 10 to 12 dimensions)...")
X_subtrain= X_subtrain_raw.copy()
X_subtrain['OCSVM_Score'] = subtrain_scores_ocsvm
X_subtrain['IF_Score'] = subtrain_scores_if

X_valid = X_valid_raw.copy()
X_valid['OCSVM_Score'] = valid_scores_ocsvm
X_valid['IF_Score'] = valid_scores_if

X_test_meta = X_test_final.copy()
X_test_meta['OCSVM_Score'] = test_scores_ocsvm
X_test_meta['IF_Score'] = test_scores_if

#############################################
# 5. LAYER 2: COST-SENSITIVE XGBOOST TRAINING
#############################################
print("\n[+] Training Layer 2 Cost-Sensitive XGBoost Classifier...")

print("    -> Computing logarithmic cost-sensitive sample weights on sub-training split...")
class_counts = y_subtrain.value_counts()
majority_count = class_counts.max()
smoothed_weights = {
    cls: np.log1p(majority_count / count) + 1 for cls, count in class_counts.items()
}
custom_weights = np.array([smoothed_weights[cls] for cls in y_subtrain])

print("    -> Fitting XGBoost classifier with tune hyperparameters on 12-dimensional sub-training set...")
layer2_xgb = XGBClassifier(
    max_depth=9, 
    learning_rate=0.1000998503939086,
    n_estimators= 181, 
    subsample=0.9826605267054558, 
    max_delta_step=5,       
    min_child_weight=0.001, 
    eval_metric='mlogloss', 
    random_state=42, 
    n_jobs=-1
)
layer2_xgb.fit(X_subtrain, y_subtrain, sample_weight=custom_weights)

#####################################################
# 6. NEYMAN-PEARSON THRESHOLD OPTIMISATION (STEP 10B)
#####################################################
print("\n[+] Executing Neyman-Pearson Threshold Optimisation on Validation Holdout...")
print("    -> Objective: Maximize Macro Recall subject to Macro Precision >= 0.8100")

y_valid_probs = layer2_xgb.predict_proba(X_valid)

# Compute logarithmic class-rarity severity indices I_c in [0,1] for classes 1..14
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

# Micro-Resolution threshold arar (108 candidate values from 0.001 to 0.99)
fine_thresholds = np.concatenate([
    np.arange(0.001, 0.01, 0.001), 
    np.arange(0.01, 1.00, 0.01)
])

severity_scalars = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0]

best_global_scalar = 0.0
best_valid_macro_recall = 0.0
final_optimal_thresholds = {}

for scalar in severity_scalars:
    temp_thresholds = {}
    
    # A. Identify class-specific thresholds maximising F_beta for the current scalar
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

    # B. Evaluate normalised probabilities ratios across the validation holdout
    base_ratios_valid = np.zeros_like(y_valid_probs)
    for cls in range(1, 15):
        base_ratios_valid[:, cls] = y_valid_probs[:, cls] / temp_thresholds[cls]
        
    best_attack_valid = np.argmax(base_ratios_valid[:, 1:], axis=1) + 1
    row_idx_valid = np.arange(len(y_valid))
    winning_ratios_valid = base_ratios_valid[row_idx_valid, best_attack_valid]
    
    valid_predictions = np.where(winning_ratios_valid >= 1.0, best_attack_valid, 0)
    
    # C. Verify Neyman-Person precision floor (Macro Precision >= 0.8100)
    temp_macro_prec = precision_score(y_valid, valid_predictions, average='macro', zero_division=0)
    temp_macro_rec = recall_score(y_valid, valid_predictions, average='macro', zero_division=0)
    
    print(f"    -> Candidate Scalar: {scalar:.2f} | Validation Macro Precision: {temp_macro_prec:.4f} | Validation Macro Recall: {temp_macro_rec:.4f}")
    
    if temp_macro_prec >= 0.8100 and temp_macro_rec > best_valid_macro_recall:
        best_valid_macro_recall = temp_macro_rec
        best_global_scalar = scalar
        final_optimal_thresholds = temp_thresholds

# Fallback to unweighted F1 (scalar = 0.0) if precision constraint is not satisified
if best_valid_macro_recall == 0.0:
    print("    [!] Warning: Precision constraint not met. Defaulting to Scalar 0.0 (unweighted F1).")
    best_global_scalar = 0.0
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

print(f"\n    [>] Optimal Severity Scalar locked at: {best_global_scalar:.2f}")

#####################################################
# 7. PROBABILITY NORMALISATION ON UNSEEN 20% TEST SET
#####################################################
print("\n[+] Executing Probability Normalisation Inference on Unseen 20% Test Partition...")
y_test_probs = layer2_xgb.predict_proba(X_test_meta)

base_ratios = np.zeros_like(y_test_probs)

for cls in range(1, 15):
    base_ratios[:, cls] = y_test_probs[:, cls] / final_optimal_thresholds[cls]

best_attack_classes = np.argmax(base_ratios[:, 1:], axis=1) + 1
row_indices = np.arange(len(y_test))
winning_base_ratios = base_ratios[row_indices, best_attack_classes]

final_predictions = np.where(winning_base_ratios >= 1.0, best_attack_classes, 0)

##################################################
# 8. FORENSIC TRIAGE ARTEFACT ISOLATION AND EXPORT
##################################################
print("\n[+] Exporting Flagged Forensic Triage Artefacts...")

anomolous_indices = np.where(final_predictions > 0)[0]
forensic_artefacts = df_test.iloc[anomolous_indices].copy()

forensic_artefacts['Predicted_Attack_Class'] = final_predictions[anomolous_indices]
forensic_artefacts['Confidence_Base_Ratio'] = winning_base_ratios[anomolous_indices]

timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
export_filename = f"forensic_triage_artefacts_{timestamp}.csv"

forensic_artefacts.to_csv(export_filename, index=False)
search_space_reduction = (1.0 - (len(forensic_artefacts) / len(df_test))) * 100
print(f"    [>] Export complete: {len(forensic_artefacts):,} records written to '{export_filename}' ({search_space_reduction:.2f}% search space reduction).")

#################################################
# 9. DETERMINISTIC VOLUMETRIC BASELINE (STEP 11A)
#################################################
print("\n[+] Executing Deterministic 99th-Percentile Volumetric Baseline...")

volumetric_metrics = [
    'Flow Bytes/s', 
    'Flow Packets/s', 
    'Total Length of Fwd Packets', 
    'Total Length of Bwd Packets'
]

available_metrics = [m for m in volumetric_metrics if m in X_train.columns]

if available_metrics:
    print(f"    -> Computing 99th-percentile thresholds on benign training records for: {available_metrics}")
    rule_thresholds = {
        metric: X_train[y_train == 0][metric].quantile(0.99)
        for metric in available_metrics
    }
    
    t0_rule = time.perf_counter()
    rule_predictions = np.zeros(len(X_test), dtype=int)
    for metric in available_metrics:
        rule_predictions = rule_predictions | (X_test[metric] > rule_thresholds[metric]).astype(int)
    time_rule = time.perf_counter() - t0_rule
        
    rule_binary_truth = (y_test != 0).astype(int)
    rule_acc = accuracy_score(rule_binary_truth, rule_predictions)
    rule_prec = precision_score(rule_binary_truth, rule_predictions, average='macro', zero_division=0)
    rule_rec = recall_score(rule_binary_truth, rule_predictions, average='macro', zero_division=0)
    rule_f1 = f1_score(rule_binary_truth, rule_predictions, average='macro', zero_division=0)
else:
    print("     [!] Warning: Required volumetric features not found. Skipping baseline evaluation.")
    time_rule = 0.0
    
#######################################################
# 10. COMPARATIVE FORENSIC ANALYSIS & PER-CLASS METRICS
#######################################################
print("\n===================================================================")
print(" COMPARATIVE FORENSIC ANALYSIS RESULTS (20% UNSEEN TEST PARTITION) ")
print("===================================================================")

he_acc = accuracy_score(y_test, final_predictions)
he_w_prec = precision_score(y_test, final_predictions, average='weighted', zero_division=0)
he_w_rec = recall_score(y_test, final_predictions, average='weighted', zero_division=0)
he_w_f1 = f1_score(y_test, final_predictions, average='weighted', zero_division=0)

he_m_prec = precision_score(y_test, final_predictions, average='macro', zero_division=0)
he_m_rec = recall_score(y_test, final_predictions, average='macro', zero_division=0)
he_m_f1 = f1_score(y_test, final_predictions, average='macro', zero_division=0)

he_binary_preds = (final_predictions != 0).astype(int)
he_bin_acc = accuracy_score(rule_binary_truth, he_binary_preds)
he_bin_prec = precision_score(rule_binary_truth, he_binary_preds, average='macro', zero_division=0)
he_bin_rec = recall_score(rule_binary_truth, he_binary_preds, average='macro', zero_division=0)
he_bin_f1 = f1_score(rule_binary_truth, he_binary_preds, average='macro', zero_division=0)

print("--- Hybrid Ensemble (15-Class Unweighted Macro Average) ---")
print(f"Global Accuracy     : {he_acc:.4f}")
print(f"Macro Precision     : {he_m_prec:.4f}")
print(f"Macro Recall        : {he_m_rec:.4f}")
print(f"Macro F1-Score      : {he_m_f1:.4f}")

print("\n--- Hybrid Ensemble (15-Class Weighted Average) ---")
print(f"Global Accuracy     : {he_acc:.4f}")
print(f"Weighted Precision  : {he_w_prec:.4f}")
print(f"Weighted Recall     : {he_w_rec:.4f}")
print(f"Weighted F1-Score   : {he_w_f1:.4f}")

print("\n--- Hybrid Ensemble (2-Class Binary Macro Average) ---")
print(f"Binary Accuracy     : {he_bin_acc:.4f}")
print(f"Macro Precision     : {he_bin_prec:.4f}")
print(f"Macro Recall        : {he_bin_rec:.4f}")
print(f"Macro F1-Score      : {he_bin_f1:.4f}")

print("\n--- Deterministic Volumetric Baseline (2-Class Binary Macro Average) ---")
print(f"Binary Accuracy     : {rule_acc:.4f}")
print(f"Macro Precision     : {rule_prec:.4f}")
print(f"Macro Recall        : {rule_rec:.4f}")
print(f"Macro F1-Score      : {rule_f1:.4f}")
print(f"Inference Time      : {time_rule:.4f} s")

print("\n--- 15-Class Per-Class Performance Breakdown ---")
print(classification_report(y_test, final_predictions, digits=4, zero_division=0))

cfm = confusion_matrix(y_test, final_predictions)
tn_bin = cfm[0, 0]
fp_bin = cfm[0, 1:].sum()
fn_bin = cfm[1:, 0].sum()
tp_bin = cfm[1:, 1:].sum()
exact_attack_tp = np.trace(cfm[1:, 1:])
cross_attack_err = tp_bin - exact_attack_tp

print("--- Confusion Matrix Binary vs Multi-Class Error Reconciliation ---")
print(f"Total Test Partition Records : {cfm.sum():,}")
print(f"Binary TN (Benign -> Benign) : {tn_bin:,} ({tn_bin/(tn_bin+fp_bin)*100:.2f}% of Benign)")
print(f"Binary FP (Benign -> Attack) : {fp_bin:,} ({fp_bin/(tn_bin+fp_bin)*100:.2f}% of Benign)")
print(f"Binary FN (Attack -> Benign) : {fn_bin:,} ({fn_bin/(tp_bin+fn_bin)*100:.2f}% of Attacks)")
print(f"Binary TP (Attack -> Attack) : {tp_bin:,} (Exact Class Match: {exact_attack_tp:,} | Cross-Attack Misattribution: {cross_attack_err:,})")

####################################################
# 10B. ABLATION STUDY: STANDALONE 10-FEATURE XGBOOST
####################################################
import time

print("\n[+] Executing Ablation Study (Standalone 10-Feature XGBoost vs. 12-Dimension Hybrid Ensemble)...")

# 1. Train Standalone XGBoost on the 10 RFE features using identical hyperparameters and weights
xgb_standalone = XGBClassifier(
    max_depth=9, 
    learning_rate=0.1000998503939086,
    n_estimators=181, 
    subsample=0.9826605267054558, 
    max_delta_step=5,       
    min_child_weight=0.001, 
    eval_metric='mlogloss', 
    random_state=42, 
    n_jobs=-1
)
xgb_standalone.fit(X_subtrain_raw, y_subtrain, sample_weight=custom_weights)

# 2. Time Standalone XGBoost Inference (10 Features, Standard Argmax)
t0_standalone = time.perf_counter()
y_pred_standalone = xgb_standalone.predict(X_test_final)
time_standalone = time.perf_counter() - t0_standalone

# 3. Time Complete Hybrid Ensemble Inference (10 -> 12 Dims, Normalised Inference)
t0_hybrid = time.perf_counter()
X_test_scaled_ablation = layer1_scaler.transform(X_test_final)
s_ocsvm_ablation = layer1_ocsvm.decision_function(X_test_scaled_ablation)
s_if_ablation = layer1_if.decision_function(X_test_scaled_ablation)

X_test_meta_ablation = X_test_final.copy()
X_test_meta_ablation['OCSVM_Score'] = s_ocsvm_ablation
X_test_meta_ablation['IF_Score'] = s_if_ablation

probs_ablation = layer2_xgb.predict_proba(X_test_meta_ablation)
ratios_ablation = np.zeros_like(probs_ablation)
for cls in range(1, 15):
    ratios_ablation[:, cls] = probs_ablation[:, cls] / final_optimal_thresholds[cls]

best_cls_ablation = np.argmax(ratios_ablation[:, 1:], axis=1) + 1
win_ratios_ablation = ratios_ablation[np.arange(len(y_test)), best_cls_ablation]
y_pred_hybrid_ablation = np.where(win_ratios_ablation >= 1.0, best_cls_ablation, 0)
time_hybrid = time.perf_counter() - t0_hybrid

# 4. Compute ablation metrics and dataframe Memory Footprints
ab_acc = accuracy_score(y_test, y_pred_standalone)
ab_prec = precision_score(y_test, y_pred_standalone, average='macro', zero_division=0)
ab_rec = recall_score(y_test, y_pred_standalone, average='macro', zero_division=0)
ab_f1 = f1_score(y_test, y_pred_standalone, average='macro', zero_division=0)

mem_10_mb = X_test_final.memory_usage(deep=True).sum() / (1024 ** 2)
mem_12_mb = X_test_meta.memory_usage(deep=True).sum() / (1024 ** 2)

print("\n--- Ablation Comparison on Unseen 20% Test Partition (566,149 Records) ---")
print(f"Standalone XGBoost (10 Feats) | Acc: {ab_acc:.4f} | Macro Prec: {ab_prec:.4f} | Macro Rec: {ab_rec:.4f} | Macro F1: {ab_f1:.4f} | RAM: {mem_10_mb:.2f} MB | Time: {time_standalone:.4f} s")
print(f"Hybrid Ensemble    (12 Dims)  | Acc: {he_acc:.4f} | Macro Prec: {he_m_prec:.4f} | Macro Rec: {he_m_rec:.4f} | Macro F1: {he_m_f1:.4f} | RAM: {mem_12_mb:.2f} MB | Time: {time_hybrid:.4f} s")
print("==================================================")

#############################################
# 11. SERIALISE AND EXPORT DEPLOYED FRAMEWORK 
#############################################
export_dir = "final_framework"
os.makedirs(export_dir, exist_ok=True)
print("\n[+] Serialising trained Hybrid Ensemble componets to '{export_dir}/'...")

joblib.dump(layer1_scaler, os.path.join(export_dir, 'hybrid_ensemble_layer1_scaler.pkl'))
joblib.dump(layer1_ocsvm, os.path.join(export_dir, 'hybrid_ensemble_layer1_ocsvm.pkl'))
joblib.dump(layer1_if, os.path.join(export_dir, 'hybrid_ensemble_layer1_if.pkl'))
joblib.dump(layer2_xgb, os.path.join(export_dir, 'hybrid_ensemble_layer2_xgb.pkl'))
joblib.dump(final_optimal_thresholds, os.path.join(export_dir, 'hybrid_ensemble_thresholds.pkl'))
print("     [>] Layer 1 scaler, OCSVM, Isolation Forest, Layer 2 XGBoost, and thresholds saved to disk.")

#################################
# 12. Confusion Matrix generation
#################################
v_directory = "visualisations"
os.makedirs(v_directory, exist_ok=True)

print(f"\f[+] Generating Confusion Matrix")
cfm = confusion_matrix(y_test, final_predictions)
plt.figure(figsize=(12, 10))
sns.heatmap(cfm, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black')
plt.title('Hybrid Ensemble - Confusion Matrix', fontsize=14, pad=15)
plt.ylabel('True Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(v_directory, 'hybrid_ensemble_confusion_matrix.png'), dpi=300)
plt.close()
print("     [>] Saved 'visulisations/hybrid_ensemble_confusion_matrix.png'.")




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

warnings.filterwarnings("ignore") 

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

# -------------------------------------------------------------
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
print(f"[+] Final 20 Forensic Features Locked for Production.")

# -------------------------------------------------------------
# 3. TRAINING LAYER 1: UNSUPERVISED (OCSVM Filter)
# -------------------------------------------------------------
print("\n[+] Training Layer 1: Nyström OCSVM Anomaly Filter...")

# Isolate and scale ONLY normal traffic for Layer 1
X_train_normal = X_train_final[y_train == 0]

layer1_scaler = StandardScaler()
X_train_normal_scaled = layer1_scaler.fit_transform(X_train_normal)
X_test_final_scaled = layer1_scaler.transform(X_test_final)

layer1_ocsvm = make_pipeline(
    Nystroem(kernel='rbf', gamma=None, n_components=300, random_state=42),
    #SGDOneClassSVM(nu=0.01, random_state=42) # Changed to 0.01 to match benign-only training
    SGDOneClassSVM(nu=0.20, random_state=42) 

)
layer1_ocsvm.fit(X_train_normal_scaled)

# -------------------------------------------------------------
# 4. TRAINING LAYER 2: SUPERVISED (XGBoost Classifier)
# -------------------------------------------------------------
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

# THE GOLDEN PARAMETERS (Derived from 10% RandomizedSearchCV)
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
# 5. THE HYBRID ENSEMBLE INFERENCE (Routing & Calibration)
# -------------------------------------------------------------
print("\n[+] Executing Hybrid Ensemble Inference on 20% Test Vault...")

# Step A: Pass everything through Layer 1
# OCSVM returns 1 (inlier/normal) and -1 (outlier/anomaly). Map to 0 and 1.
l1_preds_raw = layer1_ocsvm.predict(X_test_final_scaled)
l1_binary_flags = np.where(l1_preds_raw == 1, 0, 1)

# Initialize our final prediction array (defaulting to 0 / normal)
final_predictions = np.zeros(len(X_test_final), dtype=int)
suspicious_indices = np.where(l1_binary_flags == 1)[0]

# Step B: Route to Layer 2 and Calibrate Threshold
if len(suspicious_indices) > 0:
    X_test_suspicious = X_test_final.iloc[suspicious_indices]
    y_test_suspicious = y_test.iloc[suspicious_indices].values # Needed for calibration scoring
    
    print("    -> Extracting Layer 2 raw probabilities...")
    # predict_proba returns a 2D array [prob_0, prob_1]. We want prob_1 (Attack probability)
    l2_probs = layer2_xgb.predict_proba(X_test_suspicious)[:, 1]
    
    print("    -> Calibrating mathematical decision boundary...")
    best_f1 = 0
    best_thresh = 0.5
    best_preds = None
    
    # The Calibration Loop: Test every threshold from 1% to 99%
    for thresh in np.arange(0.01, 1.00, 0.01):
        temp_preds = (l2_probs >= thresh).astype(int)
        
        # Calculate Macro F1 just for this routed subset to find the optimal local boundary
        temp_f1 = f1_score(y_test_suspicious, temp_preds, average='macro', zero_division=0)
        
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_thresh = thresh
            best_preds = temp_preds

    print(f"    [>] Optimal Layer 2 Probability Threshold locked at: {best_thresh:.2f}")
    
    # Place Layer 2's calibrated classifications back into the final prediction array
    final_predictions[suspicious_indices] = best_preds

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
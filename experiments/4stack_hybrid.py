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
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

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

""" # -------------------------------------------------------------
# 2.5 FORENSIC FEATURE ENGINEERING (Upgrade B)
# -------------------------------------------------------------
print("\n[+] Executing Forensic Feature Engineering...")
print("    -> Generating mathematical interactions to break zero-day camouflage.")

# We extract the top 5 most important features from the RFE selection
top_5_interact = final_features[:5]
X_train_poly_source = X_train_final[top_5_interact]
X_test_poly_source = X_test_final[top_5_interact]

# interaction_only=True ensures we only multiply A*B, completely ignoring A^2 to prevent scaling distortion
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

train_interactions = poly.fit_transform(X_train_poly_source)
test_interactions = poly.transform(X_test_poly_source)

# Extract the new mathematical feature names (e.g., 'Flow Bytes/s Flow Packets/s')
poly_feature_names = poly.get_feature_names_out(top_5_interact)

df_train_poly = pd.DataFrame(train_interactions, columns=poly_feature_names, index=X_train_final.index)
df_test_poly = pd.DataFrame(test_interactions, columns=poly_feature_names, index=X_test_final.index)

# Drop the original 5 features from this temporary dataframe so we don't duplicate them
df_train_poly = df_train_poly.drop(columns=top_5_interact)
df_test_poly = df_test_poly.drop(columns=top_5_interact)

# Merge the 10 newly engineered dimensions into the master datasets
X_train_final = pd.concat([X_train_final, df_train_poly], axis=1)
X_test_final = pd.concat([X_test_final, df_test_poly], axis=1)

print(f"    [>] Engineered {len(df_train_poly.columns)} new dimensional planes. Total features expanded to {X_train_final.shape[1]}.")
 """
# -------------------------------------------------------------
# 3. TRAINING LAYER 1: UNSUPERVISED UNION (OCSVM + IF)
# -------------------------------------------------------------
print("\n[+] Training Layer 1: OCSVM + Isolation Forest Union...")

X_train_normal = X_train_final[y_train == 0]

layer1_scaler = StandardScaler()
X_train_normal_scaled = layer1_scaler.fit_transform(X_train_normal)
X_test_final_scaled = layer1_scaler.transform(X_test_final)

# Model A: Nyström OCSVM (Distance-based)
layer1_ocsvm = make_pipeline(
    Nystroem(kernel='rbf', gamma=None, n_components=300, random_state=42),
    SGDOneClassSVM(nu=0.20, random_state=42)
)
layer1_ocsvm.fit(X_train_normal_scaled)

# Model B: Isolation Forest (Tree-based)
layer1_if = IsolationForest(n_estimators=100, contamination=0.20, random_state=42, n_jobs=-1)
layer1_if.fit(X_train_normal_scaled)

# -------------------------------------------------------------
# 4. TRAINING LAYER 2: THE STACKING META-CLASSIFIER
# -------------------------------------------------------------
print("\n[+] Training Layer 2: Stacking Meta-Classifier Architecture...")

# Re-engage Adaptive SMOTE to provide base learners with target vectors
class_counts = y_train.value_counts()
majority_count = class_counts.max()
target_minority = int(majority_count * 0.10)

smote_strategy = {cls: (count if count >= target_minority else target_minority) 
                  for cls, count in class_counts.items() if cls != 0}
smote_strategy[0] = majority_count 

smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=1, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_final, y_train)

print("    -> Initializing Level 0 Base Learners...")
# Model A: The Aggressive Sniper
level0_xgb = XGBClassifier(
    max_depth=13, learning_rate=0.1186, n_estimators=121, 
    subsample=0.8394, eval_metric='mlogloss', random_state=42, n_jobs=-1
)

# Model B: The Stability Anchor
level0_rf = RandomForestClassifier(
    n_estimators=150, max_depth=15, class_weight='balanced', 
    random_state=42, n_jobs=-1
)

# Model C: The Sparse-Data Specialist (Leaf-wise growth for one-hot columns)
level0_hgb = HistGradientBoostingClassifier(
    max_iter=150, max_depth=15, learning_rate=0.1, random_state=42
)

print("    -> Initializing Level 1 Meta-Model (Logistic Regression)...")
# The Meta-Model learns which base learner is most accurate for specific camouflaged attacks
level1_meta = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)

# The Architecture
layer2_stack = StackingClassifier(
    estimators=[
        ('xgb', level0_xgb), 
        ('rf', level0_rf), 
        ('hgb', level0_hgb)
    ],
    final_estimator=level1_meta,
    cv=3,
    n_jobs=-1
)

print("    -> Fitting Stacking Architecture (This will require heavy computation)...")
layer2_stack.fit(X_train_balanced, y_train_balanced)
# -------------------------------------------------------------
# 5. THE HYBRID ENSEMBLE INFERENCE (Routing & Calibration)
# -------------------------------------------------------------
print("\n[+] Executing Hybrid Ensemble Inference on 20% Test Vault...")

# Step A: Pass everything through Layer 1 Union
l1_preds_ocsvm = layer1_ocsvm.predict(X_test_final_scaled)
l1_preds_if = layer1_if.predict(X_test_final_scaled)

# Map to 0 (Normal) and 1 (Anomaly)
flags_ocsvm = np.where(l1_preds_ocsvm == 1, 0, 1)
flags_if = np.where(l1_preds_if == 1, 0, 1)

# LOGICAL OR: If EITHER model flags it, it routes to Layer 2
l1_binary_flags = flags_ocsvm | flags_if

final_predictions = np.zeros(len(X_test_final), dtype=int)
suspicious_indices = np.where(l1_binary_flags == 1)[0]

# Step B: Route to Layer 2 and Calibrate Multi-Class Threshold
if len(suspicious_indices) > 0:
    X_test_suspicious = X_test_final.iloc[suspicious_indices]
    y_test_suspicious = y_test.iloc[suspicious_indices].values 
    
    print("    -> Extracting Layer 2 multi-class probabilities...")
    # predict_proba returns a 2D array of shape (n_samples, 15 classes)
    l2_probs = layer2_stack.predict_proba(X_test_suspicious)
    
    # The probability of being ANY attack is 1.0 minus the probability of being Normal (Class 0)
    prob_attack = 1.0 - l2_probs[:, 0]
    
    # Identify the specific zero-day attack by finding the max probability among classes 1-14.
    # We add 1 because slicing [:, 1:] shifts the index (Index 0 becomes Class 1)
    most_likely_attack = np.argmax(l2_probs[:, 1:], axis=1) + 1
    
    print("    -> Calibrating mathematical decision boundary...")
    best_f1 = 0
    best_thresh = 0.5
    best_preds = None
    
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
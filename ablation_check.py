import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

def optimize_memory(df):
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].max() <= 127 and df[col].min() >= -128:
            df[col] = df[col].astype('int8')
        else:
            df[col] = df[col].astype('int32')
    return df

print("[+] Loading Training and Testing Vaults for Ablation Study...")
df_train = optimize_memory(pd.read_csv("logs_80percent.csv"))
df_test = optimize_memory(pd.read_csv("logs_20percent.csv"))

golden_features = [
    'Total Length of Bwd Packets', 'Bwd Packet Length Mean',
    'Total Length of Fwd Packets', 'Flow Bytes/s', 'Flow Duration',
    'Fwd Packet Length Mean', 'Flow Packets/s', 'host.name_nrf',
    'host.name_udr', 'zeek.udp_conns_1,728,308,500'
]

X_train_10 = df_train[golden_features]
y_train = df_train['LabelEnc']
X_test_10 = df_test[golden_features]
y_test = df_test['LabelEnc']

# Identical subtrain split as 5_hybrid_ensemble.py
X_subtrain_10, X_valid_10, y_subtrain, y_valid = train_test_split(
    X_train_10, y_train, test_size=0.20, stratify=y_train, random_state=42
)

# Identical logarithmic weights
class_counts = y_subtrain.value_counts()
majority_count = class_counts.max()
smoothed_weights = {cls: np.log1p(majority_count / count) + 1 for cls, count in class_counts.items()}
custom_weights = np.array([smoothed_weights[cls] for cls in y_subtrain])

# Train Standalone 10-Feature XGBoost with identical tuned hyperparameters
print("[+] Training Standalone 10-Feature XGBoost...")
xgb_standalone = XGBClassifier(
    max_depth=9, learning_rate=0.1000998503939086, n_estimators=181,
    subsample=0.9826605267054558, max_delta_step=5, min_child_weight=0.001,
    eval_metric='mlogloss', random_state=42, n_jobs=-1
)
xgb_standalone.fit(X_subtrain_10, y_subtrain, sample_weight=custom_weights)

# Time Standalone 10-Feature XGBoost Inference on 20% Test Set
t0 = time.perf_counter()
y_pred_standalone = xgb_standalone.predict(X_test_10)
t_standalone = time.perf_counter() - t0

# Load Hybrid Models to time Hybrid Inference on the exact same 20% Test Set
export_dir = "final_framework"
l1_scaler = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_layer1_scaler.pkl'))
l1_ocsvm = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_layer1_ocsvm.pkl'))
l1_if = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_layer1_if.pkl'))
l2_xgb = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_layer2_xgb.pkl'))
opt_thresh = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_thresholds.pkl'))

t1 = time.perf_counter()
X_test_scaled = l1_scaler.transform(X_test_10)
s_ocsvm = l1_ocsvm.decision_function(X_test_scaled)
s_if = l1_if.decision_function(X_test_scaled)
X_test_12 = X_test_10.copy()
X_test_12['OCSVM_Score'] = s_ocsvm
X_test_12['IF_Score'] = s_if
probs = l2_xgb.predict_proba(X_test_12)
base_ratios = np.zeros_like(probs)
for cls in range(1, 15):
    base_ratios[:, cls] = probs[:, cls] / opt_thresh[cls]
best_cls = np.argmax(base_ratios[:, 1:], axis=1) + 1
win_ratios = base_ratios[np.arange(len(y_test)), best_cls]
y_pred_hybrid = np.where(win_ratios >= 1.0, best_cls, 0)
t_hybrid = time.perf_counter() - t1

mem_10 = X_test_10.memory_usage(deep=True).sum() / (1024**2)
mem_12 = X_test_12.memory_usage(deep=True).sum() / (1024**2)

print("\n=== ABLATION STUDY RESULTS (20% UNSEEN TEST SET) ===")
print(f"Standalone XGBoost (10 Feats, Standard Inference):")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_standalone):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_standalone, average='macro', zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_standalone, average='macro', zero_division=0):.4f}")
print(f"  Macro F1:  {f1_score(y_test, y_pred_standalone, average='macro', zero_division=0):.4f}")
print(f"  Memory:    {mem_10:.2f} MB | Inference Time: {t_standalone:.4f} s")

print(f"\nHybrid Ensemble (12 Dims, Normalised Inference):")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_hybrid):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_hybrid, average='macro', zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_hybrid, average='macro', zero_division=0):.4f}")
print(f"  Macro F1:  {f1_score(y_test, y_pred_hybrid, average='macro', zero_division=0):.4f}")
print(f"  Memory:    {mem_12:.2f} MB | Inference Time: {t_hybrid:.4f} s")
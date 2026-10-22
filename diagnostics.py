import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. Load full logs once to get the exact string mapping for LabelEnc (0..14)
df_label_map = pd.read_csv("logs.csv", usecols=['Label', 'LabelEnc']).drop_duplicates().sort_values('LabelEnc')
label_names = dict(zip(df_label_map['LabelEnc'], df_label_map['Label']))

# 2. Load 80% Train (for 99th percentile rule thresholds) and 20% Test
df_train = pd.read_csv("logs_80percent.csv")
df_test = pd.read_csv("logs_20percent.csv")
X_train, y_train = df_train.drop(columns=['LabelEnc']), df_train['LabelEnc']
X_test, y_test = df_test.drop(columns=['LabelEnc']), df_test['LabelEnc']

golden_features = [
    'Total Length of Bwd Packets', 'Bwd Packet Length Mean',
    'Total Length of Fwd Packets', 'Flow Bytes/s', 'Flow Duration',
    'Fwd Packet Length Mean', 'Flow Packets/s', 'host.name_nrf',
    'host.name_udr', 'zeek.udp_conns_1,728,308,500'
]
X_test_final = X_test[golden_features].astype('float32')

# 3. Load Saved Hybrid Ensemble & Run Inference
export_dir = "final_framework"
l1_scaler = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_layer1_scaler.pkl'))
l1_ocsvm = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_layer1_ocsvm.pkl'))
l1_if = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_layer1_if.pkl'))
l2_xgb = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_layer2_xgb.pkl'))
opt_thresh = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_thresholds.pkl'))

t0 = time.perf_counter()
X_test_scaled = l1_scaler.transform(X_test_final)
X_test_meta = X_test_final.copy()
X_test_meta['OCSVM_Score'] = l1_ocsvm.decision_function(X_test_scaled)
X_test_meta['IF_Score'] = l1_if.decision_function(X_test_scaled)
probs = l2_xgb.predict_proba(X_test_meta)
ratios = np.zeros_like(probs)
for cls in range(1, 15):
    ratios[:, cls] = probs[:, cls] / opt_thresh[cls]
best_cls = np.argmax(ratios[:, 1:], axis=1) + 1
win_ratios = ratios[np.arange(len(y_test)), best_cls]
final_predictions = np.where(win_ratios >= 1.0, best_cls, 0)
t_hybrid = time.perf_counter() - t0

# 4. Print 15-Class Per-Class Report
target_names = [f"Class {i} ({label_names[i]})" for i in range(15)]
print("\n=== 15-CLASS PER-CLASS METRICS ===")
print(classification_report(y_test, final_predictions, target_names=target_names, digits=4, zero_division=0))

# 5. Binary vs Multi-Class Error Breakdown
cfm = confusion_matrix(y_test, final_predictions)
tn_bin = cfm[0, 0]
fp_bin = cfm[0, 1:].sum()
fn_bin = cfm[1:, 0].sum()
tp_bin = cfm[1:, 1:].sum()
cross_attack_err = tp_bin - np.trace(cfm[1:, 1:])
print("=== CONFUSION MATRIX RECONCILIATION ===")
print(f"Total Test Records: {cfm.sum():,}")
print(f"Binary TN (Benign -> Benign): {tn_bin:,}")
print(f"Binary FP (Benign -> Any Attack): {fp_bin:,} ({fp_bin/(tn_bin+fp_bin)*100:.2f}% of Benign)")
print(f"Binary FN (Any Attack -> Benign): {fn_bin:,} ({fn_bin/(tp_bin+fn_bin)*100:.2f}% of Attacks)")
print(f"Binary TP (Any Attack -> Any Attack): {tp_bin:,} (Exact Class Match: {np.trace(cfm[1:, 1:]):,} | Cross-Attack Misclassification: {cross_attack_err:,})")

# 6. Binary Macro Comparison (Same Metric Definition as Deterministic Baseline)
y_test_bin = (y_test != 0).astype(int)
he_pred_bin = (final_predictions != 0).astype(int)
print("\n=== BINARY (2-CLASS MACRO) COMPARISON ===")
print(f"Hybrid Binary   | Acc: {accuracy_score(y_test_bin, he_pred_bin):.4f} | Macro Prec: {precision_score(y_test_bin, he_pred_bin, average='macro'):.4f} | Macro Rec: {recall_score(y_test_bin, he_pred_bin, average='macro'):.4f} | Macro F1: {f1_score(y_test_bin, he_pred_bin, average='macro'):.4f} | Time: {t_hybrid:.4f}s")

vol_metrics = ['Flow Bytes/s', 'Flow Packets/s', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets']
thresholds = {m: X_train[y_train == 0][m].quantile(0.99) for m in vol_metrics}
t1 = time.perf_counter()
rule_preds = np.zeros(len(X_test), dtype=int)
for m in vol_metrics:
    rule_preds = rule_preds | (X_test[m] > thresholds[m]).astype(int)
t_rule = time.perf_counter() - t1
print(f"Ruleset Binary  | Acc: {accuracy_score(y_test_bin, rule_preds):.4f} | Macro Prec: {precision_score(y_test_bin, rule_preds, average='macro'):.4f} | Macro Rec: {recall_score(y_test_bin, rule_preds, average='macro'):.4f} | Macro F1: {f1_score(y_test_bin, rule_preds, average='macro'):.4f} | Time: {t_rule:.4f}s")
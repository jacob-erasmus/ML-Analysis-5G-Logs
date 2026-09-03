import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score
from sklearn.frozen import FrozenEstimator
import warnings

warnings.filterwarnings('ignore')

print("\n==================================================")
print("     PHASE 9: BAYESIAN HYPERPARAMETER SWEEP       ")
print("==================================================")

# 1. LOAD YOUR DATA HERE (Only the 80% Training Vault is needed)
df_train = pd.read_csv("logs_80percent.csv")
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
X_train_raw = df_train[golden_features]
y_train = df_train['LabelEnc']

# For the sake of the script structure, assuming X_train_raw and y_train are loaded:

def objective(trial):
    # -------------------------------------------------------------
    # 1. SUGGEST HYPERPARAMETERS
    # -------------------------------------------------------------
    # Layer 1: Unsupervised Contamination Tuning
    if_contam = trial.suggest_float('if_contamination', 0.05, 0.30)
    ocsvm_nu = trial.suggest_float('ocsvm_nu', 0.05, 0.30)
    
    # Layer 2: XGBoost 12-D Alignment Tuning
    xgb_depth = trial.suggest_int('xgb_max_depth', 8, 16)
    xgb_lr = trial.suggest_float('xgb_learning_rate', 0.05, 0.20, log=True)
    xgb_estimators = trial.suggest_int('xgb_n_estimators', 80, 200)
    xgb_subsample = trial.suggest_float('xgb_subsample', 0.6, 1.0)
    
    # -------------------------------------------------------------
    # 2. STRICT INTERNAL DATA SPLITTING (No Test Set Leakage)
    # -------------------------------------------------------------
    # Split 1: Extract 20% for Optuna Validation Scoring
    X_temp, X_val, y_temp, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.20, stratify=y_train, random_state=42
    )
    # Split 2: Divide the remaining 80% into Sub-Train and Calibration
    X_sub, X_cal, y_sub, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    
    # -------------------------------------------------------------
    # 3. LAYER 1: META-FEATURE EXTRACTION
    # -------------------------------------------------------------
    X_normal = X_sub[y_sub == 0]
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    
    # Train Global Extractors using Optuna's suggestions
    ocsvm = make_pipeline(Nystroem(n_components=300, random_state=42), SGDOneClassSVM(nu=ocsvm_nu, random_state=42))
    ocsvm.fit(X_normal_scaled)
    
    iso_forest = IsolationForest(n_estimators=100, contamination=if_contam, random_state=42, n_jobs=-1)
    iso_forest.fit(X_normal_scaled)
    
    # Helper function to append features
    def append_meta_features(X_base):
        X_scaled = scaler.transform(X_base)
        X_meta = X_base.copy()
        X_meta['OCSVM_Score'] = ocsvm.decision_function(X_scaled)
        X_meta['IF_Score'] = iso_forest.decision_function(X_scaled)
        return X_meta
    
    X_sub_meta = append_meta_features(X_sub)
    X_cal_meta = append_meta_features(X_cal)
    X_val_meta = append_meta_features(X_val)
    
    # -------------------------------------------------------------
    # 4. LAYER 2: SMOTE & XGBOOST
    # -------------------------------------------------------------
    class_counts = y_sub.value_counts()
    majority_count = class_counts.max()
    target_minority = int(majority_count * 0.10)
    smote_strat = {cls: max(count, target_minority) for cls, count in class_counts.items() if cls != 0}
    smote_strat[0] = majority_count 
    
    smote = SMOTE(sampling_strategy=smote_strat, k_neighbors=1, random_state=42)
    X_sub_bal, y_sub_bal = smote.fit_resample(X_sub_meta, y_sub)
    
    model = XGBClassifier(
        max_depth=xgb_depth, learning_rate=xgb_lr, n_estimators=xgb_estimators,
        subsample=xgb_subsample, eval_metric='mlogloss', random_state=42, n_jobs=-1
    )
    model.fit(X_sub_bal, y_sub_bal)
    
    # Isotonic Calibration
    calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(model), method='isotonic')
    calibrated_model.fit(X_cal_meta, y_cal)
    
    # -------------------------------------------------------------
    # 5. SOC-CONSTRAINED EVALUATION
    # -------------------------------------------------------------
    val_probs = calibrated_model.predict_proba(X_val_meta)
    prob_attack = 1.0 - val_probs[:, 0]
    most_likely = np.argmax(val_probs[:, 1:], axis=1) + 1
    
    best_f1 = 0.0
    for thresh in np.arange(0.10, 0.95, 0.05): # Faster sweeping step
        preds = np.where(prob_attack >= thresh, most_likely, 0)
        temp_prec = precision_score(y_val, preds, average='macro', zero_division=0)
        temp_f1 = f1_score(y_val, preds, average='macro', zero_division=0)
        
        # Enforce the strict 90% boundary
        if temp_prec >= 0.90 and temp_f1 > best_f1:
            best_f1 = temp_f1
            
    # If no threshold hits 90% precision, Optuna considers this a failed configuration
    return best_f1

# -------------------------------------------------------------
# EXECUTE THE BAYESIAN SWEEP
# -------------------------------------------------------------
print("[+] Initializing Optuna Study (Hunting for Global Maximum)...")
study = optuna.create_study(direction="maximize")
# 30 trials is a solid reconnaissance sweep without burning up your hardware
study.optimize(objective, n_trials=30) 

print("\n==================================================")
print("             OPTIMIZATION COMPLETE                ")
print("==================================================")
print(f"[>] Best Validation F1-Score: {study.best_value:.4f}")
print("[>] Optimal Parameters Found:")
for key, value in study.best_params.items():
    print(f"    -> {key}: {value}")
"""
3_base_model_selection.py
Author: Jacob Erasmus
Project: Honours Research Project
Purpose: Implements the full 5-fold cross-validation pipeline for model benchmarking and selection.
Alignment with Methodology:
    - Section 3.2.1: Memory Optimisation and Datatype Downcasting.
    - Section 3.2.2: 5-Fold Stratified Cross-Validation Pipeline.
    - Section 3.2.3: Information Gain Filtering.
    - Section 3.2.4: Recursive Feature Elimination (RFE)
    - Section 3.2.5: Cost-Sensitive Learning (Logarithmic Smoothing).
    - Section 3.3.1 & 3.3.2: Supervised and Unsupervised Model Benchmarking.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline

print("--- The Master CV Pipeline (IG -> RFE -> Cost-Sensitive Learning (Log Smoothing)) ---")

###############################
# 1. LOAD THE 80% TRAINING DATA
###############################
file_path = "logs_80percent.csv"
print(f"Loading {file_path}...")
df_train = pd.read_csv(file_path)
# Downcasting
float_cols = df_train.select_dtypes(include=['float64']).columns
df_train[float_cols] = df_train[float_cols].astype('float32')
int_cols = df_train.select_dtypes(include=['int64']).columns
for col in int_cols:
    if df_train[col].max() <= 127 and df_train[col].min() >= -128:
        df_train[col] = df_train[col].astype('int8')
    else:
        df_train[col] = df_train[col].astype('int32')
X = df_train.drop(columns=['LabelEnc'])
y = df_train['LabelEnc']
print(f"Data Loaded and Optimised. Total Features: {X.shape[1]}")

########################################################
# 2. INITIALIZE THE 5-FOLD STRATIFIED CV (Section 3.2.2)
########################################################
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1

#########################
# 3. THE METHODOLOGY LOOP
#########################
for train_index, val_index in skf.split(X, y):
    print(f"\n========================================")
    print(f"        EXECUTING FOLD {fold_no} OF 5")
    print(f"========================================")
    
    # A. ISOLATE THE FOLD
    X_train_fold = X.iloc[train_index]
    y_train_fold = y.iloc[train_index]
    X_val_fold = X.iloc[val_index]
    y_val_fold = y.iloc[val_index]
    print(f"[+] Fold Isolated. Training Samples: {X_train_fold.shape[0]}")
    
    ##############################################################
    # B. METHODOLOGY STEP 3: INFORMATION GAIN (IG) (Section 3.2.3)
    ##############################################################
    print("[+] Calculating Information Gain (Entropy)...")
    ig_scores = mutual_info_classif(X_train_fold, y_train_fold, random_state=42)
    
    # Prune features with strictly negligible IG, keeping the top 50 for RFE
    ig_series = pd.Series(ig_scores, index=X_train_fold.columns)
    selected_ig_features = ig_series.sort_values(ascending=False).head(50).index.tolist()
    
    print(f"[+] IG Complete. Pruned negligible arrays. Features reduced from {X_train_fold.shape[1]} to {len(selected_ig_features)}.")
    
    # Prune the datasets using the IG selected features
    X_train_fold_ig = X_train_fold[selected_ig_features]
    X_val_fold_ig = X_val_fold[selected_ig_features]
    
    ############################################################################
    # C. METHODOLOGY STEP 4: RECURSIVE FEATURE ELIMINATION (RFE) (Section 3.2.4)
    ############################################################################
    print("[+] Executing Random Forest RFE...")
    # Using a fast RF configuration just for feature ranking
    rf_estimator = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
    
    # Aggressively prune down to the top 20 most forensic features
    rfe = RFE(estimator=rf_estimator, n_features_to_select=20, step=5)
    rfe.fit(X_train_fold_ig, y_train_fold)
    
    final_features = X_train_fold_ig.columns[rfe.support_].tolist()
    print(f"[+] RFE Complete. Final 20 Forensic Features Locked.")
    
    # Prune datasets to the final 20 features
    X_train_final = X_train_fold_ig[final_features]
    X_val_final = X_val_fold_ig[final_features]

    #########################################
    # D. LOGARITHMIC COST-SENSITIVE LEARNING (Section 3.2.5)
    #########################################
    print("[+] Calculating Log-Smoothed Class Weights...")

    #Identify class distributions
    class_counts = y_train_fold.value_counts()
    majority_count = class_counts.max()
    # implement log equation.
    log_weights_dict = {cls: np.log1p(majority_count / count) + 1 for cls, count in class_counts.items()}

    #XGBoost requires sample weights array mapped to dataset
    sample_weights_xgb = y_train_fold.map(log_weights_dict).values
    print(f"    -> Weights calculated, max weight applied to rarest class: {max(log_weights_dict.values()):.4f}")

    ############################################
    # E. SUPERVISED BENCHMARKING (Section 3.3.1)
    #############################################
    print("\n[+] Initialising Section 3.3.1: Supervised Model Benchmarking...")
    
    # Scale data primarily to ensure Logistic Regression can converge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_val_scaled = scaler.transform(X_val_final)

    # inject log weights into loss functions
    supervised_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight=log_weights_dict),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1, class_weight=log_weights_dict),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', max_depth=15, max_delta_step=5, learning_rate=0.1, random_state=42, n_jobs=-1) 
    }
    
    for model_name, model in supervised_models.items():
        print(f"    [>] Evaluating {model_name}...")
        
        if model_name == "Logistic Regression":
            model.fit(X_train_scaled, y_train_fold)
            y_pred = model.predict(X_val_scaled)
        elif model_name == "XGBoost":
            # pass log array to xgb
            model.fit(X_train_final, y_train_fold, sample_weight=sample_weights_xgb)
            y_pred = model.predict(X_val_final)
        else: #random forest
            model.fit(X_train_final, y_train_fold)
            y_pred = model.predict(X_val_final)
            
        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val_fold, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_val_fold, y_pred, average='macro', zero_division=0)
        print(f"        Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

    #####################################################
    # F. UNSUPERVISED BENCHMARKING (Section 3.3.2)
    #####################################################
    print("\n[+] Initialising Section 3.3.2: Unsupervised Model Benchmarking...")
    
    # Isolate ONLY the normal traffic (Class 0), train on normal traffic so can map the bengin network geometry
    X_train_normal = X_train_final[y_train_fold == 0]
    
    # Scale only on normal traffic to prevent anomalous outliers from skewing the mean
    unsup_scaler = StandardScaler()
    X_train_normal_scaled = unsup_scaler.fit_transform(X_train_normal)
    X_val_final_scaled = unsup_scaler.transform(X_val_final)

    print(f"    -> Training anomaly detectors on full {X_train_normal.shape[0]} scaled baseline logs...")

    unsupervised_models = {
        "Isolation Forest": IsolationForest(n_estimators=100, contamination=0.20, random_state=42, n_jobs=-1),
        "One-Class SVM": make_pipeline(
            Nystroem(kernel='rbf', gamma=None, n_components=300, random_state=42),
            SGDOneClassSVM(nu=0.20, random_state=42)
        ),
        "k-Means": KMeans(n_clusters=1, random_state=42, n_init=10)
    }

    y_val_binary = (y_val_fold != 0).astype(int)

    for model_name, model in unsupervised_models.items():
        print(f"    [>] Evaluating {model_name}...")
        
        # Train on the SCALED normal baseline
        model.fit(X_train_normal_scaled)
        
        if model_name in ["Isolation Forest", "One-Class SVM"]:
            preds = model.predict(X_val_final_scaled)
            y_pred_binary = np.where(preds == 1, 0, 1)
        elif model_name == "k-Means":
            distances = model.transform(X_val_final_scaled)
            # flag anything outside the 80th percentile of normal training distance.
            threshold = np.percentile(model.transform(X_train_normal_scaled), 80)
            y_pred_binary = (distances > threshold).astype(int).flatten()

        acc = accuracy_score(y_val_binary, y_pred_binary)
        f1 = f1_score(y_val_binary, y_pred_binary, zero_division=0)
        print(f"        Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

    print(f"\n--- FOLD {fold_no} COMPLETELY FINISHED ---")
    fold_no += 1
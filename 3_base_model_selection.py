"""
File: 3_base_model_selection.py
Author: Jacob Erasmus
Project: BSc(Hons) Computer Science Research Project (University of Pretoria)
Purpose: Implements the Stratified 5-fold cross-validation pipeline on the 80% training partition to benchmark candidate supervised and unsupervised algorithms without data leakage.
Alignment with Methodology:
    - Chapter 3 (Step 2): Memory Optimisation and Datatype Downcasting.
    - Chapter 3 (Step 3): Stratified 5-Fold Cross-Validation.
    - Chapter 3 (Step 4): Information Gain (IG) Entropy Filtering (Top 50 features)
    - Chapter 3 (Step 5): Recursive Feature Elimination (RFE, Top 20 features per fold)
    - Chapter 3 (Step 6): Logarithmic Cost-Sensitive Sample Weighting.
    - Chapter 3 (Step 7a & 7b) & Chapter 4 (Section 4.1): Supervised (15-class Macro) and Unsupervised (2-class Binary) Model Benchmarking.
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

print("=====================================================================")
print(" STRATIFIED 5-FOLD CROSS-VALIDATION BASE MODEL BENCHMARKING PIPELINE ")
print("=====================================================================")

#################################################
# 1. LOAD AND OPTIMISE THE 80% TRAINING PARTITION
#################################################
file_path = "logs_80percent.csv"
print(f"[+] Loading training partition from '{file_path}'...")
df_train = pd.read_csv(file_path)

# Reapply datatype downcasting after CSV inspection
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
print(f"[+] Training data loaded and downcasted. Records: {X.shape[0]:,} | Features: {X.shape[1]}")

##################################################
# 2. INITIALISE STRATIFIED 5-FOLD CROSS-VALIDATION
##################################################
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1

# Accumulators for 5-fold summary reporting (Mean +/- SD)
sup_results = {name: {'acc': [], 'prec': [], 'f1': []} for name in ["Logistic Regression", "Random Forest", "XGBoost"]}
unsup_results = {name: {'acc': [], 'prec': [], 'f1': []} for name in ["Isolation Forest", "One-Class SVM", "k-Means"]}

#######################################
# 3. CROSS-VALIDATION BENCHMARKING LOOP
#######################################
for train_index, val_index in skf.split(X, y):
    print(f"\n===============================================")
    print(f" EXECUTING CROSS-VALIDATION FOLD {fold_no} OF 5")
    print(f"===============================================")
    
    # A. Isolate active training and validation folds
    X_train_fold = X.iloc[train_index]
    y_train_fold = y.iloc[train_index]
    X_val_fold = X.iloc[val_index]
    y_val_fold = y.iloc[val_index]
    print(f"[+] Fold Isolated. Active training records: {X_train_fold.shape[0]:,} | Validation records: {X_val_fold.shape[0]:,}")
    
    # B. Step 4: Information Gain (IG) Filtering (fitted on active training fold)
    print("[+] Computing Information Gain on active training fold...")
    ig_scores = mutual_info_classif(X_train_fold, y_train_fold, random_state=42)
    ig_series = pd.Series(ig_scores, index=X_train_fold.columns)
    selected_ig_features = ig_series.sort_values(ascending=False).head(50).index.tolist()
    print(f"    -> IG filtering complete. Feature space reduced from {X_train_fold.shape[1]} to {len(selected_ig_features)}.")
    
    X_train_fold_ig = X_train_fold[selected_ig_features]
    X_val_fold_ig = X_val_fold[selected_ig_features]
    
    # C. Step 5: Recursive Feature Elimination (RFE) to top 20 features per fold
    print("[+] Executing Recursive Feature Eliminiation (Random Forest base estimator)...")
    rf_estimator = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
    rfe = RFE(estimator=rf_estimator, n_features_to_select=20, step=5)
    rfe.fit(X_train_fold_ig, y_train_fold)
    
    final_features = X_train_fold_ig.columns[rfe.support_].tolist()
    print(f"    -> RFE complete. Top {len(final_features)} features selected for Fold {fold_no}.")
    
    X_train_final = X_train_fold_ig[final_features]
    X_val_final = X_val_fold_ig[final_features]

    # D. Step 6: Logarithmic Cost-Sensitive Sample Weighting
    print("[+] Computing logarithmic cost-sensitive class weights...")
    class_counts = y_train_fold.value_counts()
    majority_count = class_counts.max()
    log_weights_dict = {cls: np.log1p(majority_count / count) + 1 for cls, count in class_counts.items()}
    sample_weights_xgb = y_train_fold.map(log_weights_dict).values
    print(f"    -> Weights computed. Maximum minority class weight: {max(log_weights_dict.values()):.4f}")

    # E. Step 7a: Supervised Model Benchmarking (15-class Macro-Averaging)
    print("\n[+] Evaluting Supervised Classifiers (15-Class Unweighted Macro-Averaging)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_val_scaled = scaler.transform(X_val_final)

    supervised_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight=log_weights_dict),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1, class_weight=log_weights_dict),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', max_depth=15, max_delta_step=5, learning_rate=0.1, random_state=42, n_jobs=-1) 
    }
    
    for model_name, model in supervised_models.items():   
        if model_name == "Logistic Regression":
            model.fit(X_train_scaled, y_train_fold)
            y_pred = model.predict(X_val_scaled)
        elif model_name == "XGBoost":
            model.fit(X_train_final, y_train_fold, sample_weight=sample_weights_xgb)
            y_pred = model.predict(X_val_final)
        else:
            model.fit(X_train_final, y_train_fold)
            y_pred = model.predict(X_val_final)
            
        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val_fold, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_val_fold, y_pred, average='macro', zero_division=0)

        sup_results[model_name]['acc'].append(acc)
        sup_results[model_name]['prec'].append(prec)
        sup_results[model_name]['rec'].append(rec)
        sup_results[model_name]['f1'].append(f1)

        print(f"    [>] {model_name:<20} | Accuracy: {acc:.4f} | Macro Precision: {prec:.4f} | Macro Recall: {rec:.4f} | Macro F1: {f1:.4f}")

    # F. Step 7b: Unsupervised Model Benchmarking (Binary Evaluation, Fitted on Benign Traffic Only)
    print("\n[+] Evaluating Unsupervised Models (Binary Evaluation)...")
    X_train_normal = X_train_final[y_train_fold == 0]

    unsup_scaler = StandardScaler()
    X_train_normal_scaled = unsup_scaler.fit_transform(X_train_normal)
    X_val_final_scaled = unsup_scaler.transform(X_val_final)
    print(f"    -> Fitting unsupervised models on {X_train_normal.shape[0]:,} standardised benign records...")

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
        model.fit(X_train_normal_scaled)
        
        if model_name in ["Isolation Forest", "One-Class SVM"]:
            preds = model.predict(X_val_final_scaled)
            y_pred_binary = np.where(preds == 1, 0, 1)
        elif model_name == "k-Means":
            distances = model.transform(X_val_final_scaled)
            threshold = np.percentile(model.transform(X_train_normal_scaled), 80)
            y_pred_binary = (distances > threshold).astype(int).flatten()

        acc = accuracy_score(y_val_binary, y_pred_binary)
        prec = precision_score(y_val_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_val_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_val_binary, y_pred_binary, zero_division=0)

        unsup_results[model_name]['acc'].append(acc)
        unsup_results[model_name]['prec'].append(prec)
        unsup_results[model_name]['rec'].append(rec)
        unsup_results[model_name]['f1'].append(f1)

        print(f"    [>] {model_name:<20} | Accuracy: {acc:.4f} | Binary Precision: {prec:.4f} | Binary Recall: {rec:.4f} | Binary F1: {f1:.4f}")

    print(f"\n[+] Fold {fold_no} evaluation complete.")
    fold_no += 1

##################################################
# 4. 5-FOLD CROSS-VALIDATION SUMMARY (MEAN +/- SD)
##################################################
print("\n==============================================================")
print(" 5-FOLD CROSS-VALIDATION SUMMARY RESULTS (MEAN +/- SAMPLE SD) ")
print("==============================================================")

print("\n--- Supervised Models (15-Class Macro-Averaged) ---")
for model_name, metrics in sup_results.items():
    print(
        f"{model_name:<20} | "
        f"Acc: {np.mean(metrics['acc']):.4f} +/- {np.std(metrics['acc'], ddof=1):.4f} | "
        f"Prec: {np.mean(metrics['prec']):.4f} +/- {np.std(metrics['prec'], ddof=1):.4f} | "
        f"Rec: {np.mean(metrics['rec']):.4f} +/- {np.std(metrics['rec'], ddof=1):.4f} | "
        f"F1: {np.mean(metrics['f1']):.4f} +/- {np.std(metrics['f1'], ddof=1):.4f}"
    )

print("\n--- Unsupervised Models (2-Class Binary Anomaly Detection) ---")
for model_name, metrics in unsup_results.items():
    print(
        f"{model_name:<20} | "
        f"Acc: {np.mean(metrics['acc']):.4f} +/- {np.std(metrics['acc'], ddof=1):.4f} | "
        f"Prec: {np.mean(metrics['prec']):.4f} +/- {np.std(metrics['prec'], ddof=1):.4f} | "
        f"Rec: {np.mean(metrics['rec']):.4f} +/- {np.std(metrics['rec'], ddof=1):.4f} | "
        f"F1: {np.mean(metrics['f1']):.4f} +/- {np.std(metrics['f1'], ddof=1):.4f}"
    )
print("==============================================================")

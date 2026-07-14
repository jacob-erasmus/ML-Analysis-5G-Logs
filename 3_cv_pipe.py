import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
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

print("--- Phase 3: The Master CV Pipeline (IG -> RFE -> SMOTE) ---")

# 1. LOAD THE 80% TRAINING DATA
file_path = "logs_80percent.csv"
print(f"Loading {file_path}...")
df_train = pd.read_csv(file_path)

X = df_train.drop(columns=['LabelEnc'])
y = df_train['LabelEnc']
print(f"Data Loaded. Total Features: {X.shape[1]}")

# 2. INITIALIZE THE 5-FOLD STRATIFIED CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_no = 1

# 3. THE METHODOLOGY LOOP
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
    
# -------------------------------------------------------------
    # B. METHODOLOGY STEP 3: INFORMATION GAIN (IG)
    # -------------------------------------------------------------
    print("[+] Calculating Information Gain (Entropy)...")
    ig_scores = mutual_info_classif(X_train_fold, y_train_fold, random_state=42)
    
    # We prune features with strictly negligible IG, keeping the top 50 for RFE
    ig_series = pd.Series(ig_scores, index=X_train_fold.columns)
    selected_ig_features = ig_series.sort_values(ascending=False).head(50).index.tolist()
    
    print(f"[+] IG Complete. Pruned negligible arrays. Features reduced from {X_train_fold.shape[1]} to {len(selected_ig_features)}.")
    
    # Prune the datasets using the IG selected features
    X_train_fold_ig = X_train_fold[selected_ig_features]
    X_val_fold_ig = X_val_fold[selected_ig_features]
    
    # -------------------------------------------------------------
    # C. METHODOLOGY STEP 4: RECURSIVE FEATURE ELIMINATION (RFE)
    # -------------------------------------------------------------
    print("[+] Initializing Random Forest RFE...")
    # Using a fast RF configuration just for feature ranking
    rf_estimator = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    
    # We tell RFE to aggressively prune down to the top 20 most forensic features
    rfe = RFE(estimator=rf_estimator, n_features_to_select=20, step=5)
    rfe.fit(X_train_fold_ig, y_train_fold)
    
    final_features = X_train_fold_ig.columns[rfe.support_].tolist()
    print(f"[+] RFE Complete. Final 20 Forensic Features Locked.")
    
    # Prune datasets to the final 20 features
    X_train_final = X_train_fold_ig[final_features]
    X_val_final = X_val_fold_ig[final_features]
    
# -------------------------------------------------------------
    # D. METHODOLOGY STEP 5: SMOTE (Applied ONLY to X_train_final)
    # -------------------------------------------------------------
    print("[+] Applying SMOTE to handle class imbalance...")
    
    # Identify the size of the majority class (Class 0: Normal Traffic)
    class_counts = y_train_fold.value_counts()
    majority_class_count = class_counts.max()
    
    # Create a dynamic dictionary to cap synthetic generation
    # We upsample minority classes to a maximum of 10% of the majority class size (~145k samples each)
    target_minority_size = int(majority_class_count * 0.10)
    
    smote_strategy = {}
    for cls, count in class_counts.items():
        if count == majority_class_count:
            smote_strategy[cls] = count # Leave majority alone
        elif count < target_minority_size:
            smote_strategy[cls] = target_minority_size # Upsample to the cap
        else:
            smote_strategy[cls] = count # If a minority class is already larger than the cap, leave it alone

    # Apply the capped SMOTE (k_neighbors=1 bypasses the < 5 sample trap)
    smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=1, random_state=42)
    
    try:
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_final, y_train_fold)
        print(f"[+] SMOTE Complete. Balanced Training Samples: {X_train_balanced.shape[0]}")
    except ValueError as e:
        print(f"[!] SMOTE CRITICAL ERROR: {e}")
        sys.exit()

# -------------------------------------------------------------
    # E. METHODOLOGY STEP 4.3: SUPERVISED BENCHMARKING
    # -------------------------------------------------------------
    print("\n[+] Initializing Section 4.3: Supervised Model Benchmarking...")
    
    # We added StandardScaler to help LR converge, and removed the deprecated parameters
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_val_scaled = scaler.transform(X_val_final)

    supervised_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42), # Removed n_jobs
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1) # Removed use_label_encoder
    }
    
    for model_name, model in supervised_models.items():
        print(f"    [>] Evaluating {model_name}...")
        
        if model_name == "Logistic Regression":
            model.fit(X_train_scaled, y_train_balanced)
            y_pred = model.predict(X_val_scaled)
        else:
            model.fit(X_train_balanced, y_train_balanced)
            y_pred = model.predict(X_val_final)
            
        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val_fold, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_val_fold, y_pred, average='macro', zero_division=0)
        print(f"        Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

   # -------------------------------------------------------------
    # F. METHODOLOGY STEP 4.3: UNSUPERVISED BENCHMARKING
    # -------------------------------------------------------------
    print("\n[+] Initializing Section 4.3: Unsupervised Model Benchmarking...")
    
    # Isolate ONLY the normal traffic (Class 0)
    X_train_normal = X_train_final[y_train_fold == 0]
    
    # CRITICAL FIX: Distance-based models require strictly scaled data!
    # We fit the scaler ONLY on normal traffic to prevent zero-day leakage into the mathematical mean.
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
            threshold = np.percentile(model.transform(X_train_normal_scaled), 80)
            y_pred_binary = (distances > threshold).astype(int).flatten()

        acc = accuracy_score(y_val_binary, y_pred_binary)
        f1 = f1_score(y_val_binary, y_pred_binary, zero_division=0)
        print(f"        Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

    print(f"\n--- FOLD {fold_no} COMPLETELY FINISHED ---")
    
    # LEAVE THE BREAK HERE FOR NOW
    print("\n[!] Safety break initiated. Awaiting Fold 1 Unsupervised verification.")
    break

    print(f"\n--- FOLD {fold_no} COMPLETELY FINISHED ---")
    
    # SAFETY BREAK: Keep this here so we can verify the models run on Fold 1 
    # before we commit your laptop to the full 5-fold loop.
    print("\n[!] Safety break initiated. Awaiting Fold 1 verification.")
    break 

    fold_no += 1
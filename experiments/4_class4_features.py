"""
File: 4_class4_features
Author: Jacob Erasmus
Project: UP Honours Research
Purpose: Class 4 has been identified as poor performing and evading the model, thus this program execute hierarchical feature selection to identify features
         that distingush between benign traffic (class 0) from the class 4 attacks.
Alignment with Methodology:
    - Section 4.2.1: Memory Optimisation and Dataype Downcasting.
    - Strict adherence to the 80% training set to avoid data leakage.
    - Employs Mutual Information (MI) to rank non-linear relationships.
"""

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

print("=========================================")
print("     Class 4 Feature Identification      ")
print("=========================================")

########################################
# 0. MEMORY OPTIMISATION (Section 4.2.1)
########################################
def optimise_memory(df):
    """Downcasts float64 to float32 and int64 to int8/int32 to conserve RAM."""
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].max() <= 127 and df[col].min() >= -128:
            df[col] = df[col].astype('int8')
        else:
            df[col] = df[col].astype('int32')
    return df

#########################
# 1. LOAD TRAINING DATA
#########################
print("[+] Loading 80% Training Set...")
try:
    df_train = pd.read_csv("logs_80percent.csv")
    df_train = optimise_memory(df_train)
except FileNotFoundError:
    print("[!] ERROR: 'logs_80percent.csv' not found.")
    exit()

###################################
# 2. ISOLATE CLASS 0 and CLASS 4
###################################
print("[+] Isolating Benign Traffic (Class 0) and Class 4...")
df_target = df_train[df_train['LabelEnc'].isin([0,4])]

X_target = df_target.drop(columns=['LabelEnc'])
y_target = df_target['LabelEnc']

print(f"    [>] Total Logs Evaluated: {len(df_target):,}")
print(f"    [>] Class 0 Count: {len(df_target[df_target['LabelEnc'] == 0]):,}")
print(f"    [>] Class 4 Count: {len(df_target[df_target['LabelEnc'] == 4]):,}")

#############################
# 3. CURRENT GOLDEN FEATURES
#############################
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

########################################
# 4. MUTUAL INFORMATION CLASSIFICATION
########################################
print("\n[+] Mutual Information (MI) Analysis...")
print("     -> Calculating non-linear dependenices between features and Class 4...")
# Using a random state for reproducibility
mi_scores = mutual_info_classif(X_target, y_target, random_state=42)
# Dataframe for easy sorting and viewing
mi_df = pd.DataFrame({'Feature': X_target.columns, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values(by='MI_Score', ascending=False).reset_index(drop=True)

#####################
# 5. Display Results
#####################
print("\n=====================================")
print("     Top 15 Features for Class 4     ")
print("=====================================")

for index, row in mi_df.head(15).iterrows():
    feature = row['Feature']
    score = row['MI_Score']
    # tagged the feature to see if current Golden Features are helping here
    tag = "[Current Golden Feature]" if feature in golden_features else "[New Discovery]"
    print (f"{index + 1:2d}. {score:.4f} | {feature} {tag}")

print("\n===================================")
print("[>] Analysis Complete.")
print("[>] Consider high ranked [New Discovery] features for model improvement.")

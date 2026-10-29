"""
File: 6_scalability_assessment.py
Author: Jacob Erasmus
Project: BSc(Hons) Computer Science Researcg Project (University of Pretoria)
Purpose: Executes the operational scalability assessment for the deployed framework. Across three simulated 5G batch buffer sizes (N = 5000,N = 250,000, and N = 1,000,000 across K = 20 repetitions per scenario) to
         evaluate inference latency, throughput (EPS), and empirical time complexity.
Methodology & Discussion Alignment:
    - Chapter 3 (Step 12): Operational Scalability Assessment Protocol.
    - Chapter 4 (Section 4.3): Operational Scalability Assessment and O(n) Verification.
"""

import pandas as pd
import numpy as np
import time
import joblib
import sys
import os
import warnings
warnings.filterwarnings("ignore")

print("===============================================")
print(" STEP 12: OPERATIONAL SCALABILITY ASSESSMENT ")
print("=============================================")

#############################################
# 1. TRACE-DRIVEN STREAM: LOAD FULL DATASET
#############################################
def optimise_memory(df):
    """Applies dataype downcasting to match deployed feature represenetaitons."""
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')

    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].max() <= 127 and df[col].min() >= -128:
            df[col] = df[col].astype('int8')
        else:
            df[col] = df[col].astype('int32')
    return df

print("\n[+] Initialising Trace-Driven Stream from Full Dataset...")
try:
    # pulling the entire logs.csv file
    df_full = pd.read_csv("logs.csv", low_memory=False)
    df_full = optimise_memory(df_full)
except FileNotFoundError:
    print("     [!] ERROR: 'logs.csv' not found in the root directory.")
    sys.exit()

# Add 10 Golden Features
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

X_full_stream = df_full[golden_features]
print(f"    [>] Telemetry Ready: {X_full_stream.shape[0]:,} records restricted to 10 core features.")

#############################
# 2. LOAD DEPLOYED FRAMEWORK
#############################
export_dir = "final_framework"
print(f"\n[+] Loading Framework from '{export_dir}/ directory...")

try:
    layer1_scaler = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_layer1_scaler.pkl'))
    layer1_ocsvm = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_layer1_ocsvm.pkl'))
    layer1_if = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_layer1_if.pkl'))
    layer2_xgb = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_layer2_xgb.pkl'))
    optimal_thresholds = joblib.load(os.path.join(export_dir, 'hybrid_ensemble_thresholds.pkl'))
    print("     [>] Scaler, Layer 1 (OCSVM&IF), and Layer 2 (XGBoost) and Thresholds successfully loaded.")
except FileNotFoundError:
    print(f"    [!] ERROR: Model files not found in '{export_dir}/'.")
    print("     Please run '5_hybrid_ensemble.py' first to generate the .pkl files.")
    sys.exit(1)

###################################
# 3. TRACE-DRIVEN BATCH SIMULATION
###################################
def run_simulation(scenario_name, batch_size, iterations=10):
    """
    Measures in-memory inference latency and throughput across K repetitions
    for a specified batch buffer size N.
    """
    print(f"\n[+] Scenario: {scenario_name}")
    print(f"    -> Buffer Size (N): {batch_size:,} records | Repetitions (K): {iterations}")

    latencies = []

    for i in range(iterations):
        # sample batch outside the timed inference block
        batch_df = X_full_stream.sample(n=batch_size, replace=True, random_state=42+i)

        ################################
        # START INFERENCE HARDWARE CLOCK
        ################################
        start_time = time.perf_counter()

        # Layer 1: Feauture standardistaiton and continous anomaly score extraction
        batch_scaled = layer1_scaler.transform(batch_df)
        ocsvm_scores = layer1_ocsvm.decision_function(batch_scaled)
        if_scores = layer1_if.decision_function(batch_scaled)

        # Data Augmentation (Dimensionality 10 -> 12)
        batch_meta = batch_df.copy()
        batch_meta['OCSVM_Score'] = ocsvm_scores
        batch_meta['IF_Score'] = if_scores
        
        # Layer 2: Probability Generation
        l2_probs = layer2_xgb.predict_proba(batch_meta)
        
        # Pure Normalization Inference Loop
        base_ratios = np.zeros_like(l2_probs)
        for cls in range(1, 15):
            base_ratios[:, cls] = l2_probs[:, cls] / optimal_thresholds[cls]
            
        best_attack_classes = np.argmax(base_ratios[:, 1:], axis=1) + 1
        row_indices = np.arange(len(batch_meta))
        winning_base_ratios = base_ratios[row_indices, best_attack_classes]
        
        # Trigger alert strictly if the native Base Ratio crosses the threshold (>= 1.0)
        _ = np.where(winning_base_ratios >= 1.0, best_attack_classes, 0)

        ###############################
        # STOP INFERENCE HARDWARE CLOCK
        ###############################
        end_time = time.perf_counter()

        cycle_latency = end_time - start_time
        latencies.append(cycle_latency)

    # Calculate Core Metrics & Variability
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies, ddof=1)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    iter_eps = [batch_size / l for l in latencies]
    mean_eps = batch_size / mean_latency
    std_eps = np.std(iter_eps, ddof=1)

    print(f"    [>] Inference Latency (t) : {mean_latency:.4f} ± {std_latency:.4f} s (Min: {min_latency:.4f}s, Max: {max_latency:.4f}s)")
    print(f"    [>] Throughput (EPS)      : {mean_eps:.0f} ± {std_eps:.0f} events per second")
    print(f"    [>] L1 Forwarding Load    : 100% (Layer 2 inspects all {batch_size:,} logs)")
    
    return mean_latency, mean_eps

###########################################
# 4. EXECUTING THE THREE TRAFFIC CONDITIONS
###########################################
print("\n----------------------------------------------------------------------")
print(" EXECUTING TRACE-DRIVEN 5G BATCH SCALABILITY SCENARIOS (K = 20 RUNS)  ")
print("----------------------------------------------------------------------")

# Condition 1: mIoT Synchronised Access: Simulating massive simultaneous device registrations
t_miot, eps_miot = run_simulation(
    scenario_name="mIoT Synchronised Access (Device Registrations)",
    batch_size=250000,
    iterations=20
)

# Condition 2: Control Plane Flooding: High velocity network spikes targeting NRF/UDR service requests
t_flood, eps_flood = run_simulation(
    scenario_name="Control Plane Flooding (NRF/UDR Targeting)",
    batch_size=1000000,
    iterations=20
)

# Condition 3: Baseline Signalling: typical background network operational traffic
t_base, eps_base = run_simulation(
    scenario_name="Standard Baseline Signalling Volume",
    batch_size=5000,
    iterations=20
)

####################################################
# 5. STABILITY RATION AND EMPIRICAL SCALING SUMMARY
####################################################
print("\n======================================================================")
print(" THROUGHPUT STABILITY RATIO AND EMPIRICAL O(n) SCALING SUMMARY        ")
print("======================================================================")

ratio_miot = eps_miot / eps_base
ratio_flood = eps_flood / eps_base

print(f"Baseline EPS    : {eps_base:,.0f}")
print(f"mIoT Load EPS    : {eps_miot:,.0f} (Ratio: {ratio_miot:.2f}x)")
print(f"Flood Load EPS    : {eps_flood:,.0f} (Ratio: {ratio_flood:.2f}x)")

print("\n--- Empirical Scaling Assessment --- ")
if ratio_flood >= 0.80:
    print("[+] Result: Stability Ratio >= 0.80 verified. The framework exhibits linear O(n)")
    print("    in-memory batch scaling across tested volumes up to 1,000,000 records.")
else:
    print("[!] Notice: Throughput degradation observed at peak batch volume (Ratio < 0.80).")

##########################################################
# 6. GENERATE SCALABILITY VISUALISATIONS
##########################################################
import matplotlib.pyplot as plt
import os

arch_name = "Hybrid Ensemble"
file_prefix = "hybrid_ensemble" 
vis_dir = "visualisations"
os.makedirs(vis_dir, exist_ok=True)
print(f"\n[+] Generating Scalability Visualizations in '{vis_dir}/'...")

scenarios = ['Baseline\n(5,000 logs)', 'mIoT Burst\n(250,000 logs)', 'Control Plane Flood\n(1,000,000 logs)']
batch_sizes = [5000, 250000, 1000000]

eps_data = [eps_base, eps_miot, eps_flood]
latency_data = [t_base, t_miot, t_flood]

# --- PLOT 1: THROUGHPUT (EPS) ---
print("    -> Plotting EPS Vectorisation Ceiling (Bar Chart)...")
plt.figure(figsize=(9, 6))
bars = plt.bar(scenarios, eps_data, color='#1f77b4' if file_prefix == 'seq' else '#ff7f0e', 
               edgecolor='black', width=0.5)

plt.ylabel('Throughput (Events Per Second)', fontsize=12, fontweight='bold')
plt.title(f'Operational Throughput (EPS) - {arch_name}', fontsize=14, pad=15, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(eps_data)*0.02), 
             f'{yval:,.0f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, f'{file_prefix}_eps_chart.png'), dpi=300)
plt.close()

# --- PLOT 2: O(N) LATENCY COMPLEXITY ---
print("    -> Plotting Linear Time Complexity (Line Graph)...")
plt.figure(figsize=(9, 6))
plt.plot(batch_sizes, latency_data, marker='o' if file_prefix == 'seq' else 's', 
         markersize=8, linewidth=3, color='#1f77b4' if file_prefix == 'seq' else '#ff7f0e')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Inference Latency (Seconds)', fontsize=12, fontweight='bold')
plt.xlabel('Volumetric Network Load (Logs per Buffer)', fontsize=12, fontweight='bold')
plt.title(f'Empirical Time Complexity - {arch_name}', fontsize=14, pad=15, fontweight='bold')

plt.xticks(batch_sizes, ['5,000', '250,000', '1,000,000'], fontsize=11)
plt.grid(True, which="major", linestyle='-', alpha=0.6)
plt.grid(True, which="minor", linestyle='--', alpha=0.3)

# Annotate the peak flood latency
plt.annotate(f'{t_flood:.2f}s', xy=(batch_sizes[2], t_flood), 
             xytext=(-45, 15), textcoords='offset points', 
             fontsize=11, fontweight='bold', 
             color='#1f77b4' if file_prefix == 'seq' else '#ff7f0e')

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, f'{file_prefix}_latency_chart.png'), dpi=300)
plt.close()

print(f"    [>] '{file_prefix}_eps_chart.png' and '{file_prefix}_latency_chart.png' saved successfully.")
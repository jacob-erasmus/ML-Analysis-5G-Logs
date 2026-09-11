"""
6_scalability_assessment_alt.py
Author: Jacob Erasmus
Project: UP Honours Research
Purpose: Executes the operational scalability assessment for the alternate deployed framework. Simulates variable velocity 5G network conditions to measure throughput (EPS),
         inference latency (t), and stability.
Alignment with Methodology: Section 4.5
"""

import pandas as pd
import numpy as np
import time
import joblib
import sys
import os
import warnings
warnings.filterwarnings("ignore")

print("============================================")
print(" --- OPERATIONAL SCALABILITY ASSESSMENT --- ")
print("============================================")

#############################################
# 1. TRACE-DRIVEN STREAM: LOAD FULL DATASET
#############################################
def optimise_memory(df):
    """Executes the Memory Optimisation and Downcasting Block."""
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')

    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].max() <= 127 and df[col].min() >= 128:
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
        'host.name_ausf',
        'host.name_amf',
        'zeek.udp_conns_1,728,325,600',
        'fields.vnf_connection',
        'host.name_nrf',
        'fields.vnf_weird'
]

X_full_stream = df_full[golden_features]
print(f"    [>] Stream Ready: Pool of {X_full_stream.shape[0]:,} 5G logs available.")

#############################
# 2. LOAD DEPLOYED FRAMEWORK
#############################
export_dir = "alt_deployed_framework"
print(f"\n[+] Loading Framework from '{export_dir}/ directory...")

try:
    layer1_scaler = joblib.load(os.path.join(export_dir, 'p_layer1_scaler.pkl'))
    layer1_ocsvm = joblib.load(os.path.join(export_dir, 'p_layer1_ocsvm.pkl'))
    layer1_if = joblib.load(os.path.join(export_dir, 'p_layer1_if.pkl'))
    calibrated_xgb = joblib.load(os.path.join(export_dir, 'p_layer2_xgb_calibrated.pkl'))
    print("     [>] Scaler, Layer 1 (OCSVM&IF), and Layer 2 (XGBoost) successfully loaded.")
    with open(os.path.join(export_dir, 'p_threshold.txt'), 'r') as f:
        best_thresh = float(f.read().strip())
    print(f"     [>] Threshold locked at: {best_thresh:.2f}")
except FileNotFoundError:
    print(f"    [!] ERROR: Model files not found in '{export_dir}/'.")
    print("     Please run '5_parallel_hybrid_ensemble.py' first to generate the .pkl files.")
    sys.exit()

################################################
# 3. TRACE-DRIVEN SIMULATION ENGINE (Section 4.5)
################################################
def run_simulation(scenario_name, batch_size, iterations=10):
    """
    Simulates streaming ingestion from the 5G network.
    Tests the precise hardware latency of the deployed framework.
    """
    print(f"\n[+] Executing Scenario: {scenario_name}")
    print(f"    -> Streaming Velocity: {batch_size} logs per physical buffer.")

    latencies = []

    for i in range(iterations):
        # Extract a random, unsorted batch of raw traffic from the dataset
        batch_df = X_full_stream.sample(n=batch_size, replace=True, random_state=42+i)

        ################################
        # START INFERENCE HARDWARE CLOCK
        ################################
        start_time = time.perf_counter()

        # Layer 1: Scaling and parallel distance mapping
        batch_scaled = layer1_scaler.transform(batch_df)
        ocsvm_scores = layer1_ocsvm.decision_function(batch_scaled)
        if_scores = layer1_if.decision_function(batch_scaled)

        # Data Augmentation (Dimensionality 10 -> 12)
        batch_meta = batch_df.copy()
        batch_meta['OCSVM_Score'] = ocsvm_scores
        batch_meta['IF_Score'] = if_scores
        
        # Layer 2: Calibrated Inference (Processes 100% of the batch)
        l2_probs = calibrated_xgb.predict_proba(batch_meta)
        prob_attack = 1.0 - l2_probs[:, 0]
        most_likely_attack = np.argmax(l2_probs[:, 1:], axis=1) + 1
        
        # Apply strict threshold
        _ = np.where(prob_attack >= best_thresh, most_likely_attack, 0)

        ###############################
        # STOP INFERENCE HARDWARE CLOCK
        ###############################
        end_time = time.perf_counter()

        cycle_latency = end_time - start_time
        latencies.append(cycle_latency)

    # Calculate Core Metrics
    mean_latency = np.mean(latencies)
    mean_eps = batch_size / mean_latency

    print(f"    [>] Mean Inference Latency (t) : {mean_latency:.4f} seconds")
    print(f"    [>] Throughput (EPS) : {mean_eps:.0f} events per second")
    print(f"    [>] L1 Forwarding Load         : 100% (Layer 2 inspects all {batch_size:,} logs)")
    
    return mean_latency, mean_eps

###########################################
# 4. EXECUTING THE THREE TRAFFIC CONDITIONS
###########################################
print("\n=========================================")
print(" ---SIMULATING 5G NETWORK CONDITIONS--- ")
print("========================================")

# Condition 1: mIoT Synchronised Access: Simulating massive simultaneous device registrations
t_miot, eps_miot = run_simulation(
    scenario_name="mIoT Synchronised Access (Device Registrations)",
    batch_size=250000,
    iterations=20
)

# Condition 2: Control Plane Flooding: High velocity network spikes targeting NRF/AMF service requests
t_flood, eps_flood = run_simulation(
    scenario_name="Control Plane Flooding (NRF/AMF Targeting)",
    batch_size=1000000,
    iterations=20
)

# Condition 3: Baseline Signalling: typical background network operational traffic
t_base, eps_base = run_simulation(
    scenario_name="Standard Baseline Signalling Volume",
    batch_size=5000,
    iterations=20
)

#####################
# 5. COMPLEXITY PROOF
#####################
print("\n===========================================")
print(" ---STABILITY RATIO & O(n) VERIFICATION--- ")
print("===========================================")

ratio_miot = eps_miot / eps_base
ratio_flood = eps_flood / eps_base

print(f"Baseline EPS    : {eps_base:,.0f}")
print(f"mIoT Load EPS    : {eps_miot:,.0f} (Ratio: {ratio_miot:.2f}x)")
print(f"Flood Load EPS    : {eps_flood:,.0f} (Ratio: {ratio_flood:.2f}x)")

print("\n ---O(n) Complexity Verdict--- ")
if ratio_flood >= 0.80:
    print("[SUCCESS] Deployed Framework exhibits near-perfect linear O(n) scaling.")
    print("System absorbed 100% volumetric inspection without bottlenecking.")
else: 
    print("[WARNING] Non-linear bottlenecks detected under peak laod.\nLack of a L1 triage caused latency degradation")

##################################
# 6. GRAPHS (Throughput & Latency)
##################################
import matplotlib.pyplot as plt
import os

arch_name = "Parallel Hybrid Ensemble"
file_prefix = "p" 
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


        
# Machine Learning-Based Analysis of High-Volume 5G Network Logs

This repository contains the source code, trained model artefacts, and evaluation scripts for the BSc(Hons) Computer Science research project at the University of Pretoria.

* **Author:** Jacob Erasmus
* **Supervisor:** Mr. Sheunesu Makura
* **Research Group:** Digital Forensic Science (DigiForS), Department of Computer Science, University of Pretoria
* **Full Dissertation:** [Read the Research Report]()

---

## Project Overview

The transition to virtualised fifth-generation (5G) Service-Based Architectures (SBA) generates high-volume, multidimensional network telemetry that challenges post-incident digital forensic triage. This project implements and evaluates a constraint-driven, two-layer hybrid machine learning framework designed to filter benign signalling traffic, preserve minority-class attack evidence, and scale linearly across large log buffers:

1. **Preprocessing & Dimensionality Reduction:** Applies datatype downcasting ($19.9\%$ RAM reduction), stratified 80/20 partitioning, Information Gain filtering, and cross-validated Recursive Feature Elimination (`RFECV`) to compress 178 raw features into 10 core flow and 5G SBA indicators ($94.4\%$ dimensionality reduction).
2. **Layer 1 Unsupervised Meta-Feature Extraction:** Fits a Nyström One-Class Support Vector Machine (`SGDOneClassSVM`) and an `IsolationForest` strictly on standardised benign training traffic to extract continuous spatial boundary (`OCSVM_Score`) and tree isolation depth (`IF_Score`) meta-features, expanding the input space from 10 to 12 dimensions.
3. **Layer 2 Supervised Classification & Triage Export:** Pairs a logarithmic cost-sensitive `XGBoost` classifier with Neyman-Pearson threshold optimisation and continuous probability normalisation (maximising Macro Recall on a validation holdout subject to $\text{Macro Precision} \ge 0.8100$). Records exceeding normalised class thresholds are automatically exported to a timestamped forensic triage CSV.

---

## Repository Structure

| File / Directory | Description |
| :--- | :--- |
| `1_ingest.py` | **Step 1–3:** Ingests `logs.csv`, profiles class balance and duplicates, performs datatype downcasting, and exports the stratified 80% training (`logs_80percent.csv`) and 20% test (`logs_20percent.csv`) partitions. |
| `2_eda.ipynb` | **Step 1 & 3:** Jupyter Notebook conducting Exploratory Data Analysis (EDA) strictly on `logs_80percent.csv` to verify class skew ($80.30\%$ benign, $19.70\%$ anomalous), zero-variance columns, and data density. |
| `3_base_model_selection.py` | **Steps 4–8:** Executes the leakage-free Stratified 5-Fold Cross-Validation pipeline (IG $\rightarrow$ RFE $\rightarrow$ Logarithmic Weighting) to benchmark supervised (Logistic Regression, Random Forest, XGBoost) and unsupervised (Isolation Forest, OCSVM, k-Means) base models. |
| `4_hyperparameter_feature_tuning.py` | **Step 9:** Executes global `RFECV` to lock the 10 optimal features and runs 5-fold `RandomizedSearchCV` to tune the Layer 2 XGBoost hyperparameters. |
| `5_hybrid_ensemble.py` | **Steps 10–11:** Trains the 12-dimensional Hybrid Ensemble, executes Neyman-Pearson threshold optimisation, exports flagged triage artefacts, benchmarks against the 99th-percentile deterministic baseline and standalone 10-feature XGBoost (ablation study), and saves `.pkl` files to `final_framework/`. |
| `6_scalability_assessment.py` | **Step 12:** Loads the serialised framework from `final_framework/` and executes trace-driven batch scalability tests across $N = 5,000$, $N = 250,000$, and $N = 1,000,000$ logs ($K = 20$ repetitions each). |
| `final_framework/` | Contains the serialised `.pkl` model components (`layer1_scaler`, `layer1_ocsvm`, `layer1_if`, `layer2_xgb`, and `thresholds`). |
| `visualisations/` | Output directory for generated plots (`rfe_curve.png`, `hybrid_ensemble_confusion_matrix.png`, `hybrid_ensemble_eps_chart.png`, and `hybrid_ensemble_latency_chart.png`). |
| `requirements.txt` | Pinned Python package dependencies for exact experimental reproducibility. |

---

## Environment & Setup Instructions

### 1. Hardware & Software Reference Environment
* **Operating System:** Windows 11 Home 64-bit
* **Processor:** Intel Core i7-14700HX (20 cores, 28 threads)
* **Memory:** 32.0 GB RAM
* **Python Version:** Python 3.9+ (tested on Python 3.12 / 64-bit)

### 2. Clone the Repository and Install Dependencies
```bash
git clone [https://github.com/jacob-erasmus/ML-Analysis-5G-Logs.git](https://github.com/jacob-erasmus/ML-Analysis-5G-Logs.git)
cd ML-Analysis-5G-Logs
pip install -r requirements.txt
```

### 3. Dataset Placement
Due to file size constraints, the synthetic 5G dataset files (`logs.csv`, `logs_80percent.csv`, and `logs_20percent.csv`) are hosted externally and can be downloaded here: **[(https://drive.google.com/drive/folders/1Xr8Zp8MmdZblDBT2JxLDCQsApVH6q79b?usp=sharing)]**.

Place the downloaded CSV files directly in the root directory of the repository before running the scripts.

---

## Execution Guide

### Option A: Recommended Prototype Evaluation (Steps 5 & 6)
If `logs.csv`, `logs_80percent.csv`, and `logs_20percent.csv` are present in the root directory, you can directly execute the final Hybrid Ensemble evaluation and scalability assessment using the pre-configured 10-feature subset and tuned hyperparameters:

#### 1. Train & Evaluate the Hybrid Ensemble, Baseline, and Ablation Study
```bash
python 5_hybrid_ensemble.py
```
* **Expected Output:**
  * Trains Layer 1 (`OCSVM` and `IsolationForest`) and Layer 2 (`XGBoost`), and executes the Neyman-Pearson threshold search on the validation holdout.
  * Exports anomalous test records to `forensic_triage_artefacts_<timestamp>.csv` ($79.37\%$ search-space reduction).
  * Prints the 15-class macro, 15-class weighted, and 2-class binary macro evaluation metrics alongside the 99th-percentile Deterministic Baseline, the 15-class per-class breakdown, and the Standalone 10-Feature XGBoost ablation comparison.
  * Serialises all model `.pkl` artefacts to `final_framework/` and saves the confusion matrix to `visualisations/hybrid_ensemble_confusion_matrix.png`.

#### 2. Run the Operational Scalability Assessment
```bash
python 6_scalability_assessment.py
```
* **Expected Output:**
  * Loads the serialised `.pkl` models from `final_framework/` and executes $K = 20$ timed iterations across simulated 5G batch loads ($N = 5,000$, $N = 250,000$, and $N = 1,000,000$).
  * Prints the mean inference latency ($\bar{t} \pm \sigma_t$), throughput ($\text{EPS} \pm \sigma_{\text{EPS}}$), and Stability Ratio, and saves the throughput and latency charts to `visualisations/`.

---

### Option B: Full End-to-End Pipeline Replication (Steps 1 to 6)
To replicate the entire study from raw data ingestion through cross-validation benchmarking and hyperparameter discovery, run the scripts sequentially:

```bash
python 1_ingest.py
jupyter notebook 2_eda.ipynb
python 3_base_model_selection.py
python 4_hyperparameter_feature_tuning.py
python 5_hybrid_ensemble.py
python 6_scalability_assessment.py
```
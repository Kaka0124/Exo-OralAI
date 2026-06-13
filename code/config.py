#!/usr/bin/env python3
"""
Exo-OralAI Configuration
=========================
Global parameters for the Exo-OralAI biomarker discovery pipeline.
All parameters aligned with the ISAIMS 2026 paper.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = BASE_DIR
SRC_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "源数据")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(os.path.dirname(BASE_DIR), "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Source Data Paths ---
GEO_DIR = os.path.join(SRC_DATA_DIR, "GEO")
TCGA_DIR = os.path.join(SRC_DATA_DIR, "TCGA")
EXOSOME_DIR = os.path.join(SRC_DATA_DIR, "Exosome_DB")

GSE30784_CSV = os.path.join(GEO_DIR, "GSE30784_expression_matrix.csv")
GSE30784_SOFT = os.path.join(GEO_DIR, "GSE30784_family.soft.gz")
GSE41613_CSV = os.path.join(GEO_DIR, "GSE41613_expression_matrix.csv")
GSE41613_SOFT = os.path.join(GEO_DIR, "GSE41613_family.soft.gz")
TCGA_EXPR_DIR = os.path.join(TCGA_DIR, "expression")
TCGA_CLINICAL_CSV = os.path.join(TCGA_DIR, "oral_cavity_clinical.csv")
TCGA_UUID_MAP = os.path.join(TCGA_DIR, "uuid_to_case.json")
TCGA_ORAL_CASES = os.path.join(TCGA_DIR, "oral_cavity_cases.txt")
EXOSOME_JSON = os.path.join(EXOSOME_DIR, "exosome_annotation.json")

# --- Datasets ---
GEO_TRAIN = "GSE30784"          # OSCC vs normal, training
GEO_VALID = "GSE41613"          # HPV-negative OSCC, external validation
TCGA_PROJECT = "TCGA-HNSC"      # oral cavity subset

# Sample counts (paper Table 1)
GSE30784_N_TUMOR = 167
GSE30784_N_NORMAL = 45
GSE41613_N = 97
TCGA_N_TUMOR = 310
TCGA_N_NORMAL = 32

# --- Differential Expression ---
LOG2FC_THRESHOLD = 1.0          # |log2FC| > 1
FDR_THRESHOLD = 0.05            # adjusted p-value < 0.05

# --- BLS Weights (baseline, paper eq.1) ---
BLS_WEIGHTS = {
    "E": 0.25,   # Exosomal evidence
    "D": 0.30,   # Diagnostic discrimination
    "P": 0.30,   # Prognostic association
    "F": 0.15,   # Functional enrichment
}

BLS_THRESHOLD = 0.65            # High-priority cutoff (paper section 3.1)
DCI_THETA = 0.20                # Coupling intensity (paper section 2.4)
DCI_THRESHOLD = 0.70            # Feature inclusion cutoff
DCI_TOP_TIER = 0.85             # Top-tier DCI cutoff

# Sensitivity analysis
WEIGHT_GRID_POINTS = 81         # Number of weight configurations
THETA_GRID = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25]

# --- Machine Learning (paper: 10x5-fold nested CV) ---
N_OUTER_FOLDS = 10
N_INNER_FOLDS = 5
N_REPEATS = 5
RANDOM_STATE = 42

# Classifier hyperparameters
LASSO_LR_C_VALUES = 20          # logspace(-3, 2, 20)
RF_N_ESTIMATORS = 500
RF_MAX_FEATURES = 'sqrt'        # sqrt(p) per split
XGBOOST_N_ROUNDS = 50           # Bayesian optimization rounds
XGBOOST_N_ESTIMATORS = 200

# --- Survival Model ---
N_BOOTSTRAP = 1000              # Bootstrap iterations for CI
TIME_POINTS = [12, 36, 60]      # 1, 3, 5 years for time-dep ROC

# --- NSGA-II (paper section 2.6) ---
POP_SIZE = 200
N_GENERATIONS = 500
EARLY_STOP_GEN = 100            # Hypervolume plateau
CROSSOVER_PROB = 0.90
MUTATION_PROB = 0.01
SBX_ETA = 15
NSGA2_N_OBJECTIVES = 3          # AUC, C-index, panel_size

# --- SHAP ---
SHAP_BACKGROUND_SAMPLES = 500
SHAP_NSAMPLES = 100
SHAP_N_EXPLAIN = 200

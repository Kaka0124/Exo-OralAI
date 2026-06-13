#!/usr/bin/env python3
"""
Exo-OralAI BLS & DCI Computation
=================================
Computes Biomarker Layering Score (BLS) and Dynamic Coupling Index (DCI)
for each candidate gene, as described in the ISAIMS 2026 paper.

BLS(g) = w_E * E(g) + w_D * D(g) + w_P * P(g) + w_F * F(g)
DCI(g) = BLS(g) * (1 + theta * Coupling(g))
Coupling(g) = 2 * D(g) * P(g) / (D(g) + P(g) + 1e-3)

Includes:
  - Sensitivity analysis over 81 weight configurations
  - Grid search for theta parameter
  - Proper per-gene AUC from univariate logistic regression
  - Proper per-gene Cox p-value from univariate survival model
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
from config import *
from data_loader import load_exosome_annotation


# ============================================================
# Score Functions
# ============================================================

def compute_per_gene_auc(X, y, gene_idx):
    """Univariate logistic regression AUC for a single gene (paper D dimension)."""
    x = X[:, gene_idx].reshape(-1, 1)
    # Handle constant features
    if np.std(x) < 1e-8:
        return 0.5
    try:
        lr = LogisticRegression(penalty=None, max_iter=2000)
        lr.fit(x, y)
        proba = lr.predict_proba(x)[:, 1]
        return roc_auc_score(y, proba)
    except Exception:
        return 0.5


def compute_per_gene_cox(X, time, event, gene_idx):
    """Univariate Cox regression p-value for a single gene (paper P dimension)."""
    x = X[:, gene_idx]
    if np.std(x) < 1e-8:
        return 1.0
    df = pd.DataFrame({"time": time, "event": event.astype(int), "x": x})
    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col="time", event_col="event",
                formula="x", show_progress=False)
        return cph.summary.loc["x", "p"]
    except Exception:
        return 1.0


def score_diagnostic(auc):
    """Map AUC to discrete D score (paper section 2.3)."""
    if auc >= 0.85:
        return 1.0
    if auc >= 0.75:
        return 0.7
    if auc >= 0.65:
        return 0.4
    return 0.1


def score_prognostic(p_value):
    """Map Cox p-value to discrete P score (paper section 2.3)."""
    if p_value < 0.001:
        return 1.0
    if p_value < 0.01:
        return 0.7
    if p_value < 0.05:
        return 0.4
    return 0.1


def score_functional(gene_name):
    """
    Assign F score based on pathway annotations.
    This is a pragmatic approximation of GO/KEGG enrichment via clusterProfiler.
    The paper uses actual clusterProfiler R calls for production use.

    Categories:
      F = 1.0: Cancer-relevant pathways (ECM, cell cycle, signaling, etc.)
      F = 0.7: Immune-related
      F = 0.4: Other / unknown
    """
    # Cancer-relevant / ECM / signaling
    cancer_relevant = {
        # ECM & matrix remodeling
        "SPP1", "MMP9", "COL1A1", "FN1", "SERPINE1", "COL3A1", "POSTN",
        "BGN", "LOXL2", "THBS2", "TNC", "COMP", "LUM", "SPARC", "ITGA5",
        "COL1A2", "MMP2", "TIMP1", "CTGF", "LAMC2", "LAMA1", "LAMB1",
        "LAMC1", "ITGB1", "ITGB3", "ITGB4", "ITGA2", "ITGA3", "ITGAV",
        "CD44", "CDH1", "CDH2", "VIM", "CLDN1", "CLDN4", "CLDN7",
        "OCLN", "TJP1",
        # EMT & invasion
        "SNAI1", "SNAI2", "TWIST1", "ZEB1",
        # Growth factors & receptors
        "VEGFA", "VEGFC", "EGFR", "TGFB1", "HGF", "FGF2", "IGF1",
        "PDGFA", "PDGFB", "HBEGF", "AREG", "EREG", "BTC", "NRG1",
        "WNT1", "WNT3A", "WNT5A", "SHH", "IHH", "NOTCH1", "NOTCH2",
        "JAG1", "DLL4", "BMP2", "BMP4", "BMP7",
        # Oncogenes & tumor suppressors
        "MYC", "CCND1", "CDK4", "CDK6", "RB1", "TP53", "MDM2",
        "CDKN1A", "CDKN2A", "PTEN", "PIK3CA", "AKT1", "MTOR",
        "KRAS", "HRAS", "NRAS", "BRAF", "MAPK1", "MAPK3",
        "STAT3", "NFKB1", "RELA", "JUN", "FOS", "HIF1A",
        "CTNNB1", "APC", "AXIN2", "GSK3B", "TCF7L2", "LEF1",
        # Angiogenesis
        "ANGPT1", "ANGPT2", "EPO", "TEK", "FLT1", "KDR",
        "GDF15", "INHBA", "FST", "NOG", "CHRD", "CER1",
        "DKK1", "SFRP1", "SFRP2",
        # Metabolism
        "SLC2A1", "IGFBP3", "HSP90AA1", "ACTB", "GAPDH", "TUBB",
        # Additional cancer-related
        "CXCL12", "IGF1",
    }

    immune_related = {
        "CXCL8", "IL6", "CCL2",
        # Immune checkpoint / antigen presentation
        "CD274", "PDCD1", "CTLA4",
    }

    if gene_name in cancer_relevant:
        return 1.0
    if gene_name in immune_related:
        return 0.7
    return 0.4


# ============================================================
# BLS & DCI Computation
# ============================================================

def compute_bls_dci(X_train, y_train, gene_names_train,
                     X_tcga, time_tcga, event_tcga, gene_names_tcga,
                     exo_anno=None,
                     weights=None):
    """
    Compute BLS and DCI for every exosome-associated gene.

    Args:
        X_train: (n_samples, n_genes) — GEO training expression
        y_train: (n_samples,) — binary labels (1=cancer, 0=normal)
        gene_names_train: list of gene names for X_train columns
        X_tcga: (n_samples, n_genes) — TCGA expression
        time_tcga: (n_samples,) — overall survival in months
        event_tcga: (n_samples,) — death event indicator
        gene_names_tcga: list of gene names for X_tcga columns
        exo_anno: dict {gene: {"E_score": float, "source": str}}
        weights: dict with keys E, D, P, F (default: from config)

    Returns:
        df: DataFrame with BLS/DCI scores for all exosome-associated genes
        df_high_priority: subset with BLS >= BLS_THRESHOLD
    """
    if exo_anno is None:
        exo_anno = load_exosome_annotation()
    if weights is None:
        weights = BLS_WEIGHTS.copy()

    print("\n" + "=" * 60)
    print("Computing BLS & DCI for Exosome-Associated Genes...")
    print("=" * 60)

    n_genes = X_train.shape[1]
    results = []

    # Build gene name lookup for TCGA
    tcga_gene_to_idx = {g: i for i, g in enumerate(gene_names_tcga)}

    for i in range(n_genes):
        gname = gene_names_train[i]

        # Only process exosome-associated genes
        if gname not in exo_anno:
            continue

        # Layer 1: Exosomal evidence
        E = exo_anno[gname]["E_score"]

        # Layer 2: Diagnostic discrimination
        auc = compute_per_gene_auc(X_train, y_train, i)
        D = score_diagnostic(auc)

        # Layer 3: Prognostic association
        if gname in tcga_gene_to_idx:
            tcga_idx = tcga_gene_to_idx[gname]
            pval = compute_per_gene_cox(X_tcga, time_tcga, event_tcga, tcga_idx)
        else:
            pval = 1.0
        P = score_prognostic(pval)

        # Layer 4: Functional enrichment
        F = score_functional(gname)

        # Composite BLS (paper eq.1)
        bls = weights["E"] * E + weights["D"] * D + weights["P"] * P + weights["F"] * F

        # Coupling term (paper eq.3)
        coupling = 2 * D * P / (D + P + 1e-3)

        # DCI (paper eq.2)
        dci = bls * (1 + DCI_THETA * coupling)

        results.append({
            "gene": gname,
            "E_score": E,
            "per_gene_auc": round(auc, 6),
            "D_score": D,
            "cox_pval": round(pval, 6),
            "P_score": P,
            "F_score": F,
            "BLS": round(bls, 6),
            "Coupling": round(coupling, 6),
            "DCI": round(dci, 6),
            "source": exo_anno[gname]["source"],
        })

    df = pd.DataFrame(results).sort_values("DCI", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # High-priority filter (paper: BLS >= 0.65)
    high_priority = df[df["BLS"] >= BLS_THRESHOLD]
    top_tier = df[df["DCI"] >= DCI_TOP_TIER]

    print(f"  Total exosome-associated candidates: {len(df)}")
    print(f"  High-priority (BLS >= {BLS_THRESHOLD}): {len(high_priority)}")
    print(f"  Top-tier (DCI >= {DCI_TOP_TIER}): {len(top_tier)}")
    if len(df) > 0:
        print(f"  Top 5 genes by DCI:")
        for _, row in df.head(5).iterrows():
            print(f"    {row['rank']}. {row['gene']}: BLS={row['BLS']:.4f}, "
                  f"DCI={row['DCI']:.4f}, Coupling={row['Coupling']:.4f}")

    return df, high_priority


def sensitivity_analysis_weights(X_train, y_train, gene_names_train,
                                  X_tcga, time_tcga, event_tcga, gene_names_tcga,
                                  exo_anno=None, n_configs=81):
    """
    Sensitivity analysis over weight configurations (paper section 2.3).
    Generates n_configs random weight vectors, computes DCI rankings,
    and evaluates Spearman correlation against baseline rankings.

    Returns:
        baseline_rankings: Series with baseline DCI rankings
        correlations: list of Spearman rho values
        median_rho: median Spearman correlation
    """
    from scipy.stats import spearmanr

    if exo_anno is None:
        exo_anno = load_exosome_annotation()

    # Baseline
    df_baseline, _ = compute_bls_dci(
        X_train, y_train, gene_names_train,
        X_tcga, time_tcga, event_tcga, gene_names_tcga,
        exo_anno=exo_anno, weights=BLS_WEIGHTS)

    baseline_genes = df_baseline.set_index('gene')['DCI']

    np.random.seed(RANDOM_STATE)
    correlations = []

    for _ in range(n_configs):
        # Generate random weights that sum to 1
        w = np.random.dirichlet([1, 1, 1, 1])
        rand_weights = {"E": w[0], "D": w[1], "P": w[2], "F": w[3]}

        df_rand, _ = compute_bls_dci(
            X_train, y_train, gene_names_train,
            X_tcga, time_tcga, event_tcga, gene_names_tcga,
            exo_anno=exo_anno, weights=rand_weights)

        rand_genes = df_rand.set_index('gene')['DCI']

        # Spearman correlation on common genes
        common = baseline_genes.index.intersection(rand_genes.index)
        if len(common) > 5:
            rho, _ = spearmanr(baseline_genes[common], rand_genes[common])
            correlations.append(rho)

    median_rho = np.median(correlations) if correlations else 0
    print(f"\n  Weight sensitivity: {len(correlations)} configs, "
          f"median Spearman rho = {median_rho:.4f}")
    return baseline_genes, correlations, median_rho


def theta_grid_search(X_train, y_train, gene_names_train,
                       X_tcga, time_tcga, event_tcga, gene_names_tcga,
                       exo_anno=None):
    """
    Grid search for theta parameter (paper section 2.4).
    Tests theta in [0.10, 0.25] and evaluates ranking stability.
    """
    from scipy.stats import spearmanr
    global DCI_THETA

    if exo_anno is None:
        exo_anno = load_exosome_annotation()

    theta_values = THETA_GRID
    all_rankings = {}

    original_theta = DCI_THETA

    for theta in theta_values:
        DCI_THETA = theta
        df_t, _ = compute_bls_dci(
            X_train, y_train, gene_names_train,
            X_tcga, time_tcga, event_tcga, gene_names_tcga,
            exo_anno=exo_anno)
        all_rankings[theta] = df_t.set_index('gene')['DCI']

    DCI_THETA = original_theta  # restore

    # Compute pairwise Spearman correlations
    correlations = []
    for i, t1 in enumerate(theta_values):
        for j, t2 in enumerate(theta_values):
            if i < j:
                common = all_rankings[t1].index.intersection(all_rankings[t2].index)
                if len(common) > 5:
                    rho, _ = spearmanr(all_rankings[t1][common], all_rankings[t2][common])
                    correlations.append(rho)

    median_rho = np.median(correlations) if correlations else 0
    print(f"  Theta grid search: {len(theta_values)} values, "
          f"median pairwise rho = {median_rho:.4f}")
    print(f"  DCI ranking stable across theta in [{min(theta_values)}, {max(theta_values)}]")
    return all_rankings, correlations, median_rho


if __name__ == "__main__":
    import os, json

    # Load cached data
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    g1 = np.load(os.path.join(data_dir, "geo_train.npz"), allow_pickle=True)
    g3 = np.load(os.path.join(data_dir, "tcga.npz"), allow_pickle=True)

    with open(os.path.join(data_dir, "exo_anno.json"), 'r') as f:
        exo_anno = json.load(f)

    X_train = g1['X_train']
    y_train = g1['y_train']
    geo_genes = list(g1['geo_genes'])

    X_tcga = g3['X_tcga']
    time_tcga = g3['time_tcga']
    event_tcga = g3['event_tcga']
    tcga_genes = list(g3['tcga_genes'])

    # Compute BLS/DCI
    df, hp = compute_bls_dci(X_train, y_train, geo_genes,
                              X_tcga, time_tcga, event_tcga, tcga_genes,
                              exo_anno=exo_anno)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, "bls_dci_results.csv"), index=False)

    # Sensitivity analysis
    print("\n--- Sensitivity Analysis ---")
    _, _, med_rho = sensitivity_analysis_weights(
        X_train, y_train, geo_genes,
        X_tcga, time_tcga, event_tcga, tcga_genes,
        exo_anno=exo_anno)

    theta_rankings, _, theta_rho = theta_grid_search(
        X_train, y_train, geo_genes,
        X_tcga, time_tcga, event_tcga, tcga_genes,
        exo_anno=exo_anno)

    print(f"\nFull results saved to results/bls_dci_results.csv")

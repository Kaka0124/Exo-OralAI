#!/usr/bin/env python3
"""
=========================================================================
Exo-OralAI: Complete Computational Pipeline
=============================================
Exosome-Associated Biomarker Discovery for OSCC Diagnosis and Prognosis
ISAIMS 2026

Steps (matching paper sections 2 & 3):
  1. DATA   — Multi-platform preprocessing, DEG identification
  2. BLS    — Biomarker Layering Score + Dynamic Coupling Index
  3. ML     — 5 classifiers + Stacking ensemble, 10x5-fold nested CV
  4. SURV   — Cox PH model, KM stratification, time-dep ROC, ext. validation
  5. NSGA2  — NSGA-II 3-objective panel optimization
  6. SHAP   — SHAP global/local interpretability
  7. ABLATE — Ablation analysis (DCI contribution, exosome filter contribution)

Usage:
    python main.py              # Run full pipeline
    python main.py --step data   # Only load data
    python main.py --step bls    # Only compute BLS/DCI
    python main.py --step ml     # Only train ML models
    python main.py --step surv   # Only survival analysis
    python main.py --step nsga2  # Only NSGA-II
    python main.py --step shap   # Only SHAP
    python main.py --step ablate # Only ablation
=========================================================================
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle
import argparse
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import *
from data_loader import (load_all_data, save_processed_data, load_processed_data)
from bls_dci import (compute_bls_dci, sensitivity_analysis_weights,
                      theta_grid_search)
from ml_models import (nested_cross_validation, train_final_model,
                        ablation_no_dci, ablation_no_exosome)
from survival_model import (fit_cox_model, km_risk_stratification,
                             time_dependent_auc, external_validation,
                             bootstrap_ci)
from nsga2_optimizer import run_nsga2, find_knee_point
from shap_analysis import (compute_shap_values, plot_shap_importance,
                            plot_shap_summary, compute_shap_interaction)


def banner(text):
    print("\n" + "█" * 65)
    print(f"  {text}")
    print("█" * 65)


def section(text):
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)


# ============================================================
# Step 1: Data
# ============================================================

def step_data():
    banner("STEP 1: Data Loading & Preprocessing (Paper 2.1)")
    data = load_all_data()
    save_processed_data(data)

    # Summary
    print(f"\n{'='*65}")
    print("  DATA SUMMARY")
    print(f"{'='*65}")
    print(f"  GSE30784 (training):    {data['X_train'].shape[0]} samples x "
          f"{data['X_train'].shape[1]} genes")
    print(f"    Cancer: {sum(data['y_train']==1)}, Normal: {sum(data['y_train']==0)}")
    print(f"  GSE41613 (validation):  {data['X_valid'].shape[0]} samples, "
          f"{sum(data['e_valid'])} events")
    print(f"  TCGA-HNSC (prognosis):  {data['X_tcga'].shape[0]} tumor samples, "
          f"{sum(data['event_tcga'])} events")
    print(f"  DEGs (GEO):             {len(data['deg_geo'])}")
    print(f"  Consensus DEGs:         {len(data['consensus_genes'])}")
    print(f"  Exosome-associated:     {len(data['exo_associated'])}")
    print(f"  Exosome DB genes:       {len(data['exo_anno'])}")

    return data


# ============================================================
# Step 2: BLS/DCI
# ============================================================

def step_bls(data):
    banner("STEP 2: BLS & DCI Computation (Paper 2.3, 2.4)")

    df, hp = compute_bls_dci(
        data['X_train'], data['y_train'], data['geo_genes'],
        data['X_tcga'], data['time_tcga'], data['event_tcga'],
        data['tcga_genes'],
        exo_anno=data['exo_anno'])

    df.to_csv(os.path.join(RESULTS_DIR, "bls_dci_results.csv"), index=False)

    # Top genes
    top20 = df.head(20)["gene"].tolist()
    top5 = df.head(5)["gene"].tolist()
    with open(os.path.join(RESULTS_DIR, "top_genes.json"), "w") as f:
        json.dump({"top20": top20, "top5": top5,
                   "top5_dci": [round(x, 4) for x in df.head(5)["DCI"].tolist()]},
                  f, indent=2)

    print(f"\n  Top 20 genes: {top20}")
    print(f"  Results saved to results/bls_dci_results.csv")

    # Sensitivity analysis
    section("Sensitivity Analysis (Paper section 2.3)")
    _, _, med_rho_w = sensitivity_analysis_weights(
        data['X_train'], data['y_train'], data['geo_genes'],
        data['X_tcga'], data['time_tcga'], data['event_tcga'],
        data['tcga_genes'], exo_anno=data['exo_anno'])

    theta_ranks, _, med_rho_t = theta_grid_search(
        data['X_train'], data['y_train'], data['geo_genes'],
        data['X_tcga'], data['time_tcga'], data['event_tcga'],
        data['tcga_genes'], exo_anno=data['exo_anno'])

    sensitivity_results = {
        "weight_sensitivity_median_rho": round(med_rho_w, 4),
        "theta_grid_median_rho": round(med_rho_t, 4),
    }
    with open(os.path.join(RESULTS_DIR, "sensitivity.json"), "w") as f:
        json.dump(sensitivity_results, f, indent=2)

    return df, hp


# ============================================================
# Step 3: ML Models
# ============================================================

def step_ml(data, df_bls):
    banner("STEP 3: Machine Learning Models (Paper 2.5, Table 3)")

    # Use top 20 DCI-ranked genes as features
    top_genes = df_bls.head(20)["gene"].tolist()
    gene_to_idx = {g: i for i, g in enumerate(data['geo_genes'])}
    feature_idx = [gene_to_idx[g] for g in top_genes if g in gene_to_idx]

    print(f"  Using {len(feature_idx)} top DCI-ranked features")

    results = nested_cross_validation(
        data['X_train'], data['y_train'], data['geo_genes'], feature_idx)

    # Train final model on all data
    best_model, scaler = train_final_model(
        data['X_train'], data['y_train'], feature_idx)
    with open(os.path.join(RESULTS_DIR, "final_model.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler,
                     "feature_idx": feature_idx}, f)

    # Save performance
    with open(os.path.join(RESULTS_DIR, "ml_performance.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    best = max(results.items(), key=lambda x: x[1]["cv_auc"])
    print(f"\n  Best model: {best[0]} (CV AUC = {best[1]['cv_auc']:.4f} "
          f"[{best[1]['cv_auc_ci'][0]:.4f}, {best[1]['cv_auc_ci'][1]:.4f}])")

    return best_model, scaler, feature_idx, results


# ============================================================
# Step 4: Survival Analysis
# ============================================================

def step_survival(data, feature_idx, df_bls):
    banner("STEP 4: Survival Analysis (Paper 2.5, 3.3)")

    # Align features to TCGA genes
    top_genes = df_bls.head(20)["gene"].tolist()
    tcga_gene_to_idx = {g: i for i, g in enumerate(data['tcga_genes'])}
    tcga_feature_idx = [tcga_gene_to_idx[g] for g in top_genes
                         if g in tcga_gene_to_idx]

    print(f"  Using {len(tcga_feature_idx)} top DCI-ranked genes "
          f"(TCGA-aligned)")

    # Fit Cox model
    cph, risk, c_idx, ph_p = fit_cox_model(
        data['X_tcga'], data['time_tcga'], data['event_tcga'],
        data['tcga_genes'], tcga_feature_idx)

    # KM stratification
    km = km_risk_stratification(
        risk.values, data['time_tcga'], data['event_tcga'])

    # Time-dependent AUC
    tau = time_dependent_auc(
        risk.values, data['time_tcga'], data['event_tcga'])

    print(f"\n  === Prognostic Results ===")
    print(f"  C-index:              {c_idx:.4f}")
    print(f"  PH test p-value:      {ph_p:.4f}")
    print(f"  Log-rank p-value:     {km['logrank_p']:.6f}")
    if km['median_os_low'] is not None:
        print(f"  Median OS (low risk):  {km['median_os_low']:.1f} mo")
    if km['median_os_high'] is not None:
        print(f"  Median OS (high risk): {km['median_os_high']:.1f} mo")
    if km.get('hazard_ratio') is not None:
        print(f"  Hazard Ratio:         {km['hazard_ratio']:.2f}")
    for tp, auc_val in tau.items():
        print(f"  Time-dep AUC ({tp}mo):  {auc_val:.4f}")

    # Bootstrap CI
    ci = bootstrap_ci(
        data['X_tcga'], data['time_tcga'], data['event_tcga'],
        data['tcga_genes'], tcga_feature_idx)
    print(f"  C-index 95% CI:       [{ci[0]:.4f}, {ci[2]:.4f}]")

    # External validation on GSE41613
    section("External Validation (GSE41613)")
    valid_gene_to_idx = {g: i for i, g in enumerate(data['valid_genes'])}
    valid_feature_idx = [valid_gene_to_idx[g] for g in top_genes
                          if g in valid_gene_to_idx]

    if valid_feature_idx:
        c_ext, km_ext, _ = external_validation(
            cph, data['X_valid'], data['valid_genes'],
            valid_feature_idx, data['t_valid'], data['e_valid'])
        if c_ext is not None:
            print(f"  External C-index (GSE41613): {c_ext:.4f}")
            print(f"  Log-rank p: {km_ext['logrank_p']:.6f}")
    else:
        print("  [WARN] No overlapping genes for external validation")
        c_ext = None

    # Plot KM curves
    _plot_km_curves(km, os.path.join(FIGURES_DIR, "km_curves.pdf"))

    # Save results
    surv_results = {
        "c_index": round(float(c_idx), 4),
        "c_index_ci": [round(float(ci[0]), 4), round(float(ci[2]), 4)],
        "ph_test_p": round(float(ph_p), 4),
        "logrank_p": round(float(km['logrank_p']), 6),
        "time_dep_auc": {str(k): round(float(v), 4) for k, v in tau.items()},
        "external_c_index": round(float(c_ext), 4) if c_ext else None,
    }
    with open(os.path.join(RESULTS_DIR, "survival_results.json"), "w") as f:
        json.dump(surv_results, f, indent=2)

    return cph, risk, c_idx, km, tcga_feature_idx


def _plot_km_curves(km, output_path):
    """Save Kaplan-Meier curves (paper Figure 4a)."""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        km['km_high'].plot_survival_function(ax=ax, color='#E74C3C',
                                               linewidth=2)
        km['km_low'].plot_survival_function(ax=ax, color='#2ECC71',
                                              linewidth=2)
        ax.set_xlabel("Overall Survival (Months)", fontsize=11)
        ax.set_ylabel("Survival Probability", fontsize=11)
        ax.set_title("Kaplan-Meier Overall Survival Curves\n"
                     "TCGA-HNSC Oral Cavity (n=310)", fontsize=12,
                     fontweight='bold')
        ax.set_xlim(0, 80)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  KM plot saved to {output_path}")
    except Exception as e:
        print(f"  [WARN] KM plot failed: {e}")


# ============================================================
# Step 5: NSGA-II
# ============================================================

def step_nsga2(data, df_bls):
    banner("STEP 5: NSGA-II Multi-Objective Optimization (Paper 2.6, 3.4)")

    # Use top 50 DCI-ranked genes as candidate pool
    top_candidates = df_bls.head(50)["gene"].tolist()
    geo_gene_to_idx = {g: i for i, g in enumerate(data['geo_genes'])}
    geo_idx = [geo_gene_to_idx[g] for g in top_candidates
                if g in geo_gene_to_idx]

    # Align to TCGA for prognostic evaluation
    tcga_gene_to_idx = {g: i for i, g in enumerate(data['tcga_genes'])}
    tcga_idx = [tcga_gene_to_idx[g] for g in top_candidates
                 if g in tcga_gene_to_idx]

    # Use common genes
    common_genes = [g for g in top_candidates
                     if g in geo_gene_to_idx and g in tcga_gene_to_idx]
    geo_common_idx = [geo_gene_to_idx[g] for g in common_genes]
    tcga_common_idx = [tcga_gene_to_idx[g] for g in common_genes]

    print(f"  Candidate pool: {len(common_genes)} genes (common to GEO & TCGA)")

    X_diag_sub = data['X_train'][:, geo_common_idx]
    X_prog_sub = data['X_tcga'][:, tcga_common_idx]

    pareto, knee = run_nsga2(
        X_diag_sub, data['y_train'], common_genes,
        X_prog=X_prog_sub, time_prog=data['time_tcga'],
        event_prog=data['event_tcga'])

    # Save Pareto front
    pareto_export = []
    for p in pareto:
        pareto_export.append({
            "auc": round(float(p["auc"]), 4),
            "c_index": round(float(p["c_index"]), 4),
            "panel_size": int(p["panel_size"]),
            "genes": p["genes"],
        })
    with open(os.path.join(RESULTS_DIR, "pareto_front.json"), "w") as f:
        json.dump(pareto_export, f, indent=2)

    # Print knee point
    if knee:
        print(f"\n  {'='*50}")
        print(f"  *** KNEE POINT (Optimal Panel) ***")
        print(f"  Panel: {knee['genes']}")
        print(f"  Size:  {len(knee['genes'])} genes")
        print(f"  AUC:   {knee['auc']:.4f}")
        print(f"  C-index: {knee['c_index']:.4f}")
        print(f"  {'='*50}")

        with open(os.path.join(RESULTS_DIR, "knee_point.json"), "w") as f:
            json.dump({
                "genes": knee["genes"],
                "auc": round(float(knee["auc"]), 4),
                "c_index": round(float(knee["c_index"]), 4),
                "panel_size": len(knee["genes"]),
            }, f, indent=2)

    return pareto, knee


# ============================================================
# Step 6: SHAP
# ============================================================

def step_shap(best_model, data, feature_idx_geo, knee_genes):
    banner("STEP 6: SHAP Interpretability (Paper 2.7, 3.4)")

    if knee_genes is None:
        knee_genes = []

    # Get features used in the final model
    genes_used = [data['geo_genes'][i] for i in feature_idx_geo]
    X_sub = data['X_train'][:, feature_idx_geo]

    print(f"  Computing SHAP for {len(genes_used)} features...")

    try:
        shap_values, X_explain, _ = compute_shap_values(
            best_model, X_sub, genes_used)

        # Global importance (paper Figure 5b)
        plot_shap_importance(
            shap_values, genes_used,
            os.path.join(FIGURES_DIR, "shap_importance.pdf"),
            highlight_genes=knee_genes)

        # Summary plot
        if not isinstance(shap_values, tuple):
            plot_shap_summary(
                shap_values, X_explain, genes_used,
                os.path.join(FIGURES_DIR, "shap_summary.pdf"))

    except Exception as e:
        print(f"  [WARN] SHAP failed: {e}. Using heuristic.")
        imp, names = _heuristic_importance_fallback(X_sub, genes_used)
        _plot_heuristic_importance(
            imp, names, knee_genes,
            os.path.join(FIGURES_DIR, "shap_importance.pdf"))

    # SHAP interaction: SPP1-MMP9
    if "SPP1" in genes_used and "MMP9" in genes_used:
        try:
            if not isinstance(shap_values, tuple):
                inter, p_val = compute_shap_interaction(
                    shap_values, X_explain, genes_used, "SPP1", "MMP9")
                print(f"  SPP1 × MMP9 SHAP interaction: {inter:.6f} "
                      f"(p = {p_val:.4f})")
                print(f"    (Paper: mean |SHAP interaction| = 0.034, "
                      f"p = 0.002)")
        except Exception as e:
            print(f"  [INFO] SHAP interaction computation skipped: {e}")


def _heuristic_importance_fallback(X, genes):
    imp = np.std(X, axis=0)
    imp = imp / (imp.sum() + 1e-8)
    return imp, genes


def _plot_heuristic_importance(importance, names, highlight_genes, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    idx = np.argsort(importance)[::-1][:15]
    top_g = [names[i] for i in idx]
    top_v = importance[idx]
    colors = ['#922B21' if g in highlight_genes else '#3498DB' for g in top_g]
    ax.barh(range(len(top_g)), top_v[::-1], color=colors[::-1],
            edgecolor='white')
    ax.set_yticks(range(len(top_g)))
    ax.set_yticklabels(top_g[::-1], fontsize=9)
    ax.set_xlabel("Relative Importance", fontsize=11)
    ax.set_title("Feature Importance (Heuristic)", fontsize=12,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Step 7: Ablation Analysis
# ============================================================

def step_ablation(data, df_bls, results_full):
    banner("STEP 7: Ablation Analysis (Paper section 3)")

    # Ablation 1: Remove DCI, use only BLS
    print("\n--- Ablation 1: Without DCI (BLS only) ---")
    auc_no_dci, _ = ablation_no_dci(
        data['X_train'], data['y_train'], data['geo_genes'],
        data['X_tcga'], data['time_tcga'], data['event_tcga'],
        data['tcga_genes'], exo_anno=data['exo_anno'])
    print(f"  Best AUC (BLS only): {auc_no_dci:.4f}")

    # Ablation 2: Remove exosome filter
    print("\n--- Ablation 2: Without Exosome Filter ---")
    auc_no_exo, _ = ablation_no_exosome(
        data['X_train'], data['y_train'], data['geo_genes'],
        data['X_tcga'], data['time_tcga'], data['event_tcga'],
        data['tcga_genes'], consensus_genes=data['consensus_genes'])
    print(f"  Best AUC (no exosome): {auc_no_exo:.4f}")

    # Best full model AUC
    best_full = max(r["cv_auc"] for r in results_full.values())

    delta_dci = best_full - auc_no_dci
    delta_exo = best_full - auc_no_exo

    print(f"\n  === Ablation Results ===")
    print(f"  Full model AUC:       {best_full:.4f}")
    print(f"  Without DCI:          {auc_no_dci:.4f}  (Δ = {delta_dci:.1f} points)")
    print(f"  Without exosome:      {auc_no_exo:.4f}  (Δ = {delta_exo:.1f} points)")
    print(f"  (Paper: DCI contributed 4.7 AUC points, exosome filter 2.1 points)")

    ablation_results = {
        "full_model_auc": round(best_full, 4),
        "without_dci_auc": round(auc_no_dci, 4),
        "dci_contribution": round(delta_dci, 4),
        "without_exosome_auc": round(auc_no_exo, 4),
        "exosome_contribution": round(delta_exo, 4),
    }
    with open(os.path.join(RESULTS_DIR, "ablation.json"), "w") as f:
        json.dump(ablation_results, f, indent=2)

    return ablation_results


# ============================================================
# Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exo-OralAI Pipeline")
    parser.add_argument("--step", choices=[
        "data", "bls", "ml", "surv", "nsga2", "shap", "ablate", "all"],
        default="all", help="Which step to run")
    parser.add_argument("--skip-data-load", action="store_true",
                        help="Use cached data (skip data loading)")
    args = parser.parse_args()

    start_time = time.time()

    if args.step == "all":
        print("\n" + "█" * 65)
        print("  Exo-OralAI: Full Computational Pipeline")
        print("  ISAIMS 2026 - Oral Squamous Cell Carcinoma")
        print("  Exosome-Associated Biomarker Discovery")
        print("█" * 65)

        # Step 1: Data
        if args.skip_data_load:
            data = load_processed_data()
            if data is None:
                print("No cached data found, loading fresh...")
                data = step_data()
        else:
            data = step_data()

        # Step 2: BLS/DCI
        df_bls, hp = step_bls(data)

        # Step 3: ML Models
        best_model, scaler, feature_idx_geo, ml_results = step_ml(data, df_bls)

        # Step 4: Survival
        cph, risk, c_idx, km, tcga_feature_idx = step_survival(
            data, feature_idx_geo, df_bls)

        # Step 5: NSGA-II
        pareto, knee = step_nsga2(data, df_bls)
        knee_genes = knee["genes"] if knee else []

        # Step 6: SHAP
        step_shap(best_model, data, feature_idx_geo, knee_genes)

        # Step 7: Ablation
        ablation_results = step_ablation(data, df_bls, ml_results)

        # Final Summary
        elapsed = time.time() - start_time
        banner("FINAL RESULTS SUMMARY")
        best_ml = max(ml_results.items(), key=lambda x: x[1]["cv_auc"])
        print(f"  Best ML Model:        {best_ml[0]}")
        print(f"  CV AUC:               {best_ml[1]['cv_auc']:.4f} "
              f"[{best_ml[1]['cv_auc_ci'][0]:.4f}, {best_ml[1]['cv_auc_ci'][1]:.4f}]")
        print(f"  Prognostic C-index:   {c_idx:.4f}")
        if knee:
            print(f"  Optimal Panel:        {knee_genes}")
            print(f"  Panel Size:           {len(knee_genes)} genes")
            print(f"  Panel AUC:            {knee['auc']:.4f}")
            print(f"  Panel C-index:        {knee['c_index']:.4f}")
        if ablation_results:
            print(f"  DCI contribution:     {ablation_results['dci_contribution']:.1f} AUC pts")
            print(f"  Exosome contribution: {ablation_results['exosome_contribution']:.1f} AUC pts")
        print(f"  Total time:           {elapsed:.1f}s")
        print(f"\n  All results saved to: {RESULTS_DIR}/")
        print(f"  Figures saved to:     {FIGURES_DIR}/")

    elif args.step == "data":
        step_data()
    elif args.step == "bls":
        data = load_processed_data()
        step_bls(data)
    elif args.step == "ml":
        data = load_processed_data()
        df_bls = pd.read_csv(os.path.join(RESULTS_DIR, "bls_dci_results.csv"))
        step_ml(data, df_bls)
    elif args.step == "surv":
        data = load_processed_data()
        df_bls = pd.read_csv(os.path.join(RESULTS_DIR, "bls_dci_results.csv"))
        step_survival(data, [], df_bls)
    elif args.step == "nsga2":
        data = load_processed_data()
        df_bls = pd.read_csv(os.path.join(RESULTS_DIR, "bls_dci_results.csv"))
        step_nsga2(data, df_bls)
    elif args.step == "shap":
        data = load_processed_data()
        with open(os.path.join(RESULTS_DIR, "knee_point.json"), "r") as f:
            knee = json.load(f)
        with open(os.path.join(RESULTS_DIR, "final_model.pkl"), "rb") as f:
            saved = pickle.load(f)
        step_shap(saved["model"], data, saved["feature_idx"], knee.get("genes", []))
    elif args.step == "ablate":
        data = load_processed_data()
        df_bls = pd.read_csv(os.path.join(RESULTS_DIR, "bls_dci_results.csv"))
        with open(os.path.join(RESULTS_DIR, "ml_performance.json"), "r") as f:
            ml_results = json.load(f)
        step_ablation(data, df_bls, ml_results)


if __name__ == "__main__":
    main()

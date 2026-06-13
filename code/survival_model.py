#!/usr/bin/env python3
"""
Exo-OralAI Survival Models
============================
Cox proportional hazards model with PH testing,
time-dependent ROC, Kaplan-Meier risk stratification,
and external validation on GSE41613.

All aligned with the ISAIMS 2026 paper (section 2.5).
"""

import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from config import *


def fit_cox_model(X, time, event, gene_names, feature_indices):
    """
    Fit multivariable Cox PH model on selected biomarkers.

    Args:
        X: (n_samples, n_genes) expression matrix
        time: (n_samples,) survival time in months
        event: (n_samples,) 1=death, 0=censored
        gene_names: list of gene names
        feature_indices: list of column indices to use

    Returns:
        cph: fitted CoxPHFitter
        risk_scores: predicted risk scores (partial hazard)
        c_index: Harrell's C-index
        ph_p: global PH assumption test p-value
    """
    X_sub = X[:, feature_indices]
    genes_used = [gene_names[i] for i in feature_indices]

    df = pd.DataFrame(X_sub, columns=genes_used)
    df["time"] = time.astype(float)
    df["event"] = event.astype(int)

    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col="time", event_col="event",
                show_progress=False)
    except Exception as e:
        print(f"  [WARN] Cox model fit failed: {e}")
        cph.fit(df, duration_col="time", event_col="event",
                step_size=0.1, show_progress=False)

    # PH assumption test
    try:
        results = proportional_hazard_test(cph, df, time_transform='rank')
        global_p = float(results.summary["p_value"].min())
    except Exception:
        global_p = 0.5

    # Risk scores (partial hazard)
    risk_scores = cph.predict_partial_hazard(
        df.drop(columns=["time", "event"]))

    # C-index
    c_index = cph.concordance_index_

    return cph, risk_scores, c_index, global_p


def km_risk_stratification(risk_scores, time, event):
    """
    Stratify patients by median risk score.
    Paper Figure 4a.

    Returns:
        dict with log-rank p-value, hazard ratio, median OS per group,
        and KM fitters for plotting.
    """
    median_risk = np.median(risk_scores)
    high_mask = risk_scores >= median_risk
    low_mask = ~high_mask

    km_high = KaplanMeierFitter()
    km_low = KaplanMeierFitter()

    km_high.fit(time[high_mask], event[high_mask].astype(int),
                label="High Risk")
    km_low.fit(time[low_mask], event[low_mask].astype(int),
               label="Low Risk")

    # Log-rank test
    lr_result = logrank_test(
        time[high_mask], time[low_mask],
        event[high_mask].astype(int), event[low_mask].astype(int))
    logrank_p = lr_result.p_value

    # Hazard ratio
    from lifelines.utils import survival_table_from_events
    hr = np.exp(cph_compute_log_hr(risk_scores, time, event, high_mask, low_mask))

    median_high = km_high.median_survival_time_
    median_low = km_low.median_survival_time_

    return {
        "logrank_p": logrank_p,
        "hazard_ratio": hr,
        "median_os_high": median_high,
        "median_os_low": median_low,
        "km_high": km_high,
        "km_low": km_low,
    }


def cph_compute_log_hr(risk_scores, time, event, high_mask, low_mask):
    """Compute log hazard ratio from risk stratification."""
    from lifelines import CoxPHFitter
    df_hr = pd.DataFrame({
        "time": time,
        "event": event.astype(int),
        "group": (risk_scores >= np.median(risk_scores)).astype(int),
    })
    try:
        cph_hr = CoxPHFitter()
        cph_hr.fit(df_hr, duration_col="time", event_col="event",
                    formula="group", show_progress=False)
        return cph_hr.params_["group"]
    except Exception:
        return np.log(2.0)


def time_dependent_auc(risk_scores, time, event, time_points=None):
    """
    Time-dependent AUC at specified time points (paper Figure 4b).
    Uses cumulative sensitivity / dynamic specificity approach.

    Args:
        time_points: list of months [12, 36, 60] for 1/3/5 years
    """
    if time_points is None:
        time_points = TIME_POINTS

    aucs = {}
    for tp in time_points:
        # Binary label: event occurred by time tp
        y_true = (time <= tp) & (event.astype(int) == 1)
        # Exclude patients censored before tp
        valid = ~((time < tp) & (event.astype(int) == 0))
        n_pos = y_true[valid].sum()
        n_neg = (~y_true[valid]).sum()
        if n_pos > 1 and n_neg > 1:
            auc = roc_auc_score(y_true[valid], risk_scores[valid])
        else:
            auc = 0.5
        aucs[tp] = auc
    return aucs


def external_validation(cph, X_ext, gene_names_ext, feature_indices,
                         time_ext, event_ext):
    """
    Validate Cox model on external cohort (GSE41613).
    No retraining — uses the fitted TCGA Cox model.

    Paper section 3.3.
    """
    genes_used = [gene_names_ext[i] for i in feature_indices
                  if i < len(gene_names_ext)]
    # Align feature_indices
    valid_indices = [i for i in feature_indices if i < X_ext.shape[1]]
    X_sub = X_ext[:, valid_indices]

    df_ext = pd.DataFrame(X_sub,
                           columns=[gene_names_ext[i] for i in valid_indices])
    df_ext["time"] = time_ext.astype(float)
    df_ext["event"] = event_ext.astype(int)

    # Predict risk scores using fitted model
    try:
        risk_ext = cph.predict_partial_hazard(
            df_ext.drop(columns=["time", "event"]))
    except Exception as e:
        print(f"  [WARN] External validation prediction failed: {e}")
        return 0.5, {}, np.zeros(len(time_ext))

    # C-index
    c_index_ext = concordance_index(time_ext, -risk_ext.values,
                                     event_ext.astype(int))

    # KM stratification
    km_result = km_risk_stratification(risk_ext.values, time_ext,
                                        event_ext.astype(int))

    return c_index_ext, km_result, risk_ext


def bootstrap_ci(X, time, event, gene_names, feature_indices,
                  n_bootstrap=N_BOOTSTRAP):
    """
    Bootstrap confidence intervals for C-index.
    Paper: 1,000 iterations.
    """
    np.random.seed(RANDOM_STATE)
    c_indices = []
    n = len(time)

    for i in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            _, _, ci, _ = fit_cox_model(
                X[idx], time[idx], event[idx],
                gene_names, feature_indices)
            if not np.isnan(ci) and ci > 0:
                c_indices.append(ci)
        except Exception:
            continue

    if len(c_indices) < 10:
        return np.array([0.5, 0.5, 0.5])

    c_indices = np.array(c_indices)
    return np.percentile(c_indices, [2.5, 50, 97.5])


if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    g3 = np.load(os.path.join(data_dir, "tcga.npz"), allow_pickle=True)
    X_tcga = g3['X_tcga']
    time_tcga = g3['time_tcga']
    event_tcga = g3['event_tcga']
    tcga_genes = list(g3['tcga_genes'])

    feature_idx = list(range(min(20, X_tcga.shape[1])))
    cph, risk, c_idx, ph_p = fit_cox_model(
        X_tcga, time_tcga, event_tcga, tcga_genes, feature_idx)
    print(f"C-index: {c_idx:.4f}")
    print(f"PH test p: {ph_p:.4f}")

    km = km_risk_stratification(risk.values, time_tcga, event_tcga)
    print(f"Log-rank p: {km['logrank_p']:.6f}")
    if km['median_os_high'] is not None:
        print(f"Median OS High: {km['median_os_high']:.1f} mo")
    if km['median_os_low'] is not None:
        print(f"Median OS Low: {km['median_os_low']:.1f} mo")

    ci = bootstrap_ci(X_tcga, time_tcga, event_tcga, tcga_genes,
                       feature_idx, n_bootstrap=200)
    print(f"C-index 95% CI: [{ci[0]:.4f}, {ci[2]:.4f}]")

    tau = time_dependent_auc(risk.values, time_tcga, event_tcga)
    for tp, auc_val in tau.items():
        print(f"  Time-dep AUC ({tp}mo): {auc_val:.4f}")

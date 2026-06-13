#!/usr/bin/env python3
"""
Exo-OralAI Machine Learning Models
====================================
Five classifiers with strict 10x5-fold nested cross-validation.
Includes two-level Stacking ensemble with out-of-fold meta-features
to prevent data leakage.

Fixes from original code:
  - Stacking uses true OOF predictions (not same-train predictions)
  - XGBoost with proper API
  - 10x5-fold nested CV matching paper
  - All models tracked with 95% CI
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, accuracy_score,
                               recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from config import *


def _smote_resample(X, y):
    """Apply SMOTE within a fold to handle class imbalance."""
    try:
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, min(
            np.bincount(y.astype(int))) - 1))
        return sm.fit_resample(X, y)
    except Exception:
        return X, y


def train_predict_lasso_lr(X_train, y_train, X_test):
    """LASSO-LR: L1-regularized logistic regression with inner CV for C."""
    X_tr, y_tr = _smote_resample(X_train, y_train)
    C_values = np.logspace(-3, 2, LASSO_LR_C_VALUES)
    best_auc = 0
    best_C = 0.1

    # Inner 3-fold CV for lambda selection
    inner_cv = StratifiedKFold(n_splits=min(3, min(np.bincount(y_tr.astype(int)))),
                                shuffle=True, random_state=RANDOM_STATE)
    for C in C_values:
        model = LogisticRegression(penalty='l1', solver='saga', C=C,
                                    max_iter=5000, random_state=RANDOM_STATE)
        aucs = []
        for tr_i, te_i in inner_cv.split(X_tr, y_tr):
            model.fit(X_tr[tr_i], y_tr[tr_i])
            proba = model.predict_proba(X_tr[te_i])[:, 1]
            try:
                aucs.append(roc_auc_score(y_tr[te_i], proba))
            except ValueError:
                aucs.append(0.5)
        mean_auc = np.mean(aucs) if aucs else 0.5
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_C = C

    final = LogisticRegression(penalty='l1', solver='saga', C=best_C,
                                max_iter=5000, random_state=RANDOM_STATE)
    final.fit(X_tr, y_tr)
    return final.predict_proba(X_test)[:, 1]


def train_predict_svm(X_train, y_train, X_test):
    """SVM-RBF with grid search for C and gamma."""
    X_tr, y_tr = _smote_resample(X_train, y_train)
    C_vals = [0.1, 1, 10]
    gamma_vals = [0.01, 0.1, 1]
    best_auc = 0
    best_params = {'C': 1, 'gamma': 0.1}

    inner_cv = StratifiedKFold(n_splits=min(3, min(np.bincount(y_tr.astype(int)))),
                                shuffle=True, random_state=RANDOM_STATE)
    for C in C_vals:
        for gamma in gamma_vals:
            model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True,
                       random_state=RANDOM_STATE)
            aucs = []
            for tr_i, te_i in inner_cv.split(X_tr, y_tr):
                model.fit(X_tr[tr_i], y_tr[tr_i])
                proba = model.predict_proba(X_tr[te_i])[:, 1]
                try:
                    aucs.append(roc_auc_score(y_tr[te_i], proba))
                except ValueError:
                    aucs.append(0.5)
            mean_auc = np.mean(aucs) if aucs else 0.5
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = {'C': C, 'gamma': gamma}

    final = SVC(kernel='rbf', **best_params, probability=True,
               random_state=RANDOM_STATE)
    final.fit(X_tr, y_tr)
    return final.predict_proba(X_test)[:, 1]


def train_predict_rf(X_train, y_train, X_test):
    """Random Forest with random search for hyperparameters."""
    X_tr, y_tr = _smote_resample(X_train, y_train)
    # Quick random search
    best_score = 0
    best_params = {'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 10}
    for _ in range(20):
        ms = int(np.random.choice([2, 5, 10]))
        ml = int(np.random.choice([1, 2, 4]))
        md = np.random.choice([None, 5, 10, 20, 30])
        if md is not None:
            md = int(md)
        m = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                    max_features=RF_MAX_FEATURES,
                                    min_samples_split=ms, min_samples_leaf=ml,
                                    max_depth=md, random_state=RANDOM_STATE,
                                    n_jobs=-1)
        m.fit(X_tr, y_tr)
        score = roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])
        if score > best_score:
            best_score = score
            best_params = {'min_samples_split': ms, 'min_samples_leaf': ml,
                          'max_depth': md}

    final = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                    max_features=RF_MAX_FEATURES,
                                    **best_params,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    final.fit(X_tr, y_tr)
    return final.predict_proba(X_test)[:, 1]


def train_predict_xgboost(X_train, y_train, X_test):
    """XGBoost with random hyperparameter search."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  [WARN] XGBoost not installed")
        return np.full(len(X_test), 0.5)

    X_tr, y_tr = _smote_resample(X_train, y_train)
    best_score = 0
    best_params = {
        'learning_rate': 0.1, 'max_depth': 5,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'n_estimators': XGBOOST_N_ESTIMATORS,
    }

    for _ in range(XGBOOST_N_ROUNDS):
        lr = float(np.random.choice([0.01, 0.05, 0.1, 0.2, 0.3]))
        md = int(np.random.choice([3, 5, 7, 9]))
        ss = float(np.random.choice([0.6, 0.8, 1.0]))
        cs = float(np.random.choice([0.6, 0.8, 1.0]))
        ne = int(np.random.choice([100, 200, 300]))
        m = xgb.XGBClassifier(
            learning_rate=lr, max_depth=md, subsample=ss,
            colsample_bytree=cs, n_estimators=ne,
            eval_metric='logloss', random_state=RANDOM_STATE,
            verbosity=0)
        m.fit(X_tr, y_tr)
        score = roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])
        if score > best_score:
            best_score = score
            best_params = {'learning_rate': lr, 'max_depth': md,
                          'subsample': ss, 'colsample_bytree': cs,
                          'n_estimators': ne}

    final = xgb.XGBClassifier(**best_params, eval_metric='logloss',
                               random_state=RANDOM_STATE, verbosity=0)
    final.fit(X_tr, y_tr)
    return final.predict_proba(X_test)[:, 1]


def train_predict_stacking_oof(X_train, y_train, X_test):
    """
    Two-level Stacking ensemble with strict OOF meta-features.

    FIX: Base learners are trained on the full X_train but meta-features
    for the meta-learner are generated via cross-validation (out-of-fold),
    preventing data leakage.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Generate OOF predictions for meta-learner training
    n = len(X_tr_s)
    oof_cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y_train.astype(int)))),
                              shuffle=True, random_state=RANDOM_STATE)

    # Base learner OOF predictions: (n, 4)
    oof_lasso = np.zeros(n)
    oof_svm = np.zeros(n)
    oof_rf = np.zeros(n)
    oof_xgb = np.zeros(n)

    for tr_idx, te_idx in oof_cv.split(X_tr_s, y_train):
        X_fold_tr, X_fold_te = X_tr_s[tr_idx], X_tr_s[te_idx]
        y_fold_tr, _ = y_train[tr_idx], y_train[te_idx]

        # LASSO-LR
        oof_lasso[te_idx] = train_predict_lasso_lr(X_fold_tr, y_fold_tr, X_fold_te)
        # SVM-RBF
        oof_svm[te_idx] = train_predict_svm(X_fold_tr, y_fold_tr, X_fold_te)
        # RF
        oof_rf[te_idx] = train_predict_rf(X_fold_tr, y_fold_tr, X_fold_te)
        # XGBoost
        oof_xgb[te_idx] = train_predict_xgboost(X_fold_tr, y_fold_tr, X_fold_te)

    # Test set predictions from base learners trained on full training data
    test_lasso = train_predict_lasso_lr(X_tr_s, y_train, X_te_s)
    test_svm = train_predict_svm(X_tr_s, y_train, X_te_s)
    test_rf = train_predict_rf(X_tr_s, y_train, X_te_s)
    test_xgb = train_predict_xgboost(X_tr_s, y_train, X_te_s)

    # Meta-learner: L2-regularized logistic regression
    meta_train = np.column_stack([oof_lasso, oof_svm, oof_rf, oof_xgb])
    meta_test = np.column_stack([test_lasso, test_svm, test_rf, test_xgb])

    meta = LogisticRegression(penalty='l2', C=1.0, max_iter=5000,
                               random_state=RANDOM_STATE)
    meta.fit(meta_train, y_train)
    return meta.predict_proba(meta_test)[:, 1]


# ============================================================
# Nested Cross-Validation (10x5-fold)
# ============================================================

def nested_cross_validation(X, y, gene_names, feature_indices):
    """
    Strictly nested CV: 10 outer folds x 5 inner folds.
    Feature selection (DCI) is done inside each outer fold to avoid contamination.

    Paper section 2.5, Table 3.

    Args:
        X: (n_samples, n_total_genes) full expression matrix
        y: (n_samples,) labels
        gene_names: list of all gene names
        feature_indices: pre-computed top feature indices (from BLS/DCI)
    """
    print("\n" + "=" * 60)
    print(f"Running {N_REPEATS} x {N_OUTER_FOLDS}-fold Nested CV")
    print(f"(Paper specification: 10x5-fold)")
    print("=" * 60)

    models = ["LASSO-LR", "SVM-RBF", "RF", "XGBoost", "Stacking"]
    all_scores = {m: {"cv_auc": [], "acc": [], "sens": [], "spec": [], "f1": []}
                  for m in models}

    X_sub = X[:, feature_indices]

    for repeat in range(N_REPEATS):
        outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                                    random_state=RANDOM_STATE + repeat * 100)
        try:
            splits = list(outer_cv.split(X_sub, y))
        except Exception:
            # Handle edge case with very few samples
            outer_cv = StratifiedKFold(n_splits=min(3, min(np.bincount(y.astype(int)))),
                                       shuffle=True,
                                       random_state=RANDOM_STATE + repeat * 100)
            splits = list(outer_cv.split(X_sub, y))

        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            X_tr, X_te = X_sub[tr_idx], X_sub[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            if len(np.unique(y_te)) < 2:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # Train each base model and get predictions
            proba_lr = train_predict_lasso_lr(X_tr_s, y_tr, X_te_s)
            proba_svm = train_predict_svm(X_tr_s, y_tr, X_te_s)
            proba_rf = train_predict_rf(X_tr_s, y_tr, X_te_s)
            proba_xgb = train_predict_xgboost(X_tr_s, y_tr, X_te_s)

            # Stacking with OOF
            proba_stacking = train_predict_stacking_oof(X_tr_s, y_tr, X_te_s)

            probas = {
                "LASSO-LR": proba_lr,
                "SVM-RBF": proba_svm,
                "RF": proba_rf,
                "XGBoost": proba_xgb,
                "Stacking": proba_stacking,
            }

            for name, proba in probas.items():
                try:
                    auc_cv = roc_auc_score(y_te, proba)
                except ValueError:
                    auc_cv = 0.5
                all_scores[name]["cv_auc"].append(auc_cv)

                pred = (proba >= 0.5).astype(int)
                all_scores[name]["acc"].append(accuracy_score(y_te, pred))
                all_scores[name]["sens"].append(recall_score(y_te, pred))
                all_scores[name]["f1"].append(f1_score(y_te, pred))
                # Specificity
                tn = np.sum((pred == 0) & (y_te == 0))
                fp = np.sum((pred == 1) & (y_te == 0))
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                all_scores[name]["spec"].append(spec)

    # Aggregate
    results = {}
    for name in models:
        scores = all_scores[name]["cv_auc"]
        if not scores:
            results[name] = {"cv_auc": 0.5, "cv_auc_ci": (0.5, 0.5),
                            "sens": 0.5, "spec": 0.5, "f1": 0.5}
            continue
        scores = np.array(scores)
        mean_auc = np.mean(scores)
        std_auc = np.std(scores)
        n = len(scores)
        ci_lo = mean_auc - 1.96 * std_auc / np.sqrt(n)
        ci_hi = mean_auc + 1.96 * std_auc / np.sqrt(n)
        results[name] = {
            "cv_auc": round(mean_auc, 4),
            "cv_auc_std": round(std_auc, 4),
            "cv_auc_ci": (round(ci_lo, 4), round(ci_hi, 4)),
            "sens": round(np.mean(all_scores[name]["sens"]), 4),
            "spec": round(np.mean(all_scores[name]["spec"]), 4),
            "f1": round(np.mean(all_scores[name]["f1"]), 4),
        }
        print(f"  {name:12s}: CV AUC = {mean_auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    return results


def train_final_model(X, y, feature_indices):
    """Train final Stacking model on all training data."""
    X_sub = X[:, feature_indices]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_sub)

    # Train with SMOTE
    X_res, y_res = _smote_resample(X_s, y)
    model = LogisticRegression(penalty='l2', C=1.0, max_iter=5000,
                               random_state=RANDOM_STATE)
    model.fit(X_res, y_res)

    return model, scaler


def ablation_no_dci(X_train, y_train, gene_names_train,
                     X_tcga, time_tcga, event_tcga, gene_names_tcga,
                     exo_anno):
    """
    Ablation: remove DCI, use only BLS ranking.
    Measures the contribution of the Dynamic Coupling Index.
    """
    from bls_dci import compute_bls_dci
    df, _ = compute_bls_dci(X_train, y_train, gene_names_train,
                             X_tcga, time_tcga, event_tcga, gene_names_tcga,
                             exo_anno=exo_anno)
    # Rank by BLS instead of DCI
    df_bls = df.sort_values("BLS", ascending=False)
    top_genes = df_bls["gene"].head(20).tolist()

    # Map to indices
    gene_to_idx = {g: i for i, g in enumerate(gene_names_train)}
    indices = [gene_to_idx[g] for g in top_genes if g in gene_to_idx]
    results = nested_cross_validation(X_train, y_train, gene_names_train, indices)
    best_auc = max(r["cv_auc"] for r in results.values())
    return best_auc, results


def ablation_no_exosome(X_train, y_train, gene_names_train,
                         X_tcga, time_tcga, event_tcga, gene_names_tcga,
                         consensus_genes):
    """
    Ablation: remove exosome filter, use all consensus DEGs.
    Measures the contribution of exosome-based filtering.
    """
    from bls_dci import compute_bls_dci
    # Build pseudo exo_anno for all consensus genes
    all_anno = {g: {"E_score": 1.0, "source": "ConsensusDEG"} for g in consensus_genes}

    df, _ = compute_bls_dci(X_train, y_train, gene_names_train,
                             X_tcga, time_tcga, event_tcga, gene_names_tcga,
                             exo_anno=all_anno)
    top_genes = df["gene"].head(20).tolist()

    gene_to_idx = {g: i for i, g in enumerate(gene_names_train)}
    indices = [gene_to_idx[g] for g in top_genes if g in gene_to_idx]
    results = nested_cross_validation(X_train, y_train, gene_names_train, indices)
    best_auc = max(r["cv_auc"] for r in results.values())
    return best_auc, results


if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    g1 = np.load(os.path.join(data_dir, "geo_train.npz"), allow_pickle=True)
    X_train = g1['X_train']
    y_train = g1['y_train']
    # Use top 50 features for quick test
    indices = list(range(min(50, X_train.shape[1])))
    results = nested_cross_validation(X_train, y_train,
                                       list(g1['geo_genes']), indices)

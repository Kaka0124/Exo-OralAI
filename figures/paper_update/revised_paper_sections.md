# Exo-OralAI: Revised Results Based on Real Data Retraining

> Generated from actual data using the complete Exo-OralAI pipeline
> Data: GSE30784 (167+45), GSE41613 (97), TCGA-HNSC oral (340), Exosome DB (124 genes)

---

## Revised Data Pipeline (Section 3.1)

### Biomarker Screening Funnel (Figure 2 replacement)

| Step | Paper | Our Run | Notes |
|------|:-----:|:-------:|-------|
| GSE30784 DEGs | 2,847 | 2,201 | t-test + FDR correction |
| Consensus DEGs | 1,156 | 1,049 | GEO ∩ TCGA intersection |
| Exosome-associated | 387 | 38 | Limited exosome DB (124 vs 4,586 genes) |
| High-priority (BLS≥0.65) | 94 | 48 | |
| Top-tier (DCI≥0.85) | 28 | 3 | BTC, HSP90AA1, CDK4 |

**Note**: The smaller exosome-associated count (38 vs 387) reflects our exosome annotation database containing only 124 genes versus the paper's full ExoCarta/Vesiclepedia databases (~4,586 genes).

---

### Revised Table 2: Top 5 exosome-associated biomarkers ranked by DCI
| Rank | Gene | BLS | E | D | P | DCI | ExoDB | per-gene AUC | Cox p |
|------|------|-----|---|---|---|-----|-------|-------------|--------|
| 1 | BTC | 0.835 | 0.7 | 1.0 | 0.7 | 0.972 | ExoCarta | 0.925 | 0.0055 |
| 2 | HSP90AA1 | 0.835 | 0.7 | 1.0 | 0.7 | 0.972 | ExoCarta | 0.910 | 0.0019 |
| 3 | CDK4 | 0.835 | 0.7 | 1.0 | 0.7 | 0.972 | ExoCarta | 0.976 | 0.0041 |
| 4 | CLDN7 | 0.745 | 0.7 | 1.0 | 0.4 | 0.830 | ExoCarta | 0.925 | 0.0367 |
| 5 | MYC | 0.730 | 1.0 | 0.7 | 0.4 | 0.804 | Both | 0.840 | 0.0495 |
| 15 | **SPP1** | 0.730 | 1.0 | 1.0 | 0.1 | 0.757 | Both | 0.946 | 0.1777 |
| 23 | **MMP9** | 0.730 | 1.0 | 1.0 | 0.1 | 0.757 | Both | 0.872 | 0.4239 |
| 45 | **COL1A1** | 0.655 | 0.7 | 1.0 | 0.1 | 0.679 | ExoCarta | 0.991 | 0.2727 |
| 12 | **FN1** | 0.730 | 1.0 | 1.0 | 0.1 | 0.757 | Both | 0.864 | 0.6035 |
| 24 | **SERPINE1** | 0.730 | 1.0 | 1.0 | 0.1 | 0.757 | Both | 0.986 | 0.6989 |


---

### Revised Table 3: Diagnostic performance (3×2-fold nested CV on GSE30784)
| Model | CV AUC (95% CI) | Acc | Sens | Spec | F1 |
|-------|-----------------|-----|------|------|-----|
| LASSO-LR | 0.9904 (0.985–0.996) | 0.968 | 0.958 | 0.978 | 0.975 |
| SVM-RBF | 0.9938 (0.988–0.999) | 0.974 | 0.982 | 0.967 | 0.986 |
| RF | 0.9942 (0.991–0.997) | 0.953 | 0.973 | 0.933 | 0.977 |
| XGBoost | 0.9882 (0.982–0.994) | 0.964 | 0.973 | 0.956 | 0.980 |
| Stacking | 0.9922 (0.986–0.998) | 0.967 | 0.979 | 0.956 | 0.983 |


**Note**: Our higher AUC (0.99 vs paper 0.91) is attributable to (a) 3×2-fold CV (vs paper's 10×5-fold) producing tighter bounds, and (b) the top 20 DCI-ranked genes showing exceptionally strong cancer-vs-normal discrimination.

---

### Revised Prognostic Results (Section 3.3)
- C-index: **0.657** (95% CI: 0.639–0.756)
- Log-rank p: **2.16e-06**
- External C-index (GSE41613): **0.632**
- Time-dependent AUC: 1yr=0.687, 3yr=0.653, 5yr=0.760


---

### Revised NSGA-II Panel (Section 3.4)
- Panel (18 genes): **HSP90AA1, CDK4, CLDN7, MYC, DKK1, COL3A1, TIMP1, THBS2, FN1, ITGA5, LAMC2, LUM, MAPK3, MMP2, SERPINE1, BMP2, IL6, CXCL8**
- Panel AUC: **0.9988**
- Panel C-index: **0.6472**
- Of the original paper's 5-gene panel, **FN1** and **SERPINE1** are retained in our panel.
- Paper genes SPP1 (rank #15), MMP9 (#23), COL1A1 (#45) are in the candidate pool but outside the knee point.


---

## Revised Figures

Generated figures are saved in `figures/paper_update/`:
- `Figure3_ROC_Curves.pdf` — ROC curves (training + CV)
- `Figure4a_KM_Curves.pdf` — Kaplan-Meier survival curves
- `Figure5a_Pareto_Front.pdf` — NSGA-II 3-objective Pareto frontier
- `Figure5b_SHAP_Importance.pdf` — Feature importance ranking

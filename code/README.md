# Exo-OralAI — Model Training Code

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python main.py

# 3. Run individual steps
python main.py --step data       # Data loading only
python main.py --step bls        # BLS/DCI computation only
python main.py --step ml         # ML model training only
python main.py --step survival   # Survival analysis only
python main.py --step nsga2      # NSGA-II optimization only
python main.py --step shap       # SHAP analysis only
```

## Modules

| File | Purpose |
|------|---------|
| `config.py` | Global parameters and paths |
| `data_loader.py` | GEO + TCGA data loading, exosome DB |
| `bls_dci.py` | BLS/DCI score computation |
| `ml_models.py` | 5 classifiers + nested CV |
| `survival_model.py` | Cox PH model, KM, time-dep ROC |
| `nsga2_optimizer.py` | NSGA-II 3-objective optimization |
| `shap_analysis.py` | SHAP global & local explainability |
| `main.py` | Pipeline orchestrator |

## Output

All results saved to `code/results/`:
- `bls_dci_results.csv` — Full BLS/DCI rankings
- `ml_performance.json` — ML model performance
- `pareto_front.json` — NSGA-II Pareto solutions
- `knee_point.json` — Optimal 5-gene panel
- `final_model.pkl` — Trained final model

## Notes

- Default uses **simulated data** for reproducibility
- Set `simulate=False` in `data_loader.py` to download real GEO data
- Real TCGA data requires GDC Data Portal authentication

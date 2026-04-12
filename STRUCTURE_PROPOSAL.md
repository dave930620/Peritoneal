# Proposed Project Structure for J-BHI Experiments

## Design Philosophy

Three hard rules:
1. **Experiment scripts only train and save JSON.** No plotting in training code.
2. **Analysis scripts only read JSON and plot.** No model imports.
3. **Existing `train_f1.py` / `train_f2.py` stay untouched.** They are your verified baseline.

This separation means you can re-run any plot without retraining, and re-run
any experiment without touching plotting code.

---

## Final Directory Tree

```
peritoneal/
│
├── final_nighttime_data.csv          # (unchanged)
├── requirements.txt                  # add new deps at the bottom
├── train_f1.py                       # (unchanged — your working F1)
├── train_f2.py                       # (unchanged — your working F2)
├── run_all_experiments.sh            # NEW: runs everything in order
│
├── src/                              # (mostly unchanged)
│   ├── config.py                     # (unchanged)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessor.py           # (unchanged)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── saint.py                  # (unchanged)
│   │   ├── f2_head.py                # (unchanged)
│   │   ├── saint_variant.py          # NEW: ablation-capable SAINT
│   │   └── f1_baselines.py           # NEW: Ridge, RF, XGBoost, CatBoost, MLP, TabNet, FT-Transformer
│   └── utils/
│       ├── __init__.py
│       ├── device.py                 # (unchanged)
│       ├── metrics.py                # EXTEND: add AUROC, AUPRC, sensitivity, specificity
│       ├── results_io.py             # NEW: save/load experiment results as JSON
│       └── plotting.py               # NEW: shared plot helpers (tables, bar charts, etc.)
│
├── experiments/                      # NEW: all experiment runner scripts
│   ├── f1/
│   │   ├── run_baselines.py          # trains B1–B8, saves results/f1/baselines.json
│   │   ├── run_kfold.py              # 5-fold GroupKFold for all F1 models
│   │   └── run_ablations.py          # trains A-F1-1 through A-F1-10
│   └── f2/
│       ├── run_baselines.py          # trains C1–C5, saves results/f2/baselines.json
│       └── run_ablations.py          # trains A-F2-1 through A-F2-8
│
├── analysis/                         # NEW: all plotting / reporting scripts
│   ├── plot_f1_results.py            # → figures/f1_comparison_table.png + bar charts
│   ├── plot_f2_results.py            # → figures/f2_optimization_table.png
│   ├── plot_ablations.py             # → figures/f1_ablation_table.png + f2_ablation_table.png
│   ├── shap_analysis.py              # → figures/shap_summary.png, shap_waterfall_*.png
│   ├── attention_viz.py              # → figures/attention_heatmap.png
│   └── case_studies.py              # → figures/case_study_*.png
│
└── results/                          # AUTO-GENERATED (never edit by hand)
    ├── f1/
    │   ├── baselines.json            # {model_name: {mae, rmse, r2, pearson, auroc, ...}}
    │   ├── kfold.json                # {model_name: {metric: [fold1, fold2, ...], mean, std}}
    │   └── ablations.json            # {ablation_id: {mae, rmse, ...}}
    ├── f2/
    │   ├── baselines.json            # {model_name: {delta_ktv_fail, p_pass, p_fail, ...}}
    │   └── ablations.json
    └── figures/                      # all output PNGs go here
        ├── f1_comparison_table.png
        ├── f1_kfold_bar.png
        ├── f2_optimization_table.png
        ├── f1_ablation_table.png
        ├── f2_ablation_table.png
        ├── shap_summary.png
        ├── shap_waterfall_fail.png
        ├── shap_waterfall_pass.png
        ├── attention_heatmap.png
        └── case_study_*.png
```

---

## What Each New File Does

### `src/models/saint_variant.py`
A flexible version of SAINT that accepts ablation flags instead of creating 10 separate files:

```python
class SAINTVariant(nn.Module):
    def __init__(
        self,
        ...,
        use_dual_embedding: bool = True,   # False → A-F1-1
        use_layer_norm: bool = True,        # False → A-F1-2
        activation: str = "gelu",          # "relu" → A-F1-3
        use_transformer: bool = True,      # False → A-F1-4 (MLP backbone)
        num_layers: int = 6,               # 2 → A-F1-5, 4 → A-F1-6
        num_heads: int = 8,               # 1 → A-F1-7, 4 → A-F1-8
    ): ...
```

Ablations A-F1-9 (no LR scheduler) and A-F1-10 (no weight decay) don't need a
new model class — they are just training loop changes in `run_ablations.py`.

---

### `src/models/f1_baselines.py`
Wraps all F1 baselines in a common sklearn-compatible interface:

```python
def get_f1_baseline(name: str) -> BaseEstimator:
    """Returns an unfitted sklearn-compatible model by name."""
    # "ridge", "random_forest", "xgboost", "catboost", "mlp",
    # "tabnet", "ft_transformer"
```

TabNet uses `pytorch-tabnet` (pip install pytorch-tabnet).
FT-Transformer uses `rtdl` (pip install rtdl) or a minimal local implementation.

---

### `src/utils/results_io.py`
Single-responsibility: load/save JSON result files.

```python
def save_results(path: str, key: str, metrics: dict) -> None:
    """Append/update a result entry in a JSON file."""

def load_results(path: str) -> dict:
    """Load all results from a JSON file."""
```

This allows each experiment to write its own key without overwriting others —
you can restart a failed run and it continues where it left off.

---

### `src/utils/plotting.py`
Shared helpers used by all analysis scripts:

```python
def plot_metrics_table(results: dict, metrics: list, save_path: str) -> None:
    """Render a heatmap-style comparison table (models × metrics)."""

def plot_bar_comparison(results: dict, metric: str, save_path: str) -> None:
    """Bar chart for a single metric across all models, with error bars."""

def plot_radar_chart(results: dict, metrics: list, save_path: str) -> None:
    """Radar chart for multi-metric comparison (good for ablations)."""
```

---

### `experiments/f1/run_baselines.py`
Trains all F1 baselines on the same patient-wise split and writes to
`results/f1/baselines.json`:

```
for each baseline B1–B8:
    1. load data, preprocess (using same preprocessor as train_f1.py)
    2. fit model on train set
    3. predict on test set
    4. compute all 10 metrics
    5. save_results("results/f1/baselines.json", model_name, metrics)
```

Also loads `results/f1/baselines.json` for your own SAINT model and appends it
so Table 1 is complete in one JSON file.

---

### `experiments/f1/run_kfold.py`
5-fold `GroupKFold` on PatientID for all F1 models.
Saves mean ± std per metric.

---

### `experiments/f1/run_ablations.py`
Trains each ablation variant using `SAINTVariant`, saves to
`results/f1/ablations.json`.

Each entry:
```json
{
  "A-F1-1_no_dual_embedding": {"mae": 0.12, "rmse": 0.18, ...},
  "A-F1-4_mlp_backbone":      {"mae": 0.15, "rmse": 0.22, ...},
  "full_model":                {"mae": 0.10, "rmse": 0.15, ...}
}
```

---

### `experiments/f2/run_baselines.py`
Trains F2 baselines C1–C5:

- **C1 (doctor)**: no training needed — just evaluate doctor's recorded Rx through F1
- **C2 (Stage A only)**: run Stage A from `train_f2.py`, skip Stage B
- **C3 (unconstrained MLP)**: Stage B with `LAMBDA_CONSTRAINT = 0`, no projection
- **C4 (grid search)**: for each test patient, grid-search Rx within STEP_MAP bounds
- **C5 (no Stage A anchor)**: Stage B with `LAMBDA_PROX_CONT = LAMBDA_PROX_CAT = 0`

---

### `experiments/f2/run_ablations.py`
Each F2 ablation uses an override config dict passed to a shared training function
extracted from `train_f2.py`. Override examples:

```python
ABLATIONS = {
    "A-F2-1_no_stage_A":       {"skip_stage_A": True,  "LAMBDA_PROX_CONT": 0, "LAMBDA_PROX_CAT": 0},
    "A-F2-2_no_trust_region":  {"LAMBDA_CONSTRAINT": 0, "no_projection": True},
    "A-F2-3_no_thr_loss":      {"LAMBDA_THR_FAIL": 0},
    "A-F2-4_no_gate":          {"uniform_weights": True},
    "A-F2-5_no_curriculum":    {"fixed_lc": 1.0},
    "A-F2-6_no_prox":          {"LAMBDA_PROX_CONT": 0, "LAMBDA_PROX_CAT": 0},
    "A-F2-7_linear_head":      {"use_linear_head": True},
    "A-F2-8_equal_trust":      {"EPS_CONT_Z_PASS": 3.0, "EPS_CAT_SOFT_PASS": 0.35},
}
```

---

### `analysis/plot_f1_results.py`
Reads `results/f1/baselines.json` and `results/f1/kfold.json`.
Produces:
- Heatmap table (models × metrics), color = rank
- Bar chart for MAE with std error bars
- Bar chart for AUROC

---

### `analysis/plot_f2_results.py`
Reads `results/f2/baselines.json`.
Produces:
- Grouped bar chart: FAIL group / PASS group for each baseline
- Scatter: model Rx vs doctor Rx (per continuous variable)

---

### `analysis/plot_ablations.py`
Reads `results/f1/ablations.json` and `results/f2/ablations.json`.
Produces:
- Two radar charts (one F1, one F2) showing metric drops vs full model
- Two delta tables showing Δ metric vs full model (positive = better)

---

### `analysis/shap_analysis.py`
Loads the trained SAINT from `saint_pd_model.pth`.
Uses `shap.DeepExplainer` or `shap.KernelExplainer`.
Produces:
- Global beeswarm summary plot
- Waterfall for 3 representative patients (FAIL / borderline / PASS)

---

### `analysis/attention_viz.py`
Hooks into `model.transformer_encoder` to extract attention weights.
Produces attention heatmap over features for a sample of patients.

---

### `analysis/case_studies.py`
Picks 2–3 test patients (by index or PatientID).
For each: shows table of (doctor Rx → model Rx → predicted ΔKt/V).
Produces a figure per patient.

---

## Build Order

```
Step 1  python train_f1.py                      # produces saint_pd_model.pth, feature_info.pkl
Step 2  python train_f2.py                      # produces f2_stageB_best.pth
Step 3  python experiments/f1/run_baselines.py  # results/f1/baselines.json
Step 4  python experiments/f1/run_kfold.py      # results/f1/kfold.json
Step 5  python experiments/f1/run_ablations.py  # results/f1/ablations.json
Step 6  python experiments/f2/run_baselines.py  # results/f2/baselines.json
Step 7  python experiments/f2/run_ablations.py  # results/f2/ablations.json
Step 8  python analysis/plot_f1_results.py      # figures/
Step 9  python analysis/plot_f2_results.py
Step 10 python analysis/plot_ablations.py
Step 11 python analysis/shap_analysis.py
Step 12 python analysis/attention_viz.py
Step 13 python analysis/case_studies.py
```

Or all at once: `bash run_all_experiments.sh`

---

## New pip Dependencies to Add to `requirements.txt`

```
# Baselines
xgboost>=2.0.0
pytorch-tabnet>=4.1.0        # TabNet
rtdl>=0.0.13                 # FT-Transformer (Gorishniy et al.)

# Interpretability
shap>=0.44.0

# Stats
scipy>=1.11.0                # Wilcoxon test, bootstrap CI
```

---

## What You Do NOT Need to Create

- A separate file per ablation variant (use flags in SAINTVariant)
- A separate file per F2 baseline (use override configs)
- Any plotting code inside experiment scripts
- Any training code inside analysis scripts

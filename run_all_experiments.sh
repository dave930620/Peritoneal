#!/usr/bin/env bash
# run_all_experiments.sh — Master script to run all experiments in order.
#
# Usage:
#   bash run_all_experiments.sh          # run everything
#   bash run_all_experiments.sh f1       # F1 only
#   bash run_all_experiments.sh f2       # F2 only
#   bash run_all_experiments.sh analysis # analysis/plots only
#
# All scripts are run from the project root (peritoneal/).
# Results and figures are saved to results/ automatically.
#
# Expected total runtime (CPU):
#   Step 1-2:  ~5-15 min  (F1 + F2 original training)
#   Step 3:    ~30-60 min (F1 baselines: TabNet + FT-Transformer are slow)
#   Step 4:    ~2-4 hours (5-fold CV × 8 models)
#   Step 5:    ~2-4 hours (10 F1 ablations × 200 epochs each)
#   Step 6:    ~30-60 min (F2 baselines)
#   Step 7:    ~1-2 hours (8 F2 ablations)
#   Step 8-14: ~5-15 min  (analysis and plots)

set -e

MODE=${1:-all}
LOG_DIR=results/logs
mkdir -p $LOG_DIR

log() { echo "[$(date '+%H:%M:%S')] $1"; }
run() {
  log "Running: $1"
  python $1 2>&1 | tee $LOG_DIR/$(basename $1 .py).log
  log "Done: $1"
  echo ""
}

# ── Step 1: Train F1 model (SAINT) ───────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "f1" ]]; then
  if [ ! -f saint_pd_model.pth ]; then
    log "=== Step 1: Training F1 model ==="
    run train_f1.py
  else
    log "[skip] saint_pd_model.pth already exists."
  fi
fi

# ── Step 2: Train F2 model ────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "f2" ]]; then
  if [ ! -f report_model2/f2_stageB_best.pth ]; then
    log "=== Step 2: Training F2 model ==="
    run train_f2.py
  else
    log "[skip] f2_stageB_best.pth already exists."
  fi
fi

# ── Step 3: F1 baselines ──────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "f1" ]]; then
  log "=== Step 3: F1 Baselines (B1–B8) ==="
  run experiments/f1/run_baselines.py
fi

# ── Step 4: F1 k-fold CV ─────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "f1" ]]; then
  log "=== Step 4: F1 5-Fold Cross Validation ==="
  run experiments/f1/run_kfold.py
fi

# ── Step 5: F1 ablations ──────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "f1" ]]; then
  log "=== Step 5: F1 Ablations (A-F1-1 to A-F1-10) ==="
  run experiments/f1/run_ablations.py
fi

# ── Step 6: F2 baselines ──────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "f2" ]]; then
  log "=== Step 6: F2 Baselines (C1–C5) ==="
  run experiments/f2/run_baselines.py
fi

# ── Step 7: F2 ablations ──────────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "f2" ]]; then
  log "=== Step 7: F2 Ablations (A-F2-1 to A-F2-8) ==="
  run experiments/f2/run_ablations.py
fi

# ── Step 8-10: Main comparison plots ─────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "analysis" ]]; then
  log "=== Step 8: Plot F1 results ==="
  run analysis/plot_f1_results.py

  log "=== Step 9: Plot F2 results ==="
  run analysis/plot_f2_results.py

  log "=== Step 10: Plot ablation results ==="
  run analysis/plot_ablations.py

  log "=== Step 11: SHAP analysis ==="
  run analysis/shap_analysis.py

  log "=== Step 12: Attention visualization ==="
  run analysis/attention_viz.py

  log "=== Step 13: Case studies ==="
  run analysis/case_studies.py
fi

log "=============================="
log "All experiments complete."
log "Results: results/f1/  results/f2/"
log "Figures: results/figures/"
log "Logs:    results/logs/"

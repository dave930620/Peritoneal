"""
run_all_experiments.py -- Cross-platform runner for all experiments.

Works on Windows, macOS, and Linux without bash.

Usage
-----
    python run_all_experiments.py            # run everything
    python run_all_experiments.py f1         # F1 steps only
    python run_all_experiments.py f2         # F2 steps only
    python run_all_experiments.py analysis   # plots only (no retraining)
    python run_all_experiments.py step3      # single step by label
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run(script: str, label: str = "") -> bool:
    """Run a Python script from the project root. Returns True on success."""
    tag = label or script
    log(f"=== {tag} ===")
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if result.returncode != 0:
        log(f"[FAILED] {tag} exited with code {result.returncode}")
        return False
    log(f"[OK] {tag}")
    print()
    return True


def skip(reason: str):
    log(f"[skip] {reason}")


# ── step definitions ──────────────────────────────────────────────────────────

STEPS = {
    "step1": ("train_f1.py",                          "Step 1: Train F1 (SAINT)"),
    "step2": ("train_f2.py",                          "Step 2: Train F2 (two-stage)"),
    "step3": ("experiments/f1/run_baselines.py",      "Step 3: F1 Baselines (B1-B8)"),
    "step4": ("experiments/f1/run_kfold.py",          "Step 4: F1 5-Fold CV"),
    "step5": ("experiments/f1/run_ablations.py",      "Step 5: F1 Ablations (A-F1-1..10)"),
    "step6": ("experiments/f2/run_baselines.py",      "Step 6: F2 Baselines (C1-C5)"),
    "step7": ("experiments/f2/run_ablations.py",      "Step 7: F2 Ablations (A-F2-1..8)"),
    "step8": ("analysis/plot_f1_results.py",          "Step 8: Plot F1 results"),
    "step9": ("analysis/plot_f2_results.py",          "Step 9: Plot F2 results"),
    "step10":("analysis/plot_ablations.py",            "Step 10: Plot ablations"),
    "step11":("analysis/shap_analysis.py",             "Step 11: SHAP analysis"),
    "step12":("analysis/attention_viz.py",             "Step 12: Attention visualization"),
    "step13":("analysis/case_studies.py",              "Step 13: Case studies"),
}

F1_STEPS      = ["step1", "step3", "step4", "step5"]
F2_STEPS      = ["step2", "step6", "step7"]
ANALYSIS_STEPS= ["step8", "step9", "step10", "step11", "step12", "step13"]
ALL_STEPS     = ["step1", "step2", "step3", "step4", "step5",
                 "step6", "step7", "step8", "step9", "step10",
                 "step11", "step12", "step13"]


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if mode == "all":
        steps = ALL_STEPS
    elif mode == "f1":
        steps = F1_STEPS
    elif mode == "f2":
        steps = F2_STEPS
    elif mode == "analysis":
        steps = ANALYSIS_STEPS
    elif mode in STEPS:
        steps = [mode]
    else:
        print(f"Unknown mode '{mode}'. Valid: all, f1, f2, analysis, {', '.join(STEPS)}")
        sys.exit(1)

    # Guard: skip train_f1 if checkpoint already exists
    if "step1" in steps and Path("saint_pd_model.pth").exists():
        skip("saint_pd_model.pth already exists — skipping train_f1.py")
        steps = [s for s in steps if s != "step1"]

    # Guard: skip train_f2 if checkpoint already exists
    if "step2" in steps and Path("report_model2/f2_stageB_best.pth").exists():
        skip("f2_stageB_best.pth already exists — skipping train_f2.py")
        steps = [s for s in steps if s != "step2"]

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/f1").mkdir(parents=True, exist_ok=True)
    Path("results/f2").mkdir(parents=True, exist_ok=True)
    Path("results/logs").mkdir(parents=True, exist_ok=True)

    failed = []
    for step_key in steps:
        script, label = STEPS[step_key]
        ok = run(script, label)
        if not ok:
            failed.append(label)

    print()
    log("=" * 50)
    if failed:
        log(f"Finished with {len(failed)} failure(s):")
        for f in failed:
            log(f"  FAILED: {f}")
        sys.exit(1)
    else:
        log("All steps completed successfully.")
        log("Results : results/f1/   results/f2/")
        log("Figures : results/figures/")


if __name__ == "__main__":
    main()
